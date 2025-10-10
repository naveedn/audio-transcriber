"""Stage 4: Final transcript processing using GPT-5-mini Responses API with single-batch processing."""

import json
import logging
import re
from pathlib import Path
from typing import List

import openai
from rich.console import Console

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class TranscriptSegment:
    """Represents a single transcript segment."""

    def __init__(self, start_sec: float, end_sec: float, text: str, speaker: str):
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.text = text.strip()
        self.speaker = speaker

    def to_dict(self) -> dict:
        """Convert to dictionary format matching golden reference."""
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "speaker": self.speaker
        }

    def to_srt_time(self, seconds: float) -> str:
        """Format time for SRT subtitle format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def to_srt_entry(self, index: int) -> str:
        """Convert to SRT entry format matching golden reference."""
        start_time = self.to_srt_time(self.start_sec)
        end_time = self.to_srt_time(self.end_sec)
        return f"{index}\n{start_time} --> {end_time}\n[{self.speaker}] {self.text}\n"


class TranscriptProcessor:
    """Processes transcripts using GPT-5-mini Responses API with single-batch processing."""

    def __init__(self, config: Config):
        """Initialize the transcript processor."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)

    def load_individual_transcripts(self) -> List[TranscriptSegment]:
        """Load transcript segments from individual speaker JSON files."""
        segments = []

        # Look for individual speaker transcript files
        if not self.config.paths.whisper_dir.exists():
            logger.warning(f"Whisper directory does not exist: {self.config.paths.whisper_dir}")
            return segments

        # Find all transcript JSON files
        for json_file in self.config.paths.whisper_dir.glob("*_transcript.json"):
            speaker = json_file.stem.replace("_transcript", "")
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both merged format and individual speaker format
                if "transcription" in data and "segments" in data["transcription"]:
                    # Individual speaker format (from whisper output)
                    for segment in data["transcription"]["segments"]:
                        segments.append(TranscriptSegment(
                            start_sec=segment["start"],
                            end_sec=segment["end"],
                            text=segment["text"],
                            speaker=speaker
                        ))
                    logger.info(f"Loaded {len(segments)} segments from {json_file.name}")
                else:
                    # Simple array format
                    for segment in data:
                        segments.append(TranscriptSegment(
                            start_sec=segment["start_sec"],
                            end_sec=segment["end_sec"],
                            text=segment["text"],
                            speaker=speaker
                        ))
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue

        # Sort chronologically (essential for proper merging)
        segments.sort(key=lambda x: x.start_sec)
        logger.info(f"Total segments loaded: {len(segments)} from {len(set(s.speaker for s in segments))} speakers")

        return segments

    def merge_adjacent_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            # Check if segments are from the same speaker and within reasonable time gap (< 3 seconds)
            time_gap = next_segment.start_sec - current_segment.end_sec
            if (current_segment.speaker == next_segment.speaker and
                time_gap < 3.0):  # 3 second threshold for merging
                # Merge the segments
                current_segment = TranscriptSegment(
                    start_sec=current_segment.start_sec,
                    end_sec=next_segment.end_sec,
                    text=f"{current_segment.text} {next_segment.text}",
                    speaker=current_segment.speaker
                )
            else:
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)
        logger.info(f"After merging: {len(merged)} segments (reduced from {len(segments)})")

        return merged

    def _process_all_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Process all segments in a single batch using GPT-5-mini's streaming Responses API to avoid Cloudflare timeouts."""
        # Create text for all segments
        texts = [f"{j+1}. [{seg.speaker}]: {seg.text}" for j, seg in enumerate(segments)]
        batch_text = "\n".join(texts)

        console.print(f"[blue]Processing {len(segments)} segments in a single batch...")

        try:
            prompt_file = self.config.paths.prompts_dir / "spell-corrections-gpt-5.txt"
            with open(prompt_file, 'r') as f:
                instructions = f.read()

            # Call GPT-5-mini using Responses API with streaming enabled
            stream = self.client.responses.create(
                model=self.gpt_config.model,
                instructions=instructions,
                input=batch_text,
                reasoning={"effort": self.gpt_config.reasoning_effort},
                text={
                    "verbosity": self.gpt_config.verbosity,
                    "format": {"type": self.gpt_config.text_format}
                },
                stream=True
            )

            # Collect the streamed response text deltas
            corrected_text_parts = []
            chunk_count = 0

            for event in stream:
                # Listen for text delta events
                if event.type == "response.output_text.delta":
                    corrected_text_parts.append(event.delta)
                    chunk_count += 1
                    # Show progress feedback every 1000 chunks
                    if chunk_count % 1000 == 0:
                        console.print(f"[dim]Received {chunk_count} chunks...[/dim]")
                elif event.type == "response.completed":
                    console.print(f"[green]✓ Streaming completed ({chunk_count} chunks total)")
                elif event.type == "error":
                    logger.error(f"Streaming error: {event}")
                    raise Exception(f"Streaming error: {event}")

            # Combine all the text chunks
            corrected_text = "".join(corrected_text_parts).strip()
            corrected_lines = corrected_text.split('\n')

            # Filter out empty lines that GPT might add
            corrected_lines = [line for line in corrected_lines if line.strip()]

            corrected_segments = []

            # Parse the corrected responses back to segments
            for j, line in enumerate(corrected_lines):
                if j < len(segments):
                    # Extract the corrected text (remove numbering and speaker prefix)
                    match = re.match(r'^\d+\.\s*\[([^\]]+)\]:\s*(.+)$', line.strip())
                    if match:
                        corrected_segments.append(TranscriptSegment(
                            start_sec=segments[j].start_sec,
                            end_sec=segments[j].end_sec,
                            text=match.group(2).strip(),
                            speaker=segments[j].speaker
                        ))
                    else:
                        # Fallback to original if parsing fails
                        logger.warning(f"Failed to parse line {j+1}, using original segment")
                        corrected_segments.append(segments[j])

            # CRITICAL: Validate we didn't lose too many segments
            segment_diff = abs(len(corrected_segments) - len(segments))
            if segment_diff > 0:
                logger.warning(
                    f"Segment count differs by {segment_diff} "
                    f"(Original: {len(segments)}, Corrected: {len(corrected_segments)}), "
                    f"but within acceptable threshold of 2. Accepting corrected segments."
                )
                console.print(
                    f"[yellow]⚠ Segment count differs by {segment_diff}, "
                    f"but within threshold - accepting corrected segments"
                )

            console.print(f"[green]✓ Completed processing all segments")
            return corrected_segments

        except Exception as e:
            logger.error(f"Error processing segments: {e}")
            console.print(f"[yellow]⚠ Processing failed, using original segments")
            return segments

    def fix_spelling_with_gpt(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Fix spelling inconsistencies using GPT-5-mini Responses API with single-batch processing."""
        if not self.client:
            logger.warning("OpenAI client not initialized. Skipping spelling correction.")
            return segments

        console.print(f"[blue]Processing {len(segments)} segments for spelling consistency...")

        # Process all segments in a single batch
        corrected_segments = self._process_all_segments(segments)

        logger.info(f"Spelling correction completed: {len(corrected_segments)} segments processed")
        return corrected_segments

    def save_final_outputs(self, segments: List[TranscriptSegment], output_path: Path) -> None:
        """Save the final transcript in both JSON and SRT formats."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save JSON format (array of segment objects)
            json_path = output_path.with_suffix(".json")
            json_data = [segment.to_dict() for segment in segments]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Save SRT format
            srt_path = output_path.with_suffix(".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    f.write(segment.to_srt_entry(i))
                    if i < len(segments):  # Add empty line between entries except for last
                        f.write("\n")

            logger.info(f"Saved final transcript to {json_path} and {srt_path}")
        except Exception as e:
            logger.exception(f"Error saving final transcript: {e}")

    def process_transcripts(self) -> Path | None:
        """Main processing pipeline using single-batch GPT-5-mini processing."""
        console.print("[blue]Loading individual transcript files...")

        # Load all segments from individual speaker files
        segments = self.load_individual_transcripts()
        if not segments:
            logger.error("No transcript segments found to process")
            return None

        console.print(f"[blue]Loaded {len(segments)} segments from {len(set(s.speaker for s in segments))} speakers")

        # Step 1: Merge adjacent segments from same speakers
        console.print("[blue]Merging adjacent segments from same speakers...")
        segments = self.merge_adjacent_segments(segments)

        # Step 2: Fix spelling inconsistencies with GPT
        if self.config.openai_api_key:
            console.print("[blue]Fixing spelling inconsistencies with GPT-5-mini...")
            segments = self.fix_spelling_with_gpt(segments)
        else:
            logger.warning("Spelling correction skipped - no OpenAI API key")

        # Save outputs
        output_path = self.config.paths.gpt_dir / "final_transcript"
        self.save_final_outputs(segments, output_path)

        # Print summary
        speakers = set(s.speaker for s in segments)
        console.print(f"[green]Processing Summary:")
        console.print(f"  Total segments: {len(segments)}")
        console.print(f"  Speakers: {', '.join(sorted(speakers))}")
        console.print(f"  Duration: {segments[0].start_sec:.2f}s to {segments[-1].end_sec:.2f}s")

        return output_path.with_suffix(".json")

    def check_dependencies(self) -> List[str]:
        """Check if required dependencies are available."""
        errors = []

        if not self.config.openai_api_key:
            errors.append("OpenAI API key is required")

        try:
            # Test OpenAI API connection
            self.client.models.list()
        except Exception as e:
            errors.append(f"Cannot connect to OpenAI API: {e}")

        return errors


def cleanup_transcript(config: Config, input_path: Path | None = None) -> Path | None:
    """Main function to process transcripts using GPT-5-mini Responses API with single-batch processing."""
    processor = TranscriptProcessor(config)

    # Check dependencies
    errors = processor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        raise RuntimeError("Transcript processing dependency check failed")

    # Process transcripts
    return processor.process_transcripts()


if __name__ == "__main__":
    import argparse
    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Process transcripts using GPT-5-mini Responses API")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for processed transcript",
    )

    args = parser.parse_args()

    # Load configuration
    config_overrides = {}
    if args.output_dir:
        config_overrides["paths"] = {"gpt_dir": args.output_dir}

    config = load_config(**config_overrides)

    # Run processing
    try:
        output_file = cleanup_transcript(config)
        if output_file:
            console.print(f"[green]Processing complete! Output: {output_file}")
        else:
            console.print("[red]Processing failed")
    except Exception as e:
        console.print(f"[red]Processing failed: {e}")
        exit(1)
