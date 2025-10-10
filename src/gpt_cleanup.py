"""Stage 4: Final transcript processing using the proven working approach."""

import json
import logging
import re
import time
from datetime import datetime
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
    """Processes transcripts using the proven working approach from the reference implementation."""

    def __init__(self, config: Config):
        """Initialize the transcript processor."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)

    def load_individual_transcripts(self) -> List[TranscriptSegment]:
        """Load transcript segments from individual speaker JSON files (like working version)."""
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
        """Merge adjacent segments from the same speaker (exactly like working version)."""
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

    def fix_spelling_with_gpt(self, segments: List[TranscriptSegment], batch_size: int = 100) -> List[TranscriptSegment]:
        """Fix spelling inconsistencies using GPT-4o-mini with large batches."""
        if not self.client:
            logger.warning("OpenAI client not initialized. Skipping spelling correction.")
            return segments

        console.print(f"[blue]Processing {len(segments)} segments for spelling consistency...")
        corrected_segments = []

        # Process in large batches for efficiency
        total_batches = (len(segments) + batch_size - 1) // batch_size
        console.print(f"[blue]Split into {total_batches} batches of up to {batch_size} segments each")

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Create text batch for processing
            texts = [f"{j+1}. [{seg.speaker}]: {seg.text}" for j, seg in enumerate(batch)]
            batch_text = "\n".join(texts)

            console.print(f"[blue]Processing batch {batch_num}/{total_batches} ({len(batch)} segments)...")

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model proven to work
                    messages=[
                        {
                            "role": "system",
                            "content": """You are editing a transcript. Your tasks:

1. Fix spelling errors and typos while preserving the conversational tone
2. Ensure consistent spelling of proper nouns (names, places, etc.)
3. Remove hallucinations in whisper output, where hallucinations are defined as long repeated strings (often written in caps) that would be infeasible to occur within a timeframe of a few seconds.
4. Keep the same format: "[speaker]: text"
5. Do not alter the meaning or add/remove any content except for hallucination removal as defined above
6. Maintain the original speaker labels.

Focus especially on:
- Character names and proper nouns
- Common spelling errors
- Consistency across the transcript
- Detection and removal of whisper hallucinations as specified

Respond with only the corrected text, maintaining the exact same format."""
                        },
                        {
                            "role": "user",
                            "content": batch_text
                        }
                    ],
                    max_tokens=self.gpt_config.max_tokens,
                    temperature=0.1  # Low temperature for consistent corrections
                )

                corrected_text = response.choices[0].message.content.strip()
                corrected_lines = corrected_text.split('\n')

                # Parse the corrected responses back to segments
                for j, line in enumerate(corrected_lines):
                    if j < len(batch):
                        # Extract the corrected text (remove numbering and speaker prefix)
                        match = re.match(r'^\d+\.\s*\[([^\]]+)\]:\s*(.+)$', line.strip())
                        if match:
                            corrected_segments.append(TranscriptSegment(
                                start_sec=batch[j].start_sec,
                                end_sec=batch[j].end_sec,
                                text=match.group(2).strip(),
                                speaker=batch[j].speaker  # Keep original speaker
                            ))
                        else:
                            # Fallback to original if parsing fails
                            corrected_segments.append(batch[j])

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add original segments on error
                corrected_segments.extend(batch)

            if batch_num % 5 == 0:  # Progress update every 5 batches
                console.print(f"[green]Processed {min(i + batch_size, len(segments))}/{len(segments)} segments")

        logger.info("Spelling correction completed")
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
        """Main processing pipeline using the proven working approach."""
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
            console.print("[blue]Fixing spelling inconsistencies with GPT-4o-mini...")
            segments = self.fix_spelling_with_gpt(segments, batch_size=100)
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
    """Main function to process transcripts using the proven working approach."""
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

    parser = argparse.ArgumentParser(description="Process transcripts using proven working approach")
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
