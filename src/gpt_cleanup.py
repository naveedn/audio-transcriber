"""Stage 4: Final transcript processing using the proven working approach."""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

import openai
import tiktoken
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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        try:
            encoding = tiktoken.encoding_for_model(self.gpt_config.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def _create_dynamic_batches(self, segments: List[TranscriptSegment], max_input_tokens: int = 10000) -> List[List[TranscriptSegment]]:
        """Create batches dynamically based on token count to maximize efficiency."""
        batches = []
        current_batch = []
        current_tokens = 0

        # System prompt tokens (approximate)
        system_prompt_tokens = 110

        for segment in segments:
            # Format as it will appear in the request
            segment_text = f"[{segment.speaker}]: {segment.text}"
            segment_tokens = self._estimate_tokens(segment_text)

            # Reserve tokens for response (roughly same as input)
            # Add safety margin of 20%
            total_tokens_needed = current_tokens + segment_tokens
            estimated_response_tokens = total_tokens_needed
            total_with_overhead = system_prompt_tokens + total_tokens_needed + estimated_response_tokens

            # Check if adding this segment would exceed limit
            if current_batch and (total_with_overhead * 1.2) > max_input_tokens:
                # Start new batch
                batches.append(current_batch)
                current_batch = [segment]
                current_tokens = segment_tokens
            else:
                current_batch.append(segment)
                current_tokens += segment_tokens

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch_async(
        self,
        batch: List[TranscriptSegment],
        batch_num: int,
        total_batches: int
    ) -> List[TranscriptSegment]:
        """Process a single batch asynchronously."""
        # Create text batch for processing
        texts = [f"{j+1}. [{seg.speaker}]: {seg.text}" for j, seg in enumerate(batch)]
        batch_text = "\n".join(texts)

        console.print(f"[blue]Processing batch {batch_num}/{total_batches} ({len(batch)} segments, ~{self._estimate_tokens(batch_text)} tokens)...")

        try:
            prompt_file = self.config.paths.prompts_dir / "spell-corrections.txt"
            system_prompt = open(prompt_file, 'r').read()

            # Use async OpenAI client
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.gpt_config.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": batch_text
                    }
                ],
                max_tokens=self.gpt_config.max_tokens,
                temperature=self.gpt_config.temperature
            )

            corrected_text = response.choices[0].message.content.strip()
            corrected_lines = corrected_text.split('\n')

            # Filter out empty lines that GPT might add
            corrected_lines = [line for line in corrected_lines if line.strip()]

            corrected_batch = []
            # Parse the corrected responses back to segments
            for j, line in enumerate(corrected_lines):
                if j < len(batch):
                    # Extract the corrected text (remove numbering and speaker prefix)
                    match = re.match(r'^\d+\.\s*\[([^\]]+)\]:\s*(.+)$', line.strip())
                    if match:
                        corrected_batch.append(TranscriptSegment(
                            start_sec=batch[j].start_sec,
                            end_sec=batch[j].end_sec,
                            text=match.group(2).strip(),
                            speaker=batch[j].speaker
                        ))
                    else:
                        # Fallback to original if parsing fails
                        logger.warning(f"Batch {batch_num}: Failed to parse line {j+1}, using original segment")
                        corrected_batch.append(batch[j])

            # CRITICAL: Validate we didn't lose too many segments
            # Accept batches within 2 segments of original
            segment_diff = abs(len(corrected_batch) - len(batch))
            if segment_diff > 2:
                logger.error(
                    f"Batch {batch_num}: Segment count mismatch! "
                    f"Original: {len(batch)}, Corrected: {len(corrected_batch)}, "
                    f"GPT lines: {len(corrected_lines)}. Difference of {segment_diff} exceeds threshold of 2. "
                    f"Using original batch."
                )
                console.print(
                    f"[red]⚠ Batch {batch_num}: Segment difference of {segment_diff} exceeds threshold, "
                    f"using original batch"
                )
                return batch
            elif segment_diff > 0:
                logger.warning(
                    f"Batch {batch_num}: Segment count differs by {segment_diff} "
                    f"(Original: {len(batch)}, Corrected: {len(corrected_batch)}), "
                    f"but within acceptable threshold of 2. Accepting corrected batch."
                )
                console.print(
                    f"[yellow]⚠ Batch {batch_num}: Segment count differs by {segment_diff}, "
                    f"but within threshold - accepting corrected batch"
                )

            console.print(f"[green]✓ Completed batch {batch_num}/{total_batches}")
            return corrected_batch

        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            console.print(f"[yellow]⚠ Batch {batch_num} failed, using original segments")
            return batch

    async def _process_batches_parallel(
        self,
        batches: List[List[TranscriptSegment]],
        max_concurrent: int = 10
    ) -> List[TranscriptSegment]:
        """Process multiple batches in parallel with controlled concurrency."""
        total_batches = len(batches)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(batch: List[TranscriptSegment], batch_idx: int) -> tuple[int, List[TranscriptSegment]]:
            async with semaphore:
                result = await self._process_batch_async(batch, batch_idx + 1, total_batches)
                return (batch_idx, result)

        # Process all batches with controlled concurrency
        tasks = [process_with_semaphore(batch, i) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        # Sort by batch index to maintain order
        results.sort(key=lambda x: x[0])

        # Flatten results
        corrected_segments = []
        for _, batch_segments in results:
            corrected_segments.extend(batch_segments)

        return corrected_segments

    def fix_spelling_with_gpt(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Fix spelling inconsistencies using GPT-4o-mini with optimized batching and parallel processing."""
        if not self.client:
            logger.warning("OpenAI client not initialized. Skipping spelling correction.")
            return segments

        console.print(f"[blue]Processing {len(segments)} segments for spelling consistency...")

        # Create dynamic batches based on token count
        batches = self._create_dynamic_batches(segments, max_input_tokens=8000)
        total_batches = len(batches)

        # Calculate statistics
        total_segments = sum(len(batch) for batch in batches)
        avg_segments_per_batch = total_segments / total_batches if total_batches > 0 else 0

        console.print(f"[blue]Created {total_batches} optimized batches (avg {avg_segments_per_batch:.1f} segments/batch)")
        console.print(f"[blue]Processing with up to 10 concurrent requests...")

        # Run async processing
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, create a task and run it
                import nest_asyncio
                nest_asyncio.apply()
                corrected_segments = asyncio.run(self._process_batches_parallel(batches, max_concurrent=10))
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                corrected_segments = asyncio.run(self._process_batches_parallel(batches, max_concurrent=10))

            logger.info(f"Spelling correction completed: {len(corrected_segments)} segments processed")
            return corrected_segments
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            console.print(f"[red]Parallel processing failed: {e}")
            return segments

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
