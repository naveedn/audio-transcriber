"""Stage 4: Final transcript processing using the proven working approach."""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import openai
import tiktoken
from rich.console import Console

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None

from .config import Config

console = Console()
logger = logging.getLogger(__name__)

MERGE_GAP_SECONDS = 3.0
MAX_SEGMENT_DIFF = 5
REPETITION_PATTERN = r"(.{2,}?)\1{3,}"


class TranscriptSegment:
    """Represents a single transcript segment."""

    def __init__(
        self,
        start_sec: float,
        end_sec: float,
        text: str,
        speaker: str,
    ) -> None:
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
            "speaker": self.speaker,
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
    """Process transcripts using the reference implementation flow."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcript processor."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)

    def load_individual_transcripts(self) -> list[TranscriptSegment]:
        """Load transcript segments from individual speaker JSON files."""
        segments = []

        # Look for individual speaker transcript files
        whisper_dir = self.config.paths.whisper_dir
        if not whisper_dir.exists():
            logger.warning("Whisper directory does not exist: %s", whisper_dir)
            return segments

        # Find all transcript JSON files
        for json_file in whisper_dir.glob("*_transcript.json"):
            track_name = json_file.stem.replace("_transcript", "")

            try:
                with json_file.open(encoding="utf-8") as file_obj:
                    data = json.load(file_obj)
            except (OSError, json.JSONDecodeError) as exc:
                logger.exception("Error loading %s", json_file, exc_info=exc)
                continue

            count_before = len(segments)

            # Handle both merged format and individual speaker format
            if "transcription" in data and "segments" in data["transcription"]:
                segments.extend(
                    TranscriptSegment(
                        start_sec=segment["start"],
                        end_sec=segment["end"],
                        text=segment["text"],
                        speaker=segment.get("speaker") or track_name,
                    )
                    for segment in data["transcription"]["segments"]
                )
            else:
                segments.extend(
                    TranscriptSegment(
                        start_sec=segment["start_sec"],
                        end_sec=segment["end_sec"],
                        text=segment["text"],
                        speaker=segment.get("speaker") or track_name,
                    )
                    for segment in data
                )

            new_segments = len(segments) - count_before
            logger.info("Loaded %s segments from %s", new_segments, json_file.name)

        # Sort chronologically (essential for proper merging)
        segments.sort(key=lambda x: x.start_sec)
        speaker_count = len({segment.speaker for segment in segments})
        logger.info(
            "Total segments loaded: %s from %s speakers",
            len(segments),
            speaker_count,
        )

        return segments

    def merge_adjacent_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            time_gap = next_segment.start_sec - current_segment.end_sec
            if (
                current_segment.speaker == next_segment.speaker
                and time_gap < MERGE_GAP_SECONDS
            ):
                current_segment = TranscriptSegment(
                    start_sec=current_segment.start_sec,
                    end_sec=next_segment.end_sec,
                    text=f"{current_segment.text} {next_segment.text}",
                    speaker=current_segment.speaker,
                )
            else:
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)
        logger.info(
            "After merging: %s segments (reduced from %s)",
            len(merged),
            len(segments),
        )
        return merged

    def _truncate_repetitive_sequences(self, text: str, max_repeat: int = 50) -> str:
        """Clamp repeated character sequences to avoid token overflow."""

        def replace_repetition(match: re.Match[str]) -> str:
            """Replace long repetitions with a truncated marker."""
            full_match = match.group(0)
            pattern_unit = match.group(1)

            repetitions = len(full_match) // len(pattern_unit)
            if repetitions > max_repeat:
                truncated = pattern_unit * max_repeat
                return f"{truncated}... [continues]"
            return full_match

        return re.sub(REPETITION_PATTERN, replace_repetition, text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        try:
            encoding = tiktoken.encoding_for_model(self.gpt_config.model)
            return len(encoding.encode(text))
        except (LookupError, ValueError):
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def _create_dynamic_batches(
        self, segments: list[TranscriptSegment], max_input_tokens: int = 12000
    ) -> list[list[TranscriptSegment]]:
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
            total_with_overhead = (
                system_prompt_tokens + total_tokens_needed + estimated_response_tokens
            )

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
        batch: list[TranscriptSegment],
        batch_num: int,
        total_batches: int,
    ) -> list[TranscriptSegment]:
        """Process a single batch asynchronously."""
        # Create text batch for processing, truncating excessive repetitions
        texts = []
        for j, seg in enumerate(batch):
            # Truncate repetitive sequences in the segment text
            truncated_text = self._truncate_repetitive_sequences(seg.text)
            texts.append(f"{j + 1}. [{seg.speaker}]: {truncated_text}")
        batch_text = "\n".join(texts)

        est_tokens = self._estimate_tokens(batch_text)
        console.print(
            "[blue]"
            f"Processing batch {batch_num}/{total_batches} "
            f"({len(batch)} segments, ~{est_tokens} tokens)..."
        )

        processed_batch: list[TranscriptSegment] = batch
        try:
            prompt_file = self.config.paths.prompts_dir / "spell-corrections.txt"
            with prompt_file.open(encoding="utf-8") as file_obj:
                system_prompt = file_obj.read()

            # Use async OpenAI client
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.gpt_config.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": batch_text,
                    },
                ],
                max_tokens=self.gpt_config.max_tokens,
                temperature=self.gpt_config.temperature,
            )

            corrected_text = response.choices[0].message.content.strip()
            corrected_lines = corrected_text.split("\n")

            # Filter out empty lines that GPT might add
            corrected_lines = [line for line in corrected_lines if line.strip()]

            candidate_batch: list[TranscriptSegment] = []
            # Parse the corrected responses back to segments
            for j, line in enumerate(corrected_lines):
                if j < len(batch):
                    # Extract the corrected text (remove numbering and speaker prefix)
                    match = re.match(r"^\d+\.\s*\[([^\]]+)\]:\s*(.+)$", line.strip())
                    if match:
                        candidate_batch.append(
                            TranscriptSegment(
                                start_sec=batch[j].start_sec,
                                end_sec=batch[j].end_sec,
                                text=match.group(2).strip(),
                                speaker=batch[j].speaker,
                            )
                        )
                    else:
                        # Fallback to original if parsing fails
                        logger.warning(
                            "Batch %s: Failed to parse line %s, using original segment",
                            batch_num,
                            j + 1,
                        )
                        candidate_batch.append(batch[j])

            # CRITICAL: Validate we didn't lose too many segments
            # Accept batches within 5 segments of original
            segment_diff = abs(len(candidate_batch) - len(batch))
            if segment_diff > MAX_SEGMENT_DIFF:
                # Create error output directory
                error_dir = self.config.paths.gpt_dir / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)

                # Generate timestamp-based filename
                timestamp = datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
                batch_text_file = (
                    error_dir / f"batch_{batch_num}_{timestamp}_original.txt"
                )
                corrected_text_file = (
                    error_dir / f"batch_{batch_num}_{timestamp}_corrected.txt"
                )

                # Save original batch text
                batch_text_file.write_text(batch_text, encoding="utf-8")

                # Save corrected lines output
                corrected_text_file.write_text(
                    "\n".join(corrected_lines),
                    encoding="utf-8",
                )

                logger.error(
                    (
                        "Batch %s: Segment count mismatch! Original=%s, "
                        "Corrected=%s, GPT lines=%s. Difference=%s exceeds "
                        "threshold=%s. Original saved to: %s, Corrected saved to: %s"
                    ),
                    batch_num,
                    len(batch),
                    len(candidate_batch),
                    len(corrected_lines),
                    segment_diff,
                    MAX_SEGMENT_DIFF,
                    batch_text_file,
                    corrected_text_file,
                )

                console.print(
                    "[red]⚠ Batch "
                    f"{batch_num}: Segment difference of {segment_diff} exceeds "
                    "threshold, using original batch. "
                    f"Error files saved to {error_dir}",
                )
                processed_batch = batch
            else:
                processed_batch = candidate_batch

            if 0 < segment_diff <= MAX_SEGMENT_DIFF:
                logger.warning(
                    (
                        "Batch %s: Segment count differs by %s (Original=%s, "
                        "Corrected=%s) but within threshold=%s. "
                        "Accepting corrected batch."
                    ),
                    batch_num,
                    segment_diff,
                    len(batch),
                    len(candidate_batch),
                    MAX_SEGMENT_DIFF,
                )
                console.print(
                    "[yellow]⚠ Batch "
                    f"{batch_num}: Segment count differs by {segment_diff}, "
                    "within threshold - accepting corrected batch",
                )
        except (OSError, ValueError, openai.OpenAIError) as exc:
            logger.exception("Error processing batch %s", batch_num, exc_info=exc)
            console.print(
                f"[yellow]⚠ Batch {batch_num} failed, using original segments"
            )
            processed_batch = batch

        if processed_batch is not batch:
            console.print(f"[green]✓ Completed batch {batch_num}/{total_batches}")

        return processed_batch

    async def _process_batches_parallel(
        self,
        batches: list[list[TranscriptSegment]],
        max_concurrent: int = 10,
    ) -> list[TranscriptSegment]:
        """Process multiple batches in parallel with controlled concurrency."""
        total_batches = len(batches)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(
            batch: list[TranscriptSegment], batch_idx: int
        ) -> tuple[int, list[TranscriptSegment]]:
            async with semaphore:
                result = await self._process_batch_async(
                    batch, batch_idx + 1, total_batches
                )
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

    def fix_spelling_with_gpt(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Fix spelling inconsistencies using GPT-4o-mini with batching."""
        if not self.client:
            logger.warning(
                "OpenAI client not initialized. Skipping spelling correction."
            )
            return segments

        console.print(
            f"[blue]Processing {len(segments)} segments for spelling consistency..."
        )

        # Create dynamic batches based on token count
        batches = self._create_dynamic_batches(segments, max_input_tokens=12000)
        total_batches = len(batches)

        # Calculate statistics
        total_segments = sum(len(batch) for batch in batches)
        avg_segments_per_batch = (
            total_segments / total_batches if total_batches > 0 else 0
        )

        console.print(
            "[blue]Created "
            f"{total_batches} optimized batches "
            f"(avg {avg_segments_per_batch:.1f} segments/batch)"
        )
        console.print("[blue]Processing with up to 10 concurrent requests...")

        # Run async processing
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                corrected_segments = asyncio.run(
                    self._process_batches_parallel(batches, max_concurrent=10)
                )
            else:
                if nest_asyncio is None:
                    logger.error(
                        "nest_asyncio is required to run within an existing event loop"
                    )
                    return segments

                nest_asyncio.apply()
                corrected_segments = asyncio.run(
                    self._process_batches_parallel(batches, max_concurrent=10)
                )
        except (RuntimeError, OSError, ValueError, openai.OpenAIError) as exc:
            logger.exception("Error in parallel processing", exc_info=exc)
            console.print(f"[red]Parallel processing failed: {exc}")
            return segments

        logger.info(
            "Spelling correction completed: %s segments processed",
            len(corrected_segments),
        )
        return corrected_segments

    def save_final_outputs(
        self, segments: list[TranscriptSegment], output_path: Path
    ) -> None:
        """Save the final transcript in both JSON and SRT formats."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save JSON format (array of segment objects)
            json_path = output_path.with_suffix(".json")
            json_data = [segment.to_dict() for segment in segments]

            json_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Save SRT format
            srt_path = output_path.with_suffix(".srt")
            with srt_path.open("w", encoding="utf-8") as file_obj:
                for index, segment in enumerate(segments, 1):
                    file_obj.write(segment.to_srt_entry(index))
                    if index < len(segments):
                        file_obj.write("\n")

            logger.info(
                "Saved final transcript to %s and %s",
                json_path,
                srt_path,
            )

        except OSError as exc:
            logger.exception("Error saving final transcript", exc_info=exc)

    def process_transcripts(self) -> Path | None:
        """Main processing pipeline using the proven working approach."""
        console.print("[blue]Loading individual transcript files...")

        # Load all segments from individual speaker files
        segments = self.load_individual_transcripts()
        if not segments:
            logger.error("No transcript segments found to process")
            return None

        speakers = {segment.speaker for segment in segments}
        console.print(
            f"[blue]Loaded {len(segments)} segments from {len(speakers)} speakers"
        )

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
        speakers = {segment.speaker for segment in segments}
        console.print("[green]Processing Summary:")
        console.print(f"  Total segments: {len(segments)}")
        console.print(f"  Speakers: {', '.join(sorted(speakers))}")
        console.print(
            f"  Duration: {segments[0].start_sec:.2f}s to {segments[-1].end_sec:.2f}s"
        )

        return output_path.with_suffix(".json")

    def check_dependencies(self) -> list[str]:
        """Check if required dependencies are available."""
        errors = []

        if not self.config.openai_api_key:
            errors.append("OpenAI API key is required")

        try:
            # Test OpenAI API connection
            self.client.models.list()
        except openai.OpenAIError as exc:
            errors.append(f"Cannot connect to OpenAI API: {exc}")

        return errors


def cleanup_transcript(config: Config, input_path: Path | None = None) -> Path | None:
    """Main function to process transcripts using the proven working approach."""
    del input_path
    processor = TranscriptProcessor(config)

    # Check dependencies
    errors = processor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        message = "Transcript processing dependency check failed"
        raise RuntimeError(message)

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

    parser = argparse.ArgumentParser(
        description="Process transcripts using proven working approach"
    )
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
    except (RuntimeError, OSError, openai.OpenAIError) as exc:
        console.print(f"[red]Processing failed: {exc}")
        sys.exit(1)
