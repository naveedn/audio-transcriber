"""Stage 4a: Merge transcripts from multiple speakers using GPT."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import openai
from rich.console import Console

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class TranscriptSegment:
    """Represents a single transcript segment with timing and speaker info."""

    def __init__(self, start_time: float, end_time: float, text: str, speaker: str):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text.strip()
        self.speaker = speaker

    def __str__(self) -> str:
        return f"[{self.start_time:.1f}s-{self.end_time:.1f}s] {self.speaker}: {self.text}"


class TranscriptMerger:
    """Merges transcripts from multiple speakers into a coherent dialogue using batch processing."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcript merger."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.merge_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the merge prompt template from external file."""
        prompt_file = self.config.paths.prompts_dir / "merge_segments.txt"
        try:
            with open(prompt_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load prompt template from {prompt_file}: {e}")
            # Fallback to simple inline prompt
            return """Merge these dialogue segments chronologically, resolving any overlaps:

{segments}

Provide clean dialogue in format:
SPEAKER_1: [content]
SPEAKER_2: [content]"""

    def load_transcript(self, transcript_path: Path) -> List[TranscriptSegment]:
        """Load a transcript from JSON file and return segments."""
        try:
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)

            segments = []
            speaker_name = transcript_path.stem.replace("_transcript", "")
            transcription = data.get("transcription", {})

            for segment in transcription.get("segments", []):
                segments.append(TranscriptSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"],
                    speaker=speaker_name
                ))

            return segments
        except Exception as e:
            logger.exception(f"Error loading transcript {transcript_path}: {e}")
            return []

    def find_transcript_files(self) -> List[Path]:
        """Find all transcript files to merge."""
        transcript_files = []

        if not self.config.paths.whisper_dir.exists():
            logger.warning(f"Whisper directory does not exist: {self.config.paths.whisper_dir}")
            return transcript_files

        # Look for JSON transcript files
        for pattern in ["*_transcript.json"]:
            transcript_files.extend(self.config.paths.whisper_dir.glob(pattern))

        logger.info(f"Found {len(transcript_files)} transcript files to merge")
        return sorted(transcript_files)

    def load_all_segments(self, transcript_files: List[Path]) -> List[TranscriptSegment]:
        """Load and merge all segments from transcript files."""
        all_segments = []

        for transcript_path in transcript_files:
            segments = self.load_transcript(transcript_path)
            if segments:
                all_segments.extend(segments)
                logger.info(f"Loaded {len(segments)} segments from {transcript_path.name}")
            else:
                logger.warning(f"Failed to load segments from {transcript_path}")

        # Sort by start time for chronological processing
        all_segments.sort(key=lambda x: x.start_time)
        logger.info(f"Total segments loaded: {len(all_segments)}")

        return all_segments

    def process_batch_with_retry(self, batch: List[TranscriptSegment], max_retries: int = 3) -> str:
        """Process a batch of segments with retry logic for rate limiting."""
        segments_text = "\n".join(str(segment) for segment in batch)
        prompt = self.merge_prompt_template.format(segments=segments_text)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert transcript editor specializing in multi-speaker dialogue merging."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.gpt_config.max_tokens,
                    temperature=self.gpt_config.temperature,
                )

                result = response.choices[0].message.content.strip()

                return result

            except openai.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error processing batch (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Return original segments as fallback
                    fallback = "\n".join(f"{seg.speaker}: {seg.text}" for seg in batch)
                    return fallback
                time.sleep(1)

        # Final fallback
        return "\n".join(f"{seg.speaker}: {seg.text}" for seg in batch)

    def merge_segments_in_batches(self, segments: List[TranscriptSegment], batch_size: int = 12) -> str:
        """Process segments in batches to avoid token limits and rate limiting."""
        merged_parts = []
        total_batches = (len(segments) + batch_size - 1) // batch_size

        console.print(f"[blue]Processing {len(segments)} segments in {total_batches} batches...")

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_num = i // batch_size + 1

            console.print(f"[blue]Processing batch {batch_num}/{total_batches} ({len(batch)} segments)...")

            merged_batch = self.process_batch_with_retry(batch)
            if merged_batch:
                merged_parts.append(merged_batch)

            # Progress update
            if batch_num % 5 == 0:
                console.print(f"[green]Completed {batch_num}/{total_batches} batches")

        # Combine all merged parts
        final_transcript = "\n\n".join(merged_parts)
        return final_transcript

    def save_merged_transcript(
        self,
        merged_content: str,
        segments: List[TranscriptSegment],
        output_path: Path,
    ) -> None:
        """Save the merged transcript with metadata."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create metadata
            speakers = sorted(set(seg.speaker for seg in segments))
            metadata = {
                "merged_transcript": merged_content,
                "original_speakers": speakers,
                "total_segments": len(segments),
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.gpt_config.model,
                "source_files": [str(f) for f in self.find_transcript_files()],
            }

            # Save JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save clean text
            txt_path = output_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(merged_content)

            logger.info(f"Saved merged transcript to {json_path} and {txt_path}")

        except Exception as e:
            logger.exception(f"Error saving merged transcript: {e}")

    def merge_transcripts(self, transcript_files: List[Path] | None = None) -> Path | None:
        """Main method to merge all transcripts."""
        if transcript_files is None:
            transcript_files = self.find_transcript_files()

        if not transcript_files:
            logger.warning("No transcript files found to merge")
            return None

        if len(transcript_files) == 1:
            logger.info("Only one transcript file found - no merging needed")
            return None

        console.print(f"[blue]Merging {len(transcript_files)} transcript files...")

        # Load all segments
        all_segments = self.load_all_segments(transcript_files)

        if not all_segments:
            logger.error("No segments loaded from transcript files")
            return None

        # Process in batches
        merged_content = self.merge_segments_in_batches(all_segments)

        if not merged_content:
            logger.error("GPT processing failed")
            return None

        # Save result
        output_path = self.config.paths.gpt_dir / "merged_transcript"
        self.save_merged_transcript(merged_content, all_segments, output_path)

        console.print("[green]Transcript merging complete!")
        return output_path.with_suffix(".txt")

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


def merge_transcripts(config: Config, transcript_files: List[Path] | None = None) -> Path | None:
    """Main function to merge transcript files."""
    merger = TranscriptMerger(config)

    # Check dependencies
    errors = merger.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "GPT merge dependency check failed"
        raise RuntimeError(msg)

    # Merge transcripts
    return merger.merge_transcripts(transcript_files)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Merge transcripts using GPT")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing transcript files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for merged transcript",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo",
        help="GPT model to use",
    )

    args = parser.parse_args()

    # Load configuration
    config_overrides = {}
    if args.input_dir:
        config_overrides["paths"] = {"whisper_dir": args.input_dir}
    if args.output_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["gpt_dir"] = args.output_dir
    if args.model:
        config_overrides["gpt"] = {"model": args.model}

    config = load_config(**config_overrides)

    # Run merging
    try:
        output_file = merge_transcripts(config)
        if output_file:
            console.print(f"[green]Merging complete! Output: {output_file}")
        else:
            console.print("[yellow]No merging performed")
    except Exception as e:
        console.print(f"[red]Merging failed: {e}")
        sys.exit(1)
