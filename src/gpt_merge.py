"""Stage 4a: Merge transcripts from multiple speakers using simple chronological merge."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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

    def to_dict(self) -> Dict:
        """Convert to dictionary format matching golden reference."""
        return {
            "start_sec": self.start_time,
            "end_sec": self.end_time,
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
        start_time = self.to_srt_time(self.start_time)
        end_time = self.to_srt_time(self.end_time)
        return f"{index}\n{start_time} --> {end_time}\n[{self.speaker}] {self.text}\n"


class TranscriptMerger:
    """Merges transcripts from multiple speakers into chronological order without GPT processing."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcript merger."""
        self.config = config

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

    def merge_adjacent_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Merge adjacent segments if BOTH are shorter than min_sentence_ms and
        the gap between them is < merge_gap_ms. Concatenate text with a space.
        """
        if not segments:
            return segments

        # Use whisper config parameters for consistency
        min_sentence_ms = self.config.whisper.min_sentence_ms
        merge_gap_ms = self.config.whisper.merge_sentence_gap_ms

        def duration_ms(seg: TranscriptSegment) -> int:
            return int(round((seg.end_time - seg.start_time) * 1000))

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]
            i += 1

            # Try to merge with subsequent segments
            while i < len(segments):
                next_seg = segments[i]
                gap_ms = int(round((next_seg.start_time - current.end_time) * 1000))

                # Check if both segments are short enough and gap is small enough
                should_merge = (
                    duration_ms(current) < min_sentence_ms and
                    duration_ms(next_seg) < min_sentence_ms and
                    gap_ms < merge_gap_ms
                )

                if should_merge:
                    # Merge segments
                    current.end_time = next_seg.end_time
                    current.text = (current.text.rstrip() + " " + next_seg.text.lstrip()).strip()
                    i += 1
                else:
                    break

            merged.append(current)

        return merged

    def save_merged_transcript(
        self,
        segments: List[TranscriptSegment],
        output_path: Path,
    ) -> None:
        """Save the merged transcript in both JSON and SRT formats matching golden reference."""
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

            logger.info(f"Saved merged transcript to {json_path} and {srt_path}")

        except Exception as e:
            logger.exception(f"Error saving merged transcript: {e}")

    def merge_transcripts(self, transcript_files: List[Path] | None = None) -> Path | None:
        """Main method to merge all transcripts chronologically."""
        if transcript_files is None:
            transcript_files = self.find_transcript_files()

        if not transcript_files:
            logger.warning("No transcript files found to merge")
            return None

        if len(transcript_files) == 1:
            logger.info("Only one transcript file found - no merging needed")
            return None

        console.print(f"[blue]Merging {len(transcript_files)} transcript files chronologically...")

        # Load all segments
        all_segments = self.load_all_segments(transcript_files)

        if not all_segments:
            logger.error("No segments loaded from transcript files")
            return None

        # Apply adjacent segment merging
        merged_segments = self.merge_adjacent_segments(all_segments)

        console.print(f"[blue]Merged {len(all_segments)} segments into {len(merged_segments)} final segments")

        # Save result
        output_path = self.config.paths.gpt_dir / "merged_transcript"
        self.save_merged_transcript(merged_segments, output_path)

        console.print("[green]Transcript merging complete!")
        return output_path.with_suffix(".json")

    def check_dependencies(self) -> List[str]:
        """Check if required dependencies are available."""
        # No external dependencies needed for simple chronological merge
        return []


def merge_transcripts(config: Config, transcript_files: List[Path] | None = None) -> Path | None:
    """Main function to merge transcript files."""
    merger = TranscriptMerger(config)

    # Check dependencies
    errors = merger.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "Merge dependency check failed"
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

    parser = argparse.ArgumentParser(description="Merge transcripts chronologically")
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

    args = parser.parse_args()

    # Load configuration
    config_overrides = {}
    if args.input_dir:
        config_overrides["paths"] = {"whisper_dir": args.input_dir}
    if args.output_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["gpt_dir"] = args.output_dir

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