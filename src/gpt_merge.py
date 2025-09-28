"""Stage 4a: Merge transcripts from multiple speakers using GPT."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import openai
from rich.console import Console

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class TranscriptMerger:
    """Merges transcripts from multiple speakers into a coherent dialogue."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcript merger."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)

    def load_transcript(self, transcript_path: Path) -> dict:
        """Load a transcript from JSON file."""
        try:
            with open(transcript_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"Error loading transcript {transcript_path}: {e}")
            return {}

    def find_transcript_files(self) -> list[Path]:
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

    def create_timeline_events(self, transcripts: dict[str, dict]) -> list[dict]:
        """Create a timeline of all speech events from all speakers."""
        events = []

        for speaker_name, transcript_data in transcripts.items():
            transcription = transcript_data.get("transcription", {})
            segments = transcription.get("segments", [])

            for segment in segments:
                events.append({
                    "speaker": speaker_name,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", []),
                })

        # Sort by start time
        events.sort(key=lambda x: x["start_time"])

        logger.info(f"Created timeline with {len(events)} speech events")
        return events

    def detect_overlaps(self, events: list[dict]) -> list[dict]:
        """Detect and mark overlapping speech segments."""
        for i, event in enumerate(events):
            event["overlaps"] = []

            # Check for overlaps with subsequent events
            for _j, other_event in enumerate(events[i+1:], i+1):
                # Stop checking if the other event starts after this one ends
                if other_event["start_time"] >= event["end_time"]:
                    break

                # Check for overlap
                if other_event["start_time"] < event["end_time"]:
                    overlap_start = max(event["start_time"], other_event["start_time"])
                    overlap_end = min(event["end_time"], other_event["end_time"])
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration > 0.1:  # Minimum 100ms overlap
                        event["overlaps"].append({
                            "speaker": other_event["speaker"],
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_duration": overlap_duration,
                        })

        return events

    def format_timeline_for_gpt(self, events: list[dict]) -> str:
        """Format the timeline events for GPT processing."""
        lines = []
        lines.append("TRANSCRIPT TIMELINE")
        lines.append("==================")
        lines.append("")

        for event in events:
            timestamp = f"[{event['start_time']:.1f}s - {event['end_time']:.1f}s]"
            speaker = event["speaker"].upper()
            text = event["text"]

            # Add overlap information if present
            overlap_info = ""
            if event.get("overlaps"):
                overlap_speakers = [o["speaker"] for o in event["overlaps"]]
                overlap_info = f" (OVERLAPS: {', '.join(overlap_speakers)})"

            lines.append(f"{timestamp} {speaker}{overlap_info}: {text}")

        return "\n".join(lines)

    def create_merge_prompt(self, timeline: str) -> str:
        """Create the prompt for GPT to merge and clean up the transcript."""
        return f"""You are an expert transcript editor tasked with creating a clean, accurate dialogue from multiple speaker transcripts.

Your task is to:
1. Merge the timeline into a coherent conversation
2. Resolve overlapping speech (marked with OVERLAPS) by determining who was speaking
3. Fix any obvious transcription errors or artifacts
4. Maintain the natural flow of conversation
5. Preserve all actual spoken content - do not summarize or paraphrase
6. Use clear speaker labels (SPEAKER_1, SPEAKER_2, etc.)
7. Format as a clean dialogue transcript

Guidelines:
- When speakers overlap, choose the primary speaker based on context
- Fix obvious transcription errors (repeated words, nonsense phrases)
- Keep filler words (um, uh) if they seem intentional
- Mark unclear sections with [UNCLEAR] if you cannot determine the content
- Do not add content that wasn't spoken
- Maintain chronological order

Input Timeline:
{timeline}

Please provide a clean, merged transcript in this format:

SPEAKER_1: [dialogue content]
SPEAKER_2: [dialogue content]
[continue chronologically]

Clean Merged Transcript:"""

    def process_with_gpt(self, timeline: str) -> str:
        """Process the timeline with GPT to create a merged transcript."""
        try:
            prompt = self.create_merge_prompt(timeline)

            response = self.client.chat.completions.create(
                model=self.gpt_config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert transcript editor specializing in multi-speaker dialogue cleanup and merging.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.gpt_config.max_tokens,
                temperature=self.gpt_config.temperature,
            )

            merged_transcript = response.choices[0].message.content.strip()

            # Remove the "Clean Merged Transcript:" header if present
            if "Clean Merged Transcript:" in merged_transcript:
                merged_transcript = merged_transcript.split("Clean Merged Transcript:", 1)[1].strip()

            return merged_transcript

        except Exception as e:
            logger.exception(f"Error processing with GPT: {e}")
            return ""

    def save_merged_transcript(
        self,
        merged_content: str,
        timeline_events: list[dict],
        transcripts: dict[str, dict],
        output_path: Path,
    ) -> None:
        """Save the merged transcript with metadata."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create metadata
            metadata = {
                "merged_transcript": merged_content,
                "original_speakers": list(transcripts.keys()),
                "total_events": len(timeline_events),
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.gpt_config.model,
                "source_files": [str(f) for f in self.find_transcript_files()],
                "timeline_events": timeline_events,  # Include for debugging
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

    def merge_transcripts(self, transcript_files: list[Path] | None = None) -> Path | None:
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

        # Load all transcripts
        transcripts = {}
        for transcript_path in transcript_files:
            speaker_name = transcript_path.stem.replace("_transcript", "")
            transcript_data = self.load_transcript(transcript_path)

            if transcript_data:
                transcripts[speaker_name] = transcript_data
            else:
                logger.warning(f"Failed to load transcript: {transcript_path}")

        if not transcripts:
            logger.error("No valid transcripts loaded")
            return None

        # Create timeline
        timeline_events = self.create_timeline_events(transcripts)

        if not timeline_events:
            logger.error("No speech events found in transcripts")
            return None

        # Detect overlaps
        timeline_events = self.detect_overlaps(timeline_events)

        # Format for GPT
        timeline_text = self.format_timeline_for_gpt(timeline_events)

        # Process with GPT
        console.print("[blue]Processing with GPT...")
        merged_content = self.process_with_gpt(timeline_text)

        if not merged_content:
            logger.error("GPT processing failed")
            return None

        # Save result
        output_path = self.config.paths.gpt_dir / "merged_transcript"
        self.save_merged_transcript(
            merged_content, timeline_events, transcripts, output_path,
        )

        console.print("[green]Transcript merging complete!")
        return output_path.with_suffix(".txt")

    def check_dependencies(self) -> list[str]:
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


def merge_transcripts(config: Config, transcript_files: list[Path] | None = None) -> Path | None:
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
        default="gpt-4",
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
