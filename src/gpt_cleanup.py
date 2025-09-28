"""Stage 4b: Final transcript cleanup and formatting using GPT."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List

import openai
from rich.console import Console

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class TranscriptCleaner:
    """Cleans and formats transcripts using GPT with batch processing like the working reference."""

    def __init__(self, config: Config):
        """Initialize the transcript cleaner."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.cleanup_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the cleanup prompt template from external file."""
        prompt_file = self.config.paths.prompts_dir / "cleanup_transcript.txt"
        try:
            with open(prompt_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load prompt template from {prompt_file}: {e}")
            # Fallback to simple inline prompt
            return """Fix spelling, grammar, and formatting in this transcript while preserving all content:

{transcript}

Provide cleaned transcript in same format:"""

    def load_merged_transcript(self, transcript_path: Path) -> str:
        """Load the merged transcript from file."""
        try:
            if transcript_path.suffix == ".json":
                with open(transcript_path, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("merged_transcript", "")
            else:
                with open(transcript_path, encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading transcript {transcript_path}: {e}")
            return ""

    def split_transcript_into_batches(self, transcript: str, batch_size: int = 10) -> List[str]:
        """Split transcript into batches of dialogue lines for processing."""
        lines = [line.strip() for line in transcript.split("\n") if line.strip()]

        batches = []
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i + batch_size]
            # Format with numbering like the working reference
            numbered_lines = [f"{j+1}. {line}" for j, line in enumerate(batch_lines)]
            batches.append("\n".join(numbered_lines))

        return batches

    def process_batch_with_retry(self, batch: str, max_retries: int = 3) -> str:
        """Process a batch with retry logic for rate limiting."""
        prompt = self.cleanup_prompt_template.format(transcript=batch)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert transcript editor specializing in cleaning and polishing dialogue transcripts while preserving all original content."
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
                    # Return original batch as fallback
                    return batch
                time.sleep(1)

        # Final fallback
        return batch

    def parse_cleaned_batch(self, cleaned_response: str, original_lines: List[str]) -> List[str]:
        """Parse the cleaned response back to individual lines."""
        try:
            cleaned_lines = []
            response_lines = [line.strip() for line in cleaned_response.split("\n") if line.strip()]

            for i, line in enumerate(response_lines):
                # Try to extract content after numbering (1. SPEAKER: content)
                import re
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    cleaned_lines.append(match.group(1).strip())
                else:
                    # Fallback to original if parsing fails
                    if i < len(original_lines):
                        cleaned_lines.append(original_lines[i])
                    else:
                        cleaned_lines.append(line)

            # Ensure we have the same number of lines
            while len(cleaned_lines) < len(original_lines):
                cleaned_lines.append(original_lines[len(cleaned_lines)])

            return cleaned_lines[:len(original_lines)]

        except Exception as e:
            logger.error(f"Error parsing cleaned batch: {e}")
            return original_lines

    def process_transcript_in_batches(self, transcript: str, batch_size: int = 10) -> str:
        """Process transcript in small batches like the working reference."""
        lines = [line.strip() for line in transcript.split("\n") if line.strip()]
        total_lines = len(lines)

        if total_lines == 0:
            return transcript

        console.print(f"[blue]Processing {total_lines} lines in batches of {batch_size}...")

        cleaned_lines = []
        total_batches = (total_lines + batch_size - 1) // batch_size

        for i in range(0, total_lines, batch_size):
            batch_lines = lines[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Create numbered batch for processing
            numbered_batch = "\n".join(f"{j+1}. {line}" for j, line in enumerate(batch_lines))

            console.print(f"[blue]Processing batch {batch_num}/{total_batches} ({len(batch_lines)} lines)...")

            cleaned_batch = self.process_batch_with_retry(numbered_batch)
            parsed_lines = self.parse_cleaned_batch(cleaned_batch, batch_lines)

            cleaned_lines.extend(parsed_lines)

            # Progress update every 5 batches
            if batch_num % 5 == 0:
                console.print(f"[green]Completed {batch_num}/{total_batches} batches")

        return "\n".join(cleaned_lines)

    def generate_quality_report(self, original: str, cleaned: str) -> dict:
        """Generate a simple quality report."""
        original_lines = [line.strip() for line in original.split("\n") if line.strip()]
        cleaned_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

        # Count unclear sections
        unclear_count = cleaned.count("[UNCLEAR]")

        # Estimate word counts
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())

        report = {
            "original_line_count": len(original_lines),
            "cleaned_line_count": len(cleaned_lines),
            "original_word_count": original_words,
            "cleaned_word_count": cleaned_words,
            "unclear_sections": unclear_count,
            "processing_timestamp": datetime.now().isoformat(),
        }

        return report

    def save_cleaned_transcript(
        self,
        cleaned_content: str,
        original_content: str,
        quality_report: dict,
        output_path: Path,
    ) -> None:
        """Save the cleaned transcript with metadata."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create comprehensive metadata
            metadata = {
                "final_transcript": cleaned_content,
                "quality_report": quality_report,
                "processing_details": {
                    "model_used": self.gpt_config.model,
                    "processing_timestamp": datetime.now().isoformat(),
                    "temperature": self.gpt_config.temperature,
                },
                "original_content": original_content,  # Keep for comparison
            }

            # Save comprehensive JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save clean final transcript
            txt_path = output_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            # Save quality report
            report_path = output_path.parent / f"{output_path.stem}_quality_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved cleaned transcript to {txt_path}")
            logger.info(f"Saved quality report to {report_path}")

        except Exception as e:
            logger.error(f"Error saving cleaned transcript: {e}")

    def clean_transcript(self, input_path: Path | None = None) -> Path | None:
        """Main method to clean a transcript using batch processing."""
        if input_path is None:
            # Look for merged transcript
            merged_file = self.config.paths.gpt_dir / "merged_transcript.txt"
            if not merged_file.exists():
                merged_file = self.config.paths.gpt_dir / "merged_transcript.json"

            if not merged_file.exists():
                logger.error("No merged transcript found to clean")
                return None

            input_path = merged_file

        console.print(f"[blue]Cleaning transcript: {input_path.name}")

        # Load transcript
        original_content = self.load_merged_transcript(input_path)
        if not original_content:
            logger.error("Failed to load transcript content")
            return None

        # Process in batches (always use batch processing like working reference)
        console.print("[blue]Processing transcript in batches...")
        cleaned_content = self.process_transcript_in_batches(original_content)

        if not cleaned_content:
            logger.error("GPT processing failed")
            return None

        # Generate quality report
        quality_report = self.generate_quality_report(original_content, cleaned_content)

        # Log quality metrics
        console.print("[green]Cleanup complete!")
        console.print(f"  Word count: {quality_report['original_word_count']} → {quality_report['cleaned_word_count']}")
        console.print(f"  Line count: {quality_report['original_line_count']} → {quality_report['cleaned_line_count']}")
        console.print(f"  Unclear sections: {quality_report['unclear_sections']}")

        # Save result
        output_path = self.config.paths.gpt_dir / "final_transcript"
        self.save_cleaned_transcript(
            cleaned_content, original_content, quality_report, output_path,
        )

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


def cleanup_transcript(config: Config, input_path: Path | None = None) -> Path | None:
    """Main function to clean up a transcript."""
    cleaner = TranscriptCleaner(config)

    # Check dependencies
    errors = cleaner.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        raise RuntimeError("GPT cleanup dependency check failed")

    # Clean transcript
    return cleaner.clean_transcript(input_path)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Clean transcript using GPT")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input transcript file to clean",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for cleaned transcript",
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
    if args.output_dir:
        config_overrides["paths"] = {"gpt_dir": args.output_dir}
    if args.model:
        config_overrides["gpt"] = {"model": args.model}

    config = load_config(**config_overrides)

    # Run cleanup
    try:
        output_file = cleanup_transcript(config, args.input_file)
        if output_file:
            console.print(f"[green]Cleanup complete! Output: {output_file}")
        else:
            console.print("[red]Cleanup failed")
    except Exception as e:
        console.print(f"[red]Cleanup failed: {e}")
        exit(1)
