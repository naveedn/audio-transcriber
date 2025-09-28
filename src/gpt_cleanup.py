"""Stage 4b: Final transcript cleanup and formatting using GPT."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import openai
from rich.console import Console

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class TranscriptCleaner:
    """Cleans and formats transcripts using GPT for final polish."""

    def __init__(self, config: Config):
        """Initialize the transcript cleaner."""
        self.config = config
        self.gpt_config = config.gpt
        self.client = openai.OpenAI(api_key=config.openai_api_key)

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

    def create_cleanup_prompt(self, transcript: str) -> str:
        """Create the prompt for GPT to clean up the transcript."""
        return f"""You are an expert transcript editor tasked with creating a final, polished version of this dialogue transcript. Your goal is to produce a clean, readable, and accurate transcript that maintains all original content while improving readability and fixing errors.

Please perform the following cleanup tasks:

1. **Hallucination Detection**: Flag any content that seems disconnected from natural conversation flow or appears to be AI-generated artifacts
2. **Grammar and Punctuation**: Fix grammar errors and add appropriate punctuation while preserving the natural speech patterns
3. **Speaker Consistency**: Ensure speaker labels are consistent throughout (use SPEAKER_1, SPEAKER_2, etc.)
4. **Remove Artifacts**: Clean up obvious transcription errors like:
   - Repeated words or phrases that don't make sense
   - Random characters or symbols
   - Fragmented sentences that should be combined
5. **Formatting**: Ensure consistent formatting and proper paragraph breaks
6. **Preserve Content**: Do NOT summarize, paraphrase, or remove actual spoken content
7. **Mark Uncertainties**: Use [UNCLEAR] for sections where the content is genuinely unclear
8. **Natural Flow**: Ensure the conversation flows naturally and makes sense

Guidelines:
- Keep all filler words (um, uh, well) that seem intentional
- Preserve interruptions and overlaps that are meaningful
- Fix obvious errors but don't over-edit natural speech patterns
- Flag potential hallucinations with [POSSIBLE HALLUCINATION: reason]
- Maintain chronological order
- Use clear speaker identification

Input Transcript:
{transcript}

Please provide the cleaned transcript below:

CLEANED TRANSCRIPT:"""

    def process_in_chunks(self, transcript: str) -> str:
        """Process large transcripts in chunks to avoid token limits."""
        # Split transcript into manageable chunks
        lines = transcript.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        max_chunk_length = self.gpt_config.chunk_size

        for line in lines:
            line_length = len(line)

            if current_length + line_length > max_chunk_length and current_chunk:
                # Start new chunk
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length

        # Add the last chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logger.info(f"Split transcript into {len(chunks)} chunks for processing")

        # Process each chunk
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            console.print(f"[blue]Processing chunk {i+1}/{len(chunks)}...")
            cleaned_chunk = self.process_with_gpt(chunk)
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
            else:
                logger.warning(f"Failed to process chunk {i+1}")
                cleaned_chunks.append(chunk)  # Keep original if processing fails

        return "\n\n".join(cleaned_chunks)

    def process_with_gpt(self, transcript: str) -> str:
        """Process the transcript with GPT for cleanup."""
        try:
            prompt = self.create_cleanup_prompt(transcript)

            response = self.client.chat.completions.create(
                model=self.gpt_config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert transcript editor specializing in cleaning and polishing dialogue transcripts while preserving all original content and detecting potential transcription errors or hallucinations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.gpt_config.max_tokens,
                temperature=self.gpt_config.temperature,
            )

            cleaned_transcript = response.choices[0].message.content.strip()

            # Remove the "CLEANED TRANSCRIPT:" header if present
            if "CLEANED TRANSCRIPT:" in cleaned_transcript:
                cleaned_transcript = cleaned_transcript.split("CLEANED TRANSCRIPT:", 1)[1].strip()

            return cleaned_transcript

        except Exception as e:
            logger.error(f"Error processing with GPT: {e}")
            return ""

    def extract_hallucination_flags(self, cleaned_transcript: str) -> tuple[str, list[str]]:
        """Extract and catalog any hallucination flags from the cleaned transcript."""
        hallucination_pattern = r"\[POSSIBLE HALLUCINATION: ([^\]]+)\]"

        hallucinations = []
        for match in re.finditer(hallucination_pattern, cleaned_transcript):
            hallucinations.append(match.group(1))

        # Remove the flags from the final transcript
        final_transcript = re.sub(hallucination_pattern, "[FLAGGED CONTENT]", cleaned_transcript)

        return final_transcript, hallucinations

    def generate_quality_report(self, original: str, cleaned: str, hallucinations: list[str]) -> dict:
        """Generate a quality report comparing original and cleaned transcripts."""
        original_lines = [line.strip() for line in original.split("\n") if line.strip()]
        cleaned_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

        # Count unclear sections
        unclear_count = cleaned.count("[UNCLEAR]")
        flagged_count = cleaned.count("[FLAGGED CONTENT]")

        # Estimate word counts
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())

        report = {
            "original_line_count": len(original_lines),
            "cleaned_line_count": len(cleaned_lines),
            "original_word_count": original_words,
            "cleaned_word_count": cleaned_words,
            "unclear_sections": unclear_count,
            "flagged_sections": flagged_count,
            "hallucinations_detected": len(hallucinations),
            "hallucination_details": hallucinations,
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
        """Save the cleaned transcript with metadata and quality report."""
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
                    "chunk_size": self.gpt_config.chunk_size,
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
        """Main method to clean a transcript."""
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

        # Process with GPT (in chunks if necessary)
        if len(original_content) > self.gpt_config.chunk_size:
            console.print("[blue]Large transcript detected, processing in chunks...")
            cleaned_content = self.process_in_chunks(original_content)
        else:
            console.print("[blue]Processing with GPT...")
            cleaned_content = self.process_with_gpt(original_content)

        if not cleaned_content:
            logger.error("GPT processing failed")
            return None

        # Extract hallucination flags
        final_content, hallucinations = self.extract_hallucination_flags(cleaned_content)

        # Generate quality report
        quality_report = self.generate_quality_report(
            original_content, final_content, hallucinations,
        )

        # Log quality metrics
        console.print("[green]Cleanup complete!")
        console.print(f"  Word count: {quality_report['original_word_count']} → {quality_report['cleaned_word_count']}")
        console.print(f"  Unclear sections: {quality_report['unclear_sections']}")
        console.print(f"  Flagged sections: {quality_report['flagged_sections']}")
        console.print(f"  Hallucinations detected: {quality_report['hallucinations_detected']}")

        # Save result
        output_path = self.config.paths.gpt_dir / "final_transcript"
        self.save_cleaned_transcript(
            final_content, original_content, quality_report, output_path,
        )

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
        default="gpt-4",
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
