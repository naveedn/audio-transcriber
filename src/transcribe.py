"""Stage 3: Speech-to-Text Transcription with Multiple Models."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .config import Config
from .transcription.strategy import create_transcription_strategy

console = Console()
logger = logging.getLogger(__name__)


class Transcriber:
    """Speech-to-text transcription using Strategy Pattern.

    This class delegates to different transcription strategies based on config:
    - WhisperStrategy: For Whisper models (MLX or standard)
    - ParakeetStrategy: For Parakeet models (streaming and non-streaming modes)
    """

    def __init__(self, config: Config) -> None:
        """Initialize the transcriber with appropriate strategy."""
        self.config = config
        self.model_config = config.model

        # Create the appropriate strategy based on config
        self.strategy = create_transcription_strategy(config)
        logger.info(f"Initialized transcriber with {self.strategy.__class__.__name__}")

    def load_model(self) -> None:
        """Load the transcription model via strategy."""
        self.strategy.load_model()

    def _create_srt_content(self, segments: list[dict]) -> str:
        """Create SRT subtitle content from transcription segments."""
        srt_lines = []

        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_time(segment["start"])
            end_time = self._format_srt_time(segment["end"])
            text = segment["text"].strip()

            if text:  # Only include non-empty segments
                srt_lines.extend(
                    [
                        str(i),
                        f"{start_time} --> {end_time}",
                        text,
                        "",  # Empty line between segments
                    ]
                )

        return "\n".join(srt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT subtitle format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def transcribe_file(
        self,
        audio_path: Path,
        vad_path: Path,
        progress: Progress | None = None,
        task_id: Optional = None,
    ) -> dict:
        """Transcribe a single audio file using VAD segments.

        Delegates to the configured strategy (Whisper, Parakeet, or Parakeet Streaming).
        """
        return self.strategy.transcribe_file(audio_path, vad_path, progress, task_id)

    def save_transcription(
        self,
        transcription: dict,
        output_path: Path,
        speaker_name: str,
    ) -> None:
        """Save transcription to JSON and SRT files."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = output_path.with_suffix(".json")
            transcription_data = {
                "speaker": speaker_name,
                "transcription": transcription,
                "config": self.model_config.dict(),
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            # Save SRT
            srt_path = output_path.with_suffix(".srt")
            srt_content = self._create_srt_content(transcription["segments"])

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            logger.info(f"Saved transcription to {json_path} and {srt_path}")

        except Exception as e:
            logger.exception(f"Error saving transcription to {output_path}: {e}")


class TranscriptionProcessor:
    """Main transcription processing coordinator."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcription processor."""
        self.config = config
        self.transcriber = Transcriber(config)

    def find_audio_and_vad_files(self) -> list[tuple[Path, Path]]:
        """Find matching audio and VAD files for transcription."""
        audio_vad_pairs = []

        if not self.config.paths.audio_wav_dir.exists():
            logger.warning(
                f"Audio WAV directory does not exist: {self.config.paths.audio_wav_dir}"
            )
            return audio_vad_pairs

        if not self.config.paths.silero_dir.exists():
            logger.warning(
                f"VAD directory does not exist: {self.config.paths.silero_dir}"
            )
            return audio_vad_pairs

        # Find audio files
        audio_files = []
        for pattern in ["*.wav", "*.WAV"]:
            audio_files.extend(self.config.paths.audio_wav_dir.glob(pattern))

        # Match with VAD files
        for audio_path in audio_files:
            speaker_name = audio_path.stem
            vad_path = self.config.paths.silero_dir / f"{speaker_name}_timestamps.json"

            if vad_path.exists():
                audio_vad_pairs.append((audio_path, vad_path))
            else:
                logger.warning(f"No VAD file found for {speaker_name}: {vad_path}")

        logger.info(f"Found {len(audio_vad_pairs)} audio/VAD pairs for transcription")
        return sorted(audio_vad_pairs)

    def get_output_path(self, audio_path: Path) -> Path:
        """Get the output path for transcription."""
        speaker_name = audio_path.stem
        return self.config.paths.whisper_dir / f"{speaker_name}_transcript"

    def process_files(
        self, audio_vad_pairs: list[tuple[Path, Path]] | None = None
    ) -> dict[str, Path]:
        """Process multiple audio files for transcription sequentially."""
        if audio_vad_pairs is None:
            audio_vad_pairs = self.find_audio_and_vad_files()

        if not audio_vad_pairs:
            logger.warning("No audio/VAD pairs found for transcription")
            return {}

        successful_outputs = {}
        failed_count = 0

        # Process sequentially to manage memory usage
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            for audio_path, vad_path in audio_vad_pairs:
                speaker_name = audio_path.stem
                output_path = self.get_output_path(audio_path)

                # Load VAD segments to determine total work
                try:
                    with open(vad_path) as f:
                        vad_data = json.load(f)
                        total_segments = len(vad_data.get("segments", []))
                except:
                    total_segments = 1

                task_id = progress.add_task(
                    f"[cyan]Transcribing {speaker_name}...",
                    total=total_segments,
                )

                try:
                    # Transcribe file
                    transcription = self.transcriber.transcribe_file(
                        audio_path,
                        vad_path,
                        progress,
                        task_id,
                    )

                    if transcription["segments"]:
                        # Save transcription
                        self.transcriber.save_transcription(
                            transcription,
                            output_path,
                            speaker_name,
                        )
                        successful_outputs[speaker_name] = output_path.with_suffix(
                            ".json"
                        )

                        progress.update(task_id, description=f"[green] {speaker_name}")
                    else:
                        failed_count += 1
                        progress.update(task_id, description=f"[red] {speaker_name}")

                except Exception as e:
                    logger.exception(f"Failed to transcribe {speaker_name}: {e}")
                    failed_count += 1
                    progress.update(task_id, description=f"[red] {speaker_name}")

        console.print(
            f"[green]Successfully transcribed {len(successful_outputs)} files"
        )
        if failed_count > 0:
            console.print(f"[red]Failed to transcribe {failed_count} files")

        return successful_outputs

    def check_dependencies(self) -> list[str]:
        """Check if required dependencies are available for the selected strategy."""
        errors = []

        model_name = self.config.model.model.lower()

        # Check model-specific dependencies
        if "whisper" in model_name:
            # Check Whisper dependencies
            try:
                import mlx_whisper

                logger.info("MLX Whisper is available for Apple Silicon optimization")
            except ImportError:
                try:
                    import whisper

                    logger.info("Standard Whisper is available")
                except ImportError:
                    errors.append(
                        "Neither MLX Whisper nor standard Whisper is installed"
                    )

            try:
                import librosa
            except ImportError:
                errors.append("librosa is not installed")

        elif "parakeet" in model_name:
            # Check Parakeet dependencies
            try:
                import parakeet_mlx

                logger.info("Parakeet-MLX is available")
            except ImportError:
                errors.append(
                    "parakeet-mlx is not installed (required for Parakeet models)"
                )

            try:
                import librosa
            except ImportError:
                errors.append("librosa is not installed")

            try:
                import soundfile
            except ImportError:
                errors.append("soundfile is not installed (required for Parakeet)")

        try:
            # Test initializing the transcriber strategy
            self.transcriber.load_model()
        except Exception as e:
            errors.append(f"Cannot initialize transcription model: {e}")

        return errors


def transcribe_audio(
    config: Config,
    audio_vad_pairs: list[tuple[Path, Path]] | None = None,
) -> dict[str, Path]:
    """Main function to transcribe audio files."""
    processor = TranscriptionProcessor(config)

    # Check dependencies
    errors = processor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "Whisper dependency check failed"
        raise RuntimeError(msg)

    # Process files
    return processor.process_files(audio_vad_pairs)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory containing WAV files",
    )
    parser.add_argument(
        "--vad-dir",
        type=Path,
        help="Directory containing VAD timestamp files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for transcriptions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small.en",
        help="Whisper model to use",
    )

    args = parser.parse_args()

    # Load configuration
    config_overrides = {}
    if args.audio_dir:
        config_overrides["paths"] = {"audio_wav_dir": args.audio_dir}
    if args.vad_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["silero_dir"] = args.vad_dir
    if args.output_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["whisper_dir"] = args.output_dir
    if args.model:
        config_overrides["whisper"] = {"model": args.model}

    config = load_config(**config_overrides)

    # Run transcription
    try:
        output_files = transcribe_audio(config)
        console.print(
            f"[green]Transcription complete! Generated {len(output_files)} files."
        )
    except Exception as e:
        console.print(f"[red]Transcription failed: {e}")
        sys.exit(1)
