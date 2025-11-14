"""Stage 3: Speech-to-Text Transcription with Whisper."""

import importlib
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based speech-to-text transcription."""

    def __init__(self, config: Config) -> None:
        """Initialize the Whisper transcriber."""
        self.config = config
        self.whisper_config = config.whisper
        self.model = None
        self.use_mlx = False
        self.mlx_module = None
        self._mlx_core = None

    @staticmethod
    def import_optional(module_name: str) -> object | None:
        """Import an optional module, returning None if missing."""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None

    def load_model(self) -> None:
        """Initialize an available Whisper implementation."""
        if self.use_mlx and self.mlx_module is not None:
            return
        if not self.use_mlx and self.model is not None:
            return

        mlx_module = self.import_optional("mlx_whisper")
        if mlx_module is not None:
            self.mlx_module = mlx_module
            self.use_mlx = True
            logger.info("MLX Whisper available for Apple Silicon optimization")
            return

        whisper_module = self.import_optional("whisper")
        if whisper_module is not None:
            self.model = whisper_module.load_model(self.whisper_config.model)
            self.use_mlx = False
            logger.info("Standard Whisper model loaded successfully")
            return

        msg = "Neither MLX Whisper nor standard Whisper is available"
        raise ImportError(msg) from None

    def _load_diarization_data(self, diarization_path: Path) -> dict:
        """Load Senko diarization metadata."""
        try:
            with diarization_path.open(encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Error loading diarization data from %s", diarization_path
            )
            return {}

    def _extract_audio_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """Extract audio segment from full audio array."""
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Ensure bounds are valid
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        return audio[start_sample:end_sample]

    def _transcribe_segment(
        self,
        audio_segment: np.ndarray,
    ) -> dict:
        """Transcribe a single audio segment."""
        try:
            # Ensure model is initialized
            if (self.use_mlx and self.mlx_module is None) or (
                not self.use_mlx and self.model is None
            ):
                self.load_model()

            # Apply MLX memory optimization if available
            if self.use_mlx:
                if self._mlx_core is None:
                    self._mlx_core = self.import_optional("mlx.core")
                if self._mlx_core is not None:
                    self._mlx_core.clear_cache()

            if self.use_mlx:
                # MLX Whisper - uses direct function call with model repository
                result = self.mlx_module.transcribe(
                    audio_segment,
                    path_or_hf_repo="mlx-community/whisper-small.en-mlx",
                    word_timestamps=True,
                    temperature=0,
                    condition_on_previous_text=False,
                )
            else:
                # Standard Whisper
                result = self.model.transcribe(
                    audio_segment,
                    language=self.whisper_config.language,
                    temperature=self.whisper_config.temperature,
                    word_timestamps=True,
                )

        except Exception:
            logger.exception("Error transcribing audio segment")
            return {"text": "", "segments": [], "words": []}
        return result

    def _merge_sentence_segments(self, segments: list[dict]) -> list[dict]:
        """Merge adjacent segments to create sentence-level segments."""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            # Calculate gap between segments
            gap_ms = (next_segment["start"] - current_segment["end"]) * 1000

            # Check if both segments are short enough to merge
            current_duration = (
                current_segment["end"] - current_segment["start"]
            ) * 1000
            next_duration = (next_segment["end"] - next_segment["start"]) * 1000

            same_speaker = current_segment.get("speaker") == next_segment.get("speaker")
            should_merge = (
                same_speaker
                and gap_ms <= self.whisper_config.merge_sentence_gap_ms
                and current_duration < self.whisper_config.min_sentence_ms
                and next_duration < self.whisper_config.min_sentence_ms
            )

            if should_merge:
                # Merge segments
                current_segment["end"] = next_segment["end"]
                current_segment["text"] = (
                    current_segment["text"].strip() + " " + next_segment["text"].strip()
                )

                # Merge words if available
                if "words" in current_segment and "words" in next_segment:
                    current_segment["words"].extend(next_segment["words"])
            else:
                # Save current segment and start new one
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Add the last segment
        merged.append(current_segment)

        return merged

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
        diarization_path: Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> dict:
        """Transcribe a single audio file using Senko diarization segments."""
        try:
            logger.info("Transcribing %s", audio_path.name)

            # Load audio
            audio, sample_rate = librosa.load(str(audio_path), sr=16000)

            # Load diarization segments
            diarization_data = self._load_diarization_data(diarization_path)
            diarized_segments = diarization_data.get("merged_segments", [])
            if not diarized_segments:
                fallback_vad = diarization_data.get("vad_segments", [])
                diarized_segments = [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": audio_path.stem,
                    }
                    for seg in fallback_vad
                ]

            if not diarized_segments:
                logger.warning("No diarization segments found for %s", audio_path.name)
                return {"segments": [], "text": ""}

            logger.info("Processing %s diarized segments", len(diarized_segments))

            # Transcribe each segment
            transcribed_segments = []

            for i, diar_segment in enumerate(diarized_segments):
                start_time = diar_segment["start"]
                end_time = diar_segment["end"]
                diarized_speaker = diar_segment.get("speaker") or audio_path.stem
                if diarized_speaker == audio_path.stem:
                    speaker_label = diarized_speaker
                else:
                    speaker_label = f"{audio_path.stem}:{diarized_speaker}"

                # Extract audio segment
                audio_segment = self._extract_audio_segment(
                    audio,
                    sample_rate,
                    start_time,
                    end_time,
                )

                if len(audio_segment) < sample_rate * 0.1:  # Skip very short segments
                    continue

                # Transcribe segment
                whisper_result = self._transcribe_segment(audio_segment)

                if whisper_result.get("text", "").strip():
                    segment_data = {
                        "start": start_time,
                        "end": end_time,
                        "text": whisper_result["text"].strip(),
                        "words": whisper_result.get("words", []),
                        "segment_id": i,
                        "speaker": speaker_label,
                    }
                    transcribed_segments.append(segment_data)

                # Update progress
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

            # Merge sentence segments if enabled
            if self.whisper_config.split_sentences:
                transcribed_segments = self._merge_sentence_segments(
                    transcribed_segments
                )

            # Combine all text
            full_text = " ".join(segment["text"] for segment in transcribed_segments)

            result = {
                "segments": transcribed_segments,
                "text": full_text,
                "language": self.whisper_config.language,
                "model": self.whisper_config.model,
                "total_segments": len(transcribed_segments),
            }

            logger.info(
                "Transcribed %s segments for %s",
                len(transcribed_segments),
                audio_path.name,
            )

        except Exception:
            logger.exception("Error transcribing %s", audio_path.name)
            return {"segments": [], "text": ""}
        return result

    def check_dependencies(self) -> list[str]:
        """Ensure an available Whisper backend exists."""
        errors = []

        if self.import_optional("mlx_whisper") is not None:
            logger.info("MLX Whisper is available for Apple Silicon optimization")
        elif self.import_optional("whisper") is not None:
            logger.info("Standard Whisper is available")
        else:
            errors.append("Neither MLX Whisper nor standard Whisper is installed")

        try:
            self.load_model()
        except ImportError as exc:
            errors.append(f"Cannot initialize Whisper: {exc}")

        return errors

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
            diarized = any(
                segment.get("speaker") not in (None, speaker_name)
                for segment in transcription.get("segments", [])
            )
            transcription_data = {
                "track": speaker_name,
                "speaker": speaker_name,
                "diarized": diarized,
                "transcription": transcription,
                "config": self.whisper_config.model_dump(),
            }

            json_path.write_text(
                json.dumps(transcription_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Save SRT
            srt_path = output_path.with_suffix(".srt")
            srt_content = self._create_srt_content(transcription["segments"])

            srt_path.write_text(srt_content, encoding="utf-8")

            logger.info("Saved transcription to %s and %s", json_path, srt_path)

        except OSError:
            logger.exception("Error saving transcription to %s", output_path)


class TranscriptionProcessor:
    """Main transcription processing coordinator."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcription processor."""
        self.config = config
        backend = config.transcription_backend
        self.backend = backend
        if backend == "parakeet":
            from .parakeet_transcribe import ParakeetTranscriber  # noqa: PLC0415

            self.transcriber = ParakeetTranscriber(config)
        else:
            self.transcriber = WhisperTranscriber(config)

    def find_audio_and_diarization_files(self) -> list[tuple[Path, Path]]:
        """Find matching audio and diarization files for transcription."""
        audio_diar_pairs: list[tuple[Path, Path]] = []

        if not self.config.paths.audio_wav_dir.exists():
            logger.warning(
                "Audio WAV directory does not exist: %s",
                self.config.paths.audio_wav_dir,
            )
            return audio_diar_pairs

        if not self.config.paths.diarization_dir.exists():
            logger.warning(
                "Diarization directory does not exist: %s",
                self.config.paths.diarization_dir,
            )
            return audio_diar_pairs

        # Find audio files
        audio_files = []
        for pattern in ["*.wav", "*.WAV"]:
            audio_files.extend(self.config.paths.audio_wav_dir.glob(pattern))

        # Match with diarization files
        for audio_path in audio_files:
            speaker_name = audio_path.stem
            diarization_path = (
                self.config.paths.diarization_dir / f"{speaker_name}_diarization.json"
            )

            if diarization_path.exists():
                audio_diar_pairs.append((audio_path, diarization_path))
            else:
                logger.warning(
                    "No diarization file found for %s: %s",
                    speaker_name,
                    diarization_path,
                )

        logger.info(
            "Found %s audio/diarization pairs for transcription", len(audio_diar_pairs)
        )
        return sorted(audio_diar_pairs)

    def get_output_path(self, audio_path: Path) -> Path:
        """Get the output path for transcription."""
        speaker_name = audio_path.stem
        return self.config.paths.whisper_dir / f"{speaker_name}_transcript"

    def process_files(
        self, audio_diar_pairs: list[tuple[Path, Path]] | None = None
    ) -> dict[str, Path]:
        """Process multiple audio files for transcription sequentially."""
        if audio_diar_pairs is None:
            audio_diar_pairs = self.find_audio_and_diarization_files()

        if not audio_diar_pairs:
            logger.warning("No audio/diarization pairs found for transcription")
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
            for audio_path, diarization_path in audio_diar_pairs:
                speaker_name = audio_path.stem
                output_path = self.get_output_path(audio_path)

                # Load diarization segments to determine total work
                try:
                    with diarization_path.open(encoding="utf-8") as file_obj:
                        diarization_data = json.load(file_obj)
                        total_segments = len(
                            diarization_data.get("merged_segments", [])
                        )
                except (OSError, json.JSONDecodeError):
                    total_segments = 1

                task_id = progress.add_task(
                    f"[cyan]Transcribing {speaker_name}...",
                    total=total_segments,
                )

                try:
                    # Transcribe file
                    transcription = self.transcriber.transcribe_file(
                        audio_path,
                        diarization_path,
                        progress,
                        task_id,
                    )
                    if progress and task_id is not None:
                        progress.update(task_id, completed=total_segments)

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

                except Exception:
                    logger.exception("Failed to transcribe %s", speaker_name)
                    failed_count += 1
                    progress.update(task_id, description=f"[red] {speaker_name}")

        console.print(
            f"[green]Successfully transcribed {len(successful_outputs)} files"
        )
        if failed_count > 0:
            console.print(f"[red]Failed to transcribe {failed_count} files")

        return successful_outputs

    def check_dependencies(self) -> list[str]:
        """Check if required dependencies are available."""
        if hasattr(self.transcriber, "check_dependencies"):
            return self.transcriber.check_dependencies()
        return []


def transcribe_audio(
    config: Config,
    audio_diar_pairs: list[tuple[Path, Path]] | None = None,
) -> dict[str, Path]:
    """Main function to transcribe audio files."""
    processor = TranscriptionProcessor(config)

    # Check dependencies
    errors = processor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "Transcription dependency check failed"
        raise RuntimeError(msg)

    # Process files
    return processor.process_files(audio_diar_pairs)


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
        "--diarization-dir",
        type=Path,
        help="Directory containing Senko diarization files",
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
    if args.diarization_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["diarization_dir"] = args.diarization_dir
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
    except (RuntimeError, OSError, ValueError) as exc:
        console.print(f"[red]Transcription failed: {exc}")
        sys.exit(1)
