"""Whisper transcription strategy implementation."""

import json
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from rich.progress import Progress

from ..config import Config
from .strategy import TranscriptionStrategy

logger = logging.getLogger(__name__)


class WhisperStrategy(TranscriptionStrategy):
    """Whisper-based transcription strategy (MLX or standard Whisper)."""

    def __init__(self, config: Config) -> None:
        """Initialize the Whisper strategy."""
        super().__init__(config)
        self.use_mlx = False
        self.mlx_module = None

    def load_model(self) -> None:
        """Initialize Whisper implementation (MLX Whisper doesn't require model loading)."""
        try:
            import mlx_whisper

            self.mlx_module = mlx_whisper
            self.use_mlx = True
            logger.info("MLX Whisper available for Apple Silicon optimization")

        except ImportError:
            logger.warning(
                "⚠️ MLX Whisper not available, falling back to standard Whisper"
            )
            try:
                import whisper

                self.model = whisper.load_model(self.model_config.model)
                self.use_mlx = False
                logger.info("Standard Whisper model loaded successfully")
            except ImportError:
                msg = "Neither MLX Whisper nor standard Whisper is available"
                raise ImportError(msg)
        except Exception as e:
            logger.exception(f"Failed to initialize Whisper: {e}")
            raise

    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
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
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except ImportError:
                    pass  # MLX core not available, continue without memory management

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
                    language=self.model_config.language,
                    temperature=getattr(self.model_config, "temperature", 0),
                    word_timestamps=True,
                )

            return result

        except Exception as e:
            logger.exception(f"Error transcribing audio segment: {e}")
            return {"text": "", "segments": [], "words": []}

    def transcribe_file(
        self,
        audio_path: Path,
        vad_path: Path,
        progress: Progress | None = None,
        task_id: Optional = None,
    ) -> dict:
        """Transcribe a single audio file using VAD segments."""
        try:
            logger.info(f"Transcribing {audio_path.name} with Whisper")

            # Load audio
            audio, sample_rate = librosa.load(str(audio_path), sr=16000)

            # Load VAD segments
            vad_segments = self._load_vad_segments(vad_path)
            if not vad_segments:
                logger.warning(f"No VAD segments found for {audio_path.name}")
                return {"segments": [], "text": ""}

            logger.info(f"Processing {len(vad_segments)} VAD segments")

            # Transcribe each segment
            transcribed_segments = []

            for i, vad_segment in enumerate(vad_segments):
                start_time = vad_segment["start"]
                end_time = vad_segment["end"]

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
                whisper_result = self.transcribe_segment(audio_segment, sample_rate)

                if whisper_result.get("text", "").strip():
                    segment_data = {
                        "start": start_time,
                        "end": end_time,
                        "text": whisper_result["text"].strip(),
                        "words": whisper_result.get("words", []),
                        "segment_id": i,
                    }
                    transcribed_segments.append(segment_data)

                # Update progress
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

            # Merge sentence segments if enabled
            if getattr(self.model_config, "split_sentences", True):
                transcribed_segments = self._merge_sentence_segments(
                    transcribed_segments
                )

            # Combine all text
            full_text = " ".join(segment["text"] for segment in transcribed_segments)

            result = {
                "segments": transcribed_segments,
                "text": full_text,
                "language": self.model_config.language,
                "model": self.model_config.model,
                "total_segments": len(transcribed_segments),
            }

            logger.info(
                f"Transcribed {len(transcribed_segments)} segments for {audio_path.name}"
            )
            return result

        except Exception as e:
            logger.exception(f"Error transcribing {audio_path.name}: {e}")
            return {"segments": [], "text": ""}

    def get_model_name(self) -> str:
        """Get the name of the Whisper model."""
        return self.model_config.model

    def _load_vad_segments(self, vad_path: Path) -> list[dict]:
        """Load VAD segments from JSON file."""
        try:
            with open(vad_path) as f:
                data = json.load(f)
                return data.get("segments", [])
        except Exception as e:
            logger.exception(f"Error loading VAD segments from {vad_path}: {e}")
            return []

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

    def _merge_sentence_segments(self, segments: list[dict]) -> list[dict]:
        """Merge adjacent segments to create sentence-level segments."""
        if not segments:
            return segments

        merge_gap_ms = getattr(self.model_config, "merge_sentence_gap_ms", 200)
        min_sentence_ms = getattr(self.model_config, "min_sentence_ms", 1200)

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

            should_merge = (
                gap_ms <= merge_gap_ms
                and current_duration < min_sentence_ms
                and next_duration < min_sentence_ms
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
