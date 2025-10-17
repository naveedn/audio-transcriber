"""Parakeet transcription strategy implementation (streaming and non-streaming modes)."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from rich.progress import Progress

from ..config import Config
from .strategy import TranscriptionStrategy

logger = logging.getLogger(__name__)


class ParakeetStrategy(TranscriptionStrategy):
    """Parakeet-based transcription strategy supporting both streaming and non-streaming modes."""

    def __init__(self, config: Config) -> None:
        """Initialize the Parakeet strategy."""
        super().__init__(config)
        self.parakeet_model = None
        self.sample_rate = None
        self.use_streaming = getattr(config.model, "use_streaming", False)

    def load_model(self) -> None:
        """Initialize Parakeet model."""
        try:
            from parakeet_mlx import from_pretrained

            self.parakeet_model = from_pretrained(self.model_config.model)
            # Store sample rate for audio loading
            self.sample_rate = self.parakeet_model.preprocessor_config.sample_rate

            mode = "streaming" if self.use_streaming else "non-streaming"
            logger.info(
                f"Parakeet model loaded: {self.model_config.model} "
                f"(sample_rate={self.sample_rate}Hz, mode={mode})"
            )
        except ImportError:
            msg = "Parakeet-mlx is not available. This application requires Apple Silicon and parakeet-mlx."
            raise ImportError(msg)
        except Exception as e:
            logger.exception(f"Failed to initialize Parakeet: {e}")
            raise

    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """Transcribe a single audio segment (non-streaming fallback)."""
        try:
            # Ensure model is initialized
            if self.parakeet_model is None:
                self.load_model()

            # Apply MLX memory optimization if available
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:
                pass  # MLX core not available, continue without memory management

            # Parakeet requires a file path, so we need to save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                # Write audio segment to temporary file
                sf.write(temp_path, audio_segment, sample_rate)

            try:
                # Parakeet transcription
                result = self.parakeet_model.transcribe(temp_path)

                # Convert Parakeet's AlignedResult to our expected format
                # Parakeet returns: AlignedResult with .text and .sentences
                # We need: {"text": str, "segments": list, "words": list}
                segments = []
                for sentence in result.sentences:
                    segment = {
                        "start": sentence.start,
                        "end": sentence.end,
                        "text": sentence.text,
                        "words": [
                            {
                                "word": token.text,
                                "start": token.start,
                                "end": token.end,
                            }
                            for token in sentence.tokens
                        ],
                    }
                    segments.append(segment)

                return {
                    "text": result.text,
                    "segments": segments,
                    "words": [
                        word for seg in segments for word in seg.get("words", [])
                    ],
                }
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

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
        """Transcribe a single audio file using VAD segments.

        Uses streaming mode if configured, otherwise uses non-streaming mode.
        """
        try:
            mode = "streaming" if self.use_streaming else "non-streaming"
            logger.info(f"Transcribing {audio_path.name} with Parakeet ({mode} mode)")

            # Load VAD segments
            vad_segments = self._load_vad_segments(vad_path)
            if not vad_segments:
                logger.warning(f"No VAD segments found for {audio_path.name}")
                return {"segments": [], "text": ""}

            logger.info(f"Processing {len(vad_segments)} VAD segments")

            # Choose transcription method based on config
            if self.use_streaming:
                # Load only VAD-active audio segments
                audio_segments = self._load_vad_audio_segments(audio_path, vad_segments)

                # Transcribe with streaming context
                transcribed_segments = self._transcribe_with_streaming(
                    audio_segments,
                    progress,
                    task_id,
                )
            else:
                # Non-streaming: transcribe each segment independently
                transcribed_segments = self._transcribe_non_streaming(
                    audio_path,
                    vad_segments,
                    progress,
                    task_id,
                )

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
        """Get the name of the Parakeet model."""
        return self.model_config.model

    def _transcribe_non_streaming(
        self,
        audio_path: Path,
        vad_segments: list[dict],
        progress: Progress | None = None,
        task_id: Optional = None,
    ) -> list[dict]:
        """Transcribe segments independently (non-streaming mode)."""
        # Load audio
        audio, sample_rate = librosa.load(str(audio_path), sr=16000)

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
            parakeet_result = self.transcribe_segment(audio_segment, sample_rate)

            if parakeet_result.get("text", "").strip():
                segment_data = {
                    "start": start_time,
                    "end": end_time,
                    "text": parakeet_result["text"].strip(),
                    "words": parakeet_result.get("words", []),
                    "segment_id": i,
                }
                transcribed_segments.append(segment_data)

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

        return transcribed_segments

    def _load_vad_audio_segments(
        self,
        audio_path: Path,
        vad_segments: list[dict],
    ) -> list[tuple[np.ndarray, float, float]]:
        """Load only VAD-active audio segments from file (for streaming mode).

        Args:
            audio_path: Path to audio file
            vad_segments: List of VAD segments with start/end times

        Returns:
            List of (audio_chunk, start_time, end_time) tuples
        """
        try:
            from parakeet_mlx.audio import load_audio

            # Ensure model is loaded to get sample rate
            if self.parakeet_model is None:
                self.load_model()

            # Load full audio once (parakeet's load_audio doesn't support time slicing)
            audio_full = load_audio(audio_path, self.sample_rate)

            # Extract segments from full audio
            audio_segments = []
            for segment in vad_segments:
                start_time = segment["start"]
                end_time = segment["end"]

                # Calculate sample indices
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_full), end_sample)

                # Extract segment
                audio_chunk = audio_full[start_sample:end_sample]
                audio_segments.append((audio_chunk, start_time, end_time))

            return audio_segments

        except Exception as e:
            logger.exception(f"Error loading VAD audio segments from {audio_path}: {e}")
            return []

    def _transcribe_with_streaming(
        self,
        audio_segments: list[tuple[np.ndarray, float, float]],
        progress: Progress | None = None,
        task_id: Optional = None,
    ) -> list[dict]:
        """Transcribe audio segments using streaming context.

        Args:
            audio_segments: List of (audio_chunk, start_time, end_time) tuples
            progress: Progress bar instance
            task_id: Progress task ID

        Returns:
            List of transcribed segments with adjusted timestamps
        """
        try:
            # Ensure model is initialized
            if self.parakeet_model is None:
                self.load_model()

            # Apply MLX memory optimization if available
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:
                pass  # MLX core not available, continue without memory management

            transcribed_segments = []
            segment_offset = 0  # Track which VAD segment we're processing

            # Get context frames from config
            context_frames = getattr(self.model_config, "context_frames", (256, 256))

            # Use streaming context for cross-segment awareness
            with self.parakeet_model.transcribe_stream(
                context_size=context_frames,
            ) as transcriber:
                for audio_chunk, vad_start_time, vad_end_time in audio_segments:
                    # Skip very short segments
                    duration = vad_end_time - vad_start_time
                    if duration < 0.1:  # 100ms minimum
                        segment_offset += 1
                        if progress and task_id is not None:
                            progress.update(task_id, advance=1)
                        continue

                    # Add audio to streaming context
                    transcriber.add_audio(audio_chunk)

                    # Get current result
                    result = transcriber.result

                    # Process sentences from the result
                    # Note: result.sentences contains all finalized sentences so far
                    # We need to track which ones we've already processed
                    current_sentence_count = len(result.sentences)

                    # Process only new sentences (those beyond what we've already seen)
                    for sentence in result.sentences[len(transcribed_segments) :]:
                        segment = {
                            "start": sentence.start,
                            "end": sentence.end,
                            "text": sentence.text,
                            "words": [
                                {
                                    "word": token.text,
                                    "start": token.start,
                                    "end": token.end,
                                }
                                for token in sentence.tokens
                            ],
                        }
                        transcribed_segments.append(segment)

                    # Update progress
                    segment_offset += 1
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

            return transcribed_segments

        except Exception as e:
            logger.exception(f"Error transcribing with streaming: {e}")
            return []

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
