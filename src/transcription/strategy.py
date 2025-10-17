"""Transcription strategy pattern for multiple ML models."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from rich.progress import Progress

from ..config import Config

logger = logging.getLogger(__name__)


class TranscriptionStrategy(ABC):
    """Abstract base class for transcription strategies."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcription strategy.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_config = config.model
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the transcription model."""

    @abstractmethod
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """Transcribe a single audio segment.

        Args:
            audio_segment: Audio data as numpy array
            sample_rate: Audio sample rate in Hz

        Returns:
            Dictionary with transcription results including text, segments, and words
        """

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: Path,
        vad_path: Path,
        progress: Progress | None = None,
        task_id: Optional = None,
    ) -> dict:
        """Transcribe a complete audio file using VAD segments.

        Args:
            audio_path: Path to audio file
            vad_path: Path to VAD timestamps JSON
            progress: Rich progress bar instance
            task_id: Progress task ID for updates

        Returns:
            Dictionary with full transcription results
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the transcription model.

        Returns:
            Model name string
        """


def create_transcription_strategy(config: Config) -> TranscriptionStrategy:
    """Factory function to create the appropriate transcription strategy.

    Strategy selection logic:
    - If model contains "parakeet": ParakeetStrategy (streaming or non-streaming)
    - Otherwise: WhisperStrategy (default, includes models like "small.en", "base.en")

    Args:
        config: Application configuration

    Returns:
        Appropriate TranscriptionStrategy instance
    """
    model_name = config.model.model.lower()

    # Check for Parakeet models first
    if "parakeet" in model_name:
        from .parakeet_strategy import ParakeetStrategy

        use_streaming = getattr(config.model, "use_streaming", False)
        mode = "streaming" if use_streaming else "non-streaming"
        logger.info(f"Using Parakeet transcription strategy ({mode} mode)")
        return ParakeetStrategy(config)

    # Default to Whisper for all other models
    # This includes standard Whisper models like "small.en", "base.en", "medium.en"
    # as well as explicit "whisper-*" named models
    from .whisper_strategy import WhisperStrategy

    logger.info("Using Whisper transcription strategy")
    return WhisperStrategy(config)
