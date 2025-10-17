"""Transcription module for speech-to-text processing.

This module provides a strategy pattern-based transcription system supporting
multiple models (Whisper, Parakeet) with different modes (streaming, non-streaming).
"""

from .strategy import TranscriptionStrategy, create_transcription_strategy

__all__ = [
    "TranscriptionStrategy",
    "create_transcription_strategy",
]
