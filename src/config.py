"""Configuration module for the audio processing pipeline."""

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FFmpegConfig(BaseModel):
    """Configuration for FFmpeg audio preprocessing."""

    sample_rate: int = Field(default=16000, description="Target sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels (1=mono)")
    bit_depth: str = Field(default="16", description="Audio bit depth")
    highpass_freq: int = Field(
        default=60, description="High-pass filter frequency in Hz"
    )
    gate_threshold: str = Field(default="-45dB", description="Noise gate threshold")
    gate_ratio: int = Field(default=10, description="Noise gate ratio")
    gate_attack: int = Field(default=5, description="Noise gate attack time")
    gate_release: int = Field(default=60, description="Noise gate release time")


class SenkoConfig(BaseModel):
    """Configuration for Senko diarization."""

    device: str = Field(default="auto", description="Device override for Senko")
    vad: str = Field(default="auto", description="VAD backend override")
    clustering: str = Field(
        default="auto", description="Clustering location preference for CUDA devices"
    )
    warmup: bool = Field(
        default=True, description="Warm up Senko models during initialization"
    )
    quiet: bool = Field(default=True, description="Silence Senko progress logging")
    accurate: bool | None = Field(
        default=None,
        description="Pass through to Senko accurate mode; defaults to Senko heuristics",
    )
    generate_colors: bool = Field(
        default=False, description="Request speaker colors from Senko"
    )


class WhisperConfig(BaseModel):
    """Configuration for Whisper transcription."""

    model: str = Field(default="small.en", description="Whisper model size/name")
    language: str = Field(default="en", description="Language hint")
    temperature: float = Field(default=0.0, description="Decoding temperature")
    split_sentences: bool = Field(
        default=True, description="Split into sentence segments"
    )
    min_sentence_ms: int = Field(
        default=1200, description="Minimum sentence duration for merging"
    )
    merge_sentence_gap_ms: int = Field(
        default=200, description="Max gap for sentence merging"
    )


class ParakeetConfig(BaseModel):
    """Configuration for the Parakeet CoreML transcription backend."""

    executable_path: Path = Field(
        default=Path("parakeet_bridge/.build/release/parakeet-transcriber"),
        description="Compiled Parakeet Swift bridge executable",
    )
    model_version: Literal["v2", "v3"] = Field(
        default="v2", description="Parakeet ASR model version to run"
    )
    models_root: Path | None = Field(
        default=None,
        description="Optional directory containing downloaded Parakeet models",
    )
    min_segment_seconds: float = Field(
        default=1.0, description="Minimum segment duration sent to CoreML"
    )
    language: str = Field(
        default="en",
        description="Language metadata propagated to the transcription output",
    )

    @field_validator("executable_path", "models_root", mode="before")
    @classmethod
    def resolve_paths(cls, value: object) -> object:
        """Resolve CLI paths to absolute Path instances."""
        if isinstance(value, (str, Path)):
            return Path(value).expanduser().resolve()
        return value


class GPTConfig(BaseModel):
    """Configuration for GPT post-processing."""

    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    max_tokens: int = Field(default=16384, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="Generation temperature")


class PathConfig(BaseModel):
    """Configuration for file paths."""

    inputs_dir: Path = Field(
        default=Path("inputs"), description="Input audio files directory"
    )
    outputs_dir: Path = Field(default=Path("outputs"), description="Output directory")
    audio_wav_dir: Path = Field(
        default=Path("outputs/audio-files-wav"),
        description="Preprocessed audio directory",
    )
    diarization_dir: Path = Field(
        default=Path("outputs/senko-diarization"),
        description="Senko diarization output directory",
    )
    whisper_dir: Path = Field(
        default=Path("outputs/transcripts"), description="Transcripts directory"
    )
    gpt_dir: Path = Field(
        default=Path("outputs/gpt-cleanup"), description="Final transcripts directory"
    )
    prompts_dir: Path = Field(
        default=Path("prompts"), description="GPT prompts directory"
    )
    status_file: Path = Field(
        default=Path("outputs/status.json"), description="Pipeline status file"
    )

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, value: object) -> object:
        """Resolve all paths to absolute paths."""
        if isinstance(value, (str, Path)):
            return Path(value).resolve()
        return value


class Config(BaseModel):
    """Main configuration class for the audio processing pipeline."""

    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    senko: SenkoConfig = Field(default_factory=SenkoConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    parakeet: ParakeetConfig = Field(default_factory=ParakeetConfig)
    gpt: GPTConfig = Field(default_factory=GPTConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    transcription_backend: Literal["whisper", "parakeet"] = Field(
        default="whisper",
        description="Speech-to-text backend to use for Stage 3",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    huggingface_token: str | None = Field(default=None, description="HuggingFace token")

    # Performance settings
    max_parallel_audio: int = Field(
        default=4, description="Max parallel audio preprocessing jobs"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"

    def __init__(self, **kwargs: object) -> None:
        """Initialize configuration with environment variables."""
        # Load from environment variables
        env_overrides: dict[str, object | None] = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
        }
        merged_overrides = {
            key: value for key, value in env_overrides.items() if value is not None
        }

        # Merge with provided kwargs
        merged_kwargs: dict[str, object] = {**merged_overrides, **kwargs}

        super().__init__(**merged_kwargs)

    def validate_environment(self) -> list[str]:
        """Validate required environment variables and filesystem dependencies."""
        errors = []

        # Check required API keys
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required")

        # Check required directories exist or can be created
        try:
            for dir_path in [
                self.paths.outputs_dir,
                self.paths.audio_wav_dir,
                self.paths.diarization_dir,
                self.paths.whisper_dir,
                self.paths.gpt_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Cannot create output directories: {exc}")

        # Check if inputs directory exists
        if not self.paths.inputs_dir.exists():
            errors.append(f"Input directory does not exist: {self.paths.inputs_dir}")

        return errors

    def create_directories(self) -> None:
        """Create all required output directories."""
        for dir_path in [
            self.paths.outputs_dir,
            self.paths.audio_wav_dir,
            self.paths.diarization_dir,
            self.paths.whisper_dir,
            self.paths.gpt_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def load_config(config_file: Path | None = None, **overrides: object) -> Config:
    """Load configuration from file and environment variables."""
    config_data: dict[str, object] = {}

    # Load from file if provided
    if config_file and config_file.exists():
        with config_file.open() as file_obj:
            config_data = json.load(file_obj)

    # Apply overrides
    config_data.update(dict(overrides))

    return Config(**config_data)
