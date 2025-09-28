"""Configuration module for the audio processing pipeline."""

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class FFmpegConfig(BaseModel):
    """Configuration for FFmpeg audio preprocessing."""

    sample_rate: int = Field(default=16000, description="Target sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels (1=mono)")
    bit_depth: str = Field(default="16", description="Audio bit depth")
    highpass_freq: int = Field(default=60, description="High-pass filter frequency in Hz")
    gate_threshold: str = Field(default="-45dB", description="Noise gate threshold")
    gate_ratio: int = Field(default=10, description="Noise gate ratio")
    gate_attack: int = Field(default=5, description="Noise gate attack time")
    gate_release: int = Field(default=60, description="Noise gate release time")


class SileroVADConfig(BaseModel):
    """Configuration for Silero VAD speech detection."""

    frame_ms: int = Field(default=32, description="Frame size in milliseconds (>=32)")
    block_seconds: float = Field(default=15.0, description="Streaming block size in seconds")
    threshold_start: float = Field(default=0.55, description="Start speech probability threshold")
    threshold_end: float = Field(default=0.35, description="End speech probability threshold")
    min_speech_ms: int = Field(default=300, description="Minimum speech duration to open segment")
    min_silence_ms: int = Field(default=500, description="Minimum silence to close segment")
    merge_gap_ms: int = Field(default=400, description="Merge segments if gap < this")
    pad_ms: int = Field(default=100, description="Padding for each segment")
    drop_below_ms: int = Field(default=200, description="Drop segments shorter than this")
    rms_gate_dbfs: float = Field(default=-50.0, description="RMS silence threshold in dBFS")


class WhisperConfig(BaseModel):
    """Configuration for Whisper transcription."""

    model: str = Field(default="small.en", description="Whisper model size/name")
    language: str = Field(default="en", description="Language hint")
    temperature: float = Field(default=0.0, description="Decoding temperature")
    split_sentences: bool = Field(default=True, description="Split into sentence segments")
    min_sentence_ms: int = Field(default=1200, description="Minimum sentence duration for merging")
    merge_sentence_gap_ms: int = Field(default=200, description="Max gap for sentence merging")


class GPTConfig(BaseModel):
    """Configuration for GPT post-processing."""

    model: str = Field(default="gpt-4", description="OpenAI model to use")
    max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="Generation temperature")
    chunk_size: int = Field(default=8000, description="Text chunk size for processing")


class PathConfig(BaseModel):
    """Configuration for file paths."""

    inputs_dir: Path = Field(default=Path("inputs"), description="Input audio files directory")
    outputs_dir: Path = Field(default=Path("outputs"), description="Output directory")
    audio_wav_dir: Path = Field(default=Path("outputs/audio-files-wav"), description="Preprocessed audio directory")
    silero_dir: Path = Field(default=Path("outputs/silero-timestamps"), description="VAD timestamps directory")
    whisper_dir: Path = Field(default=Path("outputs/whisper-transcripts"), description="Transcripts directory")
    gpt_dir: Path = Field(default=Path("outputs/gpt-cleanup"), description="Final transcripts directory")
    prompts_dir: Path = Field(default=Path("prompts"), description="GPT prompts directory")
    status_file: Path = Field(default=Path("outputs/status.json"), description="Pipeline status file")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Resolve all paths to absolute paths."""
        if isinstance(v, (str, Path)):
            return Path(v).resolve()
        return v


class Config(BaseModel):
    """Main configuration class for the audio processing pipeline."""

    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    silero: SileroVADConfig = Field(default_factory=SileroVADConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    gpt: GPTConfig = Field(default_factory=GPTConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    huggingface_token: str | None = Field(default=None, description="HuggingFace token")

    # Performance settings
    max_parallel_audio: int = Field(default=4, description="Max parallel audio preprocessing jobs")
    max_parallel_vad: int = Field(default=4, description="Max parallel VAD jobs")

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"

    def __init__(self, **kwargs) -> None:
        """Initialize configuration with environment variables."""
        # Load from environment variables
        env_overrides = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
        }

        # Remove None values
        env_overrides = {k: v for k, v in env_overrides.items() if v is not None}

        # Merge with provided kwargs
        kwargs = {**env_overrides, **kwargs}

        super().__init__(**kwargs)

    def validate_environment(self) -> list[str]:
        """Validate that all required environment variables and dependencies are available."""
        errors = []

        # Check required API keys
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required")

        # Check required directories exist or can be created
        try:
            for dir_path in [
                self.paths.outputs_dir,
                self.paths.audio_wav_dir,
                self.paths.silero_dir,
                self.paths.whisper_dir,
                self.paths.gpt_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directories: {e}")

        # Check if inputs directory exists
        if not self.paths.inputs_dir.exists():
            errors.append(f"Input directory does not exist: {self.paths.inputs_dir}")

        return errors

    def create_directories(self) -> None:
        """Create all required output directories."""
        for dir_path in [
            self.paths.outputs_dir,
            self.paths.audio_wav_dir,
            self.paths.silero_dir,
            self.paths.whisper_dir,
            self.paths.gpt_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def load_config(config_file: Path | None = None, **overrides) -> Config:
    """Load configuration from file and environment variables."""
    config_data = {}

    # Load from file if provided
    if config_file and config_file.exists():
        import json
        with open(config_file) as f:
            config_data = json.load(f)

    # Apply overrides
    config_data.update(overrides)

    return Config(**config_data)
