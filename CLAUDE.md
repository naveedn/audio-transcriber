# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI-based speech-to-text transcription application optimized for Apple Silicon that processes multi-track recordings into accurate transcripts with speaker identification and timestamps. The system uses a 5-stage pipeline to convert FLAC audio files into polished transcripts.

## Development Commands

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Install development dependencies
uv sync --dev
```

### Linting and Code Quality
```bash
# Set up pre-commit hooks (one-time setup for contributors)
uv run pre-commit install

# Pre-commit automatically runs on git commit
# Run manually on all files:
uv run pre-commit run --all-files

# Run ruff directly:
uv run ruff check           # Run linting
uv run ruff check --fix     # Fix auto-fixable linting issues
uv run ruff format          # Format code
```

### Running the Application
```bash
# Run CLI command
uv run transcribe --help

# Run specific commands
uv run transcribe validate    # Check dependencies and API keys
uv run transcribe status      # Show pipeline status
uv run transcribe run         # Execute full pipeline
uv run transcribe run-stage preprocess  # Run specific stage
uv run transcribe reset       # Reset pipeline status
```

### Testing
```bash
# The project currently has no tests implemented
# Test directory exists at ./tests/ but is empty
```

## Architecture

### Pipeline Stages
The system processes audio through a 5-stage pipeline:

```
Audio Files → [Stage 0] → [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4] → Final Transcript
              Bootstrap   preprocess  Senko      Whisper      GPT
```

1. **Stage 0: Bootstrap** - Model download and environment validation
2. **Stage 1: Audio Preprocess** - FLAC to 16kHz mono WAV conversion using FFmpeg
3. **Stage 2: Senko Diarization** - Speaker clustering + VAD for segmentation
4. **Stage 3: Whisper Transcribe** - Speech-to-text transcription with timestamps
5. **Stage 4: GPT Processing** - Intelligent transcript merging and cleanup

### Key Components

- **src/config.py**: Pydantic-based configuration system with validation
- **src/main.py**: CLI interface and pipeline orchestration using Click
- **src/ffmpeg_preprocess.py**: Audio preprocessing with parallel processing
- **src/senko_diarizer.py**: Senko diarization integration for speech detection
- **src/whisper_transcribe.py**: MLX Whisper for Apple Silicon optimization
- **src/gpt_merge.py**: Multi-speaker transcript merging
- **src/gpt_cleanup.py**: Final transcript post-processing

### Directory Structure
```
inputs/                    # Input FLAC audio files
outputs/
  ├── audio-files-wav/     # Converted WAV files
  ├── senko-diarization/   # Diarization bundles
  ├── whisper-transcripts/ # Raw transcriptions
  ├── gpt-cleanup/         # Final transcripts
  └── status.json          # Pipeline state tracking
prompts/                   # GPT prompt templates
src/                       # Source code
```

### Resume Functionality
The pipeline tracks progress in `outputs/status.json` and can resume from any completed stage. Each stage is independent and validates its prerequisites before execution.

## Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:
```bash
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here  # optional
```

### Default Configuration
The system uses optimal defaults from the specification:
- FFmpeg: 16kHz mono with noise gate and high-pass filter
- Senko diarization: Pyannote VAD + CAM++ embeddings with auto device selection
- Whisper: Small.en model with deterministic decoding
- GPT: OpenAI API for intelligent post-processing

## Technology Stack

- **Python 3.11+** with uv package manager
- **Audio Processing**: FFmpeg, librosa, soundfile
- **ML/AI**: Senko diarization, MLX Whisper (Apple Silicon), OpenAI API
- **CLI**: Click with Rich console interface
- **Configuration**: Pydantic with type validation
- **Async**: asyncio for I/O-bound operations

## Development Notes

### Performance Targets
- Process 3-hour multi-track recording in ≤20 minutes (20x real-time speed)
- Parallel processing for I/O-bound stages (preprocessing, VAD)
- Sequential Whisper processing to manage memory limits on MacBooks


### Code Style
- Line length: 88 characters (configured in pyproject.toml)
- Linting: Comprehensive ruff configuration with strict rules
- Type hints: Required for all functions (ANN rules enabled)
- Documentation: Google-style docstrings expected
- Pre-commit hooks: Automatically enforce code quality on commits (configured in .pre-commit-config.yaml)

## API Dependencies

- **OpenAI API**: Required for GPT-based transcript processing
- **HuggingFace**: Optional, for accessing gated models
- **MLX Whisper**: Automatic fallback to standard Whisper if unavailable
