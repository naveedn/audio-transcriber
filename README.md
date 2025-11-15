# Audio Transcriber

## 🎯 Overview

A speech-to-text transcription cli application that processes multi-track FLAC audio recordings into a single, unified transcript. Designed to be used alongside [craig.chat](https://craig.chat/). Use it for recordings like podcasts, meetings, or DND sessions. The program leverages Apple Silicon and modern Python tooling to run as fast as possible.

[![Craig's Mascot](https://craig.chat/icon-192x192.png)](https://craig.chat/)

Demo Video (and program design) here: https://www.youtube.com/watch?v=v0KZGyJARts&t=300s


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd audio-transcriber

# create an inputs folder where audio files will be placed
mkdir inputs

# Install dependencies
uv sync

# Set up pre-commit hooks (for contributors)
uv run pre-commit install

# Create environment file
cp .env.template .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# HUGGINGFACE_TOKEN=your_huggingface_token_here  # optional

# Parakeet setup (if you plan to use Parakeet TDT 0.6B v2/v3 models)
cd parakeet_bridge
git clone --depth 1 https://github.com/FluidInference/FluidAudio.git
swift build -c release --product parakeet-transcriber
cd ..
```

### Basic Usage

```bash
# Run full pipeline or resume from the last completed stage
uv run transcribe run

# Override Stage 3 backend
uv run transcribe run --transcription-backend whisper|parakeet-v2|parakeet-v3

# Validate installation and API keys
uv run transcribe validate

# Check pipeline status
uv run transcribe status

# Run specific stage
uv run transcribe run-stage preprocess

# Reset pipeline status
uv run transcribe reset

# Clean everything (reset + remove all inputs and outputs)
uv run transcribe clean

# Get help
uv run transcribe --help
```


## 📊 Pipeline Architecture

```
Audio Files → [Stage 0] → [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4] → Final Transcript
              Bootstrap   Preprocess    Senko     Transcribe    GPT
```
### 🏗️ Architecture Benefits

- **Fast**: Process 3-hour recordings in ≤20 minutes (20x real-time speed)
- **Modular**: Each stage is independent and can be developed/tested separately
- **Resumable**: Can restart from any completed stage via `outputs/status.json`
- **Debuggable**: Inspect intermediate outputs at each stage
- **Apple Silicon Optimized**: Uses MLX Whisper for M1/M2 performance

### Stage 0: Bootstrap
- **Purpose**: Download ML models and validate environment
- **Technology**: Hugging Face model downloads
- **Output**: Ready environment with all dependencies

### Stage 1: Audio Preprocess
- **Input**: Raw FLAC audio files (`inputs/<speaker-name>.flac`)
- **Purpose**: Convert to 16kHz mono WAV with noise filtering
- **Technology**: FFmpeg with parallel processing
- **Output**: Cleaned audio files (`outputs/audio-files-wav/`)

### Stage 2: Senko Diarization
- **Input**: Cleaned audio files
- **Purpose**: Detect speech activity, cluster speakers, and produce diarized segments
- **Technology**: [Senko](https://github.com/narcotic-sh/senko) diarizer (Pyannote VAD + CAM++ embeddings)
- **Output**: JSON diarization bundles (`outputs/senko-diarization/`)

### Stage 3: Speech Transcription
- **Input**: Speech segments and audio files
- **Purpose**: Convert speech to text with word-level timestamps
- **Technology**:
  - **Default**: MLX Whisper (Apple Silicon optimized) with CPU/GPU fallback
  - **Optional**: NVIDIA Parakeet TDT 0.6B V2/V3 via [FluidAudio](https://github.com/FluidInference/FluidAudio)
- **Output**: Raw transcripts with timestamps (`outputs/transcripts/`)

### Stage 4: GPT Processing
- **Input**: Raw transcripts from all speakers
- **Purpose**: Merge transcripts, clean hallucinations, format for readability
- **Technology**: OpenAI API for intelligent post-processing
- **Output**: Final polished transcripts (`outputs/gpt-cleanup/`)

## 📁 Directory Structure

```
transcribe/
├── inputs/                     # Input FLAC audio files
├── outputs/
│   ├── audio-files-wav/        # Stage 1: Converted WAV files
│   ├── senko-diarization/      # Stage 2: Diarization results + VAD windows
│   ├── transcripts/            # Stage 3: Raw transcriptions
│   ├── gpt-cleanup/            # Stage 4: Final transcripts
│   └── status.json             # Pipeline state tracking
├── prompts/                    # GPT prompt templates
├── src/                        # Source code
│   ├── main.py                 # CLI interface and orchestration
│   ├── config.py               # Configuration management
│   ├── ffmpeg_preprocess.py    # Audio preprocessing
│   ├── senko_diarizer.py       # Senko diarization integration
│   ├── whisper_transcribe.py   # Stage 3 orchestration + Whisper backend
│   ├── parakeet_transcribe.py  # Parakeet CoreML backend shim
│   ├── gpt_merge.py            # Multi-speaker merging
│   └── gpt_cleanup.py          # Final transcript cleanup
└── tests/                      # Test directory (currently empty)
```

## 🔧 Configuration

### Input Files

Place your audio files in the `inputs/` directory (`.flac` as an example):
```
inputs/
├── speaker1.flac
├── speaker2.flac
└── speaker3.flac
```

The filename (without extension) is used as the speaker label in the final transcript.

### 📝 Output Formats

Final transcripts are available in multiple formats:
- **JSON**: Machine-readable with full metadata
- **SRT**: Standard subtitle format with speaker labels

### Optimal Settings (Pre-configured)

The system includes battle-tested configurations for best results:

**FFmpeg Audio Processing:**
```bash
ffmpeg -i input.flac -ar 16000 -af "highpass=f=60,agate=threshold=-45dB:ratio=10:attack=5:release=60" output.wav
```

**Senko Diarization Defaults:**
- `device='auto'`, `vad='auto'`, and `clustering='auto'` with optional overrides via `config.senko`
- Warmup enabled by default for best throughput (can be disabled in config)
- Accurate mode automatically mirrors Senko's GPU heuristics; toggle via `config.senko.accurate`
- Diarization JSON captures merged segments, speaker centroids, and VAD windows for downstream stages

**Whisper Settings:**
- Model: small.en (optimal speed/accuracy balance)
- Temperature: 0.0 (deterministic output)
- Language: English with sentence-level segmentation

## 🛠️ Development

### Code Quality

The project uses `pre-commit` hooks to automatically run code quality checks before commits.

```bash
# Set up pre-commit hooks (one-time setup)
uv run pre-commit install

# Pre-commit will now run automatically on git commit
# You can also run it manually on all files:
uv run pre-commit run --all-files

# Or run ruff directly:
uv run ruff check           # Run linting
uv run ruff check --fix     # Fix auto-fixable issues
uv run ruff format          # Format code
```

### Technology Stack

- **Python 3.11+** with uv package manager
- **Audio Processing**: FFmpeg, librosa, soundfile
- **ML/AI**: Senko diarization, MLX Whisper (Apple Silicon), OpenAI API
- **CLI**: Click with Rich console interface
- **Configuration**: Pydantic with type validation
- **Async**: asyncio for I/O-bound operations

## 📋 Requirements

### System Dependencies
- Python 3.11+
- uv package manager
- FFmpeg (for audio processing)

### API Keys
- **OpenAI API Key**: Required for GPT-based transcript processing
- **Hugging Face Token**: Optional, for accessing gated models

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file contains valid OpenAI API key
2. **FFmpeg Not Found**: Install FFmpeg: `brew install ffmpeg` or `winget install ffmpeg` if on [windows](https://www.gyan.dev/ffmpeg/builds/)
3. **Memory Issues**: Large files may require breaking into smaller segments
4. **MLX Whisper Issues**: Falls back to standard Whisper automatically
