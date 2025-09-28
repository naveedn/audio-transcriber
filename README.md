# Transcribe

A CLI-based speech-to-text transcription application optimized for Apple Silicon that processes multi-track recordings into accurate transcripts with speaker identification and timestamps.

## 🎯 Overview

Transcribe uses a 5-stage pipeline to convert FLAC audio files into polished transcripts, designed specifically for multi-speaker recordings like podcasts or meetings. The system leverages Apple Silicon optimization and modern Python tooling for fast, high-quality transcription.

## 📊 Pipeline Architecture

```
Audio Files → [Stage 0] → [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4] → Final Transcript
              Bootstrap   Preprocess  Silero     Whisper      GPT
```

### Stage 0: Bootstrap
- **Purpose**: Download ML models and validate environment
- **Technology**: Hugging Face model downloads
- **Output**: Ready environment with all dependencies

### Stage 1: Audio Preprocess
- **Input**: Raw FLAC audio files (`inputs/<speaker-name>.flac`)
- **Purpose**: Convert to 16kHz mono WAV with noise filtering
- **Technology**: FFmpeg with parallel processing
- **Output**: Cleaned audio files (`outputs/audio-files-wav/`)

### Stage 2: Silero VAD (Voice Activity Detection)
- **Input**: Cleaned audio files
- **Purpose**: Detect speech segments and filter silence/noise
- **Technology**: Silero VAD with parallel processing per file
- **Output**: JSON files with speech timestamps (`outputs/silero-timestamps/`)

### Stage 3: Whisper Transcribe
- **Input**: Speech segments and audio files
- **Purpose**: Convert speech to text with word-level timestamps
- **Technology**: MLX Whisper (Apple Silicon optimized)
- **Output**: Raw transcripts with timestamps (`outputs/whisper-transcripts/`)

### Stage 4: GPT Processing
- **Input**: Raw transcripts from all speakers
- **Purpose**: Merge transcripts, clean hallucinations, format for readability
- **Technology**: OpenAI API for intelligent post-processing
- **Output**: Final polished transcripts (`outputs/gpt-cleanup/`)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd transcribe

# Install dependencies
uv sync

# Create environment file
cp .env.template .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# HUGGINGFACE_TOKEN=your_huggingface_token_here  # optional
```

### Basic Usage

```bash
# Validate installation and API keys
uv run transcribe validate

# Check pipeline status
uv run transcribe status

# Run full pipeline
uv run transcribe run

# Run specific stage
uv run transcribe run-stage preprocess

# Reset pipeline status
uv run transcribe reset

# Clean everything (reset + remove all inputs and outputs)
uv run transcribe clean

# Get help
uv run transcribe --help
```

### Input Files

Place your FLAC audio files in the `inputs/` directory:
```
inputs/
├── speaker1.flac
├── speaker2.flac
└── speaker3.flac
```

The filename (without extension) is used as the speaker label in the final transcript.

## 🔧 Configuration

### Optimal Settings (Pre-configured)

The system includes battle-tested configurations for best results:

**FFmpeg Audio Processing:**
```bash
ffmpeg -i input.flac -ar 16000 -af "highpass=f=60,agate=threshold=-45dB:ratio=10:attack=5:release=60" output.wav
```

**Silero VAD Parameters:**
- Threshold start: 0.6 (start speech detection)
- Threshold end: 0.4 (end speech detection)
- Min speech: 250ms (filter brief sounds)
- Min silence: 200ms (segment boundaries)
- Merge gap: 150ms (adjacent segment merging)

**Whisper Settings:**
- Model: small.en (optimal speed/accuracy balance)
- Temperature: 0.0 (deterministic output)
- Language: English with sentence-level segmentation

## 🏗️ Architecture Benefits

- **Fast**: Process 3-hour recordings in ≤20 minutes (20x real-time speed)
- **Modular**: Each stage is independent and can be developed/tested separately
- **Resumable**: Can restart from any completed stage via `outputs/status.json`
- **Debuggable**: Inspect intermediate outputs at each stage
- **Apple Silicon Optimized**: Uses MLX Whisper for M1/M2 performance

## 📁 Directory Structure

```
transcribe/
├── inputs/                    # Input FLAC audio files
├── outputs/
│   ├── audio-files-wav/       # Stage 1: Converted WAV files
│   ├── silero-timestamps/     # Stage 2: VAD speech segments
│   ├── whisper-transcripts/   # Stage 3: Raw transcriptions
│   ├── gpt-cleanup/           # Stage 4: Final transcripts
│   └── status.json            # Pipeline state tracking
├── prompts/                   # GPT prompt templates
├── src/                       # Source code
│   ├── main.py               # CLI interface and orchestration
│   ├── config.py             # Configuration management
│   ├── ffmpeg_preprocess.py  # Audio preprocessing
│   ├── vad_timestamp.py      # Silero VAD integration
│   ├── whisper_transcribe.py # MLX Whisper processing
│   ├── gpt_merge.py          # Multi-speaker merging
│   └── gpt_cleanup.py        # Final transcript cleanup
└── tests/                    # Test directory (currently empty)
```

## 🛠️ Development

### Code Quality

```bash
# Run linting
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Technology Stack

- **Python 3.11+** with uv package manager
- **Audio Processing**: FFmpeg, librosa, soundfile
- **ML/AI**: MLX Whisper (Apple Silicon), Silero VAD, OpenAI API
- **CLI**: Click with Rich console interface
- **Configuration**: Pydantic with type validation
- **Async**: asyncio for I/O-bound operations

### Performance Targets

- **Speed**: 20x real-time processing (3-hour recording in ≤20 minutes)
- **Parallel Processing**: I/O-bound stages (preprocessing, VAD) run in parallel
- **Memory Efficient**: Sequential Whisper processing respects MacBook memory limits

## 📋 Requirements

### System Dependencies
- FFmpeg (for audio processing)
- Python 3.11+
- Apple Silicon Mac (for MLX Whisper optimization)

### API Keys
- **OpenAI API Key**: Required for GPT-based transcript processing
- **Hugging Face Token**: Optional, for accessing gated models

### Python Dependencies
See `pyproject.toml` for complete list. Key dependencies:
- `mlx-whisper` - Apple Silicon optimized Whisper
- `silero-vad` - Voice activity detection
- `openai` - GPT API client
- `click` - CLI framework
- `rich` - Enhanced terminal output
- `pydantic` - Configuration validation

## 🔄 Resume Functionality

The pipeline automatically tracks progress in `outputs/status.json`. If interrupted, simply run `uv run transcribe run` again to resume from the last completed stage.

Status tracking includes:
- Completion status for each stage
- File processing progress
- Error states and recovery information

## 📝 Output Formats

Final transcripts are available in multiple formats:
- **JSON**: Machine-readable with full metadata
- **SRT**: Standard subtitle format with speaker labels

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file contains valid OpenAI API key
2. **FFmpeg Not Found**: Install FFmpeg: `brew install ffmpeg`
3. **Memory Issues**: Large files may require breaking into smaller segments
4. **MLX Whisper Issues**: Falls back to standard Whisper automatically

### Getting Help

```bash
# Check system dependencies and API connectivity
uv run transcribe validate

# View detailed pipeline status
uv run transcribe status

# Reset if pipeline is in bad state
uv run transcribe reset

# Clean everything (reset + remove all files) for fresh start
uv run transcribe clean
```

## 📄 License

This project is designed for speech-to-text transcription tasks. Please ensure you have appropriate rights to process any audio content.
