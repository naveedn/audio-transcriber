"""Stage 2: Voice Activity Detection with Silero VAD."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, TaskID

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""

    def __init__(self, config: Config) -> None:
        """Initialize the VAD processor."""
        self.config = config
        self.vad_config = config.silero
        self.model = None
        self.utils = None

    def load_model(self) -> None:
        """Load the Silero VAD model."""
        try:
            import torch

            # Load Silero VAD model
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load Silero VAD model: {e}")
            raise

    def _compute_rms_dbfs(self, audio_chunk: np.ndarray) -> float:
        """Compute RMS in dBFS for audio chunk."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)

    def _merge_and_pad_segments(self, segments: list[dict], audio_duration: float) -> list[dict]:
        """Merge segments with gap < merge_gap_ms and pad edges, matching reference implementation."""
        if not segments:
            return []

        # Sort segments by start time first
        sorted_segments = sorted(segments, key=lambda x: x["start"])

        # Convert thresholds to seconds
        gap_seconds = self.vad_config.merge_gap_ms / 1000.0
        pad_seconds = self.vad_config.pad_ms / 1000.0

        # Merge segments
        merged = []
        current = {"start": sorted_segments[0]["start"], "end": sorted_segments[0]["end"], "confidence": sorted_segments[0].get("confidence", 0.0)}

        for segment in sorted_segments[1:]:
            gap = segment["start"] - current["end"]

            if gap <= gap_seconds:
                # Merge: extend current segment to include this one
                current["end"] = max(current["end"], segment["end"])
                # Keep higher confidence if available
                if segment.get("confidence", 0.0) > current.get("confidence", 0.0):
                    current["confidence"] = segment["confidence"]
            else:
                # Gap too large, finalize current segment and start new one
                merged.append(current)
                current = {"start": segment["start"], "end": segment["end"], "confidence": segment.get("confidence", 0.0)}

        # Don't forget the last segment
        merged.append(current)

        # Apply padding after merging
        for segment in merged:
            segment["start"] = max(0, segment["start"] - pad_seconds)
            segment["end"] = min(audio_duration, segment["end"] + pad_seconds)

        return merged

    def _filter_short_segments(self, segments: list[dict]) -> list[dict]:
        """Filter out segments shorter than minimum duration."""
        min_duration = self.vad_config.drop_below_ms / 1000.0

        filtered = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration >= min_duration:
                filtered.append(segment)
            else:
                logger.debug(f"Dropping short segment: {duration:.3f}s")

        return filtered

    async def process_audio_file(
        self,
        audio_path: Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> list[dict]:
        """Process a single audio file for voice activity detection."""
        try:
            logger.info(f"Processing VAD for {audio_path.name}")

            # Load audio
            audio, sample_rate = librosa.load(str(audio_path), sr=16000)
            audio_duration = len(audio) / sample_rate

            logger.info(f"Loaded audio: {audio_duration:.2f}s, {sample_rate}Hz")

            # Ensure model is loaded
            if self.model is None:
                self.load_model()

            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)

            # Calculate frame parameters
            frame_samples = max(512, int(self.vad_config.frame_ms * sample_rate / 1000))
            block_samples = int(self.vad_config.block_seconds * sample_rate)

            # Convert ms to frame counts for hysteresis
            ms_per_frame = 1000.0 * frame_samples / sample_rate
            min_speech_frames = max(1, int(round(self.vad_config.min_speech_ms / ms_per_frame)))
            min_silence_frames = max(1, int(round(self.vad_config.min_silence_ms / ms_per_frame)))

            # State machine with frame counting
            segments = []
            in_speech = False
            above_cnt = 0
            below_cnt = 0
            start_sample = None

            for block_start in range(0, len(audio), block_samples):
                block_end = min(block_start + block_samples, len(audio))
                block_audio = audio_tensor[block_start:block_end]

                # Process frames within block
                for frame_start in range(0, len(block_audio), frame_samples):
                    frame_end = min(frame_start + frame_samples, len(block_audio))
                    frame_audio = block_audio[frame_start:frame_end]

                    # Skip if frame is too short
                    if len(frame_audio) < frame_samples:
                        continue

                    # Calculate frame boundaries in samples
                    frame_start_sample = block_start + frame_start
                    frame_end_sample = min(block_start + frame_end, len(audio))

                    # Check RMS gate first
                    rms_dbfs = self._compute_rms_dbfs(frame_audio.numpy())
                    if rms_dbfs < self.vad_config.rms_gate_dbfs:
                        prob = 0.0
                    else:
                        # Get VAD probability
                        prob = self.model(frame_audio, sample_rate).item()

                    # State machine with frame counting (hysteresis)
                    if not in_speech:
                        if prob >= self.vad_config.threshold_start:
                            above_cnt += 1
                            if above_cnt >= min_speech_frames:
                                # Start speech segment, backtrack to account for counted frames
                                start_sample = frame_end_sample - above_cnt * frame_samples
                                in_speech = True
                                below_cnt = 0
                        else:
                            above_cnt = 0
                    else:
                        if prob <= self.vad_config.threshold_end:
                            below_cnt += 1
                            if below_cnt >= min_silence_frames:
                                # End speech segment, backtrack to account for counted frames
                                end_sample = frame_end_sample - below_cnt * frame_samples
                                segments.append({
                                    "start": max(0, start_sample) / sample_rate,
                                    "end": min(end_sample, len(audio)) / sample_rate,
                                    "confidence": prob,
                                })
                                in_speech = False
                                above_cnt = 0
                                start_sample = None
                        else:
                            below_cnt = 0

            # Handle ongoing speech at end
            if in_speech and start_sample is not None:
                segments.append({
                    "start": max(0, start_sample) / sample_rate,
                    "end": audio_duration,
                    "confidence": 0.0,
                })

            # Post-process segments
            segments = self._filter_short_segments(segments)
            segments = self._merge_and_pad_segments(segments, audio_duration)

            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            logger.info(f"Found {len(segments)} speech segments in {audio_path.name}")
            return segments

        except Exception as e:
            logger.exception(f"Error processing VAD for {audio_path.name}: {e}")
            return []

    def save_segments(self, segments: list[dict], output_path: Path) -> None:
        """Save VAD segments to JSON and CSV files."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({
                    "segments": segments,
                    "total_segments": len(segments),
                    "total_speech_duration": sum(s["end"] - s["start"] for s in segments),
                    "config": self.vad_config.dict(),
                }, f, indent=2)

            # Save CSV
            csv_path = output_path.with_suffix(".csv")
            with open(csv_path, "w") as f:
                f.write("start,end,duration,confidence\n")
                for segment in segments:
                    duration = segment["end"] - segment["start"]
                    confidence = segment.get("confidence", 0.0)
                    f.write(f"{segment['start']:.3f},{segment['end']:.3f},{duration:.3f},{confidence:.3f}\n")

            logger.info(f"Saved VAD results to {json_path} and {csv_path}")

        except Exception as e:
            logger.exception(f"Error saving segments to {output_path}: {e}")


class VADProcessor:
    """Main VAD processing coordinator."""

    def __init__(self, config: Config) -> None:
        """Initialize the VAD processor."""
        self.config = config
        self.vad = SileroVAD(config)

    def find_audio_files(self) -> list[Path]:
        """Find preprocessed WAV files to process."""
        audio_files = []

        if not self.config.paths.audio_wav_dir.exists():
            logger.warning(f"Audio WAV directory does not exist: {self.config.paths.audio_wav_dir}")
            return audio_files

        # Look for WAV files
        for pattern in ["*.wav", "*.WAV"]:
            audio_files.extend(self.config.paths.audio_wav_dir.glob(pattern))

        logger.info(f"Found {len(audio_files)} audio files for VAD processing")
        return sorted(audio_files)

    def get_output_path(self, audio_path: Path) -> Path:
        """Get the output path for VAD timestamps."""
        speaker_name = audio_path.stem
        return self.config.paths.silero_dir / f"{speaker_name}_timestamps"

    async def process_files(self, audio_files: list[Path] | None = None) -> dict[str, Path]:
        """Process multiple audio files for VAD in parallel."""
        if audio_files is None:
            audio_files = self.find_audio_files()

        if not audio_files:
            logger.warning("No audio files found for VAD processing")
            return {}

        # Prepare tasks
        tasks = []
        output_mapping = {}

        with Progress() as progress:
            task_id = progress.add_task(
                "[blue]Processing VAD timestamps...",
                total=len(audio_files),
            )

            # Create semaphore to limit concurrent processes
            semaphore = asyncio.Semaphore(self.config.max_parallel_vad)

            async def process_with_semaphore(audio_path: Path) -> tuple[str, list[dict]]:
                async with semaphore:
                    segments = await self.vad.process_audio_file(
                        audio_path, progress, task_id,
                    )
                    return audio_path.stem, segments

            # Create tasks for all files
            for audio_path in audio_files:
                output_path = self.get_output_path(audio_path)
                output_mapping[audio_path.stem] = output_path
                tasks.append(process_with_semaphore(audio_path))

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Save results
        successful_outputs = {}
        failed_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"VAD task failed with exception: {result}")
                failed_count += 1
            else:
                speaker_name, segments = result
                if segments:
                    output_path = output_mapping[speaker_name]
                    self.vad.save_segments(segments, output_path)
                    successful_outputs[speaker_name] = output_path.with_suffix(".json")
                else:
                    failed_count += 1

        console.print(f"[green]Successfully processed VAD for {len(successful_outputs)} files")
        if failed_count > 0:
            console.print(f"[red]Failed VAD processing for {failed_count} files")

        return successful_outputs

    def check_dependencies(self) -> list[str]:
        """Check if required dependencies are available."""
        errors = []

        try:
            import torch
            if not torch.cuda.is_available() and not hasattr(torch.backends, "mps"):
                logger.warning("No GPU acceleration available, using CPU")
        except ImportError:
            errors.append("PyTorch is not installed")

        try:
            import librosa
        except ImportError:
            errors.append("librosa is not installed")

        try:
            # Test loading Silero VAD model
            self.vad.load_model()
        except Exception as e:
            errors.append(f"Cannot load Silero VAD model: {e}")

        return errors


async def process_vad(config: Config, audio_files: list[Path] | None = None) -> dict[str, Path]:
    """Main function to process VAD timestamps."""
    processor = VADProcessor(config)

    # Check dependencies
    errors = processor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "VAD dependency check failed"
        raise RuntimeError(msg)

    # Process files
    return await processor.process_files(audio_files)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Process VAD timestamps with Silero")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for VAD timestamps",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum parallel processes",
    )

    args = parser.parse_args()

    # Load configuration
    config_overrides = {}
    if args.input_dir:
        config_overrides["paths"] = {"audio_wav_dir": args.input_dir}
    if args.output_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["silero_dir"] = args.output_dir
    if args.max_parallel:
        config_overrides["max_parallel_vad"] = args.max_parallel

    config = load_config(**config_overrides)

    # Run VAD processing
    try:
        output_files = asyncio.run(process_vad(config))
        console.print(f"[green]VAD processing complete! Generated {len(output_files)} files.")
    except Exception as e:
        console.print(f"[red]VAD processing failed: {e}")
        sys.exit(1)
