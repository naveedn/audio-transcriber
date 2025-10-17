"""Stage 1: Audio preprocessing with FFmpeg."""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing using FFmpeg."""

    def __init__(self, config: Config) -> None:
        """Initialize the audio preprocessor."""
        self.config = config
        self.ffmpeg_config = config.ffmpeg

    def _build_ffmpeg_command(self, input_path: Path, output_path: Path) -> list[str]:
        """Build the FFmpeg command with optimal settings."""
        return [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ar",
            str(self.ffmpeg_config.sample_rate),
            "-ac",
            str(self.ffmpeg_config.channels),
            "-c:a",
            f"pcm_s{self.ffmpeg_config.bit_depth}le",
            "-af",
            (
                f"highpass=f={self.ffmpeg_config.highpass_freq},"
                f"agate=threshold={self.ffmpeg_config.gate_threshold}:"
                f"ratio={self.ffmpeg_config.gate_ratio}:"
                f"attack={self.ffmpeg_config.gate_attack}:"
                f"release={self.ffmpeg_config.gate_release}"
            ),
            "-y",  # Overwrite output files
            str(output_path),
        ]

    async def _process_single_file(
        self,
        input_path: Path,
        output_path: Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> bool:
        """Process a single audio file."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(input_path, output_path)

            logger.info(f"Processing {input_path.name} -> {output_path.name}")

            # Run FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg failed for {input_path.name}: {stderr.decode()}")
                return False

            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            logger.info(f"Successfully processed {input_path.name}")
            return True

        except Exception as e:
            logger.exception(f"Error processing {input_path.name}: {e}")
            return False

    def _get_speaker_name(self, filename: str) -> str:
        """Extract speaker name from filename (before extension)."""
        return Path(filename).stem

    def find_input_files(self) -> list[Path]:
        """Find all FLAC files in the inputs directory."""
        input_files = []

        if not self.config.paths.inputs_dir.exists():
            logger.warning(
                f"Input directory does not exist: {self.config.paths.inputs_dir}"
            )
            return input_files

        # Look for FLAC files
        for pattern in ["*.flac", "*.FLAC"]:
            input_files.extend(self.config.paths.inputs_dir.glob(pattern))

        logger.info(f"Found {len(input_files)} audio files to process")
        return sorted(input_files)

    def get_output_path(self, input_path: Path) -> Path:
        """Get the output path for a preprocessed audio file."""
        speaker_name = self._get_speaker_name(input_path.name)
        return self.config.paths.audio_wav_dir / f"{speaker_name}.wav"

    async def process_files(self, input_files: list[Path] | None = None) -> list[Path]:
        """Process multiple audio files in parallel."""
        if input_files is None:
            input_files = self.find_input_files()

        if not input_files:
            logger.warning("No input files found to process")
            return []

        # Prepare tasks
        tasks = []
        output_paths = []

        with Progress() as progress:
            task_id = progress.add_task(
                "[green]Processing audio files...",
                total=len(input_files),
            )

            # Create semaphore to limit concurrent processes
            semaphore = asyncio.Semaphore(self.config.max_parallel_audio)

            async def process_with_semaphore(input_path: Path) -> tuple[Path, bool]:
                async with semaphore:
                    output_path = self.get_output_path(input_path)
                    success = await self._process_single_file(
                        input_path,
                        output_path,
                        progress,
                        task_id,
                    )
                    return output_path, success

            # Create tasks for all files
            for input_path in input_files:
                output_path = self.get_output_path(input_path)
                output_paths.append(output_path)
                tasks.append(process_with_semaphore(input_path))

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successful_outputs = []
        failed_count = 0

        for _i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed_count += 1
            else:
                output_path, success = result
                if success and output_path.exists():
                    successful_outputs.append(output_path)
                else:
                    failed_count += 1

        console.print(f"[green]Successfully processed {len(successful_outputs)} files")
        if failed_count > 0:
            console.print(f"[red]Failed to process {failed_count} files")

        return successful_outputs

    def check_dependencies(self) -> list[str]:
        """Check if FFmpeg is available."""
        errors = []

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append("FFmpeg is not working properly")
        except FileNotFoundError:
            errors.append("FFmpeg is not installed or not in PATH")
        except subprocess.TimeoutExpired:
            errors.append("FFmpeg check timed out")
        except Exception as e:
            errors.append(f"Error checking FFmpeg: {e}")

        return errors


async def preprocess_audio(
    config: Config, input_files: list[Path] | None = None
) -> list[Path]:
    """Main function to preprocess audio files."""
    preprocessor = AudioPreprocessor(config)

    # Check dependencies
    errors = preprocessor.check_dependencies()
    if errors:
        for error in errors:
            logger.error(error)
        msg = "FFmpeg dependency check failed"
        raise RuntimeError(msg)

    # Process files
    return await preprocessor.process_files(input_files)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preprocess audio files with FFmpeg")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing FLAC files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for WAV files",
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
        config_overrides["paths"] = {"inputs_dir": args.input_dir}
    if args.output_dir:
        if "paths" not in config_overrides:
            config_overrides["paths"] = {}
        config_overrides["paths"]["audio_wav_dir"] = args.output_dir
    if args.max_parallel:
        config_overrides["max_parallel_audio"] = args.max_parallel

    config = load_config(**config_overrides)

    # Run preprocessing
    try:
        output_files = asyncio.run(preprocess_audio(config))
        console.print(
            f"[green]Preprocessing complete! Generated {len(output_files)} files."
        )
    except Exception as e:
        console.print(f"[red]Preprocessing failed: {e}")
        sys.exit(1)
