"""Main CLI interface and orchestration for the audio processing pipeline."""

import asyncio
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# from . import __version__  # TODO: Add version management
__version__ = "0.1.0"
from .config import Config, load_config

# Import stage modules with corrected names
from .ffmpeg_preprocess import AudioPreprocessor, preprocess_audio
from .gpt_cleanup import TranscriptProcessor, cleanup_transcript
from .vad_timestamp import VADProcessor, process_vad
from .whisper_transcribe import TranscriptionProcessor, transcribe_audio

console = Console()
logger = logging.getLogger(__name__)


class PipelineStatus:
    """Manages pipeline execution status and resumability."""

    def __init__(self, config: Config) -> None:
        """Initialize status manager."""
        self.config = config
        self.status_file = config.paths.status_file
        self.status = self._load_status()

    def _load_status(self) -> dict:
        """Load status from file."""
        if self.status_file.exists():
            try:
                with self.status_file.open() as file_obj:
                    return json.load(file_obj)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not load status file: %s", exc)

        return {
            "pipeline_start": None,
            "stages": {
                "stage0_bootstrap": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                },
                "stage1_preprocess": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                },
                "stage2_vad": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                },
                "stage3_whisper": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                },
                "stage4_process": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                },
            },
        }

    def save_status(self) -> None:
        """Save current status to file."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with self.status_file.open("w") as file_obj:
                json.dump(self.status, file_obj, indent=2)
        except OSError as exc:
            logger.exception("Could not save status file", exc_info=exc)

    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as started."""
        self.status["stages"][stage_name]["status"] = "running"
        self.status["stages"][stage_name]["start_time"] = datetime.now(
            tz=UTC
        ).isoformat()
        self.save_status()

    def get_stage_elapsed_time(self, stage_name: str) -> str:
        """Get elapsed time for a running stage."""
        start_time_str = self.status["stages"][stage_name].get("start_time")
        if not start_time_str:
            return "0:00"

        try:
            start_time = datetime.fromisoformat(start_time_str)
            elapsed = datetime.now(tz=UTC) - start_time
            return str(elapsed).split(".")[0]  # Remove microseconds
        except (ValueError, TypeError):
            return "0:00"

    def complete_stage(self, stage_name: str, *, success: bool = True) -> None:
        """Mark a stage as completed."""
        self.status["stages"][stage_name]["status"] = (
            "completed" if success else "failed"
        )
        self.status["stages"][stage_name]["end_time"] = datetime.now(tz=UTC).isoformat()
        self.save_status()

        # Calculate and display duration
        duration = self.get_stage_duration(stage_name)
        stage_display_name = self._get_stage_display_name(stage_name)
        if success:
            console.print(f"[green]✅ {stage_display_name} completed in {duration}")
        else:
            console.print(f"[red]❌ {stage_display_name} failed after {duration}")

    def get_stage_duration(self, stage_name: str) -> str:
        """Get duration of a completed stage."""
        start_time_str = self.status["stages"][stage_name].get("start_time")
        end_time_str = self.status["stages"][stage_name].get("end_time")

        if not start_time_str or not end_time_str:
            return "0:00"

        try:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            duration = end_time - start_time
            return str(duration).split(".")[0]  # Remove microseconds
        except (ValueError, TypeError):
            return "0:00"

    def _get_stage_display_name(self, stage_name: str) -> str:
        """Get display name for a stage."""
        stage_names = {
            "stage0_bootstrap": "Bootstrap",
            "stage1_preprocess": "Audio Preprocessing",
            "stage2_vad": "Voice Activity Detection",
            "stage3_whisper": "Speech Transcription",
            "stage4_process": "Final Processing",
        }
        return stage_names.get(stage_name, stage_name)

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is already completed."""
        return self.status["stages"][stage_name]["status"] == "completed"

    def get_resume_stage(self) -> str | None:
        """Get the next stage to run for resuming."""
        stage_order = [
            "stage0_bootstrap",
            "stage1_preprocess",
            "stage2_vad",
            "stage3_whisper",
            "stage4_process",
        ]

        for stage in stage_order:
            if not self.is_stage_completed(stage):
                return stage

        return None  # All stages completed

    def print_status(self) -> None:
        """Print current pipeline status."""
        table = Table(title="Pipeline Status")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Start Time", style="green")
        table.add_column("Duration", style="yellow")

        stage_names = {
            "stage0_bootstrap": "0. Bootstrap",
            "stage1_preprocess": "1. Audio Preprocessing",
            "stage2_vad": "2. Voice Activity Detection",
            "stage3_whisper": "3. Speech Transcription",
            "stage4_process": "4. Final Processing",
        }

        for stage_id, info in self.status["stages"].items():
            stage_name = stage_names.get(stage_id, stage_id)
            status = info["status"]
            start_time = info["start_time"] or "-"

            # Calculate duration
            duration = "-"
            if info["start_time"] and info["end_time"]:
                try:
                    start = datetime.fromisoformat(info["start_time"])
                    end = datetime.fromisoformat(info["end_time"])
                    duration = str(end - start).split(".")[0]  # Remove microseconds
                except (ValueError, TypeError):
                    duration = "-"

            # Color status
            if status == "completed":
                status = f"[green]{status}[/green]"
            elif status == "running":
                status = f"[yellow]{status}[/yellow]"
            elif status == "failed":
                status = f"[red]{status}[/red]"

            table.add_row(stage_name, status, start_time, duration)

        console.print(table)


class AudioPipeline:
    """Main audio processing pipeline orchestrator."""

    def __init__(self, config: Config) -> None:
        """Initialize the pipeline."""
        self.config = config
        self.status = PipelineStatus(config)

    def bootstrap(self) -> bool:
        """Stage 0: Bootstrap - check dependencies and download models."""
        console.print(Panel("🚀 Stage 0: Bootstrap Process", style="bold blue"))

        try:
            return self._bootstrap_impl()
        except Exception as exc:
            logger.exception("Bootstrap failed")
            console.print(f"[red]❌ Bootstrap failed: {exc}")
            return False

    def _bootstrap_impl(self) -> bool:  # noqa: C901
        """Execute bootstrap checks and return success state."""
        # Create directories
        self.config.create_directories()
        console.print("[green]✅ Created output directories")

        # Validate environment
        errors = self.config.validate_environment()
        if errors:
            for error in errors:
                console.print(f"[red]❌ {error}")
            return False
        console.print("[green]✅ Environment validation passed")

        ffmpeg_processor = AudioPreprocessor(self.config)
        errors = ffmpeg_processor.check_dependencies()
        if errors:
            for error in errors:
                console.print(f"[red]❌ FFmpeg: {error}")
            return False
        console.print("[green]✅ FFmpeg available")

        vad_processor = VADProcessor(self.config)
        errors = vad_processor.check_dependencies()
        if errors:
            for error in errors:
                console.print(f"[red]❌ Silero VAD: {error}")
            return False
        console.print("[green]✅ Silero VAD model loaded")

        whisper_processor = TranscriptionProcessor(self.config)
        errors = whisper_processor.check_dependencies()
        if errors:
            for error in errors:
                console.print(f"[red]❌ Whisper: {error}")
            return False
        console.print("[green]✅ Whisper model loaded")

        if self.config.openai_api_key:
            processor = TranscriptProcessor(self.config)
            errors = processor.check_dependencies()
            if errors:
                for error in errors:
                    console.print(f"[red]❌ OpenAI API: {error}")
                return False
            console.print("[green]✅ OpenAI API accessible")
        else:
            console.print(
                "[yellow]⚠️ OpenAI API key not provided - GPT stages will be skipped"
            )

        return True

    async def run_stage1(self) -> bool:
        """Stage 1: Audio preprocessing."""
        console.print(Panel("🎵 Stage 1: Audio Preprocessing", style="bold blue"))

        try:
            output_files = await preprocess_audio(self.config)
        except Exception as exc:
            logger.exception("Stage 1 failed")
            console.print(f"[red]❌ Stage 1 failed: {exc}")
            return False

        if output_files:
            console.print(f"[green]Processed {len(output_files)} audio files")
            return True

        console.print("[yellow]No audio files processed")
        return False

    async def run_stage2(self) -> bool:
        """Stage 2: Voice Activity Detection."""
        console.print(Panel("🎤 Stage 2: Voice Activity Detection", style="bold blue"))

        try:
            output_files = await process_vad(self.config)
        except Exception as exc:
            logger.exception("Stage 2 failed")
            console.print(f"[red]❌ Stage 2 failed: {exc}")
            return False

        if output_files:
            console.print(
                f"[green]Generated VAD timestamps for {len(output_files)} files"
            )
            return True

        console.print("[yellow]No VAD timestamps generated")
        return False

    def run_stage3(self) -> bool:
        """Stage 3: Speech transcription."""
        console.print(Panel("📝 Stage 3: Speech Transcription", style="bold blue"))

        try:
            output_files = transcribe_audio(self.config)
        except Exception as exc:
            logger.exception("Stage 3 failed")
            console.print(f"[red]❌ Stage 3 failed: {exc}")
            return False

        if output_files:
            console.print(f"[green]Transcribed {len(output_files)} files")
            return True

        console.print("[yellow]No transcriptions generated")
        return False

    def run_stage4(self) -> bool:
        """Stage 4: Final processing."""
        console.print(Panel("✨ Stage 4: Final Processing", style="bold blue"))

        if not self.config.openai_api_key:
            console.print("[yellow]⚠️ Skipping processing - no OpenAI API key")
            return True

        try:
            output_file = cleanup_transcript(self.config)
        except Exception as exc:
            logger.exception("Stage 4 failed")
            console.print(f"[red]❌ Stage 4 failed: {exc}")
            return False

        if output_file:
            console.print(f"[green]Final transcript created: {output_file.name}")
            return True

        console.print("[yellow]No processing performed")
        return True

    async def run_full_pipeline(  # noqa: C901, PLR0912, PLR0915
        self,
        stages: str | None = None,
        *,
        continue_after: bool = False,
    ) -> bool:
        """Run the complete pipeline, selected stages, or resume progress."""
        console.print(
            Panel(f"🚀 Audio Processing Pipeline v{__version__}", style="bold magenta")
        )

        # Stage name mapping
        stage_mapping = {
            "bootstrap": "stage0_bootstrap",
            "preprocess": "stage1_preprocess",
            "vad": "stage2_vad",
            "whisper": "stage3_whisper",
            "process": "stage4_process",
        }

        # Parse stages parameter
        if stages:
            # Parse comma-separated stage list
            stage_names = [name.strip() for name in stages.split(",")]

            # Validate stage names
            invalid_stages = [name for name in stage_names if name not in stage_mapping]
            if invalid_stages:
                console.print(
                    f"[red]❌ Invalid stage names: {', '.join(invalid_stages)}"
                )
                console.print(f"[cyan]Valid stages: {', '.join(stage_mapping.keys())}")
                return False

            # Convert to internal stage names
            target_stages = [stage_mapping[name] for name in stage_names]
            start_stage = target_stages[0]

            console.print(f"[cyan]Running stages: {', '.join(stage_names)}")
            if continue_after:
                console.print(
                    "[cyan]Will continue to remaining stages after completion"
                )
        else:
            # No specific stages - resume from where we left off or start from beginning
            start_stage = self.status.get_resume_stage()
            target_stages = None
            continue_after = True  # Default behavior for full pipeline

            if not start_stage:
                console.print("[green]✅ Pipeline already completed!")
                return True

            console.print(f"[cyan]Starting from: {start_stage}")

        # Record pipeline start
        if not self.status.status["pipeline_start"]:
            self.status.status["pipeline_start"] = datetime.now(tz=UTC).isoformat()
            self.status.save_status()

        # Stage execution mapping
        stage_functions = {
            "stage0_bootstrap": self.bootstrap,
            "stage1_preprocess": self.run_stage1,
            "stage2_vad": self.run_stage2,
            "stage3_whisper": self.run_stage3,
            "stage4_process": self.run_stage4,
        }

        # Get ordered list of stages to run
        stage_order = [
            "stage0_bootstrap",
            "stage1_preprocess",
            "stage2_vad",
            "stage3_whisper",
            "stage4_process",
        ]

        start_index = stage_order.index(start_stage)

        # Determine which stages to run
        if target_stages and not continue_after:
            # Run only the specified stages
            stages_to_run = target_stages
        else:
            # Run from start_stage to end. If `continue_after` is set, keep running
            # through the remaining stages once the targets are completed.
            start_index = stage_order.index(start_stage)
            stages_to_run = stage_order[start_index:]

            if target_stages and continue_after:
                # Find the last target stage and include all stages up to end
                last_target_index = max(
                    stage_order.index(stage) for stage in target_stages
                )
                stages_to_run = stage_order[start_index:]

        # Reset manually specified stages to pending (force re-run)
        if target_stages:
            for stage_name in target_stages:
                if self.status.is_stage_completed(stage_name):
                    console.print(
                        f"[yellow]🔄 Resetting {stage_name} (manually specified)"
                    )
                    self.status.status["stages"][stage_name]["status"] = "pending"
                    self.status.status["stages"][stage_name]["start_time"] = None
                    self.status.status["stages"][stage_name]["end_time"] = None

            # When continue_after is True, also reset all subsequent stages
            if continue_after:
                # Find the last target stage index
                target_indices = [stage_order.index(stage) for stage in target_stages]
                last_target_index = max(target_indices)

                # Reset all stages that come after the last target stage
                for i in range(last_target_index + 1, len(stage_order)):
                    subsequent_stage = stage_order[i]
                    if self.status.is_stage_completed(subsequent_stage):
                        console.print(
                            "[yellow]🔄 Resetting "
                            f"{subsequent_stage} (subsequent stage)"
                        )
                        self.status.status["stages"][subsequent_stage]["status"] = (
                            "pending"
                        )
                        self.status.status["stages"][subsequent_stage]["start_time"] = (
                            None
                        )
                        self.status.status["stages"][subsequent_stage]["end_time"] = (
                            None
                        )

            self.status.save_status()

        # Run stages
        for stage_name in stages_to_run:
            # Only skip completed stages if they weren't manually specified
            if self.status.is_stage_completed(stage_name) and (
                not target_stages or stage_name not in target_stages
            ):
                console.print(f"[green]✅ Skipping {stage_name} (already completed)")
                continue

            self.status.start_stage(stage_name)

            try:
                # Run the stage
                stage_func = stage_functions[stage_name]
                if asyncio.iscoroutinefunction(stage_func):
                    success = await stage_func()
                else:
                    success = stage_func()

                self.status.complete_stage(stage_name, success=success)

                if not success:
                    console.print(f"[red]❌ Pipeline stopped at {stage_name}")
                    return False

                # If this was the last target stage and we're not continuing, stop here
                if target_stages and not continue_after and stage_name in target_stages:
                    remaining_targets = [
                        s
                        for s in target_stages
                        if s in stages_to_run[stages_to_run.index(stage_name) + 1 :]
                    ]
                    if not remaining_targets:
                        console.print("[green]✅ Specified stages completed")
                        return True

            except Exception as exc:
                logger.exception("Stage %s failed", stage_name)
                self.status.complete_stage(stage_name, success=False)
                console.print(f"[red]❌ Pipeline failed at {stage_name}: {exc}")
                return False

        console.print(Panel("🎉 Pipeline completed successfully!", style="bold green"))
        return True


def setup_logging(*, verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)],
    )

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# CLI Interface
@click.group()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file",
)
@click.pass_context
def cli(ctx: click.Context, *, verbose: bool, config_file: Path | None) -> None:
    """Audio Processing Pipeline for Speech-to-Text Transcription."""
    setup_logging(verbose=verbose)

    # Load environment variables
    load_dotenv()

    # Load configuration
    try:
        config = load_config(config_file)
        ctx.ensure_object(dict)
        ctx.obj["config"] = config
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        console.print(f"[red]Configuration error: {exc}")
        sys.exit(1)


@cli.command()
@click.option(
    "--stage",
    help=(
        "Comma-separated list of stages to run "
        "(bootstrap, preprocess, vad, whisper, process)"
    ),
)
@click.option(
    "--continue",
    "continue_after",
    is_flag=True,
    help="Continue to next stages after specified stages complete",
)
@click.pass_context
def run(
    ctx: click.Context,
    stage: str | None,
    *,
    continue_after: bool,
) -> None:
    """Run the complete audio processing pipeline or specific stages."""
    config = ctx.obj["config"]
    pipeline = AudioPipeline(config)

    try:
        success = asyncio.run(
            pipeline.run_full_pipeline(stages=stage, continue_after=continue_after)
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user")
        sys.exit(1)
    except (RuntimeError, OSError, ValueError) as exc:
        console.print(f"[red]Pipeline failed: {exc}")
        sys.exit(1)

    if not success:
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline status."""
    config = ctx.obj["config"]
    status_manager = PipelineStatus(config)
    status_manager.print_status()


@cli.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset pipeline status."""
    config = ctx.obj["config"]

    if config.paths.status_file.exists():
        config.paths.status_file.unlink()
        console.print("[green]✅ Pipeline status reset")
    else:
        console.print("[yellow]⚠️ No status file found")


@cli.command()
@click.confirmation_option(
    prompt="Are you sure you want to delete all inputs and outputs?",
)
@click.pass_context
def clean(ctx: click.Context) -> None:
    """Reset pipeline and remove all inputs and outputs."""
    config = ctx.obj["config"]

    console.print("🧹 Cleaning pipeline...")

    # Reset pipeline status first
    if config.paths.status_file.exists():
        config.paths.status_file.unlink()
        console.print("[green]✅ Pipeline status reset")

    # Remove all output directories and their contents
    output_dirs = [
        config.paths.outputs_dir,
        config.paths.audio_wav_dir,
        config.paths.silero_dir,
        config.paths.whisper_dir,
        config.paths.gpt_dir,
    ]

    for dir_path in output_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            console.print(f"[green]✅ Removed: {dir_path}")

    # Remove all input files (but keep the directory)
    if config.paths.inputs_dir.exists():
        for file_path in config.paths.inputs_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
                console.print(f"[green]✅ Removed: {file_path}")

    console.print("[green]✅ Clean completed - all inputs and outputs removed")


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate configuration and dependencies."""
    config = ctx.obj["config"]
    pipeline = AudioPipeline(config)

    console.print("🔍 Validating configuration and dependencies...")
    success = pipeline.bootstrap()

    if success:
        console.print("[green]✅ All validations passed!")
    else:
        console.print("[red]❌ Validation failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
