"""Main CLI interface and orchestration for the audio processing pipeline."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# from . import __version__  # TODO: Add version management
__version__ = "0.1.0"
from .config import Config, load_config

# Import stage modules with corrected names
from .ffmpeg_preprocess import preprocess_audio
from .gpt_cleanup import cleanup_transcript
from .gpt_merge import merge_transcripts
from .vad_timestamp import process_vad
from .whisper_transcribe import transcribe_audio

console = Console()
logger = logging.getLogger(__name__)


class PipelineStatus:
    """Manages pipeline execution status and resumability."""

    def __init__(self, config: Config):
        """Initialize status manager."""
        self.config = config
        self.status_file = config.paths.status_file
        self.status = self._load_status()

    def _load_status(self) -> dict:
        """Load status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load status file: {e}")

        return {
            "pipeline_start": None,
            "stages": {
                "stage0_bootstrap": {"status": "pending", "start_time": None, "end_time": None},
                "stage1_preprocess": {"status": "pending", "start_time": None, "end_time": None},
                "stage2_vad": {"status": "pending", "start_time": None, "end_time": None},
                "stage3_whisper": {"status": "pending", "start_time": None, "end_time": None},
                "stage4a_merge": {"status": "pending", "start_time": None, "end_time": None},
                "stage4b_cleanup": {"status": "pending", "start_time": None, "end_time": None},
            },
        }

    def save_status(self) -> None:
        """Save current status to file."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, "w") as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save status file: {e}")

    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as started."""
        self.status["stages"][stage_name]["status"] = "running"
        self.status["stages"][stage_name]["start_time"] = datetime.now().isoformat()
        self.save_status()

    def complete_stage(self, stage_name: str, success: bool = True) -> None:
        """Mark a stage as completed."""
        self.status["stages"][stage_name]["status"] = "completed" if success else "failed"
        self.status["stages"][stage_name]["end_time"] = datetime.now().isoformat()
        self.save_status()

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is already completed."""
        return self.status["stages"][stage_name]["status"] == "completed"

    def get_resume_stage(self) -> str | None:
        """Get the next stage to run for resuming."""
        stage_order = [
            "stage0_bootstrap", "stage1_preprocess", "stage2_vad",
            "stage3_whisper", "stage4a_merge", "stage4b_cleanup",
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
            "stage4a_merge": "4a. Transcript Merging",
            "stage4b_cleanup": "4b. Final Cleanup",
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
                except:
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

    def __init__(self, config: Config):
        """Initialize the pipeline."""
        self.config = config
        self.status = PipelineStatus(config)

    def bootstrap(self) -> bool:
        """Stage 0: Bootstrap - check dependencies and download models."""
        console.print(Panel("🚀 Stage 0: Bootstrap Process", style="bold blue"))

        try:
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

            # Check FFmpeg
            from .ffmpeg_preprocess import AudioPreprocessor
            ffmpeg_processor = AudioPreprocessor(self.config)
            errors = ffmpeg_processor.check_dependencies()
            if errors:
                for error in errors:
                    console.print(f"[red]❌ FFmpeg: {error}")
                return False
            console.print("[green]✅ FFmpeg available")

            # Check and load Silero VAD
            from .vad_timestamp import VADProcessor
            vad_processor = VADProcessor(self.config)
            errors = vad_processor.check_dependencies()
            if errors:
                for error in errors:
                    console.print(f"[red]❌ Silero VAD: {error}")
                return False
            console.print("[green]✅ Silero VAD model loaded")

            # Check and load Whisper
            from .whisper_transcribe import TranscriptionProcessor
            whisper_processor = TranscriptionProcessor(self.config)
            errors = whisper_processor.check_dependencies()
            if errors:
                for error in errors:
                    console.print(f"[red]❌ Whisper: {error}")
                return False
            console.print("[green]✅ Whisper model loaded")

            # Check OpenAI API
            if self.config.openai_api_key:
                from .gpt_merge import TranscriptMerger
                merger = TranscriptMerger(self.config)
                errors = merger.check_dependencies()
                if errors:
                    for error in errors:
                        console.print(f"[red]❌ OpenAI API: {error}")
                    return False
                console.print("[green]✅ OpenAI API accessible")
            else:
                console.print("[yellow]⚠️ OpenAI API key not provided - GPT stages will be skipped")

            console.print("[green]✅ Bootstrap completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            console.print(f"[red]❌ Bootstrap failed: {e}")
            return False

    async def run_stage1(self) -> bool:
        """Stage 1: Audio preprocessing."""
        console.print(Panel("🎵 Stage 1: Audio Preprocessing", style="bold blue"))

        try:
            output_files = await preprocess_audio(self.config)
            if output_files:
                console.print(f"[green]✅ Processed {len(output_files)} audio files")
                return True
            console.print("[red]❌ No audio files processed")
            return False
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            console.print(f"[red]❌ Stage 1 failed: {e}")
            return False

    async def run_stage2(self) -> bool:
        """Stage 2: Voice Activity Detection."""
        console.print(Panel("🎤 Stage 2: Voice Activity Detection", style="bold blue"))

        try:
            output_files = await process_vad(self.config)
            if output_files:
                console.print(f"[green]✅ Generated VAD timestamps for {len(output_files)} files")
                return True
            console.print("[red]❌ No VAD timestamps generated")
            return False
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            console.print(f"[red]❌ Stage 2 failed: {e}")
            return False

    def run_stage3(self) -> bool:
        """Stage 3: Speech transcription."""
        console.print(Panel("📝 Stage 3: Speech Transcription", style="bold blue"))

        try:
            output_files = transcribe_audio(self.config)
            if output_files:
                console.print(f"[green]✅ Transcribed {len(output_files)} files")
                return True
            console.print("[red]❌ No transcriptions generated")
            return False
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            console.print(f"[red]❌ Stage 3 failed: {e}")
            return False

    def run_stage4a(self) -> bool:
        """Stage 4a: Transcript merging."""
        console.print(Panel("🔀 Stage 4a: Transcript Merging", style="bold blue"))

        if not self.config.openai_api_key:
            console.print("[yellow]⚠️ Skipping merge - no OpenAI API key")
            return True

        try:
            output_file = merge_transcripts(self.config)
            if output_file:
                console.print(f"[green]✅ Merged transcript created: {output_file.name}")
                return True
            console.print("[yellow]ℹ️ No merging performed (single speaker or no transcripts)")
            return True
        except Exception as e:
            logger.error(f"Stage 4a failed: {e}")
            console.print(f"[red]❌ Stage 4a failed: {e}")
            return False

    def run_stage4b(self) -> bool:
        """Stage 4b: Final cleanup."""
        console.print(Panel("✨ Stage 4b: Final Cleanup", style="bold blue"))

        if not self.config.openai_api_key:
            console.print("[yellow]⚠️ Skipping cleanup - no OpenAI API key")
            return True

        try:
            output_file = cleanup_transcript(self.config)
            if output_file:
                console.print(f"[green]✅ Final transcript created: {output_file.name}")
                return True
            console.print("[yellow]ℹ️ No cleanup performed")
            return True
        except Exception as e:
            logger.error(f"Stage 4b failed: {e}")
            console.print(f"[red]❌ Stage 4b failed: {e}")
            return False

    async def run_full_pipeline(self, stages: str | None = None, continue_after: bool = False) -> bool:
        """Run the complete pipeline, specific stages, or resume from a specific stage."""
        console.print(Panel(f"🚀 Audio Processing Pipeline v{__version__}", style="bold magenta"))

        # Stage name mapping
        stage_mapping = {
            "bootstrap": "stage0_bootstrap",
            "preprocess": "stage1_preprocess",
            "vad": "stage2_vad",
            "whisper": "stage3_whisper",
            "merge": "stage4a_merge",
            "cleanup": "stage4b_cleanup",
        }

        # Parse stages parameter
        if stages:
            # Parse comma-separated stage list
            stage_names = [name.strip() for name in stages.split(",")]

            # Validate stage names
            invalid_stages = [name for name in stage_names if name not in stage_mapping]
            if invalid_stages:
                console.print(f"[red]❌ Invalid stage names: {', '.join(invalid_stages)}")
                console.print(f"[cyan]Valid stages: {', '.join(stage_mapping.keys())}")
                return False

            # Convert to internal stage names
            target_stages = [stage_mapping[name] for name in stage_names]
            start_stage = target_stages[0]

            console.print(f"[cyan]Running stages: {', '.join(stage_names)}")
            if continue_after:
                console.print("[cyan]Will continue to remaining stages after completion")
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
            self.status.status["pipeline_start"] = datetime.now().isoformat()
            self.status.save_status()

        # Stage execution mapping
        stages = {
            "stage0_bootstrap": self.bootstrap,
            "stage1_preprocess": self.run_stage1,
            "stage2_vad": self.run_stage2,
            "stage3_whisper": self.run_stage3,
            "stage4a_merge": self.run_stage4a,
            "stage4b_cleanup": self.run_stage4b,
        }

        # Get ordered list of stages to run
        stage_order = [
            "stage0_bootstrap", "stage1_preprocess", "stage2_vad",
            "stage3_whisper", "stage4a_merge", "stage4b_cleanup",
        ]

        start_index = stage_order.index(start_stage)

        # Determine which stages to run
        if target_stages and not continue_after:
            # Run only the specified stages
            stages_to_run = target_stages
        else:
            # Run from start_stage to end (or until target stages complete if continue_after)
            start_index = stage_order.index(start_stage)
            stages_to_run = stage_order[start_index:]

            if target_stages and continue_after:
                # Find the last target stage and include all stages up to end
                last_target_index = max(stage_order.index(stage) for stage in target_stages)
                stages_to_run = stage_order[start_index:]

        # Reset manually specified stages to pending (force re-run)
        if target_stages:
            for stage_name in target_stages:
                if self.status.is_stage_completed(stage_name):
                    console.print(f"[yellow]🔄 Resetting {stage_name} (manually specified)")
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
                        console.print(f"[yellow]🔄 Resetting {subsequent_stage} (subsequent stage)")
                        self.status.status["stages"][subsequent_stage]["status"] = "pending"
                        self.status.status["stages"][subsequent_stage]["start_time"] = None
                        self.status.status["stages"][subsequent_stage]["end_time"] = None

            self.status.save_status()

        # Run stages
        for stage_name in stages_to_run:
            # Only skip completed stages if they weren't manually specified
            if self.status.is_stage_completed(stage_name) and (not target_stages or stage_name not in target_stages):
                console.print(f"[green]✅ Skipping {stage_name} (already completed)")
                continue

            self.status.start_stage(stage_name)

            try:
                # Run the stage
                stage_func = stages[stage_name]
                if asyncio.iscoroutinefunction(stage_func):
                    success = await stage_func()
                else:
                    success = stage_func()

                self.status.complete_stage(stage_name, success)

                if not success:
                    console.print(f"[red]❌ Pipeline stopped at {stage_name}")
                    return False

                # If this was the last target stage and we're not continuing, stop here
                if target_stages and not continue_after and stage_name in target_stages:
                    remaining_targets = [s for s in target_stages if s in stages_to_run[stages_to_run.index(stage_name)+1:]]
                    if not remaining_targets:
                        console.print(f"[green]✅ Specified stages completed")
                        return True

            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                self.status.complete_stage(stage_name, False)
                console.print(f"[red]❌ Pipeline failed at {stage_name}: {e}")
                return False

        console.print(Panel("🎉 Pipeline completed successfully!", style="bold green"))
        return True


def setup_logging(verbose: bool = False) -> None:
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
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), help="Configuration file")
@click.pass_context
def cli(ctx, verbose: bool, config_file: Path | None):
    """Audio Processing Pipeline for Speech-to-Text Transcription."""
    setup_logging(verbose)

    # Load environment variables
    load_dotenv()

    # Load configuration
    try:
        config = load_config(config_file)
        ctx.ensure_object(dict)
        ctx.obj["config"] = config
    except Exception as e:
        console.print(f"[red]Configuration error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--stage", help="Comma-separated list of stages to run (bootstrap,preprocess,vad,whisper,merge,cleanup)")
@click.option("--continue", "continue_after", is_flag=True, help="Continue to next stages after specified stages complete")
@click.pass_context
def run(ctx, stage: str | None, continue_after: bool):
    """Run the complete audio processing pipeline or specific stages."""
    config = ctx.obj["config"]
    pipeline = AudioPipeline(config)

    try:
        success = asyncio.run(pipeline.run_full_pipeline(stages=stage, continue_after=continue_after))
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}")
        sys.exit(1)




@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status."""
    config = ctx.obj["config"]
    status_manager = PipelineStatus(config)
    status_manager.print_status()


@cli.command()
@click.pass_context
def reset(ctx):
    """Reset pipeline status."""
    config = ctx.obj["config"]

    if config.paths.status_file.exists():
        config.paths.status_file.unlink()
        console.print("[green]✅ Pipeline status reset")
    else:
        console.print("[yellow]⚠️ No status file found")


@cli.command()
@click.pass_context
def validate(ctx):
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
