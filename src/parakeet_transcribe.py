"""Parakeet CoreML transcription backend."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .whisper_transcribe import WhisperTranscriber

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


class ParakeetTranscriber:
    """Invoke the Swift Parakeet bridge for Senko-aware transcription."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.parakeet_config = config.parakeet
        self.cli_path = self._resolve_cli_path()
        # Reuse the Whisper helper for formatting + persistence.
        self._saver = WhisperTranscriber(config)

    def _resolve_cli_path(self) -> Path:
        """Locate the Parakeet CLI, falling back to known Swift build folders."""
        configured = self.parakeet_config.executable_path
        binary_name = configured.name or "parakeet-transcriber"

        candidates = [configured]

        project_root = Path(__file__).resolve().parent.parent
        bridge_root = project_root / "parakeet_bridge"
        build_root = bridge_root / ".build"

        # SwiftPM on macOS typically places binaries under `.build/<triple>/release`.
        candidates.extend(
            build_root / triple / "release" / binary_name
            for triple in ("arm64-apple-macosx", "x86_64-apple-macosx")
        )

        # Older SwiftPM layouts use `.build/release`.
        candidates.append(build_root / "release" / binary_name)

        for candidate in candidates:
            if candidate.exists():
                if candidate != configured:
                    logger.info("Resolved Parakeet CLI to %s", candidate)
                return candidate

        return configured

    def _build_command(
        self,
        audio_path: Path,
        diarization_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Construct the CLI command for the Swift bridge."""
        cmd = [
            str(self.cli_path),
            "--audio",
            str(audio_path),
            "--diarization",
            str(diarization_path),
            "--output",
            str(output_path),
            "--language",
            self.parakeet_config.language,
            "--model-version",
            self.parakeet_config.model_version,
            "--min-seconds",
            str(self.parakeet_config.min_segment_seconds),
        ]

        if self.parakeet_config.models_root:
            cmd.extend(["--models-root", str(self.parakeet_config.models_root)])

        return cmd

    def _env(self) -> dict[str, str]:
        """Prepare environment variables for the Swift CLI."""
        env = os.environ.copy()
        token = self.config.huggingface_token or env.get("HF_TOKEN")
        if token and "HF_TOKEN" not in env:
            env["HF_TOKEN"] = token
        return env

    def transcribe_file(
        self,
        audio_path: Path,
        diarization_path: Path,
        progress: object | None = None,
        task_id: object | None = None,
    ) -> dict[str, Any]:
        """Transcribe a single audio file by shelling out to the Swift binary."""
        del progress, task_id

        if not self.cli_path.exists():
            msg = f"Parakeet CLI missing at {self.cli_path}"
            raise FileNotFoundError(msg)

        with tempfile.TemporaryDirectory(prefix="parakeet-") as tmpdir:
            tmp_output = Path(tmpdir) / "transcription.json"
            cmd = self._build_command(audio_path, diarization_path, tmp_output)
            logger.info("Running Parakeet CLI: %s", " ".join(cmd))
            result = subprocess.run(  # noqa: S603
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=self._env(),
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                logger.error("Parakeet CLI failed: %s", stderr or stdout)
                message = f"Parakeet CLI failed with exit code {result.returncode}"
                raise RuntimeError(message)

            try:
                payload = json.loads(tmp_output.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                msg = f"Parakeet CLI did not produce output at {tmp_output}"
                raise RuntimeError(msg) from exc
            except json.JSONDecodeError as exc:
                logger.exception("Invalid JSON from Parakeet CLI")
                message = "Parakeet CLI produced invalid JSON"
                raise RuntimeError(message) from exc

        return payload

    def save_transcription(
        self,
        transcription: dict[str, Any],
        output_path: Path,
        speaker_name: str,
    ) -> None:
        """Reuse Whisper's save routine for JSON + SRT outputs."""
        self._saver.save_transcription(transcription, output_path, speaker_name)

    def check_dependencies(self) -> list[str]:
        """Validate that the Swift CLI is available."""
        errors: list[str] = []

        if not self.cli_path.exists():
            errors.append(
                f"Parakeet CLI not found at {self.cli_path}. "
                "Build it via `cd parakeet_bridge && swift build -c release`."
            )
        elif not os.access(self.cli_path, os.X_OK):
            errors.append(
                f"Parakeet CLI at {self.cli_path} is not executable. "
                "Recompile or fix permissions."
            )

        return errors
