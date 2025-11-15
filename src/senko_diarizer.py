"""Stage 2: Speaker diarization via Senko."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

from .config import Config

console = Console()
logger = logging.getLogger(__name__)


class SenkoDiarizationProcessor:
    """Run Senko diarization across preprocessed audio tracks."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.paths = config.paths
        self._diarizer = None

    def _build_diarizer(self, *, warmup_override: bool | None = None):
        import senko

        senko_cfg = self.config.senko
        diarizer_kwargs = {
            "device": senko_cfg.device,
            "vad": senko_cfg.vad,
            "clustering": senko_cfg.clustering,
            "quiet": senko_cfg.quiet,
            "warmup": senko_cfg.warmup,
        }
        if warmup_override is not None:
            diarizer_kwargs["warmup"] = warmup_override

        return senko.Diarizer(**diarizer_kwargs)

    def _get_diarizer(self):
        if self._diarizer is None:
            self._diarizer = self._build_diarizer()
        return self._diarizer

    def find_audio_files(self) -> list[Path]:
        audio_dir = self.paths.audio_wav_dir
        if not audio_dir.exists():
            logger.warning("Audio directory does not exist: %s", audio_dir)
            return []

        audio_files: list[Path] = []
        for pattern in ("*.wav", "*.WAV"):
            audio_files.extend(audio_dir.glob(pattern))
        return sorted(audio_files)

    def _serialize_result(
        self,
        audio_path: Path,
        diarization_result: dict[str, Any],
    ) -> dict[str, Any]:
        import senko

        speaker_centroids = {
            speaker_id: centroid.tolist()
            for speaker_id, centroid in diarization_result.get(
                "speaker_centroids", {}
            ).items()
        }
        vad_segments = [
            {"start": float(start), "end": float(end)}
            for start, end in diarization_result.get("vad", [])
        ]

        payload: dict[str, Any] = {
            "track": audio_path.stem,
            "wav_path": str(audio_path),
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "raw_segments": diarization_result.get("raw_segments", []),
            "raw_speakers_detected": diarization_result.get("raw_speakers_detected", 0),
            "merged_segments": diarization_result.get("merged_segments", []),
            "merged_speakers_detected": diarization_result.get(
                "merged_speakers_detected", 0
            ),
            "speaker_centroids": speaker_centroids,
            "timing_stats": diarization_result.get("timing_stats", {}),
            "vad_segments": vad_segments,
        }

        if diarization_result.get("speaker_color_sets"):
            payload["speaker_color_sets"] = diarization_result["speaker_color_sets"]

        payload["senko_version"] = getattr(senko, "__version__", "unknown")
        return payload

    def _save_result(self, audio_path: Path, payload: dict[str, Any]) -> Path:
        output_path = (
            self.paths.diarization_dir / f"{audio_path.stem}_diarization.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def process_files(self, audio_files: list[Path] | None = None) -> dict[str, Path]:
        if audio_files is None:
            audio_files = self.find_audio_files()

        if not audio_files:
            logger.warning("No audio files available for diarization")
            return {}

        diarizer = self._get_diarizer()
        senko_cfg = self.config.senko

        successful_outputs: dict[str, Path] = {}
        failed = 0

        with Progress() as progress:
            task_id = progress.add_task(
                "[blue]Running Senko diarization...", total=len(audio_files)
            )

            for audio_path in audio_files:
                try:
                    result = diarizer.diarize(
                        str(audio_path),
                        accurate=senko_cfg.accurate,
                        generate_colors=senko_cfg.generate_colors,
                    )
                except Exception:
                    logger.exception("Diarization failed for %s", audio_path.name)
                    failed += 1
                    progress.update(task_id, advance=1)
                    continue

                if not result:
                    logger.warning("No speakers detected for %s", audio_path.name)
                    progress.update(task_id, advance=1)
                    continue

                payload = self._serialize_result(audio_path, result)
                output_path = self._save_result(audio_path, payload)
                successful_outputs[audio_path.stem] = output_path
                progress.update(task_id, advance=1)

        if failed:
            console.print(f"[red]Failed diarization for {failed} files")

        return successful_outputs

    def check_dependencies(self) -> list[str]:
        errors: list[str] = []

        try:
            diarizer = self._build_diarizer(warmup_override=False)
            # Trigger immediate cleanup to avoid holding GPU/CPU memory twice.
            del diarizer
        except ModuleNotFoundError as exc:
            errors.append(f"Senko not installed: {exc}")
        except Exception as exc:
            errors.append(f"Unable to initialize Senko diarizer: {exc}")

        return errors


def diarize_audio(
    config: Config, audio_files: list[Path] | None = None
) -> dict[str, Path]:
    """Run Senko diarization synchronously."""
    processor = SenkoDiarizationProcessor(config)
    return processor.process_files(audio_files)
