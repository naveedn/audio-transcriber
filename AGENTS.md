# Repository Guidelines

## Project Structure & Module Organization
Core pipeline code lives under `src/`, anchored by the `src/main.py` CLI orchestrator and stage modules such as `src/ffmpeg_preprocess.py`, `src/vad_timestamp.py`, and `src/gpt_cleanup.py`. Targeted utilities and scripts belong in the same package tree so Ruff can lint them. `tests/` currently hosts regression scripts; add new suites there and keep fixtures alongside. Audio inputs should stay in `inputs/`, while generated artifacts land in `outputs/` and remain untracked. Prompt templates reside in `prompts/`, and experiment notes live in `instructions/` and `CLAUDE.md`.

## Build, Test, and Development Commands
- `uv sync` installs runtime and dev dependencies from `pyproject.toml`.
- `uv run transcribe validate` confirms environment readiness, model downloads, and API keys.
- `uv run transcribe run` executes the full transcription pipeline; use `uv run transcribe run-stage <stage>` for focused debugging.
- `uv run pre-commit run --all-files` applies Ruff lint, formatting, and other hooks before committing.
- `uv run python tests/test_truncation.py` runs the current regression check; extend with `uv run python -m pytest` once pytest suites land.

## Coding Style & Naming Conventions
Adhere to Ruff’s configuration: 4-space indentation, 88-character lines, and Google-style docstrings where public. Modules stay snake_case (`src/gpt_merge.py`), classes use CapWords, and constants are UPPER_SNAKE. Favor type hints and dataclasses for config surfaces, and prefer `pathlib.Path` over raw strings for file access. Run `uv run ruff check --fix` followed by `uv run ruff format` before opening a PR.

## Testing Guidelines
Keep regression scripts deterministic and silent unless they detect anomalies. Name future tests `test_<feature>.py` in `tests/` and ensure they are discoverable by pytest. Store lightweight fixtures in `tests/assets/` (create if needed) and document expected outputs near the test. After modifying a stage, rerun `uv run transcribe validate` or a targeted `run-stage` command to confirm downstream behavior.

## Commit & Pull Request Guidelines
Existing history favors short, present-tense summaries (`Adds issues to bead issue tracker`). Group related changes and avoid mixed concerns. Pull requests should describe stage impacts, list any required secrets, and attach sample transcript diffs or timing metrics when applicable. Confirm pre-commit hooks passed, link issues or prompt experiments, and note any manual validation steps (`uv run transcribe status`).

## Security & Configuration Tips
Never commit `.env`, raw transcripts, or large audio assets. Validate keys locally with `uv run transcribe validate` before sharing logs. Rotate credentials if they appear in error output, and reference sensitive prompts by filename instead of pasting content.
BEFORE ANYTHING ELSE: run 'bd onboard' and follow the instructions
