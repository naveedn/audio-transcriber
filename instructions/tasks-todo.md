## Todos

### code clean up
- refactor the main file to not have a numerical stage mapping, allow the order to be defined by array index ordering or explicitly by the user.
- fix linter errors
- add ruff as a pre-commit hook.
- add unit tests

### visual bugs
- fix bug in showing loading bars during transcribe step

### Additions
- ffmpeg audio-track Merge
- hallucination analyzer for manual fixing
- additional DND specific formatting on gpt-cleanup as a sub-task
- speaker diarization

### Architecture
- experiment with a local LLM such as llama for the text processing to compare performance and quality.
