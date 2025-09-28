## Todos
- fix continue flag to reset subsequent stage statuses (When the user runs uv run transcribe run --stage merge --continue, the system should reset all subsequent stages (like stage4b_cleanup) to "pending" status so they get re-run. Currently, the system only resets the explicitly specified stages,
  but with --continue, it should also reset all stages that come after the specified ones)
- test the fixed continue flag behavior
- Update transcript merger step to use the same algorithm as original file (aka do not use chatGPT)
- test the step and commit the updated files once complete
- Merge adjacent segments after whisper processing if BOTH are shorter than min_sentence_ms and the gap between them is < merge_gap_ms. Concatenate with a space. See the whisper second pass chunked from vad audio file for reference. /Users/naveednadjmabadi/code/dnd-podcast-transcriber/audio-processing-pipeline/whisper-second-pass/whisper_chunked_from_vad.py Lines 236-264
- test the step and commit the updated files once complete
- For cleanup (stage 4b), batch sizes must be bigger, run this step on the largest chunks possible without getting rate limited. Evaluate if chatGPT 4.1 or chatGPT 5 should be used instead.
- test the prior step and commit the updated files once complete
- Make sure speakers in the merged_segment retained their original names (zaboombafool, feliznaveedad, etc)
- test this fix and commit the updated files once complete
- Add the historical prompts from git history into the prompts folder, for experimentation on output
- Add readme
- refactor the main file to not have a numerical stage mapping, allow the order to be defined by array index ordering or explicitly by the user.
- commit the new files.
- debug why the whisper step keeps downloading a model instead of using one that is cached on the filesystem?


### Future TODO:
- add ruff as a pre-commit hook.
- add unit tests
