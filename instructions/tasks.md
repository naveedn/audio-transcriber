I can see the issue! When the user runs uv run transcribe run --stage merge --continue, the system should reset all subsequent stages (like stage4b_cleanup) to "pending" status so they get re-run. Currently, the system only resets the explicitly specified stages,
  but with --continue, it should also reset all stages that come after the specified ones.
  ⎿  Session limit reached ∙ resets 2am
     /upgrade to increase your usage limit.

  Todos
  ☐ Fix continue flag to reset subsequent stage statuses
  ☐ Test the fixed continue flag behavior



- fix continue flag to reset subsequent stage statuses
- test the fixed continue flag behavior
- Update transcript merger step to use the same algorithm as original file (aka do not use chatGPT)
- test the step and commit the updated files once complete
- For cleanup (stage 4b), batch sizes must be bigger, run this step on the largest chunks possible without getting rate limited. Evaluate if chatGPT 4.1 or chatGPT 5 should be used instead.
- test the prior step and commit the updated files once complete
- Make sure speakers in the merged_segment retained their original names (zaboombafool, feliznaveedad, etc)
- test this fix and commit the updated files once complete
- Add the historical prompts from git history into the prompts folder, for experimentation on output
- Add readme
- refactor the main file to not have a numerical stage mapping, allow the order to be defined by array index ordering or explicitly by the user.
- commit the new files.
