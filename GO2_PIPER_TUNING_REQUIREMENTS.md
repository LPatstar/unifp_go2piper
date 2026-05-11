# GO2_PIPER_TUNING_REQUIREMENTS

## Purpose

This note defines the required workflow and constraints for future Go2+Piper tuning work.

It is intentionally short and operational so a new agent window can quickly understand:

- what to tune,
- what not to tune directly,
- what evidence to gather before changing anything,
- and how to record each tuning attempt afterward.

## Important Tuning Stance

Do not interpret "focused tuning" as "only make tiny safe changes."

- A tuning round may include a small number of hypothesis-driven, moderately bold reward-related changes when the available evidence is suggestive but not yet conclusive.
- These changes must still be logically grounded in `eval` results, WandB trends, multi-run comparison, or B2+Z1 baseline differences; they should not be random or careless edits.
- The goal is to avoid getting stuck in repeated micro-tweaks that leave obviously poor behavior largely unchanged.

## Environment Assumption

All Go2+Piper tuning, evaluation, and WandB export commands should be run in the project's Conda environment:

- `unifp`

In practice, prefer one of:

- `conda activate unifp`
- `conda run -n unifp <command>`

Do not assume the system/default Python has the required Isaac Gym / PyTorch / NumPy stack.

## Key References

Before tuning, use these files for context:

- `EVAL_METRICS.md`
  - Use this to interpret `eval` outputs and understand what each benchmark score means.
- `GO2_PIPER_CONFIG_REVIEW.md`
  - Use this to understand inherited config structure and which areas are still likely to need review.

Important constraint:

- These reference files are for context.
- Actual tuning focus should remain reward design and reward scales first, not physical robot parameters.

## Context-Efficient File Reading Rule

To save context in future tuning sessions:

- when reading eval results, read `summary.json` first
- when reading WandB export results, read `ai_ready.json` first

Unless there is a clear and specific need, do not read the companion markdown or other export files.

In practice this means:

- prefer `eval_reports/.../summary.json`
- prefer `wandb_exports/.../ai_ready.json`
- avoid reading `summary.md`, `history.jsonl`, `history.csv`, `config.json`, or `summary.json` from WandB export unless the tuning task truly needs them

## Tuning Scope

Primary tuning target:

- reward terms and reward scales
- especially reward items visible in WandB training curves
- especially how those curves compare against:
  - at least the previous two Go2+Piper runs
  - and, if those two runs do not yet reveal a clear issue, the previous three Go2+Piper runs
  - historical B2+Z1 baseline runs

Typical direct edit target:

- `legged_gym/envs/go2/go2_piper_pos_force_config.py`

Most likely section to modify:

- reward scales and related reward configuration

## Parameters That Must Not Be Directly Modified By The Agent

The following categories are user-controlled and should not be directly changed by the agent during tuning:

- robot size / geometry
- goal EE physical workspace
- force ranges
- COM / mass / inertia related physical settings
- geometric offsets
- other physics-shaped parameters tied to real robot embodiment

The agent may:

- point out these parameters,
- explain why they may matter,
- suggest possible future manual changes,

but should not directly edit them unless the user explicitly asks for that.

## Required Pre-Tuning Workflow

Before changing any training config:

Important eval execution rule:

- For each checkpoint that needs evaluation, run exactly one `eval` command.
- If repeated scenario aggregation is needed, use `--eval_repeats=3` within that single command rather than launching multiple separate `eval` runs for the same `.pt`.
- Avoid generating duplicate report folders for the same checkpoint unless there is a specific reason such as an interrupted run, an evaluation-code change, or an explicit user request.

1. Run one fresh `eval` on the latest model checkpoint.
   - Use the latest checkpoint by default unless the user explicitly asks for a specific run/checkpoint.
   - The purpose is to get a current benchmark snapshot before making any tuning decision, and the default pre-tuning check should not evaluate only one checkpoint.
   - Instead, identify:
     - the largest available checkpoint in the latest run, for example `36600.pt`
     - and the checkpoint that is `10000` iterations earlier when it exists, for example `26600.pt`
   - Evaluate both checkpoints before tuning, and use `--eval_repeats=3` for each so the report aggregates three repeated passes of each scripted scenario.
   - For each selected checkpoint, this still means one `eval` command per `.pt`, not multiple reruns that create redundant reports.
   - This serves two purposes:
     - reduce the chance of making a tuning decision based on one noisy eval
     - and catch cases where the very latest checkpoint has already started to train past its best behavior
   - If the `latest - 10000` checkpoint does not exist, state that clearly and evaluate only the latest checkpoint with `--eval_repeats=3`.

2. Check whether the latest training run already has a compact WandB export.
   - If not, run `fetch_wandb_data.py` first.
   - Prefer the compact `ai_ready.json` output for quick analysis.
   - `fetch_wandb_data.py` may take a noticeably long time to finish.
   - Do not wait only briefly and then conclude that no new export was generated.
   - Before deciding the export is missing or failed, allow enough time for the fetch/export step to complete and then check the expected output directory again.

3. Check whether the sample being analyzed already has the joint tracking plots produced by `play --draw`.
   - If not, run one `play` with `--draw` for that sample first.
   - Standard command:
     - `cd legged_gym/scripts`
     - `python play_go2piperposforce.py --load_run=<run_name> --draw`
   - Use `--draw_steps <N>` only when the default recording window is clearly not enough.
   - The plots are saved under `play_draws/`.
   - In normal tuning workflow, directly inspect the saved PNG plots themselves.
   - Keep the generated leg and arm joint command-vs-actual plots together with the rest of the analysis material.

4. Analyze all three:
   - latest `eval` result
   - latest training curves / exported WandB data
   - latest `play --draw` plots

5. Compare against:
   - at least the previous two Go2+Piper training runs
   - and, if those comparisons still do not reveal a likely issue, expand to the previous three Go2+Piper training runs
   - the historical B2+Z1 baseline training data when useful

If `tuning_records/` does not yet contain any prior tuning markdown notes, treat the session as the first tuning round:

- comparison against a previous tuning note is not required
- still compare against the latest available Go2+Piper run/checkpoint evidence
- still use historical B2+Z1 baseline data when it helps interpret current behavior

## Analysis Rules

When using training curves and exported WandB data:

- compare absolute values together with full-curve trend shape
- compare early-stage learning speed, mid-stage transition behavior, and late-stage saturation / regression / collapse
- pay attention to individual reward components, not only total reward
- do not rely mainly on only the "last N sampled points" such as the last 10 points
- a short tail-window summary may be used as a quick helper, but it must not replace reading the full curve / full sampled history
- the main tuning judgment should use the global trajectory of training data, including:
  - inflection points where dominant reward terms change
  - whether one reward term becomes too strong and starts suppressing others
- when possible, reason from the whole available WandB history first, then use tail summaries only as supporting evidence
- if a later checkpoint looks worse than an earlier checkpoint, do not immediately treat that as a reward-design failure; first check whether the full curve indicates late-training degradation or overtraining

When reading `eval` results:

- pay especially close attention to `success rate`, because it is a high-priority indicator of whether the behavior is actually succeeding rather than only looking numerically better
- in many cases, `success rate` should carry more practical weight than a small improvement in RMSE or score if those improvements do not translate into actual successful outcomes
- however, do not look at `success rate` in isolation
- still read the other eval signals together, especially:
  - EE RMSE / tracking error
  - base velocity and yaw tracking error
  - posture / stability / slip / collision metrics
  - and the case score composition
- the correct goal is not "only maximize success rate", but "treat success rate as a top-priority outcome while using the rest of the eval metrics to explain why it is high or low"

When reading `play --draw` results:

- compare the saved PNG command-vs-actual plots carefully at the joint level
- in normal tuning workflow, these plots should be treated as required evidence rather than optional nice-to-have visuals
- do not only ask whether `actual` follows `command`
- also inspect whether the `command` trajectories themselves already look problematic
- before concluding that a visible pattern is a real problem, check the y-axis scale carefully
- make sure the apparent issue is not just a very small numerical variation visually amplified by a tight vertical range
- when possible, compare the same style of draw plot against other runs, especially the B2+Z1 baseline, before deciding that a suspected issue is truly significant
- pay special attention to:
  - significant command oscillation / chatter
  - visibly noisy or jagged command trajectories
  - persistent lag
  - overshoot
  - oscillation
  - saturation / clipping-like behavior
  - large steady-state mismatch
  - whether the leg and arm show different tracking quality patterns
- if the command itself is already unstable or oscillatory, treat that as an important clue that the reward design may be encouraging a bad solution
- similarly, if `actual` appears not to track `command`, verify from the axis scale and comparison runs that the mismatch is materially large rather than a negligible fluctuation
- treat the draw plots as one of the most important diagnostic signals for deciding what reward-related problem may be happening underneath
- use them together with `eval` and WandB data, not as a standalone decision source

Important caution:

- if reward scales were changed between runs, direct numeric comparison becomes less reliable
- in that case, place more weight on:
  - curve shape,
  - consistency,
  - eval behavior,
  - benchmark outcomes,
  - and what the reward decomposition suggests about policy behavior

The agent should reason carefully about what the observable signals imply, rather than relying on naive one-number comparisons.

Important baseline interpretation:

- the historical B2+Z1 training result is considered a healthy / meaningful reference
- in particular, the B2+Z1 run is known to be a strong training result whose eval scores are almost all above `90`, so it should be treated as a high-quality baseline rather than a weak loose reference
- even when reward scale changes make absolute WandB numbers not directly comparable, the B2+Z1 run is still useful for:
  - trend shape,
  - improvement speed,
  - stability vs. oscillation,
  - saturation behavior,
  - and which reward terms appear to dominate training
- in other words, B2+Z1 may still indicate what a "more normal" reward balance looks like, even when exact scalar values no longer transfer cleanly
- if some reward curves or dominance relationships differ strongly from B2+Z1, treat that difference itself as an important diagnostic signal that must be explained
- when such differences appear, think carefully about multiple possible causes instead of jumping to one fixed explanation:
  - some differences may come from real embodiment differences, such as robot size, inertia, joint loading, or torque scale
  - other differences may come from reward imbalance, poor reward coupling, or a policy learning the wrong priority
- when comparing the current Go2+Piper run against B2+Z1, pay special attention to whether the same kinds of reward terms are leading progress, and whether unexpectedly weak / over-dominant terms reveal a more subtle tuning problem
- do not fall into a narrow loop of repeatedly micro-tweaking only the same few recently changed reward scales if the global training evidence does not support that direction
- always use the full training history, the multi-run comparison, and the B2+Z1 high-quality baseline together to strip away superficial symptoms and identify the real bottleneck
- it is acceptable, and often necessary, to iterate on several competing explanations before deciding which reward term actually deserves the next change
- this is one key reason WandB data and `eval` results should be read together:
  - WandB helps explain which reward terms are driving policy learning
  - `eval` confirms whether that learning actually turns into the intended behavior
  - using both together makes it easier to catch subtle mismatches that either signal alone could miss

## Preferred Decision Logic

The tuning loop should generally be:

1. Inspect latest `eval`
2. Inspect latest compact WandB export
3. Inspect latest `play --draw` plots
4. Compare with at least the previous two Go2+Piper runs
5. If those two runs do not expose a clear issue, expand comparison to the previous three Go2+Piper runs
6. Compare with B2+Z1 baseline when informative
7. Form a reward-focused hypothesis
8. Change a focused, coherent reward-related config set that meaningfully tests the hypothesis, including a small number of moderately bold adjustments when the evidence is suggestive but not yet conclusive
9. Record the rationale and exact changes

## Post-Tuning Record Requirement

After each tuning round, record the result as a new markdown file under:

- `tuning_records/`

Record language requirement:

- tuning record markdown files under `tuning_records/` should be written in Chinese for easier review
- these tuning notes are local working records and do not need to be committed to git unless the user explicitly asks

Each record should include at least:

- date / run being tuned
- checkpoint or run used as the starting point
- previous eval folder path
- eval summary before tuning
- key WandB observations
- exact config changes made
- reasoning behind the changes
- expected effect
- suggested next run name / run suffix
- later follow-up result if available

## Naming Suggestion For Future Tuning Notes

Use one file per tuning round, for example:

- `tuning_records/2026-04-05_reward_scale_adjustment_01.md`
- `tuning_records/2026-04-05_tracking_reward_rebalance.md`

## Summary

For future Go2+Piper tuning:

- tune rewards first
- do not directly edit physical embodiment parameters
- always evaluate first
- fetch compact WandB data if missing
- if `play --draw` plots are missing for the sample being analyzed, generate them first and treat them as a key diagnostic signal
- compare against at least the previous two Go2+Piper runs, and expand to the previous three if the first two do not reveal the issue
- compare against B2+Z1 evidence carefully
- record every tuning round in `tuning_records/`
