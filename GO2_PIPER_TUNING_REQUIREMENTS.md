# GO2_PIPER_TUNING_REQUIREMENTS

## Purpose

This note defines the required workflow and constraints for future Go2+Piper tuning work.

It is intentionally short and operational so a new agent window can quickly understand:

- what to tune,
- what not to tune directly,
- what evidence to gather before changing anything,
- and how to record each tuning attempt afterward.

## Key References

Before tuning, use these files for context:

- `GO2_PIPER_EVAL_METRICS.md`
  - Use this to interpret `eval` outputs and understand what each benchmark score means.
- `GO2_PIPER_CONFIG_REVIEW.md`
  - Use this to understand inherited config structure and which areas are still likely to need review.

Important constraint:

- These reference files are for understanding context.
- Actual tuning focus should be reward design and reward scales first, not physical robot parameters.

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
- especially reward items that are visible in WandB training curves
- especially how those curves compare against:
  - the immediately previous Go2+Piper run
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

1. Run one fresh `eval` on the latest model checkpoint.
   - Use the latest checkpoint by default unless the user explicitly asks for a specific run/checkpoint.
   - The purpose is to get a current benchmark snapshot before making any tuning decision.

2. Check whether the latest training run already has a compact WandB export.
   - If not, run `fetch_wandb_data.py` first.
   - Prefer the compact `ai_ready.json` output for quick analysis.

3. Analyze both:
   - latest `eval` result
   - latest training curves / exported WandB data

4. Compare against:
   - the previous Go2+Piper training run
   - the historical B2+Z1 baseline training data when useful

## Analysis Rules

When using training curves and exported WandB data:

- compare both absolute values and trend shapes
- compare improvement speed, stability, saturation, oscillation, and collapse patterns
- pay attention to individual reward components, not only total reward

Important caution:

- if reward scales were changed between runs, direct numeric comparison becomes less reliable
- in that case, place more weight on:
  - curve shape,
  - consistency,
  - eval behavior,
  - benchmark outcomes,
  - and what the reward decomposition suggests about policy behavior

The agent should reason carefully about what the observable signals imply, rather than relying on naive one-number comparisons.

## Preferred Decision Logic

The tuning loop should generally be:

1. Inspect latest `eval`
2. Inspect latest compact WandB export
3. Compare with previous Go2+Piper run
4. Compare with B2+Z1 baseline when informative
5. Form a reward-focused hypothesis
6. Change only the smallest justified reward-related config
7. Record the rationale and exact changes

## Post-Tuning Record Requirement

After each tuning round, record the result as a new markdown file under:

- `tuning_records/`

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
- compare against previous Go2+Piper and B2+Z1 evidence carefully
- record every tuning round in `tuning_records/`
