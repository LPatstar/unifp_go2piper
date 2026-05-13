# AGENTS.md

## Project Identity

This repository is a UniFP-based loco-manipulation workspace for whole-body position-force control in Isaac Gym.

The main project direction is to use UniFP's position-force controller as a low-level execution mode for legged manipulation, with B2+Z1 as the prioritized robot path, and build toward high-level loco-manipulation door-opening tasks.

Current position-force tasks:
- `b2z1_pos_force`
- `go2_piper_pos_force`

The Go2+Piper adaptation is an early side path and useful secondary robot mode, not the final identity of the repository.

The repo is not a clean greenfield project. It keeps a large amount of upstream UniFP / B2+Z1 structure, and the B2+Z1 path should remain the default priority unless the user explicitly asks for Go2+Piper work.

## Environment Assumptions

- Python `3.8`
- Isaac Gym Preview 4
- Linux, typically Ubuntu `20.04` or `22.04`
- PyTorch and CUDA configured separately before running scripts

This repo is not meaningfully testable in a generic sandbox without a working Isaac Gym install and GPU runtime.

## Main Entry Points

Train:
```bash
cd legged_gym/scripts
python train_b2z1posforce.py --headless
```

Go2+Piper train:
```bash
cd legged_gym/scripts
python train_go2piperposforce.py --headless
```

High-level door teacher train:
```bash
cd legged_gym/scripts
python train_b2z1dooropen.py --headless
```

Play:
```bash
cd legged_gym/scripts
python play_b2z1posforce.py --load_run=<run_name>
```

High-level door teacher play:
```bash
cd legged_gym/scripts
python play_b2z1dooropen.py --load_run=<run_name> --checkpoint=<N>
```

Go2+Piper play:
```bash
cd legged_gym/scripts
python play_go2piperposforce.py --load_run=<run_name>
```

Play with joint tracking draw export:
```bash
cd legged_gym/scripts
python play_go2piperposforce.py --load_run=<run_name> --draw
```

Keyboard teleop / manual inspection:
```bash
cd legged_gym/scripts
python keyplay_posforce.py --load_run=<run_name>
```

Keyboard teleop with draw export:
```bash
cd legged_gym/scripts
python keyplay_posforce.py --load_run=<run_name> --draw
```

Automated evaluation:
```bash
cd legged_gym/scripts
python eval_posforce.py --load_run=<run_name> --headless
```

## Code Map

Go2-specific config:
- `legged_gym/envs/go2/go2_piper_pos_force_config.py`

Go2 task alias:
- `legged_gym/envs/go2/legged_robot_go2_piper_pos_force.py`

Shared environment implementation that actually contains most runtime logic:
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`

Shared base config inherited by Go2:
- `legged_gym/envs/b2/b2z1_pos_force_config.py`

Training stack:
- `legged_gym/b2_gym_learn/ppo_cse_pf/`
- `StateTeacherActorCritic` in `legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py` is the plain state-based high-level teacher policy class used by `b2z1_door_open`; its MLP widths follow the doorgym teacher shape `[512, 256, 128]`, while the original `ActorCritic` remains the UniFP low-level adaptation actor-critic.

Task registration:
- `legged_gym/envs/__init__.py`
- `legged_gym/utils/task_registry.py`

High-level door migration:
- `legged_gym/envs/door/b2z1_door_open_config.py`
- `legged_gym/envs/door/legged_robot_b2z1_door_open.py`
- `legged_gym/envs/door/unifp_low_level_adapter.py`
- `legged_gym/envs/door/door_asset_adapter.py`
- `legged_gym/scripts/train_b2z1dooropen.py`
- `legged_gym/scripts/play_b2z1dooropen.py`
- `legged_gym/scripts/play_b2z1dooropen_asset.py`
- `legged_gym/scripts/play_b2z1dooropen_walk.py`
- `legged_gym/scripts/play_b2z1dooropen_scripted.py`

## Architectural Notes

- Treat UniFP position-force control as the low-level controller layer for future door-opening loco-manipulation work.
- For new high-level task integration, assume B2+Z1 first unless the user explicitly requests Go2+Piper.
- Do not frame repository-level changes as primarily a B2+Z1-to-Go2+Piper migration. That migration exists, but it is only one supporting thread.
- `legged_robot_go2_piper_pos_force.py` is only a thin alias to the shared B2+Z1 environment class.
- Real Go2+Piper behavior is split between:
  - Go2 config overrides in `go2_piper_pos_force_config.py`
  - shared control / observation / viewer / force logic in `legged_robot_b2z1_pos_force.py`
- The Go2 config keeps the upstream control layout:
  - 12 leg joints plus the first 5 arm joints are policy-controlled
  - the final 3 Piper wrist / gripper joints are fixed-PD controlled
- Many changes that look "Go2-specific" still belong in the shared B2 environment because that is where command handling, force injection, debug drawing, viewer hotkeys, and rollout behavior live.

## Force And Command Conventions

- Commanded force and externally applied force are separate paths.
- Commanded EE force and commanded base force are stored in `commands`.
- External disturbance forces are applied through `self.forces`.
- In `play`, green arrows represent commanded force and blue arrows represent applied external force.
- For Go2+Piper, gripper force ranges are narrower than the inherited B2+Z1 defaults.

## Script Behavior Notes

- `train_b2z1dooropen.py` trains the state-based high-level door teacher for `b2z1_door_open` while freezing the configured B2+Z1 UniFP low-level policy.
- `b2z1_door_open` uses `StateTeacherActorCritic` instead of the low-level adaptation `ActorCritic`, so the high-level teacher is a direct state PPO policy and does not train a low-level latent estimator.
- `play_b2z1dooropen.py` loads a trained high-level door teacher checkpoint and prints door progress metrics during rollout.
- `play_b2z1dooropen_asset.py` validates door asset loading, actor/DOF/tensor shapes, and basic zero-action stepping. It defaults to `--low_level_policy_mode zero` so it does not require a low-level checkpoint.
- `play_b2z1dooropen_walk.py` keeps the door in the scene and sends base-forward high-level commands through the adapter to inspect whether the frozen low-level still walks near the door.
- `play_b2z1dooropen_scripted.py` is a scripted high-level sanity check, not a policy-performance benchmark. `--joint_assist` directly advances door/handle DOFs and should only be used to validate state/reward plumbing.
- Door teacher training logs door-specific episode metrics such as `door_success_rate`, `door_open_ratio`, `handle_open_ratio`, `open_stage_rate`, `closest_ee_handle_dist`, and `base_door_dist`.
- `train_b2z1posforce.py` and `play_b2z1posforce.py` are the B2+Z1 entry points and default to `b2z1_pos_force`.
- `train_go2piperposforce.py` and `play_go2piperposforce.py` are the Go2+Piper entry points and default to `go2_piper_pos_force`. The Go2 play wrapper enables force visualization / play-side command-force behavior through module flags.
- `eval_posforce.py` and `keyplay_posforce.py` are generic position-force entry points and default to `b2z1_pos_force`; pass `--task=go2_piper_pos_force` when using them on Go2+Piper runs.
- `play_go2piperposforce.py --draw` runs a short rollout, saves joint command-vs-actual plots for one front-left leg and the task-relevant arm joints, then exits. The shared play script auto-resolves the correct arm joint names for Go2+Piper and B2+Z1.
- `keyplay_posforce.py` creates a single-env teleop setup, disables most randomization, and relies on viewer keyboard events for command updates. In `--draw` mode it records joint command-vs-actual traces and uses `X` to save plots and exit.
- `eval_posforce.py` is the main reproducible checkpoint benchmark. Prefer it over ad hoc `play` sessions when comparing runs.
- `play_b2z1posforce.py`, `play_go2piperposforce.py`, `keyplay_posforce.py`, and `eval_posforce.py` all use checkpoint-resume loading. If `--load_run` and `--checkpoint` are omitted, the shared loader resolves to the latest run directory and then the latest saved `model_*.pt` checkpoint inside it.

## Play Draw Notes

- `play --draw` is the standard joint-level diagnostic path for tuning analysis.
- Default command:
  - `python play_go2piperposforce.py --load_run=<run_name> --draw`
- Useful optional flag:
  - `--draw_steps <N>` to control how many play steps are recorded before the script exits
- Output directory:
  - `play_draws/`
- Expected files:
  - one leg joint tracking plot
  - one arm joint tracking plot
- During tuning, inspect the PNG plots directly and judge both:
  - whether actual tracks command
  - and whether command itself already shows obvious oscillation or other bad control patterns
- When a tuning sample does not already have draw plots, generate them before forming a tuning conclusion.

## Keyplay Notes

- `keyplay` intentionally disables random command resampling and random force events.
- Keyplay controls are split between:
  - `legged_gym/scripts/keyplay_posforce.py`
  - viewer-event handling inside `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`
- Numeric viewer hotkeys are disabled in key-command mode to avoid collisions with numpad EE controls.
- In both ordinary keyplay and `keyplay --draw`, `V` toggles viewer sync. `X` is the dedicated exit key, and in `keyplay --draw` it saves plots before exiting.

## Evaluation Notes

- Automated evaluation writes reports under `eval_reports/`.
- Report folder names include the resolved run name and checkpoint so they can be matched back to tuning notes.
- `summary.json` is the machine-readable artifact and `summary.md` is the human-readable report.
- The report section named `Runtime Quality` summarizes runtime stability / posture / contact cleanliness / slip / smoothness.
- `Overall` is the weighted benchmark total, not the same quantity as `Runtime Quality`.
- `fetch_wandb_data.py --sync` keeps the normal compact local export and additionally creates a new `*_tb_sync` WandB run from the local TensorBoard event file so online charts use the real iteration-aligned axis.

## Working Rules For Future Changes

- Preserve the broader door-opening loco-manipulation direction when updating docs, configs, or task structure.
- Prefer B2+Z1 as the primary robot/task path for new high-level integration work unless the request says otherwise.
- Preserve compatibility with the inherited B2+Z1 implementation. Any Go2+Piper change should avoid breaking the original B2+Z1 task, scripts, config expectations, or shared environment behavior unless an explicit compatibility break is requested.
- When changing Go2 behavior, first decide whether it belongs in:
  - Go2 config overrides
  - the shared B2 environment implementation
  - the script layer
- If changing keyboard control, force injection, viewer drawing, or command interpretation, inspect the shared B2 environment file before assuming the Go2 wrapper contains the logic.
- If changing evaluation metrics or report structure, keep `README.md` and `EVAL_METRICS.md` aligned.
- Prefer targeted edits. This fork inherits duplicated concepts and upstream naming that are easy to break with broad refactors.

## Local Artifacts And Ignore Guidance

Do not commit these local or generated artifacts:
- `eval_reports/`
- `wandb/`
- `wandb_exports/`
- `logs/`
- `__pycache__/`
- local Isaac Gym install directories
- local paper PDFs or extracted text notes

## Validation Expectations

- For code-only changes, validate by reading the affected config, script, and shared environment paths together.
- For behavior changes, prefer running the narrowest relevant entry point:
  - `play` for quick viewer inspection
  - `keyplay` for manual command-path debugging
  - `eval` for reproducible checkpoint comparison
- Do not first attempt Isaac Gym runtime commands in the default sandbox when the task clearly needs real execution. For `play`, `keyplay`, `eval`, or other commands that rely on GPU access, Isaac Gym native bindings, viewer access, or writable PyTorch extension caches such as `~/.cache/torch_extensions`, request escalated execution immediately instead of failing once in the sandbox and retrying afterward.
- If Isaac Gym is unavailable, state that runtime validation could not be executed rather than guessing.

## Persistent 3-Subagent Tuning Pattern

This repository now has a standing shorthand workflow:

- If the user says:
  - `根据3subagents范式进行一轮新的调参`
  - or equivalent wording about `3 subagents`
- interpret that as the following required collaboration pattern unless the user explicitly overrides part of it.

### Scope Of The Pattern

This pattern is used not only for narrow reward tuning, but also for broader:

- diagnosis
- mechanism analysis
- training-result review
- transferability analysis
- tuning + small code/config changes

In other words, `3subagents范式` means a structured evidence-driven multi-agent workflow, not only “change one reward number”.

### Required Up-Front Reading

Before any diagnosis or modification, all participating agents should first read and absorb at least:

- `AGENTS.md`
- `README.md`
- `GO2_PIPER_TUNING_REQUIREMENTS.md`
- `EVAL_METRICS.md`
- `GO2_PIPER_CONFIG_REVIEW.md`
- the latest relevant root-level analysis md files
- the latest relevant files under `tuning_records/`

When a specific analysis md is named by the user, treat it as a primary input rather than a light reference.

### Default Agent Roles

Unless the user explicitly reassigns them, use exactly 3 agents with this meaning:

- Agent 1: primary diagnosis and final integration
  - build the first evidence chain
  - propose the first version of the tuning / modification plan
  - after feedback, integrate accepted comments into the final single output
- Agent 2: strict review and error correction
  - challenge Agent 1
  - look for misread docs, weak evidence, causality errors, bad parameter direction, missing risks, and spec violations
- Agent 3: supplemental diagnosis and expansion
  - look for missed mechanisms, deeper root causes, additional parameters, training-setting issues, code-path issues, and high-risk but worth-recording ideas

### Required Execution Order

The workflow is sequential, not parallel debate from the start:

1. Agent 1 forms the first diagnosis and first proposal.
2. Agent 2 reviews Agent 1 and points out problems.
3. Agent 3 supplements with additional diagnosis and alternatives.
4. The main agent integrates the three threads and produces the single final result.

Do not collapse this into one blended opinion.
The final output should clearly state:

- what Agent 1 first thought
- what Agent 2 corrected or rejected
- what Agent 3 added
- what was finally accepted
- what was not accepted, and why

### Evidence Standard

All three agents should ground their reasoning in available evidence first:

- `eval_reports/.../summary.json`
- `wandb_exports/.../ai_ready.json`
- `play_draws/...`
- relevant config and runtime code
- prior tuning records
- prior root-level analysis notes

Prefer existing artifacts when available.
If the user explicitly says not to rerun the latest `eval` or `play`, do not rerun them just to complete the ritual; rely on current artifacts and state that constraint clearly.

### Constraint Mode Switch

There are now two standing modes for future rounds:

1. Constrained tuning mode
   - Triggered when the user says things like:
     - `遵循调参文档约束`
     - `按调参文档要求`
     - or equivalent wording
   - In this mode:
     - use `GO2_PIPER_TUNING_REQUIREMENTS.md` as an actual constraint document
     - keep the usual reward-first stance
     - do not directly modify user-controlled physical embodiment parameters unless the user explicitly asks
     - still use the 3-subagent workflow above

2. Unconstrained analysis/tuning mode
   - Triggered when the user says things like:
     - `不受调参文档约束`
     - `这次不受 tuning requirements 限制`
     - or equivalent wording
   - In this mode:
     - still read `GO2_PIPER_TUNING_REQUIREMENTS.md` for methodology, comparison habits, and output structure
     - but do not treat its parameter-scope restrictions as binding
     - any reward/config/mechanism/code-path change may be considered if justified by evidence

If the user does not specify, follow the wording of the request:

- if they ask for standard tuning, assume constrained mode
- if they emphasize broader mechanism fixing or say the document is only reference, use unconstrained mode

### Output Expectations Under This Pattern

When the task is a true tuning round:

- keep exactly one final tuning record for the round
- update that one file rather than creating many iterative variants
- write the tuning record in Chinese
- include:
  - key observed phenomena
  - cause analysis
  - exact implemented changes
  - evidence used
  - Agent 2 / Agent 3 feedback absorption
  - final conclusion
  - bold ideas not implemented
  - physical/high-risk suggestions not implemented

When the task is broader analysis rather than a tuning round:

- a root-level analysis md is acceptable
- still preserve the same evidence chain and multi-agent structure

### Practical Interpretation Notes

- Do not mechanically chase the currently lowest benchmark case if a more important newly introduced mechanism problem exists.
- Separate:
  - true bottlenecks
  - metric blind spots
  - task-definition knobs
  - deployment transferability risks
- If a proposed parameter changes both task definition and controller behavior, call that out explicitly; do not present it as a clean one-dimensional fix.
