# B2Z1 Door-Opening High-Level Integration Plan

## Purpose

This document is an AI-facing implementation plan for adding a high-level door-opening loco-manipulation task on top of the existing UniFP low-level position-force controller.

The intended task is:

- `b2z1_door_open`: a high-level training task for B2+Z1 door opening.

It should not be treated as a replacement for `b2z1_pos_force`. The expected first version is:

```text
high-level door policy
  -> low-level command adapter
  -> frozen UniFP B2+Z1 position-force low-level policy
  -> 17-D robot joint action
  -> robot + door simulation
  -> door-opening reward
```

The main engineering goal is to keep high-level and low-level modules decoupled so the low-level executor can later be swapped, finetuned, or replaced by another robot mode.

## Current Interface Facts

### Existing UniFP Low-Level Task

Current low-level task:

- `b2z1_pos_force`

Relevant files:

- `legged_gym/envs/b2/b2z1_pos_force_config.py`
- `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`
- `legged_gym/b2_gym_learn/ppo_cse_pf/`

The low-level policy action is not a high-level command. It is:

- 17-D policy action
- 12 leg joints + 5 arm joints
- interpreted as PD target offsets through the existing low-level control path

The existing command buffer is:

```text
commands[0:3]   = base velocity command: vx, vy, yaw_rate
commands[3:6]   = EE spherical position command
commands[6:9]   = EE orientation delta command
commands[9:12]  = commanded EE force
commands[12:15] = commanded base force
```

Important existing behavior:

- `env.step(actions)` expects the 17-D low-level policy action, not a high-level door action.
- The current environment can internally resample base commands, EE goals, and force events.
- For high-level control, random command resampling and random force events must be disabled or bypassed.
- `teleop_mode` and `key_command_mode` already show that external command driving is possible.

### Door High-Level Command Shape

The high-level door policy should not directly output UniFP joint actions.

Recommended first high-level action:

```text
action[0:3] = EE local Cartesian target delta
action[3:6] = EE local RPY target delta
action[6]   = gripper/contact mode command
action[7]   = base forward velocity command
action[8]   = base yaw velocity command
```

Optional force-aware extension:

```text
action[9:12] = EE commanded force in local/yaw-aligned frame
```

The force-aware extension should be added only after the position-only high-level loop is running.

## Required Abstraction

Add an explicit adapter boundary instead of letting the high-level task write all low-level internals directly.

Suggested conceptual API:

```text
LowLevelCommand:
  base_vel_local: [vx, vy, yaw_rate]
  ee_goal_local_cart: [x, y, z]
  ee_goal_local_rpy: [roll, pitch, yaw]
  ee_force_cmd_local: optional [fx, fy, fz]
  base_force_cmd_local: optional [fx, fy, fz]
  gripper_cmd: optional scalar or mode

LowLevelState:
  base pose / velocity
  EE pose / velocity
  current low-level commands
  low-level latent force estimate if available
  contact / force diagnostics if available

UniFPLowLevelExecutor:
  set_command(command)
  get_state()
  step_lowlevel()
```

The high-level door environment should depend on this adapter contract, not on raw UniFP command indices.

## Implementation Phases

## Phase 0: Freeze The Low-Level Baseline

Goal:

- Establish the exact `b2z1_pos_force` checkpoint and config used as the first frozen low-level executor.

Tasks:

- Pick a B2+Z1 checkpoint.
- Run or reuse `eval_posforce.py` report for the checkpoint.
- Record the run name, checkpoint, and important low-level limitations.
- Decide whether the first door experiments use state-based teacher PPO only.

Do not:

- Finetune low-level yet.
- Change reward weights before the high-level integration can run.

## Phase 1: Build The Low-Level Command Adapter

Goal:

- Make UniFP low-level externally commandable by a high-level task.

Tasks:

- Add a command adapter that maps high-level command fields to UniFP low-level buffers.
- Disable internal random command resampling in high-level-controlled mode.
- Disable random force events by default in door high-level mode.
- Provide a method to update EE target from local Cartesian/RPY command.
- Keep the existing `b2z1_pos_force` behavior unchanged for normal train/play/eval.

Expected adapter mapping:

```text
base_vel_local -> commands[:, 0:3]
ee_goal_local_cart -> curr_ee_goal_cart / curr_ee_goal_sphere / commands[:, 3:6]
ee_goal_local_rpy -> curr_ee_goal_orn_delta_rpy / commands[:, 6:9]
ee_force_cmd_local -> current_Fxyz_gripper_cmd / commands[:, 9:12]
base_force_cmd_local -> current_Fxyz_base_cmd / commands[:, 12:15]
```

Validation:

- A scripted high-level command sequence can move the base and EE target without random resampling overriding it.
- Existing `play_b2z1posforce.py`, `keyplay_posforce.py`, and `eval_posforce.py` still work.

## Phase 2: Add Robot + Door Simulation Task Shell

Goal:

- Create a new high-level task environment with both B2+Z1 and a door actor.

Suggested task name:

- `b2z1_door_open`

Tasks:

- Add a door task config.
- Add door assets and metadata paths.
- Create actor 0 as robot and actor 1 as door.
- Split tensor views into robot and door slices:
  - robot root state
  - door root state
  - robot DOF state
  - door DOF state
  - robot rigid bodies
  - door handle rigid body
- Pad robot torques / targets to full `robot + door` DOF count before sending tensors to Isaac Gym.
- Add passive or scripted door/handle dynamics:
  - handle spring
  - door resistance
  - door lock before handle press

High-risk area:

- UniFP currently assumes many tensors are robot-only. Adding a door actor changes root-state, DOF, rigid-body, and actor indexing assumptions. Do this with explicit robot/door slices rather than broad edits.

## Phase 3: Add Door State Observation

Goal:

- Train a state-based high-level teacher before adding visual policy complexity.

Recommended observation fields:

- handle pose in base/yaw-aligned local frame
- grasp goal position
- approach direction
- lever rotation direction
- door opening direction
- EE pose and EE-to-goal offset
- door hinge angle
- handle joint angle
- door opening ratio
- handle opening ratio
- base-to-door or base-to-handle distance
- current low-level command
- current EE target
- robot proprioception summary or selected low-level state
- optional low-level force estimate / force diagnostics

Avoid:

- Duplicating the full low-level observation unless there is a specific reason.
- Hard-coding B2+Z1-only internals in the high-level policy input when an adapter state field can expose the same information.

## Phase 4: Add High-Level Door Rewards And Termination

Goal:

- Train a door-opening policy using sparse success plus dense stage rewards.

Recommended reward stages:

- `approach_handle`: progress in EE-to-grasp-goal distance
- `ee_align_handle`: gripper orientation alignment with handle frame
- `lever_press`: handle joint progress before door opening
- `door_open_progress`: hinge progress after handle is pressed
- `door_open_success`: one-time sparse success
- `base_command_penalty`: suppress unnecessary base motion near the door
- `action_rate`: high-level command smoothness
- `force_penalty` or `force_direction_reward`: only after force action is enabled

Recommended termination:

- base posture failure inherited from low-level safety criteria
- IK/EE target infeasibility if applicable
- sustained door-open success
- EE too far from handle after a grace period
- base too far from the door after a grace period
- timeout

## Phase 5: Train Position-Only High-Level Teacher

Goal:

- Validate that the frozen UniFP low-level can support door approach, handle contact, and some opening behavior using only position/orientation/base commands.

Setup:

- Freeze low-level policy.
- Train only high-level policy.
- Start with state observations, no camera.
- Start with B2+Z1 only.
- Keep Go2+Piper out of the first implementation unless explicitly requested.

Success criteria:

- The policy can approach the handle.
- The policy can keep the base stable near the door.
- The policy can press the handle or produce repeatable contact attempts.
- Door metrics are logged clearly enough to diagnose failure.

Expected limitation:

- Pure position control may struggle with sustained handle pressure and pulling under contact.

## Phase 6: Add Force-Aware High-Level Control

Goal:

- Use UniFP's extra force information and commanded-force path to make door opening more robust.

Candidate high-level force actions:

- 1-D force along handle press direction
- 1-D force along door opening direction
- 3-D EE local force command

Recommended staged design:

1. Start with 1-D force along known task direction.
2. Add force magnitude limits and smoothness penalties.
3. Add observation of estimated/measured EE force.
4. Reward force direction alignment, not just force magnitude.
5. Penalize excessive force and unstable base reaction.

Potential benefits:

- Better handle press detection.
- More robust pulling/pushing after contact.
- Reduced dependence on exact position target placement.
- Better behavior under door friction and stiffness randomization.

Important distinction:

- UniFP's `self.forces` is currently an external force injection path.
- Door opening needs contact/interaction force information.
- Do not confuse commanded/external force buffers with measured handle contact force.

Recommended future improvement:

- Add or aggregate real EE/handle contact force signals.
- Expose them through `LowLevelState`.
- Keep force sensing and force command fields separate in the adapter API.

## Phase 7: Low-Level Finetuning Only If Needed

Goal:

- Improve contact-rich door manipulation only after high-level integration proves the bottleneck is low-level capability.

Possible low-level improvements:

- Contact-aware position-force finetuning.
- Door-handle contact randomization.
- Force-command tracking around handle-like obstacles.
- Low-level robustness to sustained EE contact while base stands near a door.
- Gripper/contact mode support if the current gripper handling is insufficient.

Do not start here. First prove what the frozen low-level can and cannot do.

## Current Low-Level Sufficiency Assessment

The current UniFP B2+Z1 low-level is sufficient for a first integration attempt because it already supports:

- base velocity commands
- EE position target tracking
- EE orientation target tracking
- commanded EE force fields
- commanded base force fields
- force estimator / latent diagnostics in the PPO stack
- external command paths through keyplay/eval style control

It is not sufficient as a final door-opening low-level because:

- it lacks a clean high-level command API
- gripper open/close is not yet a clean high-level primitive
- door contact is outside the main training distribution
- commanded/external force buffers are not the same as real door contact force sensing
- robot-only tensor assumptions must be fixed before adding door actors

## Modularity Rules For Future Agents

- Keep `b2z1_pos_force` usable as a standalone low-level task.
- Put high-level door task logic in a separate task/module.
- Put command conversion in an adapter layer.
- Do not let high-level policy code depend directly on raw UniFP command indices.
- Do not let low-level policy code depend on door reward or high-level PPO internals.
- Treat force command, external force, estimated force, and measured contact force as separate concepts.
- Prefer B2+Z1 first. Add Go2+Piper only after the B2+Z1 path is working or if explicitly requested.

## Minimal First Milestone

The first useful milestone is not a successful learned door policy. It is:

```text
scripted high-level door command
  -> adapter writes UniFP low-level commands
  -> frozen low-level policy runs
  -> robot and door simulate in one env
  -> door/handle metrics update correctly
  -> no random low-level command generator overwrites the high-level command
```

Only after this milestone should high-level PPO training be added.

