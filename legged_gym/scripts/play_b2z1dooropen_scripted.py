import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym  # noqa: F401
import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


WALK_TO_DOOR = 0
MOVE_TO_PREGRASP = 1
MOVE_TO_HANDLE = 2
CLOSE_AND_SEAT = 3
PRESS_HANDLE = 4
PUSH_DOOR = 5
HOLD_OPEN = 6

PHASE_NAMES = {
    WALK_TO_DOOR: "walk",
    MOVE_TO_PREGRASP: "pregrasp",
    MOVE_TO_HANDLE: "handle",
    CLOSE_AND_SEAT: "close",
    PRESS_HANDLE: "press",
    PUSH_DOOR: "push",
    HOLD_OPEN: "hold",
}
PHASE_NAME_TO_ID = {name: phase_id for phase_id, name in PHASE_NAMES.items()}


CUSTOM_PARAMETERS = [
    {"name": "--steps", "type": int, "default": 700, "help": "Number of scripted high-level steps."},
    {
        "name": "--start_phase",
        "type": str,
        "default": "walk",
        "help": "Initial scripted phase: walk, pregrasp, handle, close, press, push, or hold.",
    },
    {
        "name": "--walk_stop_distance",
        "type": float,
        "default": 0.75,
        "help": "Switch from walk to pregrasp once base-to-handle distance is below this value.",
    },
    {
        "name": "--walk_speed",
        "type": float,
        "default": 0.18,
        "help": "Forward base velocity command during the scripted walk phase.",
    },
    {
        "name": "--walk_timeout_steps",
        "type": int,
        "default": 180,
        "help": "Maximum scripted walk steps before switching to pregrasp.",
    },
    {
        "name": "--debug_targets",
        "action": "store_true",
        "default": False,
        "help": "Print scripted target offsets relative to the handle grasp goal.",
    },
    {
        "name": "--low_level_policy_mode",
        "type": str,
        "default": None,
        "help": "Frozen low-level mode: checkpoint or zero. Use zero for adapter-only smoke tests.",
    },
    {
        "name": "--low_level_load_run",
        "type": str,
        "default": None,
        "help": "B2+Z1 low-level run folder under logs/b2z1_pos_force. Use -1 for latest.",
    },
    {
        "name": "--low_level_checkpoint",
        "type": int,
        "default": None,
        "help": "B2+Z1 low-level checkpoint number. Use -1 for latest.",
    },
    {
        "name": "--pin_base",
        "action": "store_true",
        "default": False,
        "help": "Keep the robot base fixed for door/arm sanity checks.",
    },
    {
        "name": "--joint_assist",
        "action": "store_true",
        "default": False,
        "help": "Script door/handle DOFs after contact phases to validate state/reward plumbing.",
    },
    {
        "name": "--allow_resets",
        "action": "store_true",
        "default": False,
        "help": "Allow normal task termination during the scripted sequence.",
    },
]


def _parse_start_phase(name):
    phase_name = str(name).strip().lower()
    if phase_name not in PHASE_NAME_TO_ID:
        raise ValueError(
            f"Unknown --start_phase '{name}'. Expected one of: {', '.join(PHASE_NAME_TO_ID.keys())}"
        )
    return PHASE_NAME_TO_ID[phase_name]


def _set_phase(phase, phase_steps, mask, new_phase):
    if torch.any(mask):
        phase[mask] = new_phase
        phase_steps[mask] = 0


def _phase_counts(phase):
    return {
        "walk": int((phase == WALK_TO_DOOR).sum().item()),
        "pregrasp": int((phase == MOVE_TO_PREGRASP).sum().item()),
        "handle": int((phase == MOVE_TO_HANDLE).sum().item()),
        "close": int((phase == CLOSE_AND_SEAT).sum().item()),
        "press": int((phase == PRESS_HANDLE).sum().item()),
        "push": int((phase == PUSH_DOOR).sum().item()),
        "hold": int((phase == HOLD_OPEN).sum().item()),
    }


def _pin_robot_base(env, target_root_states):
    env.root_states[:] = target_root_states
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._all_root_states_tensor),
        gymtorch.unwrap_tensor(env.robot_actor_ids),
        len(env.robot_actor_ids),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._update_door_derived_state()


def _scripted_joint_assist(env, phase):
    press_like = phase == PRESS_HANDLE
    if torch.any(press_like):
        target_handle = torch.maximum(
            env._door_dof_pos[press_like, 1] + 0.03,
            env.door_dof_lower[press_like, 1]
            + float(env.cfg.door.handle_press_threshold_ratio)
            * (env.door_dof_upper[press_like, 1] - env.door_dof_lower[press_like, 1])
            + 0.08,
        )
        target_handle = torch.minimum(target_handle, env.door_dof_upper[press_like, 1])
        env._door_dof_pos[press_like, 1] = target_handle
        env._door_dof_vel[press_like, 1] = 0.0

    push_like = ((phase == PUSH_DOOR) | (phase == HOLD_OPEN)) & env.open_door_stage
    if torch.any(push_like):
        hinge_sign = -1.0 if float(getattr(env.cfg.door, "door_hinge_open_sign", 1.0)) < 0.0 else 1.0
        target_door = env._door_dof_pos[push_like, 0] + 0.025 * hinge_sign
        target_door = torch.maximum(target_door, env.door_dof_lower[push_like, 0])
        target_door = torch.minimum(target_door, env.door_dof_upper[push_like, 0])
        env._door_dof_pos[push_like, 0] = target_door
        env._door_dof_vel[push_like, 0] = 0.0

    assisted = press_like | push_like
    if torch.any(assisted):
        actor_ids = env.door_actor_ids[assisted]
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env._all_dof_state_tensor),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env._update_door_derived_state()
    return int(assisted.sum().item())


def _configure_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    if args.low_level_policy_mode is not None:
        env_cfg.low_level.policy_mode = args.low_level_policy_mode
    if args.low_level_load_run is not None:
        env_cfg.low_level.load_run = -1 if args.low_level_load_run == "-1" else args.low_level_load_run
    if args.low_level_checkpoint is not None:
        env_cfg.low_level.checkpoint = args.low_level_checkpoint
    env_cfg.env.episode_length_s = max(
        env_cfg.env.episode_length_s,
        args.steps * env_cfg.sim.dt * env_cfg.control.decimation + 1.0,
    )
    return env_cfg


def main():
    args = get_args(custom_parameters=CUSTOM_PARAMETERS)
    if "--task" not in sys.argv:
        args.task = "b2z1_door_open"

    env_cfg = _configure_env(args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    if not args.allow_resets:
        def _no_script_reset():
            env.reset_buf.zero_()
            env.time_out_buf.zero_()

        env.check_termination = _no_script_reset

    pinned_root_states = env.root_states.clone()
    pinned_root_states[:, 7:13] = 0.0

    start_phase = _parse_start_phase(args.start_phase)
    phase = torch.full((env.num_envs,), start_phase, device=env.device, dtype=torch.long)
    phase_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    for step in range(args.steps):
        phase_steps += 1
        target_pos = env.get_pregrasp_goal_world(offset=0.22)
        target_rot = env.ee_orn.clone()
        aligned = phase >= MOVE_TO_HANDLE
        target_rot[aligned] = env.handle_target_rot[aligned]
        gripper_open = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        base_x_cmd = torch.zeros(env.num_envs, device=env.device)
        yaw_cmd = torch.zeros(env.num_envs, device=env.device)

        walk_mask = phase == WALK_TO_DOOR
        pregrasp_mask = phase == MOVE_TO_PREGRASP
        handle_mask = phase == MOVE_TO_HANDLE
        close_mask = phase == CLOSE_AND_SEAT
        press_mask = phase == PRESS_HANDLE
        push_mask = phase == PUSH_DOOR
        hold_mask = phase == HOLD_OPEN

        if torch.any(walk_mask):
            target_pos[walk_mask] = env.get_pregrasp_goal_world(offset=0.26)[walk_mask]
            base_x_cmd[walk_mask] = float(args.walk_speed)
        if torch.any(pregrasp_mask):
            target_pos[pregrasp_mask] = env.get_pregrasp_goal_world(offset=0.20)[pregrasp_mask]
        if torch.any(handle_mask):
            target_pos[handle_mask] = (env.grasp_goal_world + env.handle_approach_dir_world * 0.035)[handle_mask]
        if torch.any(close_mask):
            target_pos[close_mask] = (env.grasp_goal_world + env.handle_approach_dir_world * 0.005)[close_mask]
            gripper_open[close_mask] = False
        if torch.any(press_mask):
            target_pos[press_mask] = (
                env.grasp_goal_world
                + env.handle_approach_dir_world * 0.002
                + env.handle_rotate_dir_world * 0.045
            )[press_mask]
            gripper_open[press_mask] = False
        if torch.any(push_mask):
            target_pos[push_mask] = (
                env.grasp_goal_world
                + env.door_open_dir_world * 0.05
                + env.handle_rotate_dir_world * 0.025
            )[push_mask]
            gripper_open[push_mask] = False
        if torch.any(hold_mask):
            target_pos[hold_mask] = (
                env.grasp_goal_world
                + env.door_open_dir_world * 0.035
                + env.handle_rotate_dir_world * 0.015
            )[hold_mask]
            gripper_open[hold_mask] = False

        target_offset = target_pos - env.grasp_goal_world
        target_debug = {
            "target_dist": round(torch.norm(target_offset, dim=1).mean().item(), 4),
            "target_approach": round(torch.sum(target_offset * env.handle_approach_dir_world, dim=1).mean().item(), 4),
            "target_rotate": round(torch.sum(target_offset * env.handle_rotate_dir_world, dim=1).mean().item(), 4),
            "target_open": round(torch.sum(target_offset * env.door_open_dir_world, dim=1).mean().item(), 4),
            "ee_target_dist": round(torch.norm(target_pos - env.ee_pos, dim=1).mean().item(), 4),
        }

        actions = env.scripted_actions_from_world_targets(target_pos, target_rot, gripper_open, base_x_cmd, yaw_cmd)
        env.step(actions)
        if args.pin_base:
            _pin_robot_base(env, pinned_root_states)
        assisted_count = 0
        if args.joint_assist:
            assisted_count = _scripted_joint_assist(env, phase)

        walk_done = (phase == WALK_TO_DOOR) & (
            (env.base_door_dis < float(args.walk_stop_distance))
            | (phase_steps > int(args.walk_timeout_steps))
        )
        pregrasp_done = (phase == MOVE_TO_PREGRASP) & (env.curr_dist < 0.20)
        handle_done = (phase == MOVE_TO_HANDLE) & ((env.curr_dist < 0.10) | (phase_steps > 60))
        close_done = (phase == CLOSE_AND_SEAT) & (phase_steps > 18)
        press_done = (phase == PRESS_HANDLE) & (
            env.open_door_stage
            | (env.handle_open_ratio > float(env.cfg.door.handle_press_threshold_ratio) * 0.92)
            | (phase_steps > 60)
        )
        push_done = (phase == PUSH_DOOR) & (env.door_open_success | (phase_steps > 160))

        _set_phase(phase, phase_steps, walk_done, MOVE_TO_PREGRASP)
        _set_phase(phase, phase_steps, pregrasp_done, MOVE_TO_HANDLE)
        _set_phase(phase, phase_steps, handle_done, CLOSE_AND_SEAT)
        _set_phase(phase, phase_steps, close_done, PRESS_HANDLE)
        _set_phase(phase, phase_steps, press_done, PUSH_DOOR)
        _set_phase(phase, phase_steps, push_done, HOLD_OPEN)

        if step % 25 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "dist": round(env.curr_dist.mean().item(), 4),
                    "base_door": round(env.base_door_dis.mean().item(), 4),
                    "cmd_vx": round(env.commands[:, 0].mean().item(), 4),
                    "base_vx": round(env.base_lin_vel[:, 0].mean().item(), 4),
                    "door_dof": round(env._door_dof_pos[:, 0].mean().item(), 4),
                    "handle_dof": round(env._door_dof_pos[:, 1].mean().item(), 4),
                    "handle_ratio": round(env.handle_open_ratio.mean().item(), 4),
                    "door_ratio": round(env.door_open_ratio.mean().item(), 4),
                    "success": int(env.door_open_success.sum().item()),
                    "assist_envs": assisted_count,
                    "phase": _phase_counts(phase),
                    **(target_debug if args.debug_targets else {}),
                },
            )

        if bool(torch.all(env.door_open_success)):
            print("All environments reached the door-open success threshold.")
            break


if __name__ == "__main__":
    main()
