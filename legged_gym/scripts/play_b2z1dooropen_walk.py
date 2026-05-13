import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CUSTOM_PARAMETERS = [
    {"name": "--steps", "type": int, "default": 240, "help": "Number of high-level walking smoke steps."},
    {"name": "--resample_interval", "type": int, "default": 60, "help": "Steps between new forward command samples."},
    {"name": "--speed_min", "type": float, "default": 0.05, "help": "Minimum forward velocity command in m/s."},
    {"name": "--speed_max", "type": float, "default": 0.25, "help": "Maximum forward velocity command in m/s."},
    {
        "name": "--low_level_policy_mode",
        "type": str,
        "default": None,
        "help": "Frozen low-level mode: checkpoint or zero. Default comes from b2z1_door_open config.",
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
]


def _configure_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.test = True
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    if args.low_level_policy_mode is not None:
        env_cfg.low_level.policy_mode = args.low_level_policy_mode
    if args.low_level_load_run is not None:
        env_cfg.low_level.load_run = -1 if args.low_level_load_run == "-1" else args.low_level_load_run
    if args.low_level_checkpoint is not None:
        env_cfg.low_level.checkpoint = args.low_level_checkpoint
    return env_cfg


def _sample_forward_commands(num_envs, device, speed_min, speed_max):
    return torch.empty(num_envs, device=device).uniform_(float(speed_min), float(speed_max))


def main():
    args = get_args(custom_parameters=CUSTOM_PARAMETERS)
    if "--task" not in sys.argv:
        args.task = "b2z1_door_open"

    env_cfg = _configure_env(args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    print("Loaded door assets:", env.door_asset_names)
    if hasattr(env, "low_level_policy_path") and env.low_level_policy_path is not None:
        print(f"Frozen low-level policy: {env.low_level_policy_path}")

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    base_forward_scale = max(float(getattr(env.cfg.high_level, "base_forward_scale", 0.35)), 1e-6)
    forward_cmd = _sample_forward_commands(env.num_envs, env.device, args.speed_min, args.speed_max)
    actions[:, 7] = torch.clamp(forward_cmd / base_forward_scale, -1.0, 1.0)

    resample_interval = max(1, int(args.resample_interval))
    for step in range(int(args.steps)):
        if step % resample_interval == 0:
            forward_cmd = _sample_forward_commands(env.num_envs, env.device, args.speed_min, args.speed_max)
            actions[:, 7] = torch.clamp(forward_cmd / base_forward_scale, -1.0, 1.0)

        env.step(actions)

        if step % 30 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "target_vx": forward_cmd[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "command_x": env.commands[: min(4, env.num_envs), 0].detach().cpu().tolist(),
                    "base_lin_vel_x": env.base_lin_vel[: min(4, env.num_envs), 0].detach().cpu().tolist(),
                    "base_height": env.base_pos[: min(4, env.num_envs), 2].detach().cpu().tolist(),
                    "base_door_dis": env.base_door_dis[: min(4, env.num_envs)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
