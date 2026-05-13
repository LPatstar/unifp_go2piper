import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CUSTOM_PARAMETERS = [
    {"name": "--steps", "type": int, "default": 240, "help": "Number of zero-action high-level steps."},
    {
        "name": "--low_level_policy_mode",
        "type": str,
        "default": "zero",
        "help": "Frozen low-level mode for asset smoke tests. Defaults to zero.",
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


def main():
    args = get_args(custom_parameters=CUSTOM_PARAMETERS)
    if "--task" not in sys.argv:
        args.task = "b2z1_door_open"

    env_cfg = _configure_env(args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    print("Loaded door assets:", env.door_asset_names)
    print("Door actor count:", int(env.door_actor_ids.numel()))
    print("Door DOF count:", env.num_door_dofs)
    print("Door body / handle body:", env.door_body_name, env.handle_body_name)
    print("Rigid body indices:", {"door_body_idx": env.door_body_idx, "handle_body_idx": env.handle_body_idx})
    print(
        "Tensor shapes:",
        {
            "root_states": tuple(env.root_states.shape),
            "door_root_states": tuple(env.door_root_states.shape),
            "dof_pos": tuple(env.dof_pos.shape),
            "door_dof_pos": tuple(env._door_dof_pos.shape),
            "rigid_state": tuple(env.rigid_state.shape),
            "full_rigid_state": tuple(env._full_rigid_state.shape),
            "grasp_goal_world": tuple(env.grasp_goal_world.shape),
            "obs": tuple(env.obs_buf.shape),
        },
    )

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(int(args.steps)):
        env.step(zero_actions)
        if step % 60 == 0:
            print(
                f"[step {step:04d}]",
                {
                    "base_door_dis": env.base_door_dis[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "door_dof": env._door_dof_pos[: min(4, env.num_envs), 0].detach().cpu().tolist(),
                    "handle_dof": env._door_dof_pos[: min(4, env.num_envs), 1].detach().cpu().tolist(),
                    "door_open_ratio": env.door_open_ratio[: min(4, env.num_envs)].detach().cpu().tolist(),
                    "handle_open_ratio": env.handle_open_ratio[: min(4, env.num_envs)].detach().cpu().tolist(),
                },
            )


if __name__ == "__main__":
    main()
