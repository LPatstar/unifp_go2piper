import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CUSTOM_PARAMETERS = [
    {"name": "--steps", "type": int, "default": 2000, "help": "Number of high-level policy steps to play."},
    {"name": "--report_interval", "type": int, "default": 25, "help": "Door metric print interval."},
    {"name": "--stop_on_all_success", "action": "store_true", "default": False, "help": "Stop when every env reaches door success once."},
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
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
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
    return env_cfg, train_cfg


def _metric_snapshot(env):
    return {
        "dist": round(env.curr_dist.mean().item(), 4),
        "closest": round(torch.where(env.closest_dist >= 0.0, env.closest_dist, env.curr_dist).mean().item(), 4),
        "door_ratio": round(env.door_open_ratio.mean().item(), 4),
        "handle_ratio": round(env.handle_open_ratio.mean().item(), 4),
        "open_stage": round(env.open_door_stage.to(dtype=torch.float).mean().item(), 4),
        "success": int(env.door_open_success.sum().item()),
        "base_door_dis": round(env.base_door_dis.mean().item(), 4),
    }


def play(args):
    env_cfg, train_cfg = _configure_env(args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if hasattr(env, "low_level_policy_path") and env.low_level_policy_path is not None:
        print(f"Frozen low-level policy: {env.low_level_policy_path}")
    print("Initial door metrics:", _metric_snapshot(env))

    success_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    policy_info = {}
    report_interval = max(1, int(args.report_interval))
    for step in range(int(args.steps)):
        actions = policy(obs, policy_info)
        obs, _, _, _ = env.step(actions.detach())
        success_seen |= env.door_open_success

        if step % report_interval == 0:
            print(f"[step {step:04d}]", _metric_snapshot(env))
        if args.stop_on_all_success and bool(torch.all(success_seen)):
            print(f"All environments reached door success by step {step}.")
            break

    print("Final door metrics:", _metric_snapshot(env))


if __name__ == "__main__":
    args = get_args(custom_parameters=CUSTOM_PARAMETERS)
    if "--task" not in sys.argv:
        args.task = "b2z1_door_open"
    play(args)
