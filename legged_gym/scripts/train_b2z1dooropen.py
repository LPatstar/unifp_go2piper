import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


DOOR_OPEN_CUSTOM_PARAMETERS = [
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


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    if args.low_level_policy_mode is not None:
        env_cfg.low_level.policy_mode = args.low_level_policy_mode
    if args.low_level_load_run is not None:
        env_cfg.low_level.load_run = -1 if args.low_level_load_run == "-1" else args.low_level_load_run
    if args.low_level_checkpoint is not None:
        env_cfg.low_level.checkpoint = args.low_level_checkpoint

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    args = get_args(custom_parameters=DOOR_OPEN_CUSTOM_PARAMETERS)
    if "--task" not in sys.argv:
        args.task = "b2z1_door_open"
    args.headless = True
    train(args)

