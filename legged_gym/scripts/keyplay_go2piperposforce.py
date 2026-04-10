import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import print_env_control_gains


PRINT_EVERY_STEPS = 30


def print_key_help():
    print("Keyplay controls for go2_piper:")
    print("  W/S : forward velocity +/-")
    print("  A/D : lateral velocity +/-")
    print("  Q/E : yaw rate +/- (Q is CCW)")
    print("  Numpad 8/2 : EE target x +/-")
    print("  Numpad 4/6 : EE target y +/-")
    print("  Numpad 9/3 : EE target z +/-")
    print("  Numpad 0   : set EE target to current EE pose")
    print("  J/K : EE force command x -/+")
    print("  O/I : base force command x -/+")
    print("  R   : reset base motion commands to zero")
    print("  Numpad 5   : reset EE target to home")
    print("  N   : reset force commands to zero")
    print("  F : toggle follow camera")
    print("  V : toggle viewer sync")
    print("  SPACE : pause")


def format_status(env):
    ee_target_local = env.key_command_ee_local_cart[0].detach().cpu().numpy()
    return (
        f"base_cmd: vx={env.commands[0, 0].item():+.2f}, "
        f"vy={env.commands[0, 1].item():+.2f}, "
        f"yaw={env.commands[0, 2].item():+.2f} | "
        f"ee_target_local: x={ee_target_local[0]:+.3f}, "
        f"y={ee_target_local[1]:+.3f}, "
        f"z={ee_target_local[2]:+.3f} | "
        f"ee_force_x={env.current_Fxyz_gripper_cmd[0, 0].item():+.1f}N | "
        f"base_force_x={env.current_Fxyz_base_cmd[0, 0].item():+.1f}N"
    )


def keyplay(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.teleop_mode = True
    env_cfg.env.key_command_mode = True
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

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print_env_control_gains(env)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.play = True
    env.enable_gripper_cmd_force = False
    env.enable_play_immediate_gripper_cmd_force = False
    env.enable_random_force_events = False
    env._update_key_command_ee_goal()

    print_key_help()
    print(format_status(env))

    policy_info = {}
    for i in range(100 * int(env.max_episode_length)):
        actions = policy(obs, policy_info)
        obs, rews, dones, infos = env.step(actions.detach())

        if i % PRINT_EVERY_STEPS == 0:
            print(format_status(env))


if __name__ == "__main__":
    args = get_args()
    if not getattr(args, "task", None):
        args.task = "go2_piper_pos_force"
    keyplay(args)
