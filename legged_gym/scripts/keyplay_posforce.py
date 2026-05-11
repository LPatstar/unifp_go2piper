import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym

from legged_gym.envs import *
import legged_gym.scripts.play_b2z1posforce as play_impl
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import print_env_control_gains


PRINT_EVERY_STEPS = 30


def print_key_help(draw_mode=False, torque_recording=False):
    print("Keyplay controls for position-force task:")
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
    if draw_mode:
        print("  X : save draw plots and exit")
    elif torque_recording:
        print("  X : save torque record and exit")
    else:
        print("  X : exit keyplay")
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
    draw_mode = bool(getattr(args, "draw", False))
    rec_torque = bool(getattr(args, "rec_torque", False))

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

    if draw_mode:
        draw_leg_joint_names = play_impl.resolve_draw_leg_joint_names(env)
        draw_arm_joint_names = play_impl.resolve_draw_arm_joint_names(env)
        leg_joint_indices = play_impl.collect_joint_indices(env, draw_leg_joint_names)
        arm_joint_indices = play_impl.collect_joint_indices(env, draw_arm_joint_names)
        draw_time_axis = []
        leg_cmd_series = {name: [] for name in draw_leg_joint_names}
        leg_act_series = {name: [] for name in draw_leg_joint_names}
        arm_cmd_series = {name: [] for name in draw_arm_joint_names}
        arm_act_series = {name: [] for name in draw_arm_joint_names}

    torque_record = play_impl.init_torque_record(env) if rec_torque else None

    print_key_help(draw_mode=draw_mode, torque_recording=rec_torque)
    print(format_status(env))

    policy_info = {}
    for i in range(100 * int(env.max_episode_length)):
        actions = policy(obs, policy_info)
        obs, rews, dones, infos = env.step(actions.detach())
        if torque_record is not None:
            play_impl.append_torque_record(torque_record, env, i)

        if draw_mode:
            command_targets = play_impl.compute_command_targets(env)
            actual_dof_pos = env.dof_pos[0].detach().cpu().numpy()
            draw_time_axis.append((i + 1) * env.dt)
            for joint_name, joint_idx in zip(draw_leg_joint_names, leg_joint_indices):
                leg_cmd_series[joint_name].append(float(command_targets[joint_idx]))
                leg_act_series[joint_name].append(float(actual_dof_pos[joint_idx]))
            for joint_name, joint_idx in zip(draw_arm_joint_names, arm_joint_indices):
                arm_cmd_series[joint_name].append(float(command_targets[joint_idx]))
                arm_act_series[joint_name].append(float(actual_dof_pos[joint_idx]))

        if i % PRINT_EVERY_STEPS == 0:
            print(format_status(env))

        if env.key_exit_requested:
            break

    if draw_mode:
        model_metadata = play_impl.resolve_model_metadata(train_cfg)
        output_dir = play_impl.build_draw_output_dir(args, model_metadata)
        leg_output_path = os.path.join(output_dir, "leg_joint_tracking.png")
        arm_output_path = os.path.join(output_dir, "arm_joint_tracking.png")
        play_impl.save_joint_tracking_plot(
            draw_time_axis,
            draw_leg_joint_names,
            leg_cmd_series,
            leg_act_series,
            "Front-left Leg Joint Command vs Actual (absolute rad)",
            leg_output_path,
        )
        play_impl.save_joint_tracking_plot(
            draw_time_axis,
            draw_arm_joint_names,
            arm_cmd_series,
            arm_act_series,
            "Arm Joint Command vs Actual (absolute rad)",
            arm_output_path,
        )
        print(f"Saved leg tracking plot to: {leg_output_path}")
        print(f"Saved arm tracking plot to: {arm_output_path}")
    if torque_record is not None:
        model_metadata = play_impl.resolve_model_metadata(train_cfg)
        play_impl.save_torque_record(torque_record, args, model_metadata)


if __name__ == "__main__":
    args = get_args(custom_parameters=play_impl.DRAW_JOINT_CUSTOM_PARAMETERS)
    if not getattr(args, "task", None):
        args.task = "b2z1_pos_force"
    keyplay(args)
