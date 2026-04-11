import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import sys
sys.path.append(unitree_rl_gym_path)
from legged_gym import LEGGED_GYM_ROOT_DIR
from datetime import datetime
import re

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path, print_env_control_gains

import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

ENABLE_PLAY_CMD_FORCE = False
DRAW_JOINT_CUSTOM_PARAMETERS = [
    {"name": "--draw", "action": "store_true", "default": False, "help": "Record joint command/actual trajectories for a short play window and save plots."},
    {"name": "--draw_steps", "type": int, "default": 1000, "help": "Number of play steps to record before saving draw plots. Default: 1000."},
    {"name": "--draw_dir", "type": str, "default": "play_draws", "help": "Directory for saved play draw plots. Default: play_draws/."},
    {"name": "--rec_torque", "action": "store_true", "default": False, "help": "Record Piper joint torques during play/keyplay and save them on exit."},
    {"name": "--torque_dir", "type": str, "default": "play_torques", "help": "Directory for saved Piper torque records. Default: play_torques/."},
]
PREFERRED_DRAW_LEG_JOINT_GROUPS = [
    ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
]
PREFERRED_DRAW_ARM_JOINT_GROUPS = [
    [f"piper_joint{i}" for i in range(1, 6)],
    ["z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll"],
]


def get_play_args(task_default=None):
    args = get_args(custom_parameters=DRAW_JOINT_CUSTOM_PARAMETERS)
    if task_default and "--task" not in sys.argv:
        args.task = task_default
    return args


def safe_path_component(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def resolve_model_metadata(train_cfg):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    resolved_model_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    resolved_run_name = os.path.basename(os.path.dirname(resolved_model_path))
    resolved_model_file = os.path.basename(resolved_model_path)
    resolved_checkpoint = "latest"
    if resolved_model_file.startswith("model_") and resolved_model_file.endswith(".pt"):
        resolved_checkpoint = resolved_model_file[len("model_"):-len(".pt")]
    return {
        "resolved_run_name": resolved_run_name,
        "resolved_checkpoint": resolved_checkpoint,
    }


def compute_command_targets(env, actions=None):
    full_targets = env.default_dof_pos[0].clone()
    if actions is None:
        actions = getattr(env, "control_actions", env.actions)
    scaled_actions = actions[0].detach() * env.motor_strength[0] * env.cfg.control.action_scale
    full_targets[:env.num_torques] = scaled_actions + env.default_dof_pos_wo_gripper[0]
    if hasattr(env, "_get_pd_equivalent_gripper_targets"):
        full_targets[env.num_torques:] = env._get_pd_equivalent_gripper_targets()[0].detach()
    return full_targets.detach().cpu().numpy()


def collect_joint_indices(env, joint_names):
    return [env.dof_names.index(name) for name in joint_names]


def pick_first_available_joint_group(dof_names, joint_groups):
    for joint_group in joint_groups:
        if all(joint_name in dof_names for joint_name in joint_group):
            return list(joint_group)
    return None


def resolve_draw_leg_joint_names(env):
    resolved_joint_names = pick_first_available_joint_group(env.dof_names, PREFERRED_DRAW_LEG_JOINT_GROUPS)
    if resolved_joint_names is not None:
        return resolved_joint_names

    fl_joint_names = [joint_name for joint_name in env.dof_names if joint_name.startswith("FL_")]
    if len(fl_joint_names) >= 3:
        return fl_joint_names[:3]

    leg_dof_count = min(len(env.dof_names), int(getattr(env.cfg.env, "num_leg_dofs", 0)))
    if leg_dof_count >= 3:
        return list(env.dof_names[:3])

    raise ValueError(f"Unable to resolve draw leg joints from dof_names: {env.dof_names}")


def resolve_draw_arm_joint_names(env):
    resolved_joint_names = pick_first_available_joint_group(env.dof_names, PREFERRED_DRAW_ARM_JOINT_GROUPS)
    if resolved_joint_names is not None:
        return resolved_joint_names

    leg_dof_count = int(getattr(env.cfg.env, "num_leg_dofs", 0))
    policy_arm_dof_count = max(0, int(getattr(env, "num_torques", 0)) - leg_dof_count)
    fallback_count = min(5, policy_arm_dof_count)
    if fallback_count > 0:
        fallback_joint_names = env.dof_names[leg_dof_count:leg_dof_count + fallback_count]
        if len(fallback_joint_names) == fallback_count:
            return list(fallback_joint_names)

    raise ValueError(f"Unable to resolve draw arm joints from dof_names: {env.dof_names}")


def build_draw_output_dir(args, model_metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        args.draw_dir,
        f"{args.task}_{safe_path_component(model_metadata['resolved_run_name'])}_ckpt{safe_path_component(model_metadata['resolved_checkpoint'])}_{timestamp}",
    )
    os.makedirs(root, exist_ok=True)
    return root


def build_torque_output_dir(args, model_metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        args.torque_dir,
        f"{args.task}_{safe_path_component(model_metadata['resolved_run_name'])}_ckpt{safe_path_component(model_metadata['resolved_checkpoint'])}_{timestamp}",
    )
    os.makedirs(root, exist_ok=True)
    return root


def resolve_piper_torque_joint_names(env):
    piper_joint_names = [joint_name for joint_name in env.dof_names if joint_name.startswith("piper_joint")]
    if piper_joint_names:
        return piper_joint_names
    arm_start = int(getattr(env, "num_leg_dofs", 0))
    return list(env.dof_names[arm_start:])


def init_torque_record(env):
    joint_names = resolve_piper_torque_joint_names(env)
    joint_indices = collect_joint_indices(env, joint_names)
    return {
        "joint_names": joint_names,
        "joint_indices": joint_indices,
        "time_s": [],
        "torque": [],
    }


def append_torque_record(record, env, step_idx):
    torques = env.torques[0, record["joint_indices"]].detach().cpu().numpy()
    record["time_s"].append(float((step_idx + 1) * env.dt))
    record["torque"].append(torques.astype(np.float32, copy=True))


def save_torque_record(record, args, model_metadata):
    output_dir = build_torque_output_dir(args, model_metadata)
    torque_array = np.asarray(record["torque"], dtype=np.float32)
    time_array = np.asarray(record["time_s"], dtype=np.float32)
    joint_names = np.asarray(record["joint_names"], dtype=str)

    npz_path = os.path.join(output_dir, "piper_joint_torques.npz")
    csv_path = os.path.join(output_dir, "piper_joint_torques.csv")
    np.savez(npz_path, time_s=time_array, torque=torque_array, joint_names=joint_names)

    header = "time_s," + ",".join(record["joint_names"])
    if torque_array.size:
        csv_data = np.column_stack([time_array, torque_array])
        np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="")
    else:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")

    print(f"Saved Piper torque npz to: {npz_path}")
    print(f"Saved Piper torque csv to: {csv_path}")
    return output_dir


def save_joint_tracking_plot(time_axis, joint_names, cmd_series, act_series, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, joint_name in enumerate(joint_names):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(time_axis, cmd_series[joint_name], linestyle="--", color=color, linewidth=1.5, label=f"{joint_name}-cmd")
        ax.plot(time_axis, act_series[joint_name], linestyle="-", color=color, linewidth=1.8, label=f"{joint_name}-act")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint Angle [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print_env_control_gains(env)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    model_metadata = resolve_model_metadata(train_cfg)
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:

        import copy
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        os.makedirs(path, exist_ok=True)
        adaptation_module_path = os.path.join(path, 'adaptation_module.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_encoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_module_path)
        print('Exported policy as jit script to: ', adaptation_module_path)

        adaptation_decoder_path = os.path.join(path, 'adaptation_decoder.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_decoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_decoder_path)
        print('Exported policy as jit script to: ', adaptation_decoder_path)

        actor_body_path = os.path.join(path, 'actor_body.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor_body).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(actor_body_path)
        print('Exported policy as jit script to: ', actor_body_path)

    draw_mode = bool(getattr(args, "draw", False))
    visual_pred_enabled = VISUAL_PRED and not draw_mode

    if visual_pred_enabled:
        fig_ee_force = plt.figure()
        ax_ee_force = fig_ee_force.add_subplot(111, projection='3d')
        vector1_ee_force = np.array([0, 0, 0])
        vector2_ee_force = np.array([0, 0, 0])
        ax_ee_force.set_xlim([-0.7, 0.7])
        ax_ee_force.set_ylim([-0.7, 0.7])
        ax_ee_force.set_zlim([-0.7, 0.7])
        ax_ee_force.set_xlabel('X axis')
        ax_ee_force.set_ylabel('Y axis')
        ax_ee_force.set_zlabel('Z axis')
        line1_ee_force, = ax_ee_force.plot([0, vector1_ee_force[0]], [0, vector1_ee_force[1]], [0, vector1_ee_force[2]], marker='o', label='ee_force_pred')
        line2_ee_force, = ax_ee_force.plot([0, vector2_ee_force[0]], [0, vector2_ee_force[1]], [0, vector2_ee_force[2]], marker='o', label='ee_force_gt')
        ax_ee_force.legend()

        fig_base_force = plt.figure()
        ax_base_force = fig_base_force.add_subplot(111, projection='3d')
        vector1_base_force = np.array([0, 0, 0])
        vector2_base_force = np.array([0, 0, 0])
        ax_base_force.set_xlim([-0.7, 0.7])
        ax_base_force.set_ylim([-0.7, 0.7])
        ax_base_force.set_zlim([-0.7, 0.7])
        ax_base_force.set_xlabel('X axis')
        ax_base_force.set_ylabel('Y axis')
        ax_base_force.set_zlabel('Z axis')
        line1_base_force, = ax_base_force.plot([0, vector1_base_force[0]], [0, vector1_base_force[1]], [0, vector1_base_force[2]], marker='o', label='base_force_pred')
        line2_base_force, = ax_base_force.plot([0, vector2_base_force[0]], [0, vector2_base_force[1]], [0, vector2_base_force[2]], marker='o', label='base_force_gt')
        ax_base_force.legend()

        fig_linvel = plt.figure()
        ax_linvel = fig_linvel.add_subplot(111, projection='3d')
        vector1_linvel = np.array([0, 0, 0])
        vector2_linvel = np.array([0, 0, 0])
        ax_linvel.set_xlim([-2, 2])
        ax_linvel.set_ylim([-2, 2])
        ax_linvel.set_zlim([-2, 2])
        ax_linvel.set_xlabel('X axis')
        ax_linvel.set_ylabel('Y axis')
        ax_linvel.set_zlabel('Z axis')
        line1_linvel, = ax_linvel.plot([0, vector1_linvel[0]], [0, vector1_linvel[1]], [0, vector1_linvel[2]], marker='o', label='linvel_pred')
        line2_linvel, = ax_linvel.plot([0, vector2_linvel[0]], [0, vector2_linvel[1]], [0, vector2_linvel[2]], marker='o', label='linvel_gt')
        ax_linvel.legend()

        fig_eepos = plt.figure()
        ax_eepos = fig_eepos.add_subplot(111, projection='3d')
        vector1_eepos = np.array([0, 0, 0])
        vector2_eepos = np.array([0, 0, 0])
        ax_eepos.set_xlim([-2, 2])
        ax_eepos.set_ylim([-2, 2])
        ax_eepos.set_zlim([-2, 2])
        ax_eepos.set_xlabel('X axis')
        ax_eepos.set_ylabel('Y axis')
        ax_eepos.set_zlabel('Z axis')
        line1_eepos, = ax_eepos.plot([0, vector1_eepos[0]], [0, vector1_eepos[1]], [0, vector1_eepos[2]], marker='o', label='eepos_pred')
        line2_eepos, = ax_eepos.plot([0, vector2_eepos[0]], [0, vector2_eepos[1]], [0, vector2_eepos[2]], marker='o', label='eepos_gt')
        ax_eepos.legend()
    

    env.play = True
    env.enable_gripper_cmd_force = ENABLE_PLAY_CMD_FORCE
    env.enable_play_immediate_gripper_cmd_force = ENABLE_PLAY_CMD_FORCE
    if env.viewer:
        print("Viewer controls: press F to toggle follow camera on the current robot; press X to save and exit.")
    if draw_mode:
        draw_steps = max(1, int(getattr(args, "draw_steps", 1000)))
        draw_leg_joint_names = resolve_draw_leg_joint_names(env)
        draw_arm_joint_names = resolve_draw_arm_joint_names(env)
        leg_joint_indices = collect_joint_indices(env, draw_leg_joint_names)
        arm_joint_indices = collect_joint_indices(env, draw_arm_joint_names)
        draw_time_axis = []
        leg_cmd_series = {name: [] for name in draw_leg_joint_names}
        leg_act_series = {name: [] for name in draw_leg_joint_names}
        arm_cmd_series = {name: [] for name in draw_arm_joint_names}
        arm_act_series = {name: [] for name in draw_arm_joint_names}
        total_steps = draw_steps
    else:
        total_steps = 100 * int(env.max_episode_length)

    torque_record = init_torque_record(env) if bool(getattr(args, "rec_torque", False)) else None

    policy_info = {}
    for i in range(total_steps):
        actions = policy(obs, policy_info)
        # breakpoint()
        if FIX_COMMAND:
            env.commands[:, 0] = 0.    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.
            # env.gait_indices[:] = 0.
        obs, rews, dones, infos = env.step(actions.detach())
        if torque_record is not None:
            append_torque_record(torque_record, env, i)
        if draw_mode:
            command_targets = compute_command_targets(env)
            actual_dof_pos = env.dof_pos[0].detach().cpu().numpy()
            draw_time_axis.append((i + 1) * env.dt)
            for joint_name, joint_idx in zip(draw_leg_joint_names, leg_joint_indices):
                leg_cmd_series[joint_name].append(float(command_targets[joint_idx]))
                leg_act_series[joint_name].append(float(actual_dof_pos[joint_idx]))
            for joint_name, joint_idx in zip(draw_arm_joint_names, arm_joint_indices):
                arm_cmd_series[joint_name].append(float(command_targets[joint_idx]))
                arm_act_series[joint_name].append(float(actual_dof_pos[joint_idx]))

        if visual_pred_enabled:
            ee_force_pred = policy_info["latents"][0, 6:9]
            vector1_ee_force = ee_force_pred
            vector2_ee_force = env.forces_local[0, env.gripper_idx].detach().cpu().numpy() * env.obs_scales.ee_force
            print("ee_force_pred:", ee_force_pred*100)
            print("ee_force_ext:", vector2_ee_force*100)
            line1_ee_force.set_data([0, vector1_ee_force[0]], [0, vector1_ee_force[1]])
            line1_ee_force.set_3d_properties([0, vector1_ee_force[2]])

            line2_ee_force.set_data([0, vector2_ee_force[0]], [0, vector2_ee_force[1]])
            line2_ee_force.set_3d_properties([0, vector2_ee_force[2]])

            base_force_pred = policy_info["latents"][0, 9:12]
            vector1_base_force = base_force_pred
            vector2_base_force = env.forces_local[0,env.robot_base_idx].detach().cpu().numpy() * env.obs_scales.base_force
            print("ee_base_pred:", base_force_pred*100)
            print("ee_base_ext:", vector2_base_force*100)
            # line1_base_force.set_data([0, vector1_base_force[0]], [0, vector1_base_force[1]])
            # line1_base_force.set_3d_properties([0, vector1_base_force[2]])

            # line2_base_force.set_data([0, vector2_base_force[0]], [0, vector2_base_force[1]])
            # line2_base_force.set_3d_properties([0, vector2_base_force[2]])
            

            # linvel_pred = policy_info["latents"][0, 0:3]
            # vector1_linvel = linvel_pred
            # vector2_linvel = env.base_lin_vel[0].detach().cpu().numpy() * env.obs_scales.lin_vel
            # line1_linvel.set_data([0, vector1_linvel[0]], [0, vector1_linvel[1]])
            # line1_linvel.set_3d_properties([0, vector1_linvel[2]])

            # line2_linvel.set_data([0, vector2_linvel[0]], [0, vector2_linvel[1]])
            # line2_linvel.set_3d_properties([0, vector2_linvel[2]])

            # eepos_pred = policy_info["latents"][0, 3:6]
            # vector1_eepos = eepos_pred
            
            # vector2_eepos = np.array([env.ee_pos_sphe_arm[0,0].detach().cpu().numpy() * env.obs_scales.ee_sphe_radius_cmd,
            #                                 env.ee_pos_sphe_arm[0,1].detach().cpu().numpy() * env.obs_scales.ee_sphe_pitch_cmd,
            #                                 env.ee_pos_sphe_arm[0,2].detach().cpu().numpy() * env.obs_scales.ee_sphe_yaw_cmd])
            # line1_eepos.set_data([0, vector1_eepos[0]], [0, vector1_eepos[1]])
            # line1_eepos.set_3d_properties([0, vector1_eepos[2]])
            # line2_eepos.set_data([0, vector2_eepos[0]], [0, vector2_eepos[1]])
            # line2_eepos.set_3d_properties([0, vector2_eepos[2]])
            # plt.draw()
            # plt.pause(0.001)

        if getattr(env, "key_exit_requested", False):
            break

    if draw_mode:
        output_dir = build_draw_output_dir(args, model_metadata)
        leg_output_path = os.path.join(output_dir, "leg_joint_tracking.png")
        arm_output_path = os.path.join(output_dir, "arm_joint_tracking.png")
        save_joint_tracking_plot(
            draw_time_axis,
            draw_leg_joint_names,
            leg_cmd_series,
            leg_act_series,
            "Front-left Leg Joint Command vs Actual (absolute rad)",
            leg_output_path,
        )
        save_joint_tracking_plot(
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
        save_torque_record(torque_record, args, model_metadata)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = False
    VISUAL_PRED = True
    args = get_play_args(task_default="b2z1_pos_force")
    play(args)
