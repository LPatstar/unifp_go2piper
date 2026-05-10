import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import isaacgym  # noqa: F401
from isaacgym import gymutil
from isaacgym.torch_utils import quat_apply
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.b2.legged_robot_b2z1_pos_force import (
    INDEX_BASE_FORCE_X,
    INDEX_BASE_FORCE_Z,
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Z,
)
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path, print_env_control_gains, set_seed
from legged_gym.utils.isaacgym_utils import sphere2cart


TRACKING_WINDOW_S = 0.25
LAST_WINDOW_RMSE_METRICS = (
    "nominal_ee_error",
    "compensated_ee_error",
    "base_nominal_vel_error",
    "base_comp_vel_error",
    "yaw_rate_error",
    "roll_pitch_abs",
    "base_ang_vel_xy_norm",
)


@dataclass
class Phase:
    duration_s: float
    ee_target_local: np.ndarray
    base_cmd: np.ndarray
    ee_force_cmd_local: np.ndarray
    base_force_cmd_local: np.ndarray
    ee_ext_force_local: np.ndarray
    base_ext_force_local: np.ndarray
    collect: bool = True
    primary_collect: bool = True
    tag: str = "main"


@dataclass
class Scenario:
    name: str
    phases: List[Phase]
    primary_metric: str
    success_threshold: float


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_piper_pos_force", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Experiment name override."},
        {"name": "--run_name", "type": str, "help": "Run name override."},
        {"name": "--load_run", "type": str, "help": "Run name to load."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number to load."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Disable viewer."},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "help": "Number of eval envs."},
        {"name": "--seed", "type": int, "help": "Random seed override."},
        {"name": "--flat_terrain", "action": "store_true", "default": False, "help": "Use flat terrain."},
        {"name": "--max_iterations", "type": int, "help": "Unused training override."},
        {"name": "--observe_gait_commands", "action": "store_true", "help": "Unused gait command flag."},
        {
            "name": "--eval_case",
            "type": str,
            "default": "all",
            "help": "Case to run: all, position_only, hybrid_force_position, arm_force_estimation, base_force_estimation, base_disturbance, mixed_whole_body",
        },
        {"name": "--eval_repeats", "type": int, "default": 1, "help": "Repeat each scripted scenario this many times."},
        {"name": "--output_dir", "type": str, "default": "eval_reports", "help": "Directory for eval outputs."},
        {"name": "--no_report", "action": "store_true", "default": False, "help": "Run evaluation without exporting summary files."},
    ]
    args = gymutil.parse_arguments(description="Go2+Piper evaluation", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def error_to_score(error: float, good: float, bad: float) -> float:
    if math.isnan(error):
        return 0.0
    if error <= good:
        return 100.0
    if error >= bad:
        return 0.0
    return 100.0 * (bad - error) / (bad - good)


def ratio_to_score(ratio: float, good: float, bad: float) -> float:
    return error_to_score(ratio, good, bad)


def force_relative_accuracy(mae: float, target_norm: float) -> float:
    if mae is None or target_norm is None or math.isnan(mae) or math.isnan(target_norm) or target_norm <= 1e-6:
        return float("nan")
    return float(np.clip(100.0 * (1.0 - mae / target_norm), 0.0, 100.0))


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def rms_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def format_float(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def make_progress_bar(current: int, total: int, width: int = 36) -> str:
    if total <= 0:
        total = 1
    current = max(0, min(current, total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def show_progress(status_line: str, current: int, total: int, first_update: bool = False):
    percent = 100.0 * current / max(total, 1)
    progress_line = f"{make_progress_bar(current, total)} {current}/{total} ({percent:5.1f}%)"
    if sys.stdout.isatty():
        if first_update:
            sys.stdout.write(status_line + "\n")
            sys.stdout.write(progress_line + "\n")
        else:
            sys.stdout.write("\x1b[2F")
            sys.stdout.write("\r\x1b[2K" + status_line + "\n")
            sys.stdout.write("\r\x1b[2K" + progress_line + "\n")
        sys.stdout.flush()
    else:
        print(status_line)
        print(progress_line)


def build_repeat_seeds(case_names, eval_repeats):
    if eval_repeats <= 1:
        return {}
    num_seeds = len(case_names) * eval_repeats
    seed_values = np.random.SeedSequence().generate_state(num_seeds, dtype=np.uint32)
    seed_values = [int(seed % np.uint32(2**31 - 1)) for seed in seed_values]
    seeds = {}
    cursor = 0
    for case_name in case_names:
        seeds[case_name] = seed_values[cursor:cursor + eval_repeats]
        cursor += eval_repeats
    return seeds


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def phase(
    duration_s: float,
    ee_target_local,
    base_cmd=(0.0, 0.0, 0.0),
    ee_force_cmd_local=(0.0, 0.0, 0.0),
    base_force_cmd_local=(0.0, 0.0, 0.0),
    ee_ext_force_local=(0.0, 0.0, 0.0),
    base_ext_force_local=(0.0, 0.0, 0.0),
    collect=True,
    primary_collect=True,
    tag="main",
):
    return Phase(
        duration_s=duration_s,
        ee_target_local=np.asarray(ee_target_local, dtype=np.float32),
        base_cmd=np.asarray(base_cmd, dtype=np.float32),
        ee_force_cmd_local=np.asarray(ee_force_cmd_local, dtype=np.float32),
        base_force_cmd_local=np.asarray(base_force_cmd_local, dtype=np.float32),
        ee_ext_force_local=np.asarray(ee_ext_force_local, dtype=np.float32),
        base_ext_force_local=np.asarray(base_ext_force_local, dtype=np.float32),
        collect=collect,
        primary_collect=primary_collect,
        tag=tag,
    )


def build_scenarios(home_local: np.ndarray) -> Dict[str, List[Scenario]]:
    forward_high = home_local + np.array([0.10, 0.00, 0.08], dtype=np.float32)
    forward_low = home_local + np.array([0.08, 0.00, -0.06], dtype=np.float32)
    left_reach = home_local + np.array([0.05, 0.10, 0.02], dtype=np.float32)
    right_reach = home_local + np.array([0.05, -0.10, 0.02], dtype=np.float32)

    hybrid_front = home_local + np.array([0.08, 0.00, 0.03], dtype=np.float32)
    hybrid_left = home_local + np.array([0.05, 0.08, 0.03], dtype=np.float32)
    hybrid_right = home_local + np.array([0.05, -0.08, 0.00], dtype=np.float32)

    scenarios = {
        "position_only": [
            Scenario(
                name="pos_forward_high",
                primary_metric="nominal_ee_error",
                success_threshold=0.05,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(2.0, forward_high, tag="track"),
                ],
            ),
            Scenario(
                name="pos_forward_low",
                primary_metric="nominal_ee_error",
                success_threshold=0.05,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(2.0, forward_low, tag="track"),
                ],
            ),
            Scenario(
                name="pos_left_reach",
                primary_metric="nominal_ee_error",
                success_threshold=0.05,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(2.0, left_reach, tag="track"),
                ],
            ),
            Scenario(
                name="pos_right_reach",
                primary_metric="nominal_ee_error",
                success_threshold=0.05,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(2.0, right_reach, tag="track"),
                ],
            ),
        ],
        "hybrid_force_position": [
            Scenario(
                name="hybrid_front_x_force",
                primary_metric="compensated_ee_error",
                success_threshold=0.06,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.8, hybrid_front, primary_collect=False, tag="pre_force"),
                    phase(
                        1.4,
                        hybrid_front,
                        ee_force_cmd_local=(10.0, 0.0, 0.0),
                        ee_ext_force_local=(10.0, 0.0, 0.0),
                        tag="force_track",
                    ),
                ],
            ),
            Scenario(
                name="hybrid_left_z_force",
                primary_metric="compensated_ee_error",
                success_threshold=0.06,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.8, hybrid_left, primary_collect=False, tag="pre_force"),
                    phase(
                        1.4,
                        hybrid_left,
                        ee_force_cmd_local=(0.0, 0.0, -8.0),
                        ee_ext_force_local=(0.0, 0.0, -8.0),
                        tag="force_track",
                    ),
                ],
            ),
            Scenario(
                name="hybrid_right_xy_force",
                primary_metric="compensated_ee_error",
                success_threshold=0.06,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.8, hybrid_right, primary_collect=False, tag="pre_force"),
                    phase(
                        1.4,
                        hybrid_right,
                        ee_force_cmd_local=(6.0, -6.0, 0.0),
                        ee_ext_force_local=(6.0, -6.0, 0.0),
                        tag="force_track",
                    ),
                ],
            ),
        ],
        "arm_force_estimation": [
            Scenario(
                name="arm_force_x_probe",
                primary_metric="estimator_ee_force_mae",
                success_threshold=3.0,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        ee_ext_force_local=(10.0, 0.0, 0.0),
                        tag="force_probe",
                    ),
                ],
            ),
            Scenario(
                name="arm_force_z_probe",
                primary_metric="estimator_ee_force_mae",
                success_threshold=3.0,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        ee_ext_force_local=(0.0, 0.0, -8.0),
                        tag="force_probe",
                    ),
                ],
            ),
            Scenario(
                name="arm_force_xy_probe",
                primary_metric="estimator_ee_force_mae",
                success_threshold=3.0,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        ee_ext_force_local=(6.0, -6.0, 0.0),
                        tag="force_probe",
                    ),
                ],
            ),
        ],
        "base_force_estimation": [
            Scenario(
                name="base_force_x_probe",
                primary_metric="estimator_base_force_mae",
                success_threshold=2.5,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        base_ext_force_local=(6.0, 0.0, 0.0),
                        tag="force_probe",
                    ),
                ],
            ),
            Scenario(
                name="base_force_y_probe",
                primary_metric="estimator_base_force_mae",
                success_threshold=2.5,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        base_ext_force_local=(0.0, 6.0, 0.0),
                        tag="force_probe",
                    ),
                ],
            ),
            Scenario(
                name="base_force_xy_probe",
                primary_metric="estimator_base_force_mae",
                success_threshold=2.5,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.5, home_local, primary_collect=False, tag="zero_probe"),
                    phase(
                        1.2,
                        home_local,
                        base_ext_force_local=(-5.0, 4.0, 0.0),
                        tag="force_probe",
                    ),
                ],
            ),
        ],
        "base_disturbance": [
            Scenario(
                name="base_forward_x_disturbance",
                primary_metric="base_comp_vel_error",
                success_threshold=0.08,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.8, home_local, base_cmd=(0.25, 0.0, 0.0), primary_collect=False, tag="nominal"),
                    phase(
                        1.6,
                        home_local,
                        base_cmd=(0.25, 0.0, 0.0),
                        base_force_cmd_local=(4.0, 0.0, 0.0),
                        base_ext_force_local=(4.0, 0.0, 0.0),
                        tag="disturbance",
                    ),
                ],
            ),
            Scenario(
                name="base_lateral_y_disturbance",
                primary_metric="base_comp_vel_error",
                success_threshold=0.08,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(0.8, home_local, base_cmd=(0.0, 0.18, 0.0), primary_collect=False, tag="nominal"),
                    phase(
                        1.6,
                        home_local,
                        base_cmd=(0.0, 0.18, 0.0),
                        base_force_cmd_local=(0.0, 4.0, 0.0),
                        base_ext_force_local=(0.0, 4.0, 0.0),
                        tag="disturbance",
                    ),
                ],
            ),
            Scenario(
                name="base_yaw_tracking",
                primary_metric="yaw_rate_error",
                success_threshold=0.10,
                phases=[
                    phase(0.6, home_local, collect=False, tag="warmup"),
                    phase(1.8, home_local, base_cmd=(0.0, 0.0, 0.25), tag="yaw_track"),
                ],
            ),
        ],
        "mixed_whole_body": [
            Scenario(
                name="mixed_whole_body_sequence",
                primary_metric="compensated_ee_error",
                success_threshold=0.08,
                phases=[
                    phase(0.8, home_local, collect=False, tag="warmup"),
                    phase(1.0, left_reach, base_cmd=(0.20, 0.00, 0.0), tag="reach_move"),
                    phase(
                        1.2,
                        forward_high,
                        base_cmd=(0.18, 0.08, 0.0),
                        ee_force_cmd_local=(8.0, 0.0, 0.0),
                        ee_ext_force_local=(8.0, 0.0, 0.0),
                        tag="reach_force",
                    ),
                    phase(
                        1.2,
                        right_reach,
                        base_cmd=(0.10, -0.12, 0.15),
                        base_force_cmd_local=(3.0, 0.0, 0.0),
                        base_ext_force_local=(3.0, 0.0, 0.0),
                        tag="move_disturbance",
                    ),
                ],
            ),
        ],
    }
    return scenarios


def reset_env_for_eval(env):
    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    env.enable_gripper_cmd_force = False
    env.enable_play_immediate_gripper_cmd_force = False
    env.enable_random_force_events = False
    env.play = True
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)
    obs, _, _, _ = env.step(zero_actions)
    return obs


def sync_force_commands(env):
    env.commands[:, INDEX_EE_FORCE_X:(INDEX_EE_FORCE_Z + 1)] = env.current_Fxyz_gripper_cmd
    env.commands[:, INDEX_BASE_FORCE_X:(INDEX_BASE_FORCE_Z + 1)] = env.current_Fxyz_base_cmd


def apply_profile(env, profile: Phase):
    num_envs = env.num_envs
    device = env.device
    env_ids = torch.arange(num_envs, device=device)

    ee_target = torch.tensor(profile.ee_target_local, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    env._set_key_command_ee_goal_local_cart(env_ids, ee_target)
    env._update_key_command_ee_goal()

    base_cmd = torch.tensor(profile.base_cmd, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    env.commands[:, 0:3] = base_cmd

    ee_force_cmd = torch.tensor(profile.ee_force_cmd_local, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    base_force_cmd = torch.tensor(profile.base_force_cmd_local, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    env.current_Fxyz_gripper_cmd[:, :] = ee_force_cmd
    env.current_Fxyz_base_cmd[:, :] = base_force_cmd
    sync_force_commands(env)

    ee_ext_force_local = torch.tensor(profile.ee_ext_force_local, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    base_ext_force_local = torch.tensor(profile.base_ext_force_local, device=device, dtype=torch.float).unsqueeze(0).repeat(num_envs, 1)
    ee_ext_force_global = quat_apply(env.base_yaw_quat, ee_ext_force_local)
    base_ext_force_global = quat_apply(env.base_yaw_quat, base_ext_force_local)

    env.forces[:, env.gripper_idx, :3] = ee_ext_force_global
    env.forces[:, env.robot_base_idx, :3] = base_ext_force_global


def refresh_policy_observation(env, obs):
    if not isinstance(obs, dict) or "obs" not in obs:
        return obs

    obs_tensor = obs["obs"]
    if obs_tensor is None:
        return obs

    latest_frame = obs_tensor[:, -env.num_single_obs:]
    latest_frame[:, -15:] = (env.commands * env.commands_scale)[:, :15]
    return obs


def compute_step_metrics(env, latent_pred_tensor: torch.Tensor, gt_obs_pred_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    nominal_ee_error = torch.norm(env.ee_pos - env.curr_ee_goal_cart_world, dim=1)

    gripper_cmd_global = quat_apply(env.base_yaw_quat, env.current_Fxyz_gripper_cmd)
    compensated_ee_target = env.curr_ee_goal_cart_world + (env.forces[:, env.gripper_idx, :3] + gripper_cmd_global) / env.gripper_force_kps
    compensated_ee_error = torch.norm(env.ee_pos - compensated_ee_target, dim=1)

    base_force_local = env.forces_local[:, env.robot_base_idx]
    compensated_base_cmd = env.commands[:, :2] + (base_force_local[:, :2] + env.current_Fxyz_base_cmd[:, :2]) / env.base_force_kds[:, :2]
    base_nominal_vel_error = torch.norm(env.base_lin_vel[:, :2] - env.commands[:, :2], dim=1)
    base_comp_vel_error = torch.norm(env.base_lin_vel[:, :2] - compensated_base_cmd, dim=1)
    yaw_rate_error = torch.abs(env.base_ang_vel[:, 2] - env.commands[:, 2])

    roll_abs = torch.abs(env.base_euler_xyz[:, 0])
    pitch_abs = torch.abs(env.base_euler_xyz[:, 1])

    foot_contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    foot_vel_xy = torch.norm(env.rigid_state[:, env.feet_indices, 7:9], dim=-1)
    foot_slip_events = torch.sum((foot_contact & (foot_vel_xy > 0.20)).float(), dim=1)
    foot_contact_count = torch.sum(foot_contact.float(), dim=1)

    collision_events = torch.any(torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1, dim=1).float()

    action_delta_rms = torch.sqrt(torch.mean(torch.square(env.actions[:, :env.num_torques] - env.last_actions[:, :env.num_torques]), dim=1))
    torque_delta_rms = torch.sqrt(torch.mean(torch.square(env.torques - env.last_torques), dim=1))
    torque_rms = torch.sqrt(torch.mean(torch.square(env.torques), dim=1))
    energy_proxy = torch.mean(torch.abs(env.torques * env.dof_vel), dim=1)

    pred = latent_pred_tensor.to(dtype=torch.float, device=env.device)
    gt = gt_obs_pred_tensor.to(dtype=torch.float, device=env.device)

    obs_scales = env.obs_scales
    sphere_scale = torch.tensor(
        [obs_scales.ee_sphe_radius_cmd, obs_scales.ee_sphe_pitch_cmd, obs_scales.ee_sphe_yaw_cmd],
        device=env.device,
        dtype=torch.float,
    )
    pred_base_vel = pred[:, 0:3] / obs_scales.lin_vel
    gt_base_vel = gt[:, 0:3] / obs_scales.lin_vel
    pred_ee_sphere = pred[:, 3:6] / sphere_scale
    gt_ee_sphere = gt[:, 3:6] / sphere_scale
    pred_ee_local_cart = sphere2cart(pred_ee_sphere)
    gt_ee_local_cart = sphere2cart(gt_ee_sphere)
    pred_ee_force = pred[:, 6:9] / obs_scales.ee_force
    gt_ee_force = gt[:, 6:9] / obs_scales.ee_force
    pred_base_force = pred[:, 9:12] / obs_scales.base_force
    gt_base_force = gt[:, 9:12] / obs_scales.base_force
    pred_ee_force_norm = torch.norm(pred_ee_force, dim=1)
    gt_ee_force_norm = torch.norm(gt_ee_force, dim=1)
    pred_base_force_norm = torch.norm(pred_base_force, dim=1)
    gt_base_force_norm = torch.norm(gt_base_force, dim=1)

    estimator_base_vel_mae = torch.norm(pred_base_vel - gt_base_vel, dim=1)
    estimator_ee_pos_mae = torch.norm(pred_ee_local_cart - gt_ee_local_cart, dim=1)
    estimator_ee_force_mae = torch.norm(pred_ee_force - gt_ee_force, dim=1)
    estimator_base_force_mae = torch.norm(pred_base_force - gt_base_force, dim=1)

    return {
        "nominal_ee_error": nominal_ee_error,
        "compensated_ee_error": compensated_ee_error,
        "base_nominal_vel_error": base_nominal_vel_error,
        "base_comp_vel_error": base_comp_vel_error,
        "yaw_rate_error": yaw_rate_error,
        "roll_abs": roll_abs,
        "pitch_abs": pitch_abs,
        "foot_slip_events": foot_slip_events,
        "foot_contact_count": foot_contact_count,
        "collision_events": collision_events,
        "action_delta_rms": action_delta_rms,
        "torque_delta_rms": torque_delta_rms,
        "torque_rms": torque_rms,
        "energy_proxy": energy_proxy,
        "estimator_base_vel_mae": estimator_base_vel_mae,
        "estimator_ee_pos_mae": estimator_ee_pos_mae,
        "estimator_ee_force_mae": estimator_ee_force_mae,
        "estimator_base_force_mae": estimator_base_force_mae,
        "estimator_ee_force_pred_norm": pred_ee_force_norm,
        "estimator_ee_force_gt_norm": gt_ee_force_norm,
        "estimator_base_force_pred_norm": pred_base_force_norm,
        "estimator_base_force_gt_norm": gt_base_force_norm,
    }


def append_case_records(records, metrics, active_mask, tag):
    active_cpu = active_mask.detach().cpu().numpy().astype(bool)
    for name, tensor in metrics.items():
        values = tensor.detach().cpu().numpy()
        if values.ndim == 0:
            values = np.asarray([float(values)], dtype=np.float32)
        records["all"][name].extend(values[active_cpu].tolist())
        records["tags"][tag][name].extend(values[active_cpu].tolist())


def stack_or_empty(values: List[np.ndarray], num_envs: int) -> np.ndarray:
    if not values:
        return np.empty((0, num_envs), dtype=np.float32)
    return np.stack(values, axis=0)


def metric_values_for_case(case_name: str, tags: Dict[str, Dict[str, List[float]]], all_metrics: Dict[str, List[float]], metric_name: str) -> List[float]:
    if case_name == "position_only":
        return tags["track"].get(metric_name, all_metrics[metric_name])
    if case_name == "hybrid_force_position":
        return tags["force_track"].get(metric_name, all_metrics[metric_name])
    if case_name == "base_disturbance":
        if metric_name == "yaw_rate_error":
            return tags["yaw_track"].get(metric_name, [])
        if metric_name in ("base_nominal_vel_error", "base_comp_vel_error"):
            return tags["disturbance"].get(metric_name, all_metrics[metric_name])
    return all_metrics[metric_name]


def rmse_values_for_case(case_name: str, case_records: Dict, metric_name: str) -> List[float]:
    last_window_values = metric_values_for_case(
        case_name,
        case_records.get("last_window_tags", defaultdict(lambda: defaultdict(list))),
        case_records.get("last_window_metrics", defaultdict(list)),
        metric_name,
    )
    if last_window_values:
        return last_window_values
    return metric_values_for_case(case_name, case_records["tags"], case_records["all"], metric_name)


def append_last_window_values(dst: List[float], metric_matrix: np.ndarray, last_window_steps: int):
    if metric_matrix.size == 0:
        return
    for env_idx in range(metric_matrix.shape[1]):
        env_values = metric_matrix[:, env_idx]
        valid = ~np.isnan(env_values)
        env_values = env_values[valid]
        if env_values.size == 0:
            continue
        window = env_values[-last_window_steps:] if env_values.size >= last_window_steps else env_values
        dst.extend(window.astype(np.float64).tolist())


def compute_settling_times(error_matrix: np.ndarray, dt: float, threshold: float, dwell_s: float = 0.25) -> List[float]:
    if error_matrix.size == 0:
        return []
    dwell_steps = max(1, int(round(dwell_s / dt)))
    settling_times = []
    for env_idx in range(error_matrix.shape[1]):
        env_errors = error_matrix[:, env_idx]
        valid = ~np.isnan(env_errors)
        env_errors = env_errors[valid]
        if env_errors.size < dwell_steps:
            settling_times.append(float("nan"))
            continue
        settling_idx = float("nan")
        for i in range(env_errors.size - dwell_steps + 1):
            if np.all(env_errors[i:i + dwell_steps] < threshold):
                settling_idx = i * dt
                break
        settling_times.append(float(settling_idx))
    return settling_times


def compute_segment_success(error_matrix: np.ndarray, threshold: float, last_window_steps: int) -> List[float]:
    if error_matrix.size == 0:
        return []
    successes = []
    for env_idx in range(error_matrix.shape[1]):
        env_errors = error_matrix[:, env_idx]
        valid = ~np.isnan(env_errors)
        env_errors = env_errors[valid]
        if env_errors.size == 0:
            successes.append(0.0)
            continue
        window = env_errors[-last_window_steps:] if env_errors.size >= last_window_steps else env_errors
        successes.append(float(np.mean(window) < threshold))
    return successes


def summarize_case(case_name: str, case_records: Dict, dt: float) -> Dict[str, float]:
    all_metrics = case_records["all"]
    tags = case_records["tags"]
    success_flags = case_records["segment_success"]
    settling_times = case_records["settling_times"]
    reset_flags = case_records["reset_flags"]

    nominal_ee_rmse_m = rms_or_nan(rmse_values_for_case(case_name, case_records, "nominal_ee_error"))
    compensated_ee_rmse_m = rms_or_nan(rmse_values_for_case(case_name, case_records, "compensated_ee_error"))
    base_nominal_vel_rmse = rms_or_nan(rmse_values_for_case(case_name, case_records, "base_nominal_vel_error"))
    base_comp_vel_rmse = rms_or_nan(rmse_values_for_case(case_name, case_records, "base_comp_vel_error"))
    yaw_rate_rmse = rms_or_nan(rmse_values_for_case(case_name, case_records, "yaw_rate_error"))
    roll_pitch_rms = rms_or_nan(rmse_values_for_case(case_name, case_records, "roll_pitch_abs"))
    base_ang_vel_xy_rms = rms_or_nan(rmse_values_for_case(case_name, case_records, "base_ang_vel_xy_norm"))
    dynamic_baseline_roll_pitch_rms = rms_or_nan(tags["dynamic_baseline"].get("roll_pitch_abs", []))
    dynamic_oracle_roll_pitch_rms = rms_or_nan(tags["dynamic_oracle"].get("roll_pitch_abs", []))

    posture_abs = []
    posture_abs.extend(all_metrics["roll_abs"])
    posture_abs.extend(all_metrics["pitch_abs"])
    posture_mean_rad = mean_or_nan(posture_abs)
    posture_quality_score = error_to_score(posture_mean_rad, 0.08, 0.35)

    slip_ratio = 0.0
    if case_records["foot_contact_total"] > 0:
        slip_ratio = case_records["foot_slip_total"] / case_records["foot_contact_total"]
    collision_ratio = 0.0
    if case_records["active_step_count"] > 0:
        collision_ratio = case_records["collision_step_count"] / case_records["active_step_count"]

    contact_cleanliness_score = ratio_to_score(collision_ratio, 0.00, 0.30)
    foot_slip_score = ratio_to_score(slip_ratio, 0.05, 0.50)
    survival_rate = percentage(sum(1.0 - f for f in reset_flags), len(reset_flags)) if reset_flags else 0.0

    action_delta_rms = mean_or_nan(all_metrics["action_delta_rms"])
    torque_delta_rms = mean_or_nan(all_metrics["torque_delta_rms"])
    torque_rms = mean_or_nan(all_metrics["torque_rms"])
    energy_proxy = mean_or_nan(all_metrics["energy_proxy"])
    smoothness_score = 0.6 * error_to_score(action_delta_rms, 0.03, 0.30) + 0.4 * error_to_score(torque_delta_rms, 2.0, 40.0)

    success_rate = percentage(sum(success_flags), len(success_flags)) if success_flags else 0.0
    settling_time_s = mean_or_nan([t for t in settling_times if not math.isnan(t)])

    disturbance_band_accuracy = float("nan")
    if "disturbance" in tags and len(tags["disturbance"]["base_comp_vel_error"]) > 0:
        values = np.asarray(tags["disturbance"]["base_comp_vel_error"], dtype=np.float64)
        disturbance_band_accuracy = percentage(np.sum(values < 0.10), values.size)

    force_band_accuracy = float("nan")
    if "force_track" in tags and len(tags["force_track"]["compensated_ee_error"]) > 0:
        values = np.asarray(tags["force_track"]["compensated_ee_error"], dtype=np.float64)
        force_band_accuracy = percentage(np.sum(values < 0.06), values.size)

    case_summary = {
        "case_name": case_name,
        "tracking_rmse_window_s": TRACKING_WINDOW_S,
        "success_rate_pct": success_rate,
        "settling_time_s": settling_time_s,
        "nominal_ee_rmse_cm": nominal_ee_rmse_m * 100.0 if not math.isnan(nominal_ee_rmse_m) else float("nan"),
        "compensated_ee_rmse_cm": compensated_ee_rmse_m * 100.0 if not math.isnan(compensated_ee_rmse_m) else float("nan"),
        "base_nominal_vel_rmse_mps": base_nominal_vel_rmse,
        "base_comp_vel_rmse_mps": base_comp_vel_rmse,
        "yaw_rate_rmse_radps": yaw_rate_rmse,
        "survival_rate_pct": survival_rate,
        "posture_mean_rad": posture_mean_rad,
        "posture_quality_score_pct": posture_quality_score,
        "foot_slip_ratio_pct": slip_ratio * 100.0,
        "foot_slip_score_pct": foot_slip_score,
        "collision_step_ratio_pct": collision_ratio * 100.0,
        "contact_cleanliness_score_pct": contact_cleanliness_score,
        "action_delta_rms": action_delta_rms,
        "torque_delta_rms": torque_delta_rms,
        "torque_rms": torque_rms,
        "energy_proxy_mean": energy_proxy,
        "smoothness_score_pct": smoothness_score,
        "disturbance_band_accuracy_pct": disturbance_band_accuracy,
        "force_band_accuracy_pct": force_band_accuracy,
        "estimator_by_tag": summarize_estimator_tags(tags),
    }

    if case_name == "position_only":
        case_summary["tracking_score_pct"] = error_to_score(nominal_ee_rmse_m, 0.03, 0.12)
        case_summary["case_score_pct"] = (
            0.40 * case_summary["success_rate_pct"]
            + 0.30 * case_summary["tracking_score_pct"]
            + 0.15 * case_summary["posture_quality_score_pct"]
            + 0.15 * case_summary["contact_cleanliness_score_pct"]
        )
    elif case_name == "hybrid_force_position":
        case_summary["tracking_score_pct"] = error_to_score(compensated_ee_rmse_m, 0.03, 0.14)
        band = case_summary["force_band_accuracy_pct"]
        band = 0.0 if math.isnan(band) else band
        case_summary["case_score_pct"] = (
            0.35 * case_summary["success_rate_pct"]
            + 0.35 * case_summary["tracking_score_pct"]
            + 0.15 * band
            + 0.15 * case_summary["survival_rate_pct"]
        )
    elif case_name == "arm_force_estimation":
        force_diag = summarize_force_estimator_branch(tags["force_probe"], "ee")
        case_summary["force_estimator_branch"] = "ee"
        case_summary["force_estimation_nonzero_accuracy_pct"] = force_diag["nonzero_force_accuracy_pct"]
        case_summary["force_estimation_mae_all_n"] = force_diag["mae_all_n"]
        case_summary["force_estimation_mae_nonzero_n"] = force_diag["mae_nonzero_n"]
        case_summary["force_estimation_nonzero_gt_norm_mean_n"] = force_diag["nonzero_gt_force_norm_mean_n"]
        case_summary["force_estimation_nonzero_pred_norm_mean_n"] = force_diag["nonzero_pred_force_norm_mean_n"]
        accuracy = 0.0 if math.isnan(force_diag["nonzero_force_accuracy_pct"]) else force_diag["nonzero_force_accuracy_pct"]
        case_summary["tracking_score_pct"] = accuracy
        case_summary["case_score_pct"] = (
            0.70 * accuracy
            + 0.20 * case_summary["success_rate_pct"]
            + 0.10 * case_summary["survival_rate_pct"]
        )
    elif case_name == "base_force_estimation":
        force_diag = summarize_force_estimator_branch(tags["force_probe"], "base")
        case_summary["force_estimator_branch"] = "base"
        case_summary["force_estimation_nonzero_accuracy_pct"] = force_diag["nonzero_force_accuracy_pct"]
        case_summary["force_estimation_mae_all_n"] = force_diag["mae_all_n"]
        case_summary["force_estimation_mae_nonzero_n"] = force_diag["mae_nonzero_n"]
        case_summary["force_estimation_nonzero_gt_norm_mean_n"] = force_diag["nonzero_gt_force_norm_mean_n"]
        case_summary["force_estimation_nonzero_pred_norm_mean_n"] = force_diag["nonzero_pred_force_norm_mean_n"]
        accuracy = 0.0 if math.isnan(force_diag["nonzero_force_accuracy_pct"]) else force_diag["nonzero_force_accuracy_pct"]
        case_summary["tracking_score_pct"] = accuracy
        case_summary["case_score_pct"] = (
            0.70 * accuracy
            + 0.20 * case_summary["success_rate_pct"]
            + 0.10 * case_summary["survival_rate_pct"]
        )
    elif case_name == "base_disturbance":
        case_summary["tracking_score_pct"] = error_to_score(base_comp_vel_rmse, 0.04, 0.35)
        band = case_summary["disturbance_band_accuracy_pct"]
        band = 0.0 if math.isnan(band) else band
        yaw_score = error_to_score(yaw_rate_rmse, 0.05, 0.40)
        case_summary["case_score_pct"] = (
            0.35 * case_summary["tracking_score_pct"]
            + 0.25 * band
            + 0.20 * yaw_score
            + 0.20 * case_summary["survival_rate_pct"]
        )
    elif case_name == "mixed_whole_body":
        case_summary["tracking_score_pct"] = error_to_score(compensated_ee_rmse_m, 0.04, 0.16)
        base_score = error_to_score(base_comp_vel_rmse, 0.05, 0.35)
        case_summary["case_score_pct"] = (
            0.25 * case_summary["tracking_score_pct"]
            + 0.20 * base_score
            + 0.20 * case_summary["survival_rate_pct"]
            + 0.15 * case_summary["posture_quality_score_pct"]
            + 0.10 * case_summary["contact_cleanliness_score_pct"]
            + 0.10 * case_summary["smoothness_score_pct"]
        )
    else:
        case_summary["tracking_score_pct"] = 0.0
        case_summary["case_score_pct"] = 0.0

    return case_summary


def summarize_force_estimator_branch(records: Dict[str, List[float]], branch: str, nonzero_threshold_n: float = 1e-3) -> Dict[str, float]:
    mae_values = np.asarray(records.get(f"estimator_{branch}_force_mae", []), dtype=np.float64)
    pred_norm = np.asarray(records.get(f"estimator_{branch}_force_pred_norm", []), dtype=np.float64)
    gt_norm = np.asarray(records.get(f"estimator_{branch}_force_gt_norm", []), dtype=np.float64)

    if gt_norm.size == 0:
        return {
            "sample_count": 0,
            "nonzero_sample_count": 0,
            "nonzero_sample_pct": 0.0,
            "gt_force_norm_mean_n": float("nan"),
            "pred_force_norm_mean_n": float("nan"),
            "nonzero_gt_force_norm_mean_n": float("nan"),
            "nonzero_pred_force_norm_mean_n": float("nan"),
            "mae_all_n": float("nan"),
            "mae_nonzero_n": float("nan"),
            "nonzero_force_accuracy_pct": float("nan"),
            "nonzero_force_score_pct": 0.0,
            "zero_force_false_positive_pred_norm_n": float("nan"),
        }

    nonzero_mask = gt_norm > nonzero_threshold_n
    zero_mask = ~nonzero_mask
    nonzero_mae = mean_or_nan(mae_values[nonzero_mask].tolist()) if np.any(nonzero_mask) else float("nan")
    zero_pred_norm = mean_or_nan(pred_norm[zero_mask].tolist()) if np.any(zero_mask) else float("nan")
    gt_norm_mean = float(np.mean(gt_norm))
    pred_norm_mean = float(np.mean(pred_norm))
    nonzero_gt_norm_mean = mean_or_nan(gt_norm[nonzero_mask].tolist()) if np.any(nonzero_mask) else float("nan")
    nonzero_pred_norm_mean = mean_or_nan(pred_norm[nonzero_mask].tolist()) if np.any(nonzero_mask) else float("nan")
    nonzero_accuracy = force_relative_accuracy(nonzero_mae, nonzero_gt_norm_mean)

    return {
        "sample_count": int(gt_norm.size),
        "nonzero_sample_count": int(np.sum(nonzero_mask)),
        "nonzero_sample_pct": percentage(np.sum(nonzero_mask), gt_norm.size),
        "gt_force_norm_mean_n": gt_norm_mean,
        "pred_force_norm_mean_n": pred_norm_mean,
        "nonzero_gt_force_norm_mean_n": nonzero_gt_norm_mean,
        "nonzero_pred_force_norm_mean_n": nonzero_pred_norm_mean,
        "mae_all_n": mean_or_nan(mae_values.tolist()),
        "mae_nonzero_n": nonzero_mae,
        "nonzero_force_accuracy_pct": nonzero_accuracy,
        "nonzero_force_score_pct": 0.0 if math.isnan(nonzero_accuracy) else nonzero_accuracy,
        "zero_force_false_positive_pred_norm_n": zero_pred_norm,
    }


def force_branch_score(all_score: float, nonzero_score: float, nonzero_count: int) -> float:
    if nonzero_count <= 0 or math.isnan(nonzero_score):
        return all_score
    return 0.5 * all_score + 0.5 * nonzero_score


def summarize_estimator_tags(tags: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    tag_summaries = {}
    for tag, records in tags.items():
        ee_diag = summarize_force_estimator_branch(records, "ee")
        base_diag = summarize_force_estimator_branch(records, "base")
        if ee_diag["sample_count"] > 0 or base_diag["sample_count"] > 0:
            tag_summaries[tag] = {
                "ee_force": ee_diag,
                "base_force": base_diag,
            }
    return tag_summaries


def summarize_estimator(estimator_records: Dict[str, List[float]]) -> Dict[str, float]:
    base_vel_mae = mean_or_nan(estimator_records["estimator_base_vel_mae"])
    ee_pos_mae = mean_or_nan(estimator_records["estimator_ee_pos_mae"])
    ee_force_mae = mean_or_nan(estimator_records["estimator_ee_force_mae"])
    base_force_mae = mean_or_nan(estimator_records["estimator_base_force_mae"])
    ee_force_diag = summarize_force_estimator_branch(estimator_records, "ee")
    base_force_diag = summarize_force_estimator_branch(estimator_records, "base")
    ee_force_all_score = error_to_score(ee_force_mae, 3.0, 20.0)
    base_force_all_score = error_to_score(base_force_mae, 3.0, 20.0)
    ee_force_score = force_branch_score(
        ee_force_all_score,
        ee_force_diag["nonzero_force_score_pct"],
        ee_force_diag["nonzero_sample_count"],
    )
    base_force_score = force_branch_score(
        base_force_all_score,
        base_force_diag["nonzero_force_score_pct"],
        base_force_diag["nonzero_sample_count"],
    )

    summary = {
        "base_velocity_estimation_mae_mps": base_vel_mae,
        "ee_position_estimation_mae_cm": ee_pos_mae * 100.0 if not math.isnan(ee_pos_mae) else float("nan"),
        "ee_force_estimation_mae_n": ee_force_mae,
        "base_force_estimation_mae_n": base_force_mae,
        "ee_force_estimator_diagnostics": ee_force_diag,
        "base_force_estimator_diagnostics": base_force_diag,
        "base_velocity_estimation_score_pct": error_to_score(base_vel_mae, 0.03, 0.30),
        "ee_position_estimation_score_pct": error_to_score(ee_pos_mae, 0.03, 0.15),
        "ee_force_estimation_all_sample_score_pct": ee_force_all_score,
        "base_force_estimation_all_sample_score_pct": base_force_all_score,
        "ee_force_estimation_score_pct": ee_force_score,
        "base_force_estimation_score_pct": base_force_score,
    }
    summary["estimator_overall_score_pct"] = (
        0.25 * summary["base_velocity_estimation_score_pct"]
        + 0.25 * summary["ee_position_estimation_score_pct"]
        + 0.25 * summary["ee_force_estimation_score_pct"]
        + 0.25 * summary["base_force_estimation_score_pct"]
    )
    return summary


def summarize_runtime_quality(global_records: Dict[str, List[float]], global_counts: Dict[str, float]) -> Dict[str, float]:
    posture_abs = []
    posture_abs.extend(global_records["roll_abs"])
    posture_abs.extend(global_records["pitch_abs"])
    posture_mean_rad = mean_or_nan(posture_abs)
    slip_ratio = 0.0
    if global_counts["foot_contact_total"] > 0:
        slip_ratio = global_counts["foot_slip_total"] / global_counts["foot_contact_total"]
    collision_ratio = 0.0
    if global_counts["active_step_count"] > 0:
        collision_ratio = global_counts["collision_step_count"] / global_counts["active_step_count"]

    action_delta_rms = mean_or_nan(global_records["action_delta_rms"])
    torque_delta_rms = mean_or_nan(global_records["torque_delta_rms"])
    energy_proxy = mean_or_nan(global_records["energy_proxy"])
    torque_rms = mean_or_nan(global_records["torque_rms"])

    posture_quality_score = error_to_score(posture_mean_rad, 0.08, 0.35)
    foot_slip_score = ratio_to_score(slip_ratio, 0.05, 0.50)
    contact_cleanliness_score = ratio_to_score(collision_ratio, 0.00, 0.30)
    smoothness_score = 0.6 * error_to_score(action_delta_rms, 0.03, 0.30) + 0.4 * error_to_score(torque_delta_rms, 2.0, 40.0)
    stability_score = (
        0.40 * global_counts["survival_rate_pct"]
        + 0.25 * posture_quality_score
        + 0.20 * contact_cleanliness_score
        + 0.15 * foot_slip_score
    )

    return {
        "survival_rate_pct": global_counts["survival_rate_pct"],
        "posture_mean_rad": posture_mean_rad,
        "posture_quality_score_pct": posture_quality_score,
        "foot_slip_ratio_pct": slip_ratio * 100.0,
        "foot_slip_score_pct": foot_slip_score,
        "collision_step_ratio_pct": collision_ratio * 100.0,
        "contact_cleanliness_score_pct": contact_cleanliness_score,
        "action_delta_rms": action_delta_rms,
        "torque_delta_rms": torque_delta_rms,
        "torque_rms": torque_rms,
        "energy_proxy_mean": energy_proxy,
        "smoothness_score_pct": smoothness_score,
        "stability_score_pct": stability_score,
    }


def run_scenario(env, policy, obs, scenario: Scenario, estimator_records, global_records, global_counts):
    num_envs = env.num_envs
    dt = env.dt
    scenario_records = {
        "all": defaultdict(list),
        "tags": defaultdict(lambda: defaultdict(list)),
        "last_window_metrics": defaultdict(list),
        "last_window_tags": defaultdict(lambda: defaultdict(list)),
        "segment_success": [],
        "settling_times": [],
        "reset_flags": [],
        "foot_slip_total": 0.0,
        "foot_contact_total": 0.0,
        "collision_step_count": 0.0,
        "active_step_count": 0.0,
    }
    segment_primary_errors = []
    segment_window_metric_rows = defaultdict(list)
    segment_window_tag_rows = defaultdict(lambda: defaultdict(list))
    alive_mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    any_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    policy_info = {}

    for ph in scenario.phases:
        steps = max(1, int(round(ph.duration_s / dt)))
        for _ in range(steps):
            apply_profile(env, ph)
            obs = refresh_policy_observation(env, obs)
            gt_obs_pred_tensor = env.obs_pred.detach().clone()
            actions = policy(obs, policy_info)
            obs, _, dones, _ = env.step(actions.detach())
            latent_pred_tensor = torch.tensor(policy_info["latents"], device=env.device, dtype=torch.float)
            active_mask = alive_mask & (~dones.bool())
            any_reset_mask |= dones.bool()

            if ph.collect and torch.any(active_mask):
                metrics = compute_step_metrics(env, latent_pred_tensor, gt_obs_pred_tensor)
                append_case_records(scenario_records, metrics, active_mask, ph.tag)
                append_case_records({"all": estimator_records, "tags": defaultdict(lambda: defaultdict(list))}, metrics, active_mask, "all")
                append_case_records({"all": global_records, "tags": defaultdict(lambda: defaultdict(list))}, metrics, active_mask, "all")

                if ph.primary_collect:
                    active_cpu = active_mask.detach().cpu().numpy().astype(bool)
                    primary_values = metrics[scenario.primary_metric].detach().cpu().numpy()
                    row = np.full(num_envs, np.nan, dtype=np.float32)
                    row[active_cpu] = primary_values[active_cpu]
                    segment_primary_errors.append(row)
                    for metric_name in LAST_WINDOW_RMSE_METRICS:
                        if metric_name not in metrics:
                            continue
                        metric_values = metrics[metric_name].detach().cpu().numpy()
                        metric_row = np.full(num_envs, np.nan, dtype=np.float32)
                        metric_row[active_cpu] = metric_values[active_cpu]
                        segment_window_metric_rows[metric_name].append(metric_row)
                        segment_window_tag_rows[ph.tag][metric_name].append(metric_row)

                scenario_records["foot_slip_total"] += float(torch.sum(metrics["foot_slip_events"][active_mask]).item())
                scenario_records["foot_contact_total"] += float(torch.sum(metrics["foot_contact_count"][active_mask]).item())
                scenario_records["collision_step_count"] += float(torch.sum(metrics["collision_events"][active_mask]).item())
                scenario_records["active_step_count"] += float(torch.sum(active_mask).item())

                global_counts["foot_slip_total"] += float(torch.sum(metrics["foot_slip_events"][active_mask]).item())
                global_counts["foot_contact_total"] += float(torch.sum(metrics["foot_contact_count"][active_mask]).item())
                global_counts["collision_step_count"] += float(torch.sum(metrics["collision_events"][active_mask]).item())
                global_counts["active_step_count"] += float(torch.sum(active_mask).item())

            alive_mask &= (~dones.bool())
            if not torch.any(alive_mask):
                break
        if not torch.any(alive_mask):
            break

    error_matrix = stack_or_empty(segment_primary_errors, num_envs)
    last_window_steps = max(1, int(round(TRACKING_WINDOW_S / dt)))
    for metric_name, rows in segment_window_metric_rows.items():
        metric_matrix = stack_or_empty(rows, num_envs)
        append_last_window_values(scenario_records["last_window_metrics"][metric_name], metric_matrix, last_window_steps)
    for tag, tag_rows in segment_window_tag_rows.items():
        for metric_name, rows in tag_rows.items():
            metric_matrix = stack_or_empty(rows, num_envs)
            append_last_window_values(scenario_records["last_window_tags"][tag][metric_name], metric_matrix, last_window_steps)
    scenario_records["segment_success"].extend(compute_segment_success(error_matrix, scenario.success_threshold, last_window_steps))
    scenario_records["settling_times"].extend(compute_settling_times(error_matrix, dt, scenario.success_threshold))
    scenario_records["reset_flags"].extend(any_reset_mask.detach().cpu().numpy().astype(np.float32).tolist())

    global_counts["trial_count"] += float(num_envs)
    global_counts["trial_resets"] += float(torch.sum(any_reset_mask).item())
    return obs, scenario_records


def merge_case_records(dst, src):
    for name, values in src["all"].items():
        dst["all"][name].extend(values)
    for tag, tag_dict in src["tags"].items():
        for name, values in tag_dict.items():
            dst["tags"][tag][name].extend(values)
    for name, values in src["last_window_metrics"].items():
        dst["last_window_metrics"][name].extend(values)
    for tag, tag_dict in src["last_window_tags"].items():
        for name, values in tag_dict.items():
            dst["last_window_tags"][tag][name].extend(values)
    dst["segment_success"].extend(src["segment_success"])
    dst["settling_times"].extend(src["settling_times"])
    dst["reset_flags"].extend(src["reset_flags"])
    dst["foot_slip_total"] += src["foot_slip_total"]
    dst["foot_contact_total"] += src["foot_contact_total"]
    dst["collision_step_count"] += src["collision_step_count"]
    dst["active_step_count"] += src["active_step_count"]


def make_empty_case_records():
    return {
        "all": defaultdict(list),
        "tags": defaultdict(lambda: defaultdict(list)),
        "last_window_metrics": defaultdict(list),
        "last_window_tags": defaultdict(lambda: defaultdict(list)),
        "segment_success": [],
        "settling_times": [],
        "reset_flags": [],
        "foot_slip_total": 0.0,
        "foot_contact_total": 0.0,
        "collision_step_count": 0.0,
        "active_step_count": 0.0,
    }


def safe_path_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def output_paths(args, metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = safe_path_name(metadata["resolved_run_name"])
    safe_checkpoint = safe_path_name(metadata["resolved_checkpoint"])
    output_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        args.output_dir,
        f"{args.task}_{safe_run_name}_ckpt{safe_checkpoint}_{timestamp}",
    )
    os.makedirs(output_root, exist_ok=True)
    return {
        "root": output_root,
        "json": os.path.join(output_root, "summary.json"),
        "md": os.path.join(output_root, "summary.md"),
    }


def resolve_model_metadata(args, train_cfg):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    resolved_model_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    resolved_run_dir = os.path.dirname(resolved_model_path)
    resolved_run_name = os.path.basename(resolved_run_dir)
    resolved_model_file = os.path.basename(resolved_model_path)

    resolved_checkpoint = -1
    if resolved_model_file.startswith("model_") and resolved_model_file.endswith(".pt"):
        checkpoint_str = resolved_model_file[len("model_"):-len(".pt")]
        try:
            resolved_checkpoint = int(checkpoint_str)
        except ValueError:
            resolved_checkpoint = checkpoint_str

    return {
        "requested_load_run": args.load_run if args.load_run is not None else train_cfg.runner.load_run,
        "requested_checkpoint": args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint,
        "resolved_run_name": resolved_run_name,
        "resolved_checkpoint": resolved_checkpoint,
        "resolved_model_file": resolved_model_file,
        "resolved_model_path": resolved_model_path,
    }


def format_force_diag(diag: Dict[str, float]) -> str:
    return (
        f"samples={diag['sample_count']}, "
        f"nonzero={diag['nonzero_sample_count']} ({format_float(diag['nonzero_sample_pct'], 1)} %), "
        f"gt_norm={format_float(diag['gt_force_norm_mean_n'], 2)} N, "
        f"pred_norm={format_float(diag['pred_force_norm_mean_n'], 2)} N, "
        f"nonzero_acc={format_float(diag['nonzero_force_accuracy_pct'], 2)} %, "
        f"nonzero_gt_norm={format_float(diag['nonzero_gt_force_norm_mean_n'], 2)} N, "
        f"nonzero_pred_norm={format_float(diag['nonzero_pred_force_norm_mean_n'], 2)} N, "
        f"mae_all={format_float(diag['mae_all_n'], 2)} N, "
        f"mae_nonzero={format_float(diag['mae_nonzero_n'], 2)} N, "
        f"zero_gt_pred_norm={format_float(diag['zero_force_false_positive_pred_norm_n'], 2)} N"
    )


def build_markdown_report(metadata, case_summaries, estimator_summary, runtime_quality, overall_summary):
    lines = [
        "# Go2+Piper Automated Evaluation Report",
        "",
        "## Metadata",
        f"- Task: `{metadata['task']}`",
        f"- Requested load run: `{metadata['requested_load_run']}`",
        f"- Requested checkpoint: `{metadata['requested_checkpoint']}`",
        f"- Resolved run: `{metadata['resolved_run_name']}`",
        f"- Resolved checkpoint: `{metadata['resolved_checkpoint']}`",
        f"- Model file: `{metadata['resolved_model_file']}`",
        f"- Model path: `{metadata['resolved_model_path']}`",
        f"- Num envs: `{metadata['num_envs']}`",
        f"- Eval repeats: `{metadata['eval_repeats']}`",
        f"- Requested seed: `{metadata['requested_seed']}`",
        f"- Base seed: `{metadata['base_seed']}`",
        f"- Repeat seed mode: `{metadata['repeat_seed_mode']}`",
        f"- Dt: `{format_float(metadata['dt'], 4)}` s",
        f"- Terrain: `{metadata['terrain']}`",
        "",
        "## Overall",
        f"- Overall score: `{format_float(overall_summary['overall_score_pct'], 2)} %`",
        f"- Position score: `{format_float(overall_summary['position_score_pct'], 2)} %`",
        f"- Hybrid score: `{format_float(overall_summary['hybrid_score_pct'], 2)} %`",
        f"- Arm force estimation score: `{format_float(overall_summary['arm_force_estimation_score_pct'], 2)} %`",
        f"- Base force estimation score: `{format_float(overall_summary['base_force_estimation_score_pct'], 2)} %`",
        f"- Base disturbance score: `{format_float(overall_summary['base_score_pct'], 2)} %`",
        f"- Mixed whole-body score: `{format_float(overall_summary['mixed_score_pct'], 2)} %`",
        f"- Estimator score: `{format_float(overall_summary['estimator_score_pct'], 2)} %`",
        f"- Stability score: `{format_float(overall_summary['stability_score_pct'], 2)} %`",
        f"- Smoothness score: `{format_float(overall_summary['smoothness_score_pct'], 2)} %`",
        "",
        "## Case Summaries",
    ]

    for case_name, summary in case_summaries.items():
        lines.extend([
            f"### {case_name}",
            f"- Case score: `{format_float(summary['case_score_pct'], 2)} %`",
            f"- Success rate: `{format_float(summary['success_rate_pct'], 2)} %`",
            f"- Tracking score: `{format_float(summary['tracking_score_pct'], 2)} %`",
            f"- Tracking RMSE window: `{format_float(summary['tracking_rmse_window_s'], 3)} s`",
            f"- Survival rate: `{format_float(summary['survival_rate_pct'], 2)} %`",
            f"- Settling time: `{format_float(summary['settling_time_s'], 3)} s`",
            f"- Nominal EE RMSE: `{format_float(summary['nominal_ee_rmse_cm'], 2)} cm`",
            f"- Compensated EE RMSE: `{format_float(summary['compensated_ee_rmse_cm'], 2)} cm`",
            f"- Base nominal velocity RMSE: `{format_float(summary['base_nominal_vel_rmse_mps'], 3)} m/s`",
            f"- Base compensated velocity RMSE: `{format_float(summary['base_comp_vel_rmse_mps'], 3)} m/s`",
            f"- Yaw rate RMSE: `{format_float(summary['yaw_rate_rmse_radps'], 3)} rad/s`",
            f"- Posture quality: `{format_float(summary['posture_quality_score_pct'], 2)} %`",
            f"- Contact cleanliness: `{format_float(summary['contact_cleanliness_score_pct'], 2)} %`",
            f"- Foot slip ratio: `{format_float(summary['foot_slip_ratio_pct'], 2)} %`",
            f"- Smoothness: `{format_float(summary['smoothness_score_pct'], 2)} %`",
        ])
        if "force_estimator_branch" in summary:
            lines.extend([
                f"- Force estimator branch: `{summary['force_estimator_branch']}`",
                f"- Force estimation nonzero accuracy: `{format_float(summary['force_estimation_nonzero_accuracy_pct'], 2)} %`",
                f"- Force estimation MAE all samples: `{format_float(summary['force_estimation_mae_all_n'], 2)} N`",
                f"- Force estimation MAE nonzero samples: `{format_float(summary['force_estimation_mae_nonzero_n'], 2)} N`",
                f"- Nonzero force norm mean: gt `{format_float(summary['force_estimation_nonzero_gt_norm_mean_n'], 2)} N`, pred `{format_float(summary['force_estimation_nonzero_pred_norm_mean_n'], 2)} N`",
            ])
        nonzero_tag_lines = []
        for tag, tag_diag in summary.get("estimator_by_tag", {}).items():
            ee_diag = tag_diag["ee_force"]
            base_diag = tag_diag["base_force"]
            if ee_diag["nonzero_sample_count"] > 0 or base_diag["nonzero_sample_count"] > 0:
                nonzero_tag_lines.extend([
                    f"- `{tag}` EE force estimator: {format_force_diag(ee_diag)}",
                    f"- `{tag}` base force estimator: {format_force_diag(base_diag)}",
                ])
        if nonzero_tag_lines:
            lines.append("- Force-estimator diagnostics on nonzero-force tags:")
            lines.extend(nonzero_tag_lines)
        lines.append("")

    lines.extend([
        "## Estimator",
        f"- Overall estimator score: `{format_float(estimator_summary['estimator_overall_score_pct'], 2)} %`",
        f"- Base velocity estimation MAE: `{format_float(estimator_summary['base_velocity_estimation_mae_mps'], 3)} m/s`",
        f"- EE position estimation MAE: `{format_float(estimator_summary['ee_position_estimation_mae_cm'], 2)} cm`",
        f"- EE force estimation MAE: `{format_float(estimator_summary['ee_force_estimation_mae_n'], 2)} N`",
        f"- Base force estimation MAE: `{format_float(estimator_summary['base_force_estimation_mae_n'], 2)} N`",
        f"- EE force all-sample score: `{format_float(estimator_summary['ee_force_estimation_all_sample_score_pct'], 2)} %`",
        f"- EE force nonzero-aware score: `{format_float(estimator_summary['ee_force_estimation_score_pct'], 2)} %`",
        f"- Base force all-sample score: `{format_float(estimator_summary['base_force_estimation_all_sample_score_pct'], 2)} %`",
        f"- Base force nonzero-aware score: `{format_float(estimator_summary['base_force_estimation_score_pct'], 2)} %`",
        f"- EE force diagnostics: {format_force_diag(estimator_summary['ee_force_estimator_diagnostics'])}",
        f"- Base force diagnostics: {format_force_diag(estimator_summary['base_force_estimator_diagnostics'])}",
        "",
        "## Runtime Quality",
        f"- Stability score: `{format_float(runtime_quality['stability_score_pct'], 2)} %`",
        f"- Survival rate: `{format_float(runtime_quality['survival_rate_pct'], 2)} %`",
        f"- Posture quality: `{format_float(runtime_quality['posture_quality_score_pct'], 2)} %`",
        f"- Contact cleanliness: `{format_float(runtime_quality['contact_cleanliness_score_pct'], 2)} %`",
        f"- Foot slip score: `{format_float(runtime_quality['foot_slip_score_pct'], 2)} %`",
        f"- Smoothness score: `{format_float(runtime_quality['smoothness_score_pct'], 2)} %`",
        f"- Energy proxy mean: `{format_float(runtime_quality['energy_proxy_mean'], 4)}`",
        "",
    ])
    return "\n".join(lines)


def print_console_summary(case_summaries, estimator_summary, runtime_quality, overall_summary, output_files):
    print("\n=== Automated Evaluation Summary ===")
    print(f"Overall score        : {format_float(overall_summary['overall_score_pct'], 2)} %")
    print(f"Position score       : {format_float(overall_summary['position_score_pct'], 2)} %")
    print(f"Hybrid score         : {format_float(overall_summary['hybrid_score_pct'], 2)} %")
    print(f"Arm force est. score : {format_float(overall_summary['arm_force_estimation_score_pct'], 2)} %")
    print(f"Base force est. score: {format_float(overall_summary['base_force_estimation_score_pct'], 2)} %")
    print(f"Base score           : {format_float(overall_summary['base_score_pct'], 2)} %")
    print(f"Mixed score          : {format_float(overall_summary['mixed_score_pct'], 2)} %")
    print(f"Estimator score      : {format_float(overall_summary['estimator_score_pct'], 2)} %")
    print(f"Stability score      : {format_float(overall_summary['stability_score_pct'], 2)} %")
    print(f"Smoothness score     : {format_float(overall_summary['smoothness_score_pct'], 2)} %")
    print("")
    for case_name, summary in case_summaries.items():
        if "force_estimator_branch" in summary:
            print(
                f"[{case_name}] success={format_float(summary['success_rate_pct'], 2)} % | "
                f"case_score={format_float(summary['case_score_pct'], 2)} % | "
                f"nonzero_acc={format_float(summary['force_estimation_nonzero_accuracy_pct'], 2)} % | "
                f"MAE_all={format_float(summary['force_estimation_mae_all_n'], 2)} N | "
                f"MAE_nonzero={format_float(summary['force_estimation_mae_nonzero_n'], 2)} N"
            )
        else:
            print(
                f"[{case_name}] success={format_float(summary['success_rate_pct'], 2)} % | "
                f"case_score={format_float(summary['case_score_pct'], 2)} % | "
                f"EE_RMSE={format_float(summary['compensated_ee_rmse_cm'], 2)} cm | "
                f"base_RMSE={format_float(summary['base_comp_vel_rmse_mps'], 3)} m/s"
            )
    print("")
    print(
        "Estimator MAE        : "
        f"base_vel={format_float(estimator_summary['base_velocity_estimation_mae_mps'], 3)} m/s, "
        f"ee_pos={format_float(estimator_summary['ee_position_estimation_mae_cm'], 2)} cm, "
        f"ee_force={format_float(estimator_summary['ee_force_estimation_mae_n'], 2)} N, "
        f"base_force={format_float(estimator_summary['base_force_estimation_mae_n'], 2)} N"
    )
    print(
        "Arm/EE force estimate: "
        f"nonzero_acc={format_float(estimator_summary['ee_force_estimator_diagnostics']['nonzero_force_accuracy_pct'], 2)} %, "
        f"MAE_all={format_float(estimator_summary['ee_force_estimator_diagnostics']['mae_all_n'], 2)} N, "
        f"MAE_nonzero={format_float(estimator_summary['ee_force_estimator_diagnostics']['mae_nonzero_n'], 2)} N"
    )
    print(
        "Base force estimate  : "
        f"nonzero_acc={format_float(estimator_summary['base_force_estimator_diagnostics']['nonzero_force_accuracy_pct'], 2)} %, "
        f"MAE_all={format_float(estimator_summary['base_force_estimator_diagnostics']['mae_all_n'], 2)} N, "
        f"MAE_nonzero={format_float(estimator_summary['base_force_estimator_diagnostics']['mae_nonzero_n'], 2)} N"
    )
    print(
        "Runtime quality      : "
        f"survival={format_float(runtime_quality['survival_rate_pct'], 2)} %, "
        f"slip={format_float(runtime_quality['foot_slip_ratio_pct'], 2)} %, "
        f"collision={format_float(runtime_quality['collision_step_ratio_pct'], 2)} %"
    )
    print("")
    if output_files is None:
        print("Report export        : disabled (`--no_report`)")
    else:
        print(f"Saved JSON report to : {output_files['json']}")
        print(f"Saved Markdown to    : {output_files['md']}")


def run_evaluation(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else min(env_cfg.env.num_envs, 8)
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

    home_local = env.key_command_ee_local_cart[0].detach().cpu().numpy().astype(np.float32)
    scenario_bank = build_scenarios(home_local)

    selected_cases = list(scenario_bank.keys()) if args.eval_case == "all" else [args.eval_case]
    invalid_cases = [case for case in selected_cases if case not in scenario_bank]
    if invalid_cases:
        raise ValueError(f"Unknown eval case(s): {invalid_cases}")

    case_records = {case: make_empty_case_records() for case in selected_cases}
    repeat_seeds = build_repeat_seeds(selected_cases, args.eval_repeats)
    estimator_records = defaultdict(list)
    global_records = defaultdict(list)
    global_counts = {
        "foot_slip_total": 0.0,
        "foot_contact_total": 0.0,
        "collision_step_count": 0.0,
        "active_step_count": 0.0,
        "trial_count": 0.0,
        "trial_resets": 0.0,
        "survival_rate_pct": 0.0,
    }

    total_scenarios = sum(len(scenario_bank[case_name]) for case_name in selected_cases) * args.eval_repeats
    completed_scenarios = 0
    first_progress_update = True

    for case_idx, case_name in enumerate(selected_cases, start=1):
        case_scenarios = scenario_bank[case_name]
        for repeat_idx in range(args.eval_repeats):
            repeat_seed = None
            if args.eval_repeats > 1:
                repeat_seed = repeat_seeds[case_name][repeat_idx]
                set_seed(repeat_seed)
            for scenario_idx, scenario in enumerate(case_scenarios, start=1):
                seed_text = f" | seed {repeat_seed}" if repeat_seed is not None else ""
                status_line = (
                    f"Current case {case_idx}/{len(selected_cases)}: {case_name} | "
                    f"repeat {repeat_idx + 1}/{args.eval_repeats} | "
                    f"scenario {scenario_idx}/{len(case_scenarios)}: {scenario.name}"
                    f"{seed_text}"
                )
                show_progress(status_line, completed_scenarios, total_scenarios, first_update=first_progress_update)
                first_progress_update = False
                obs = reset_env_for_eval(env)
                obs, scenario_result = run_scenario(env, policy, obs, scenario, estimator_records, global_records, global_counts)
                merge_case_records(case_records[case_name], scenario_result)
                completed_scenarios += 1

    show_progress("Evaluation complete.", completed_scenarios, total_scenarios, first_update=first_progress_update)
    if sys.stdout.isatty():
        print("")

    global_counts["survival_rate_pct"] = percentage(
        global_counts["trial_count"] - global_counts["trial_resets"],
        global_counts["trial_count"],
    )

    case_summaries = {case_name: summarize_case(case_name, records, env.dt) for case_name, records in case_records.items()}
    estimator_summary = summarize_estimator(estimator_records)
    runtime_quality = summarize_runtime_quality(global_records, global_counts)

    overall_summary = {
        "position_score_pct": case_summaries.get("position_only", {}).get("case_score_pct", 0.0),
        "hybrid_score_pct": case_summaries.get("hybrid_force_position", {}).get("case_score_pct", 0.0),
        "arm_force_estimation_score_pct": case_summaries.get("arm_force_estimation", {}).get("case_score_pct", 0.0),
        "base_force_estimation_score_pct": case_summaries.get("base_force_estimation", {}).get("case_score_pct", 0.0),
        "base_score_pct": case_summaries.get("base_disturbance", {}).get("case_score_pct", 0.0),
        "mixed_score_pct": case_summaries.get("mixed_whole_body", {}).get("case_score_pct", 0.0),
        "estimator_score_pct": estimator_summary["estimator_overall_score_pct"],
        "stability_score_pct": runtime_quality["stability_score_pct"],
        "smoothness_score_pct": runtime_quality["smoothness_score_pct"],
    }
    weights = {
        "position_score_pct": 0.16,
        "hybrid_score_pct": 0.20,
        "arm_force_estimation_score_pct": 0.10,
        "base_force_estimation_score_pct": 0.10,
        "base_score_pct": 0.16,
        "mixed_score_pct": 0.08,
        "estimator_score_pct": 0.12,
        "stability_score_pct": 0.06,
        "smoothness_score_pct": 0.02,
    }
    case_component_map = {
        "position_score_pct": "position_only",
        "hybrid_score_pct": "hybrid_force_position",
        "arm_force_estimation_score_pct": "arm_force_estimation",
        "base_force_estimation_score_pct": "base_force_estimation",
        "base_score_pct": "base_disturbance",
        "mixed_score_pct": "mixed_whole_body",
    }
    enabled_components = {
        key: value for key, value in overall_summary.items()
        if key not in case_component_map or case_component_map[key] in selected_cases
    }
    enabled_weight_sum = sum(weights[key] for key in enabled_components.keys())
    overall_summary["overall_score_pct"] = 0.0
    if enabled_weight_sum > 0:
        for key, value in enabled_components.items():
            overall_summary["overall_score_pct"] += (weights[key] / enabled_weight_sum) * value

    metadata = {
        "task": args.task,
        "num_envs": env.num_envs,
        "eval_repeats": args.eval_repeats,
        "requested_seed": args.seed,
        "base_seed": env_cfg.seed,
        "repeat_seed_mode": "random_per_case_repeat" if args.eval_repeats > 1 else "default_seed",
        "repeat_seeds": repeat_seeds,
        "dt": env.dt,
        "terrain": "flat" if args.flat_terrain else "default_eval",
        "env_cfg": class_to_dict(env_cfg),
    }
    metadata.update(resolve_model_metadata(args, train_cfg))
    result_payload = {
        "metadata": metadata,
        "cases": case_summaries,
        "estimator": estimator_summary,
        "runtime_quality": runtime_quality,
        "overall": overall_summary,
    }
    output_files = None
    if not args.no_report:
        output_files = output_paths(args, metadata)
        with open(output_files["json"], "w", encoding="utf-8") as f:
            json.dump(make_json_safe(result_payload), f, indent=2, ensure_ascii=False)

        markdown = build_markdown_report(metadata, case_summaries, estimator_summary, runtime_quality, overall_summary)
        with open(output_files["md"], "w", encoding="utf-8") as f:
            f.write(markdown + "\n")

    print_console_summary(case_summaries, estimator_summary, runtime_quality, overall_summary, output_files)


if __name__ == "__main__":
    args = get_eval_args()
    run_evaluation(args)
