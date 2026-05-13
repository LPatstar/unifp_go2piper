import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import eval_posforce as full_eval
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import set_seed


TRACKING_WINDOW_S = 0.25
POSITION_SUCCESS_THRESHOLD_M = 0.01
BASE_VEL_SUCCESS_THRESHOLD_MPS = 0.05
BASE_YAW_SUCCESS_THRESHOLD_RADPS = 0.10
ARM_FORCE_SUCCESS_THRESHOLD_N = 6.0

AXIS_NAMES = ("x", "y", "z")
CASE_ORDER = (
    "position_only_static",
    "base_command_tracking",
    "arm_force_estimation",
    "position_only_moving",
)
CASE_ALIASES = {
    "position_only": "position_only_static",
    "position_only_static": "position_only_static",
    "position_only_no_base_cmd": "position_only_static",
    "base_command_tracking": "base_command_tracking",
    "command_tracking": "base_command_tracking",
    "vx_vy_yaw_command_tracking": "base_command_tracking",
    "arm_force_estimation": "arm_force_estimation",
    "position_only_moving": "position_only_moving",
    "position_only_with_vx_vy": "position_only_moving",
}


@dataclass
class LiteScenario:
    name: str
    metric_kind: str
    phases: List[full_eval.Phase]
    active_axis: Optional[int] = None


def format_float(value: float, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def percentage_from_flags(flags: List[float]) -> float:
    if not flags:
        return float("nan")
    return 100.0 * float(np.sum(flags)) / float(len(flags))


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_values(values: np.ndarray) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    return arr[~np.isnan(arr)].tolist()


def resolve_selected_cases(eval_case: str) -> List[str]:
    if eval_case == "all":
        return list(CASE_ORDER)
    case_name = CASE_ALIASES.get(eval_case)
    if case_name is None:
        valid = ", ".join(["all"] + sorted(CASE_ALIASES.keys()))
        raise ValueError(f"Unknown compact eval case `{eval_case}`. Valid cases: {valid}")
    return [case_name]


def build_position_scenarios(home_local: np.ndarray, moving: bool = False) -> List[LiteScenario]:
    targets = [
        ("forward_high", home_local + np.array([0.10, 0.00, 0.08], dtype=np.float32)),
        ("forward_low", home_local + np.array([0.08, 0.00, -0.06], dtype=np.float32)),
        ("left_reach", home_local + np.array([0.05, 0.10, 0.02], dtype=np.float32)),
        ("right_reach", home_local + np.array([0.05, -0.10, 0.02], dtype=np.float32)),
    ]
    base_cmds = [
        np.array([0.18, 0.00, 0.0], dtype=np.float32),
        np.array([0.12, 0.00, 0.0], dtype=np.float32),
        np.array([0.10, 0.08, 0.0], dtype=np.float32),
        np.array([0.10, -0.08, 0.0], dtype=np.float32),
    ]
    scenarios = []
    for idx, (name, target) in enumerate(targets):
        base_cmd = base_cmds[idx] if moving else np.zeros(3, dtype=np.float32)
        scenario_name = f"{'moving_' if moving else ''}pos_{name}"
        scenarios.append(
            LiteScenario(
                name=scenario_name,
                metric_kind="position",
                phases=[
                    full_eval.phase(0.6, home_local, collect=False, tag="warmup"),
                    full_eval.phase(2.0, target, base_cmd=base_cmd, tag="track"),
                ],
            )
        )
    return scenarios


def build_base_command_scenarios(home_local: np.ndarray) -> List[LiteScenario]:
    command_specs = [
        ("vx_forward", (0.25, 0.00, 0.00)),
        ("vy_left", (0.00, 0.18, 0.00)),
        ("yaw_left", (0.00, 0.00, 0.25)),
        ("vx_vy_yaw", (0.20, 0.10, 0.20)),
    ]
    return [
        LiteScenario(
            name=f"cmd_{name}",
            metric_kind="base_command",
            phases=[
                full_eval.phase(0.6, home_local, collect=False, tag="warmup"),
                full_eval.phase(1.8, home_local, base_cmd=base_cmd, tag="command_track"),
            ],
        )
        for name, base_cmd in command_specs
    ]


def make_force_probe_scenario(name: str, target_local: np.ndarray, force_local: np.ndarray, active_axis: int) -> LiteScenario:
    return LiteScenario(
        name=name,
        metric_kind="arm_force",
        active_axis=active_axis,
        phases=[
            full_eval.phase(0.6, target_local, collect=False, tag="warmup"),
            full_eval.phase(0.5, target_local, primary_collect=False, tag="zero_probe"),
            full_eval.phase(1.2, target_local, ee_ext_force_local=force_local, tag="force_probe"),
        ],
    )


def build_b2z1_arm_force_scenarios(home_local: np.ndarray, seed: Optional[int]) -> List[LiteScenario]:
    rng = np.random.default_rng(1 if seed is None else seed)
    position_offsets = rng.uniform(
        low=np.array([0.02, -0.10, -0.07], dtype=np.float32),
        high=np.array([0.12, 0.10, 0.08], dtype=np.float32),
        size=(5, 3),
    ).astype(np.float32)
    target_positions = [home_local + offset for offset in position_offsets]
    axis_specs = [
        ("x", 0, np.array([-60.0, -30.0, -15.0, 30.0, 60.0], dtype=np.float32)),
        ("y", 1, np.array([-40.0, -20.0, -10.0, 20.0, 40.0], dtype=np.float32)),
        ("z", 2, np.array([-40.0, -20.0, -10.0, 20.0, 40.0], dtype=np.float32)),
    ]
    scenarios = []
    for axis_name, axis_idx, force_values in axis_specs:
        for position_idx, (target_local, force_value) in enumerate(zip(target_positions, force_values), start=1):
            force_local = np.zeros(3, dtype=np.float32)
            force_local[axis_idx] = force_value
            scenarios.append(
                make_force_probe_scenario(
                    f"arm_force_{axis_name}_pos{position_idx}_{force_value:+.0f}n",
                    target_local,
                    force_local,
                    axis_idx,
                )
            )
    return scenarios


def build_generic_arm_force_scenarios(home_local: np.ndarray, force_limit: float) -> List[LiteScenario]:
    force_values = np.array([-force_limit, -0.5 * force_limit, 0.5 * force_limit, force_limit], dtype=np.float32)
    target_offsets = [
        np.array([0.05, 0.00, 0.03], dtype=np.float32),
        np.array([0.08, 0.05, 0.00], dtype=np.float32),
        np.array([0.08, -0.05, 0.02], dtype=np.float32),
        np.array([0.03, 0.00, -0.04], dtype=np.float32),
    ]
    scenarios = []
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        for position_idx, force_value in enumerate(force_values, start=1):
            target_local = home_local + target_offsets[position_idx - 1]
            force_local = np.zeros(3, dtype=np.float32)
            force_local[axis_idx] = force_value
            scenarios.append(
                make_force_probe_scenario(
                    f"arm_force_{axis_name}_pos{position_idx}_{force_value:+.0f}n",
                    target_local,
                    force_local,
                    axis_idx,
                )
            )
    return scenarios


def build_arm_force_scenarios(home_local: np.ndarray, task_name: str, env, seed: Optional[int]) -> List[LiteScenario]:
    if task_name == "b2z1_pos_force":
        return build_b2z1_arm_force_scenarios(home_local, seed)
    force_low, force_high = env.cfg.commands.max_push_force_xyz_gripper_ext
    force_limit = float(min(abs(force_low), abs(force_high)))
    return build_generic_arm_force_scenarios(home_local, force_limit)


def build_scenario_bank(home_local: np.ndarray, task_name: str, env, seed: Optional[int]) -> Dict[str, List[LiteScenario]]:
    return {
        "position_only_static": build_position_scenarios(home_local, moving=False),
        "base_command_tracking": build_base_command_scenarios(home_local),
        "arm_force_estimation": build_arm_force_scenarios(home_local, task_name, env, seed),
        "position_only_moving": build_position_scenarios(home_local, moving=True),
    }


def decode_ee_forces(env, policy_info, gt_obs_pred_tensor):
    if "latents" not in policy_info:
        raise RuntimeError("Policy did not populate `policy_info['latents']`; cannot evaluate force estimator.")
    pred = torch.tensor(policy_info["latents"], device=env.device, dtype=torch.float)
    gt = gt_obs_pred_tensor.to(dtype=torch.float, device=env.device)
    pred_ee_force = pred[:, 6:9] / env.obs_scales.ee_force
    gt_ee_force = gt[:, 6:9] / env.obs_scales.ee_force
    return pred_ee_force, gt_ee_force


def compute_step_metrics(env, scenario: LiteScenario, policy_info, gt_obs_pred_tensor) -> Dict[str, torch.Tensor]:
    if scenario.metric_kind == "position":
        return {"ee_distance_m": torch.norm(env.ee_pos - env.curr_ee_goal_cart_world, dim=1)}

    if scenario.metric_kind == "base_command":
        return {
            "vx_abs_error_mps": torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0]),
            "vy_abs_error_mps": torch.abs(env.base_lin_vel[:, 1] - env.commands[:, 1]),
            "yaw_abs_error_radps": torch.abs(env.base_ang_vel[:, 2] - env.commands[:, 2]),
        }

    if scenario.metric_kind == "arm_force":
        pred_ee_force, gt_ee_force = decode_ee_forces(env, policy_info, gt_obs_pred_tensor)
        force_abs_error = torch.abs(pred_ee_force - gt_ee_force)
        active_axis = scenario.active_axis
        if active_axis is None:
            raise ValueError(f"Arm force scenario `{scenario.name}` has no active axis.")
        return {
            "force_active_abs_error_n": force_abs_error[:, active_axis],
            "force_x_abs_error_n": force_abs_error[:, 0],
            "force_y_abs_error_n": force_abs_error[:, 1],
            "force_z_abs_error_n": force_abs_error[:, 2],
        }

    raise ValueError(f"Unknown metric kind `{scenario.metric_kind}`.")


def append_rows(rows: Dict[str, List[np.ndarray]], metrics: Dict[str, torch.Tensor], active_mask: torch.Tensor, num_envs: int):
    active_cpu = active_mask.detach().cpu().numpy().astype(bool)
    for metric_name, metric_tensor in metrics.items():
        values = metric_tensor.detach().cpu().numpy()
        row = np.full(num_envs, np.nan, dtype=np.float32)
        row[active_cpu] = values[active_cpu]
        rows[metric_name].append(row)


def final_window_means(rows: Dict[str, List[np.ndarray]], num_envs: int, last_window_steps: int) -> Dict[str, np.ndarray]:
    final_values = {}
    for metric_name, metric_rows in rows.items():
        if not metric_rows:
            final_values[metric_name] = np.full(num_envs, np.nan, dtype=np.float32)
            continue
        metric_matrix = np.stack(metric_rows, axis=0)
        env_values = []
        for env_idx in range(num_envs):
            values = metric_matrix[:, env_idx]
            values = values[~np.isnan(values)]
            if values.size == 0:
                env_values.append(np.nan)
                continue
            window = values[-last_window_steps:] if values.size >= last_window_steps else values
            env_values.append(float(np.mean(window)))
        final_values[metric_name] = np.asarray(env_values, dtype=np.float32)
    return final_values


def run_lite_scenario(env, policy, obs, scenario: LiteScenario):
    num_envs = env.num_envs
    dt = env.dt
    policy_info = {}
    rows = defaultdict(list)
    alive_mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)

    for ph in scenario.phases:
        steps = max(1, int(round(ph.duration_s / dt)))
        for _ in range(steps):
            full_eval.apply_profile(env, ph)
            obs = full_eval.refresh_policy_observation(env, obs)
            gt_obs_pred_tensor = env.obs_pred.detach().clone()
            actions = policy(obs, policy_info)
            obs, _, dones, _ = env.step(actions.detach())
            full_eval.apply_profile_ee_orientation(env, ph)
            active_mask = alive_mask & (~dones.bool())

            if ph.collect and ph.primary_collect and torch.any(active_mask):
                metrics = compute_step_metrics(env, scenario, policy_info, gt_obs_pred_tensor)
                append_rows(rows, metrics, active_mask, num_envs)

            alive_mask &= (~dones.bool())
            if not torch.any(alive_mask):
                break
        if not torch.any(alive_mask):
            break

    last_window_steps = max(1, int(round(TRACKING_WINDOW_S / dt)))
    return obs, final_window_means(rows, num_envs, last_window_steps)


def make_empty_case_summary(case_name: str) -> Dict:
    return {
        "case_name": case_name,
        "success_flags": [],
        "mae_values": [],
        "component_values": defaultdict(list),
        "axis_values": defaultdict(list),
        "scenario_count": 0,
    }


def update_case_summary(summary: Dict, scenario: LiteScenario, final_values: Dict[str, np.ndarray]):
    summary["scenario_count"] += 1

    if scenario.metric_kind == "position":
        distances = final_values.get("ee_distance_m", np.asarray([], dtype=np.float32))
        summary["mae_values"].extend(finite_values(distances))
        for value in distances:
            summary["success_flags"].append(float((not np.isnan(value)) and value <= POSITION_SUCCESS_THRESHOLD_M))
        return

    if scenario.metric_kind == "base_command":
        vx_errors = final_values.get("vx_abs_error_mps", np.asarray([], dtype=np.float32))
        vy_errors = final_values.get("vy_abs_error_mps", np.asarray([], dtype=np.float32))
        yaw_errors = final_values.get("yaw_abs_error_radps", np.asarray([], dtype=np.float32))
        summary["component_values"]["vx_abs_error_mps"].extend(finite_values(vx_errors))
        summary["component_values"]["vy_abs_error_mps"].extend(finite_values(vy_errors))
        summary["component_values"]["yaw_abs_error_radps"].extend(finite_values(yaw_errors))
        for vx_error, vy_error, yaw_error in zip(vx_errors, vy_errors, yaw_errors):
            success = (
                not np.isnan(vx_error)
                and not np.isnan(vy_error)
                and not np.isnan(yaw_error)
                and vx_error <= BASE_VEL_SUCCESS_THRESHOLD_MPS
                and vy_error <= BASE_VEL_SUCCESS_THRESHOLD_MPS
                and yaw_error <= BASE_YAW_SUCCESS_THRESHOLD_RADPS
            )
            summary["success_flags"].append(float(success))
        return

    if scenario.metric_kind == "arm_force":
        active_errors = final_values.get("force_active_abs_error_n", np.asarray([], dtype=np.float32))
        axis_name = AXIS_NAMES[scenario.active_axis]
        summary["mae_values"].extend(finite_values(active_errors))
        summary["axis_values"][axis_name].extend(finite_values(active_errors))
        for value in active_errors:
            summary["success_flags"].append(float((not np.isnan(value)) and value <= ARM_FORCE_SUCCESS_THRESHOLD_N))
        return

    raise ValueError(f"Unknown metric kind `{scenario.metric_kind}`.")


def finalize_case_summary(summary: Dict) -> Dict:
    case_name = summary["case_name"]
    success_rate = percentage_from_flags(summary["success_flags"])

    if case_name in ("position_only_static", "position_only_moving"):
        mae_m = mean_or_nan(summary["mae_values"])
        summary.update(
            {
                "criterion": f"last {TRACKING_WINDOW_S:.2f}s mean EE-target distance <= {POSITION_SUCCESS_THRESHOLD_M * 100.0:.1f} cm",
                "success_rate_pct": success_rate,
                "mae": {"ee_distance_cm": mae_m * 100.0 if not math.isnan(mae_m) else float("nan")},
            }
        )
        label = "Position-only static" if case_name == "position_only_static" else "Position-only with vx/vy"
        summary["lines"] = [
            f"[{case_name}] Success Rate: {format_float(success_rate, 2)} % | Criterion: {summary['criterion']}",
            f"[{case_name}] MAE: EE-target distance = {format_float(summary['mae']['ee_distance_cm'], 2)} cm",
        ]
        summary["label"] = label
        return summary

    if case_name == "base_command_tracking":
        vx_mae = mean_or_nan(summary["component_values"]["vx_abs_error_mps"])
        vy_mae = mean_or_nan(summary["component_values"]["vy_abs_error_mps"])
        yaw_mae = mean_or_nan(summary["component_values"]["yaw_abs_error_radps"])
        summary.update(
            {
                "criterion": (
                    f"last {TRACKING_WINDOW_S:.2f}s mean |vx error| <= {BASE_VEL_SUCCESS_THRESHOLD_MPS:.2f} m/s, "
                    f"|vy error| <= {BASE_VEL_SUCCESS_THRESHOLD_MPS:.2f} m/s, "
                    f"|yaw-rate error| <= {BASE_YAW_SUCCESS_THRESHOLD_RADPS:.2f} rad/s"
                ),
                "success_rate_pct": success_rate,
                "mae": {
                    "vx_abs_error_mps": vx_mae,
                    "vy_abs_error_mps": vy_mae,
                    "yaw_abs_error_radps": yaw_mae,
                },
            }
        )
        summary["lines"] = [
            f"[{case_name}] Success Rate: {format_float(success_rate, 2)} % | Criterion: {summary['criterion']}",
            f"[{case_name}] MAE: vx = {format_float(vx_mae, 3)} m/s, vy = {format_float(vy_mae, 3)} m/s, yaw = {format_float(yaw_mae, 3)} rad/s",
        ]
        summary["label"] = "Base vx/vy/yaw command tracking"
        return summary

    if case_name == "arm_force_estimation":
        overall_mae = mean_or_nan(summary["mae_values"])
        axis_mae = {axis: mean_or_nan(summary["axis_values"][axis]) for axis in AXIS_NAMES}
        summary.update(
            {
                "criterion": f"last {TRACKING_WINDOW_S:.2f}s mean active-axis force estimator AE <= {ARM_FORCE_SUCCESS_THRESHOLD_N:.1f} N",
                "success_rate_pct": success_rate,
                "mae": {
                    "overall_active_axis_abs_error_n": overall_mae,
                    "x_active_axis_abs_error_n": axis_mae["x"],
                    "y_active_axis_abs_error_n": axis_mae["y"],
                    "z_active_axis_abs_error_n": axis_mae["z"],
                },
            }
        )
        summary["lines"] = [
            f"[{case_name}] Success Rate: {format_float(success_rate, 2)} % | Criterion: {summary['criterion']}",
            f"[{case_name}] MAE: overall = {format_float(overall_mae, 2)} N, x = {format_float(axis_mae['x'], 2)} N, y = {format_float(axis_mae['y'], 2)} N, z = {format_float(axis_mae['z'], 2)} N",
        ]
        summary["label"] = "Arm force estimation"
        return summary

    raise ValueError(f"Unknown case `{case_name}`.")


def compact_summary_for_json(summary: Dict) -> Dict:
    return {
        "case_name": summary["case_name"],
        "label": summary["label"],
        "criterion": summary["criterion"],
        "success_rate_pct": summary["success_rate_pct"],
        "mae": summary["mae"],
        "scenario_count": summary["scenario_count"],
        "sample_count": len(summary["success_flags"]),
        "lines": summary["lines"],
    }


def output_paths(args, metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = full_eval.safe_path_name(metadata["resolved_run_name"])
    safe_checkpoint = full_eval.safe_path_name(metadata["resolved_checkpoint"])
    output_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        args.output_dir,
        f"{args.task}_lite_{safe_run_name}_ckpt{safe_checkpoint}_{timestamp}",
    )
    os.makedirs(output_root, exist_ok=True)
    return {
        "root": output_root,
        "json": os.path.join(output_root, "summary.json"),
        "md": os.path.join(output_root, "summary.md"),
    }


def build_markdown_report(metadata: Dict, task_summaries: Dict[str, Dict]) -> str:
    lines = [
        "# Compact Position-Force Evaluation Report",
        "",
        "## Metadata",
        f"- Script: `{metadata['script']}`",
        f"- Task: `{metadata['task']}`",
        f"- Requested load run: `{metadata['requested_load_run']}`",
        f"- Requested checkpoint: `{metadata['requested_checkpoint']}`",
        f"- Resolved run: `{metadata['resolved_run_name']}`",
        f"- Resolved checkpoint: `{metadata['resolved_checkpoint']}`",
        f"- Model file: `{metadata['resolved_model_file']}`",
        f"- Model path: `{metadata['resolved_model_path']}`",
        f"- Generated at: `{metadata['generated_at']}`",
        f"- Num envs: `{metadata['num_envs']}`",
        f"- Eval repeats: `{metadata['eval_repeats']}`",
        f"- Requested seed: `{metadata['requested_seed']}`",
        f"- Base seed: `{metadata['base_seed']}`",
        f"- Dt: `{format_float(metadata['dt'], 4)}` s",
        f"- Terrain: `{metadata['terrain']}`",
        "",
        "## Results",
    ]
    for case_name in task_summaries:
        lines.append(f"### {case_name}")
        lines.extend(f"- {line}" for line in task_summaries[case_name]["lines"])
        lines.append("")
    return "\n".join(lines).rstrip()


def print_console_summary(task_summaries: Dict[str, Dict], output_files):
    for case_name in task_summaries:
        for line in task_summaries[case_name]["lines"]:
            print(line)
    if output_files is not None:
        print(f"Report JSON: {output_files['json']}")
        print(f"Report Markdown: {output_files['md']}")


def setup_eval_env(args):
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
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.play = True
    env.enable_gripper_cmd_force = False
    env.enable_play_immediate_gripper_cmd_force = False
    env.enable_random_force_events = False
    env._update_key_command_ee_goal()
    return env_cfg, train_cfg, env, policy, obs


def run_evaluation(args):
    selected_cases = resolve_selected_cases(args.eval_case)
    env_cfg, train_cfg, env, policy, obs = setup_eval_env(args)

    home_local = env.key_command_ee_local_cart[0].detach().cpu().numpy().astype(np.float32)
    scenario_bank = build_scenario_bank(home_local, args.task, env, env_cfg.seed)
    repeat_seeds = full_eval.build_repeat_seeds(selected_cases, args.eval_repeats)
    raw_summaries = {case_name: make_empty_case_summary(case_name) for case_name in selected_cases}
    total_scenarios = sum(len(scenario_bank[case_name]) for case_name in selected_cases) * args.eval_repeats
    completed_scenarios = 0
    first_progress_update = True

    for case_idx, case_name in enumerate(selected_cases, start=1):
        scenarios = scenario_bank[case_name]
        for repeat_idx in range(args.eval_repeats):
            repeat_seed = None
            if args.eval_repeats > 1:
                repeat_seed = repeat_seeds[case_name][repeat_idx]
                set_seed(repeat_seed)
            for scenario_idx, scenario in enumerate(scenarios, start=1):
                seed_text = f" | seed {repeat_seed}" if repeat_seed is not None else ""
                status_line = (
                    f"Current case {case_idx}/{len(selected_cases)}: {case_name} | "
                    f"repeat {repeat_idx + 1}/{args.eval_repeats} | "
                    f"scenario {scenario_idx}/{len(scenarios)}: {scenario.name}"
                    f"{seed_text}"
                )
                full_eval.show_progress(status_line, completed_scenarios, total_scenarios, first_update=first_progress_update)
                first_progress_update = False
                obs = full_eval.reset_env_for_eval(env)
                obs, final_values = run_lite_scenario(env, policy, obs, scenario)
                update_case_summary(raw_summaries[case_name], scenario, final_values)
                completed_scenarios += 1

    full_eval.show_progress("Evaluation complete.", completed_scenarios, total_scenarios, first_update=first_progress_update)
    if sys.stdout.isatty():
        print("")

    task_summaries = {
        case_name: compact_summary_for_json(finalize_case_summary(raw_summaries[case_name]))
        for case_name in selected_cases
    }

    metadata = {
        "script": "eval_posforce_lite.py",
        "task": args.task,
        "num_envs": env.num_envs,
        "eval_repeats": args.eval_repeats,
        "requested_seed": args.seed,
        "base_seed": env_cfg.seed,
        "repeat_seed_mode": "random_per_case_repeat" if args.eval_repeats > 1 else "default_seed",
        "repeat_seeds": repeat_seeds,
        "dt": env.dt,
        "terrain": "flat" if args.flat_terrain else "default_eval",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata.update(full_eval.resolve_model_metadata(args, train_cfg))

    output_files = None
    if not args.no_report:
        output_files = output_paths(args, metadata)
        payload = {
            "metadata": metadata,
            "tasks": task_summaries,
        }
        with open(output_files["json"], "w", encoding="utf-8") as f:
            json.dump(full_eval.make_json_safe(payload), f, indent=2, ensure_ascii=False)
        with open(output_files["md"], "w", encoding="utf-8") as f:
            f.write(build_markdown_report(metadata, task_summaries) + "\n")

    print_console_summary(task_summaries, output_files)


if __name__ == "__main__":
    run_evaluation(full_eval.get_eval_args())
