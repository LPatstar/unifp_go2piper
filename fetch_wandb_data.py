#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = "UniFP"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "wandb_exports"
LOCAL_RUN_RE = re.compile(r"^(?P<timestamp>[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2})(?:_(?P<run_suffix>.*))?$")
RESERVED_HISTORY_KEYS = {"_step", "_runtime", "_timestamp", "global_step"}
TIME_AXIS_SUFFIX = "/time"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch WandB run metadata/history for the latest local training run or a specified run."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Target run to fetch. Accepts an exact WandB run name, a local logs directory name, or a short local run suffix.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"WandB project name. Default: {DEFAULT_PROJECT}",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity/team. If omitted, the script tries to infer it from your WandB login state.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory used to store exported run data. Default: {DEFAULT_OUTPUT_DIR.name}/",
    )
    parser.add_argument(
        "--history_stride",
        type=int,
        default=1000,
        help="Keep one history row every N training iterations. Default: 1000.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="Also sync the selected local TensorBoard run back to WandB as a new *_tb_sync run with correct iteration-aligned steps.",
    )
    parser.add_argument(
        "--no_fetch",
        action="store_true",
        default=False,
        help="Skip local export file generation. If used with --sync, only the WandB sync is performed.",
    )
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "items"):
        try:
            return {str(k): json_safe(v) for k, v in value.items()}
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, int, float, bool)):
        try:
            return {str(k): json_safe(v) for k, v in vars(value).items()}
        except Exception:
            pass
    return value


def log_progress(message: str):
    print(f"[fetch_wandb_data] {message}", flush=True)


def load_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "The `wandb` package is not installed in the current Python environment. "
            "Install it first, for example with `pip install wandb`."
        ) from exc
    return wandb


def iter_local_training_dirs(logs_root: Path):
    if not logs_root.exists():
        return []

    candidates = []
    for experiment_dir in logs_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "exported":
                continue
            match = LOCAL_RUN_RE.match(run_dir.name)
            if not match:
                continue
            candidates.append(run_dir)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def parse_local_run_dir(run_dir: Path):
    match = LOCAL_RUN_RE.match(run_dir.name)
    if not match:
        return None

    timestamp = match.group("timestamp")
    run_suffix = match.group("run_suffix") or ""
    experiment_name = run_dir.parent.name
    wandb_run_name = f"{timestamp}_{experiment_name}_{run_suffix}"

    return {
        "source": "local_logs",
        "experiment_name": experiment_name,
        "local_run_dir": str(run_dir),
        "local_dir_name": run_dir.name,
        "timestamp": timestamp,
        "run_suffix": run_suffix,
        "wandb_run_name": wandb_run_name,
        "mtime": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
    }


def resolve_target_from_local_logs(run_name: Optional[str], logs_root: Path):
    local_dirs = iter_local_training_dirs(logs_root)
    if not local_dirs:
        return None

    if run_name is None:
        return parse_local_run_dir(local_dirs[0])

    exact_basename_matches = [d for d in local_dirs if d.name == run_name]
    if exact_basename_matches:
        return parse_local_run_dir(exact_basename_matches[0])

    for d in local_dirs:
        parsed = parse_local_run_dir(d)
        if parsed is None:
            continue
        if parsed["run_suffix"] == run_name:
            return parsed
        if parsed["wandb_run_name"] == run_name:
            return parsed

    return None


def infer_entity(api, explicit_entity: Optional[str]):
    if explicit_entity:
        return explicit_entity

    env_entity = os.environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity

    for attr_name in ("default_entity", "entity"):
        attr_value = getattr(api, attr_name, None)
        if isinstance(attr_value, str) and attr_value:
            return attr_value

    viewer = getattr(api, "viewer", None)
    if callable(viewer):
        try:
            viewer = viewer()
        except TypeError:
            viewer = None

    if isinstance(viewer, dict):
        for key in ("entity", "entityName", "username", "name"):
            value = viewer.get(key)
            if isinstance(value, str) and value:
                return value

    if viewer is not None:
        for attr_name in ("entity", "entityName", "username", "name"):
            value = getattr(viewer, attr_name, None)
            if isinstance(value, str) and value:
                return value

    return None


def find_run_by_name(api, project_path: str, target_names: List[str]):
    log_progress(f"Searching WandB runs in `{project_path}` for: {target_names}")
    runs = api.runs(project_path, order="-created_at")
    target_names = [name for name in target_names if name]
    for run in runs:
        candidate_names = {
            getattr(run, "name", None),
            getattr(run, "display_name", None),
            getattr(run, "id", None),
        }
        if any(name in candidate_names for name in target_names):
            return run
    return None


def should_drop_summary_key(key: str) -> bool:
    lower = key.lower()
    return (
        lower.startswith("system/")
        or lower.startswith("system.")
        or "fan" in lower
        or lower.startswith("_wandb")
    )


def build_allowed_history_keys(summary_payload) -> List[str]:
    allowed = []
    for key in summary_payload.keys():
        if should_drop_summary_key(str(key)):
            continue
        allowed.append(str(key))
    return sorted(set(allowed))


def build_metric_history_keys(allowed_keys: List[str]) -> List[str]:
    return [key for key in allowed_keys if key not in RESERVED_HISTORY_KEYS]


def build_iteration_history_keys(allowed_keys: List[str]) -> List[str]:
    metric_keys = build_metric_history_keys(allowed_keys)
    return [key for key in metric_keys if not key.endswith(TIME_AXIS_SUFFIX)]


def row_step_value(row, fallback_index: int) -> int:
    step_value = row.get("global_step", row.get("_step", fallback_index))
    try:
        return int(step_value)
    except (TypeError, ValueError):
        return fallback_index


def filter_history_row(row, allowed_keys: List[str]):
    selected = {}
    for key in RESERVED_HISTORY_KEYS:
        if key in row:
            selected[key] = row[key]
    for key in allowed_keys:
        if key in row:
            selected[key] = row[key]
    return selected


def compact_history_row(row: dict):
    return {key: value for key, value in row.items() if value is not None}


def history_identity(row, fallback_index: int):
    if "global_step" in row and row["global_step"] is not None:
        return ("global_step", row["global_step"])
    if "_step" in row and row["_step"] is not None:
        return ("_step", row_step_value(row, fallback_index))
    if "_runtime" in row and row["_runtime"] is not None:
        return ("_runtime", row["_runtime"])
    if "_timestamp" in row and row["_timestamp"] is not None:
        return ("_timestamp", row["_timestamp"])
    return ("index", fallback_index)


def merge_row_into_history(row: dict, merged_rows: List[dict], merged_lookup: dict, fallback_index: int):
    identity = history_identity(row, fallback_index)
    if identity not in merged_lookup:
        merged_lookup[identity] = len(merged_rows)
        merged_rows.append(dict(row))
        return

    target = merged_rows[merged_lookup[identity]]
    for key, value in row.items():
        if value is None:
            continue
        target[key] = value


def history_sort_key(row: dict):
    if row.get("global_step") is not None:
        return (0, row["global_step"])
    if row.get("_step") is not None:
        return (1, row["_step"])
    if row.get("_runtime") is not None:
        return (2, row["_runtime"])
    if row.get("_timestamp") is not None:
        return (3, row["_timestamp"])
    return (4, 0)


def find_local_event_files(local_run_dir: str):
    run_dir = Path(local_run_dir)
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("events.out.tfevents.*"))


def load_tensorboard_event_accumulator(local_run_dir: str):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None

    event_files = find_local_event_files(local_run_dir)
    event_source = local_run_dir if len(event_files) != 1 else str(event_files[0])
    size_guidance = {
        "scalars": 0,
        "histograms": 1,
        "images": 1,
        "audio": 1,
        "compressedHistograms": 1,
        "tensors": 1,
    }
    accumulator = event_accumulator.EventAccumulator(event_source, size_guidance=size_guidance)
    accumulator.Reload()
    return accumulator


def tensorboard_iteration_scalar_tags(accumulator, allowed_keys: Optional[List[str]] = None, use_all_scalar_tags: bool = False):
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if not scalar_tags:
        return [], []

    if use_all_scalar_tags or not allowed_keys:
        iteration_keys = sorted(key for key in scalar_tags if not key.endswith(TIME_AXIS_SUFFIX))
        ignored_time_keys = sorted(key for key in scalar_tags if key.endswith(TIME_AXIS_SUFFIX))
        return iteration_keys, ignored_time_keys

    iteration_keys = build_iteration_history_keys(allowed_keys)
    skipped_time_keys = [key for key in build_metric_history_keys(allowed_keys) if key.endswith(TIME_AXIS_SUFFIX)]
    available_time_keys = [key for key in skipped_time_keys if key in scalar_tags]
    available_iteration_keys = [key for key in iteration_keys if key in scalar_tags]
    return available_iteration_keys, available_time_keys


def scalar_event_value(event):
    if hasattr(event, "value"):
        return event.value
    if hasattr(event, "tensor_proto"):
        return getattr(event, "tensor_proto", None)
    return None


def merge_scalar_events(events_by_key, stride: int):
    merged_rows = []
    row_lookup = {}
    first_wall_time = None

    for key, events in events_by_key.items():
        for event in events:
            step = int(event.step)
            if first_wall_time is None:
                first_wall_time = float(event.wall_time)
            if step not in row_lookup:
                row = {
                    "global_step": step,
                    "_step": step,
                    "_timestamp": float(event.wall_time),
                    "_runtime": float(event.wall_time) - first_wall_time,
                }
                row_lookup[step] = row
                merged_rows.append(row)
            row = row_lookup[step]
            row["_timestamp"] = max(row["_timestamp"], float(event.wall_time))
            row["_runtime"] = max(row["_runtime"], float(event.wall_time) - first_wall_time)
            row[key] = scalar_event_value(event)

    merged_rows.sort(key=lambda row: row["global_step"])

    sampled_rows = []
    last_sampled_step = None
    stride = max(1, int(stride))
    for row in merged_rows:
        step = row["global_step"]
        if step % stride != 0:
            continue
        sampled_rows.append(compact_history_row(row))
        last_sampled_step = step

    if merged_rows:
        final_row = compact_history_row(merged_rows[-1])
        if last_sampled_step != final_row["global_step"]:
            sampled_rows.append(final_row)

    return sampled_rows


def fetch_history_rows_from_tensorboard(local_run_dir: str, allowed_keys: Optional[List[str]], stride: int, use_all_scalar_tags: bool = False):
    accumulator = load_tensorboard_event_accumulator(local_run_dir)
    if accumulator is None:
        return None

    available_iteration_keys, available_time_keys = tensorboard_iteration_scalar_tags(
        accumulator,
        allowed_keys,
        use_all_scalar_tags=use_all_scalar_tags,
    )

    if not available_iteration_keys:
        return None

    log_progress(
        f"Using local TensorBoard events from `{local_run_dir}` "
        f"({len(available_iteration_keys)} iteration keys; ignoring {len(available_time_keys)} /time-axis keys)."
    )

    iteration_events = {}
    for idx, key in enumerate(available_iteration_keys, start=1):
        log_progress(f"Loading TensorBoard scalar {idx}/{len(available_iteration_keys)}: {key}")
        iteration_events[key] = accumulator.Scalars(key)

    history_rows = merge_scalar_events(iteration_events, stride)
    log_progress(f"Local TensorBoard history ready: {len(history_rows)} sampled iteration rows.")
    return {
        "history": history_rows,
        "source": "local_tensorboard",
        "available_iteration_keys": available_iteration_keys,
        "ignored_time_keys": available_time_keys,
    }


def build_sync_run_name(base_name: str):
    if not base_name:
        base_name = "tensorboard_sync"
    if base_name.endswith("_tb_sync"):
        return base_name
    return f"{base_name}_tb_sync"


def sync_tensorboard_history_to_wandb(
    wandb,
    entity: str,
    project: str,
    local_resolution: dict,
    base_run_name: str,
    config_payload: dict,
    summary_payload: dict,
):
    if local_resolution is None:
        raise SystemExit("`--sync` requires a local run under `logs/` so the TensorBoard event file can be read.")

    sync_history_bundle = fetch_history_rows_from_tensorboard(
        local_resolution["local_run_dir"],
        allowed_keys=None,
        stride=1,
        use_all_scalar_tags=True,
    )
    if sync_history_bundle is None:
        raise SystemExit(
            f"Could not load local TensorBoard iteration history for sync from `{local_resolution['local_run_dir']}`."
        )

    history_rows = sync_history_bundle["history"]
    if not history_rows:
        raise SystemExit("TensorBoard sync requested, but no iteration-aligned scalar rows were found.")

    sync_run_name = build_sync_run_name(base_run_name)
    log_progress(
        f"Syncing {len(history_rows)} full TensorBoard iteration rows to WandB as `{sync_run_name}` "
        f"under `{entity}/{project}`."
    )

    sync_config = json_safe(dict(config_payload))
    sync_config.update(
        {
            "tb_sync_source": "local_tensorboard",
            "tb_sync_local_run_dir": local_resolution["local_run_dir"],
            "tb_sync_original_run_name": base_run_name,
        }
    )

    sync_run = wandb.init(
        project=project,
        entity=entity,
        name=sync_run_name,
        config=sync_config,
        tags=["tb_sync", "tensorboard_sync", "iteration_axis"],
        resume="never",
    )
    sync_run.define_metric("global_step")
    sync_run.define_metric("*", step_metric="global_step")

    total_rows = len(history_rows)
    for idx, row in enumerate(history_rows, start=1):
        step = int(row["global_step"])
        payload = {
            key: value
            for key, value in row.items()
            if key not in {"_step", "_runtime", "_timestamp"} and value is not None
        }
        payload["global_step"] = step
        sync_run.log(payload, step=step)
        if idx == 1 or idx == total_rows or idx % 500 == 0:
            log_progress(f"WandB sync progress: {idx}/{total_rows} rows logged.")

    clean_summary = json_safe(
        {str(k): v for k, v in summary_payload.items() if not should_drop_summary_key(str(k))}
    )
    if clean_summary:
        sync_run.summary.update(clean_summary)
    sync_run.summary["tb_sync_local_run_dir"] = local_resolution["local_run_dir"]
    sync_run.summary["tb_sync_original_run_name"] = base_run_name
    sync_run.summary["tb_sync_history_row_count"] = total_rows
    sync_run.summary["tb_sync_iteration_key_count"] = len(sync_history_bundle["available_iteration_keys"])

    sync_url = getattr(sync_run, "url", None)
    sync_id = getattr(sync_run, "id", None)
    sync_run.finish()
    log_progress(f"WandB TensorBoard sync complete: {sync_run_name}")

    return {
        "enabled": True,
        "run_name": sync_run_name,
        "run_id": sync_id,
        "run_url": sync_url,
        "history_source": sync_history_bundle["source"],
        "history_row_count": total_rows,
        "iteration_key_count": len(sync_history_bundle["available_iteration_keys"]),
        "ignored_time_axis_keys": sync_history_bundle["ignored_time_keys"],
    }


def fetch_history_rows(run, allowed_keys: List[str], stride: int):
    merged_rows = []
    merged_lookup = {}
    fallback_index = 0
    stride = max(1, int(stride))
    metric_keys = build_iteration_history_keys(allowed_keys)
    total_keys = len(metric_keys)
    request_reserved_keys = ["global_step", "_step", "_runtime", "_timestamp"]
    for key_idx, key in enumerate(metric_keys, start=1):
        log_progress(f"Fetching history key {key_idx}/{total_keys}: {key}")
        row_count_before = len(merged_rows)
        for row in run.scan_history(keys=[key] + request_reserved_keys):
            if not row:
                continue
            row = dict(row)
            value = row.get(key)
            if value is None:
                continue
            filtered = filter_history_row(row, allowed_keys)
            merge_row_into_history(filtered, merged_rows, merged_lookup, fallback_index)
            fallback_index += 1
        log_progress(
            f"Finished key {key_idx}/{total_keys}: {key} | merged rows now {len(merged_rows)} "
            f"(delta {len(merged_rows) - row_count_before})"
        )

    if not merged_rows:
        return []

    log_progress(f"Merging complete. Sorting {len(merged_rows)} merged rows.")
    merged_rows.sort(key=history_sort_key)

    sampled_rows = []
    last_sampled_step = None
    for idx, row in enumerate(merged_rows):
        step = row_step_value(row, idx)
        if step % stride != 0:
            continue
        compact_row = compact_history_row(row)
        if any(key not in RESERVED_HISTORY_KEYS for key in compact_row.keys()):
            sampled_rows.append(compact_row)
        last_sampled_step = step

    final_row = merged_rows[-1]
    final_step = row_step_value(final_row, len(merged_rows) - 1)
    if last_sampled_step != final_step:
        compact_final_row = compact_history_row(final_row)
        if any(key not in RESERVED_HISTORY_KEYS for key in compact_final_row.keys()):
            sampled_rows.append(compact_final_row)

    log_progress(f"Downsampling complete. Kept {len(sampled_rows)} rows with stride={stride}.")
    return {
        "history": sampled_rows,
        "source": "wandb_api",
        "available_iteration_keys": metric_keys,
        "ignored_time_keys": [key for key in build_metric_history_keys(allowed_keys) if key.endswith(TIME_AXIS_SUFFIX)],
    }


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def scalar_to_csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(json_safe(value), ensure_ascii=False)


def write_history_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def write_history_csv(path: Path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_to_csv_cell(row.get(key)) for key in fieldnames})


def safe_dir_name(name: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main():
    args = parse_args()
    log_progress(
        f"Starting export with args: run_name={args.run_name}, project={args.project}, "
        f"stride={args.history_stride}, sync={args.sync}, no_fetch={args.no_fetch}"
    )
    wandb = load_wandb()
    log_progress("Imported wandb successfully.")

    logs_root = REPO_ROOT / "logs"
    local_resolution = resolve_target_from_local_logs(args.run_name, logs_root)
    if local_resolution is not None:
        log_progress(
            f"Resolved target from local logs: dir={local_resolution['local_dir_name']} -> "
            f"wandb_run_name={local_resolution['wandb_run_name']}"
        )
    elif args.run_name:
        log_progress(f"No local logs match found for `{args.run_name}`. Falling back to direct WandB name lookup.")
    else:
        log_progress("No local run specified. Will fall back to latest local run or latest WandB run.")

    api = wandb.Api(timeout=30)
    log_progress("Connected to WandB API.")
    entity = infer_entity(api, args.entity)
    if not entity:
        raise SystemExit(
            "Could not infer the WandB entity/team automatically. "
            "Please rerun with `--entity <your_entity>`."
        )
    log_progress(f"Using WandB entity `{entity}`.")

    project_path = f"{entity}/{args.project}"
    log_progress(f"Resolved project path: {project_path}")

    target_names = []
    resolution_info = {}
    if local_resolution is not None:
        target_names.append(local_resolution["wandb_run_name"])
        resolution_info = local_resolution
    elif args.run_name:
        target_names.append(args.run_name)
        resolution_info = {
            "source": "explicit_name",
            "requested_name": args.run_name,
        }
    else:
        log_progress("Querying latest WandB run because no target name was resolved.")
        runs = api.runs(project_path, order="-created_at")
        run = next(iter(runs), None)
        if run is None:
            raise SystemExit(f"No WandB runs found under `{project_path}`.")
        resolution_info = {
            "source": "wandb_latest",
            "requested_name": "latest",
        }
    if target_names:
        run = find_run_by_name(api, project_path, target_names)
        if run is None and not (args.sync and local_resolution is not None):
            raise SystemExit(
                f"Could not find a WandB run under `{project_path}` matching any of: {target_names}"
            )
    if run is not None:
        log_progress(f"Resolved WandB run: {run.name} (id={getattr(run, 'id', None)})")
    else:
        log_progress("No existing WandB run matched. Continuing with local TensorBoard context for export/sync.")

    log_progress("Fetching summary/config payloads.")
    summary_payload = dict(getattr(run, "summary", {})) if run is not None else {}
    config_payload = dict(getattr(run, "config", {})) if run is not None else {}
    allowed_history_keys = build_allowed_history_keys(summary_payload)
    log_progress(f"Keeping {len(allowed_history_keys)} history keys after summary-based filtering.")
    history_bundle = None
    if local_resolution is not None:
        event_files = find_local_event_files(local_resolution["local_run_dir"])
        if event_files:
            log_progress(f"Found {len(event_files)} local TensorBoard event file(s).")
            history_bundle = fetch_history_rows_from_tensorboard(
                local_resolution["local_run_dir"],
                allowed_history_keys,
                args.history_stride,
            )
        else:
            log_progress("No local TensorBoard event files found. Falling back to WandB API history.")
    if history_bundle is None:
        log_progress("Fetching history through WandB API fallback.")
        history_bundle = fetch_history_rows(run, allowed_history_keys, args.history_stride)
    history_rows = history_bundle["history"]

    output_run_name = getattr(run, "name", None) or local_resolution["wandb_run_name"]
    run_metadata = {
        "entity": entity,
        "project": args.project,
        "project_path": project_path,
        "run_id": getattr(run, "id", None) if run is not None else None,
        "run_name": getattr(run, "name", None) if run is not None else output_run_name,
        "run_display_name": getattr(run, "display_name", None) if run is not None else None,
        "run_url": getattr(run, "url", None) if run is not None else None,
        "run_state": getattr(run, "state", None) if run is not None else None,
        "created_at": getattr(run, "created_at", None) if run is not None else None,
        "resolution": resolution_info,
        "history_stride": args.history_stride,
        "history_keys": history_bundle["available_iteration_keys"],
        "ignored_time_axis_keys": history_bundle["ignored_time_keys"],
        "history_source": history_bundle["source"],
        "iteration_history_keys": history_bundle["available_iteration_keys"],
        "history_row_count": len(history_rows),
        "exported_at": datetime.now().isoformat(),
    }

    if args.sync:
        sync_result = sync_tensorboard_history_to_wandb(
            wandb=wandb,
            entity=entity,
            project=args.project,
            local_resolution=local_resolution,
            base_run_name=output_run_name,
            config_payload=config_payload,
            summary_payload=summary_payload,
        )
        run_metadata["wandb_sync"] = sync_result

    run_output_dir = None
    if not args.no_fetch:
        output_dir = Path(args.output_dir).expanduser().resolve()
        run_output_dir = output_dir / safe_dir_name(output_run_name)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_progress(f"Writing exported files to {run_output_dir}")

        write_json(run_output_dir / "run_info.json", run_metadata)
        write_json(run_output_dir / "summary.json", summary_payload)
        write_json(run_output_dir / "config.json", config_payload)
        write_history_jsonl(run_output_dir / "history.jsonl", history_rows)
        write_history_csv(run_output_dir / "history.csv", history_rows)
        write_json(
            run_output_dir / "ai_ready.json",
            {
                "run_info": run_metadata,
                "summary": summary_payload,
                "config": config_payload,
                "history": history_rows,
            },
        )
        log_progress("All export files written successfully.")
    else:
        log_progress("Local fetch/export skipped because --no_fetch was set.")

    print(f"Resolved WandB run : {output_run_name}")
    print(f"Project path       : {project_path}")
    print(f"Output directory   : {run_output_dir if run_output_dir is not None else '<skipped by --no_fetch>'}")
    print(f"History source     : {history_bundle['source']}")
    print(f"History rows       : {len(history_rows)} (stride={args.history_stride})")
    print(f"History keys       : {len(history_bundle['available_iteration_keys'])} exported on iteration axis")
    print(f"Ignored /time keys : {len(history_bundle['ignored_time_keys'])}")
    if args.sync:
        print(f"Synced WandB run   : {run_metadata['wandb_sync']['run_name']}")
        print(f"Sync row count     : {run_metadata['wandb_sync']['history_row_count']}")


if __name__ == "__main__":
    main()
