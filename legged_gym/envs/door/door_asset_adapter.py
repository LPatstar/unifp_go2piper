import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List

from legged_gym import LEGGED_GYM_ROOT_DIR


@dataclass(frozen=True)
class DoorAssetSpec:
    """Asset metadata used by the high-level door task.

    This object intentionally carries metadata only. Isaac Gym actor creation
    belongs in the high-level environment, and low-level UniFP control should
    not depend on any of these fields.
    """

    name: str
    root_dir: str
    urdf_path: str
    bounding_box: Dict
    handle_bounding: Dict
    dof_lower: List[float]
    dof_upper: List[float]


def resolve_asset_root(root: str) -> str:
    return root.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)


def load_door_asset_specs(asset_root: str, asset_names: List[str]) -> List[DoorAssetSpec]:
    resolved_root = resolve_asset_root(asset_root)
    specs = []
    for name in asset_names:
        asset_dir = os.path.join(resolved_root, name)
        bounding_path = os.path.join(asset_dir, "bounding_box.json")
        handle_path = os.path.join(asset_dir, "handle_bounding.json")
        urdf_path = os.path.join(asset_dir, "mobility.urdf")
        with open(bounding_path, "r", encoding="utf-8") as f:
            bounding_box = json.load(f)
        with open(handle_path, "r", encoding="utf-8") as f:
            handle_bounding = json.load(f)
        dof_lower, dof_upper = parse_urdf_joint_limits(urdf_path)
        specs.append(
            DoorAssetSpec(
                name=name,
                root_dir=asset_dir,
                urdf_path=urdf_path,
                bounding_box=bounding_box,
                handle_bounding=handle_bounding,
                dof_lower=dof_lower,
                dof_upper=dof_upper,
            )
        )
    return specs


def parse_urdf_joint_limits(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    lower = []
    upper = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            lower.append(0.0)
            upper.append(0.0)
            continue
        lower.append(float(limit.attrib.get("lower", 0.0)))
        upper.append(float(limit.attrib.get("upper", 0.0)))
    return lower, upper

