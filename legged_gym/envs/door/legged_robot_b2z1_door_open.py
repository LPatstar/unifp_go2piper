import os
from collections import deque

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    to_torch,
)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfgPPO
from legged_gym.envs.b2.legged_robot_b2z1_pos_force import (
    LeggedRobot_b2z1_pos_force,
    euler_from_quat,
    get_euler_xyz_tensor,
    sphere2cart,
    torch_rand_float,
)
from legged_gym.envs.door.b2z1_door_open_config import B2Z1DoorOpenRoughCfg
from legged_gym.envs.door.door_asset_adapter import load_door_asset_specs
from legged_gym.envs.door.unifp_low_level_adapter import UniFPLowLevelCommandAdapter
from legged_gym.utils.helpers import class_to_dict, get_load_path


def _wrap_to_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def _orientation_error(desired, current):
    quat_diff = quat_mul(desired, quat_conjugate(current))
    return quat_diff[:, 0:3] * torch.sign(quat_diff[:, 3]).unsqueeze(-1)


class LeggedRobot_b2z1_door_open(LeggedRobot_b2z1_pos_force):
    """High-level B2+Z1 door-opening task backed by a frozen UniFP low-level.

    The class intentionally has two public contracts:

    - high-level contract exposed to PPO: 9-D door actions and compact state obs
    - low-level contract hidden behind ``UniFPLowLevelCommandAdapter``: UniFP
      command buffers plus a frozen 17-D low-level policy

    The current migration keeps the physical door actor and its tensors inside
    this class. The frozen UniFP low-level continues to see robot-only state and
    command buffers through the adapter.
    """

    def __init__(self, cfg: B2Z1DoorOpenRoughCfg, sim_params, physics_engine, sim_device, headless):
        self._door_high_level_ready = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._low_level_action_dim = int(self.actions.shape[1])
        self._low_level_num_obs = int(self.obs_buf.shape[1])
        self._low_level_num_privileged_obs = int(self.privileged_obs_buf.shape[1])
        self._low_level_num_pred_obs = int(self.obs_pred.shape[1])
        self._low_level_num_single_obs = int(self.cfg.env.num_single_obs)
        self._low_level_obs_buf = torch.zeros(
            self.num_envs, self._low_level_num_obs, dtype=torch.float, device=self.device
        )
        self._low_level_privileged_obs_buf = torch.zeros(
            self.num_envs, self._low_level_num_privileged_obs, dtype=torch.float, device=self.device
        )
        self._low_level_obs_pred = torch.zeros(
            self.num_envs, self._low_level_num_pred_obs, dtype=torch.float, device=self.device
        )

        self.low_level_command_adapter = UniFPLowLevelCommandAdapter(self, self.cfg.high_level)
        self.low_level_command_adapter.configure_external_control()
        self._ensure_force_gain_defaults()
        self.low_level_policy = self._build_low_level_policy()

        self._install_high_level_public_buffers()
        self._init_door_task_buffers()
        self._prepare_door_reward_function()
        self._door_high_level_ready = True

    def _install_high_level_public_buffers(self):
        self.num_actions = int(self.cfg.high_level.num_actions)
        self.num_obs = int(self.cfg.high_level.num_observations)
        self.num_privileged_obs = int(self.cfg.high_level.num_privileged_obs)
        self.num_pred_obs = int(self.cfg.high_level.num_pred_obs)
        self.num_single_obs = int(self.cfg.high_level.num_single_obs)
        self._high_level_current_obs_dim = int(getattr(self.cfg.high_level, "num_current_obs", self.num_obs))
        self._high_level_reserved_obs_dim = int(getattr(self.cfg.high_level, "num_reserved_obs", 0))
        if self._high_level_current_obs_dim + self._high_level_reserved_obs_dim != self.num_obs:
            raise ValueError(
                "b2z1_door_open observation contract mismatch: "
                f"current({self._high_level_current_obs_dim}) + "
                f"reserved({self._high_level_reserved_obs_dim}) != num_obs({self.num_obs})"
            )

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device
        )
        self.obs_pred = torch.zeros(self.num_envs, self.num_pred_obs, dtype=torch.float, device=self.device)
        self.high_level_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.high_level_last_actions = torch.zeros_like(self.high_level_actions)

    def _build_low_level_policy(self):
        mode = str(getattr(self.cfg.low_level, "policy_mode", "checkpoint")).lower()
        if mode in {"zero", "zeros", "none"}:
            self.low_level_policy_path = None
            return None
        if mode != "checkpoint":
            raise ValueError(f"Unsupported low-level policy_mode: {mode}")

        policy_cfg = class_to_dict(B2Z1PosForceRoughCfgPPO.policy)
        actor_critic = ActorCritic(
            self._low_level_num_obs,
            self._low_level_num_privileged_obs,
            self._low_level_num_pred_obs,
            self._low_level_num_single_obs,
            self._low_level_action_dim,
            **policy_cfg,
        ).to(self.device)

        log_root = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            getattr(self.cfg.low_level, "experiment_name", "b2z1_pos_force"),
        )
        try:
            load_path = get_load_path(
                log_root,
                load_run=getattr(self.cfg.low_level, "load_run", -1),
                checkpoint=getattr(self.cfg.low_level, "checkpoint", -1),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve the frozen low-level checkpoint. "
                "Set cfg.low_level.load_run/checkpoint, or use "
                "--low_level_policy_mode zero for adapter-only smoke tests."
            ) from exc

        loaded = torch.load(load_path, map_location=self.device)
        actor_critic.load_state_dict(loaded["model_state_dict"])
        actor_critic.eval()
        for param in actor_critic.parameters():
            param.requires_grad_(False)
        self.low_level_policy_path = load_path
        return actor_critic.act_inference

    def _use_physical_door(self):
        return bool(getattr(self.cfg.door, "use_physical_actor", False))

    def _load_door_asset_specs_once(self):
        if not hasattr(self, "door_asset_specs"):
            self.door_asset_specs = load_door_asset_specs(self.cfg.door.asset_root, list(self.cfg.door.asset_names))
            self.door_asset_names = [spec.name for spec in self.door_asset_specs]
        return self.door_asset_specs

    def _ensure_force_gain_defaults(self):
        """Keep UniFP force-observation math finite when force randomization is disabled."""
        gripper_kp = float(getattr(self.cfg.commands, "gripper_force_kp_range", [200.0, 200.0])[0])
        base_kp = float(getattr(self.cfg.commands, "base_force_kp_range", [200.0, 200.0])[0])
        gripper_kd_range = getattr(self.cfg.commands, "gripper_force_kd_range", [50.0, 50.0])
        base_kd_range = getattr(self.cfg.commands, "base_force_kd_range", [50.0, 50.0])
        gripper_kd = gripper_kp * float(getattr(self.cfg.commands, "gripper_prop_kd", 0.0))
        if gripper_kd <= 0.0:
            gripper_kd = float(gripper_kd_range[0])
        base_kd = float(base_kd_range[0])
        self.gripper_force_kps[:] = max(gripper_kp, 1e-6)
        self.gripper_force_kds[:] = max(gripper_kd, 1e-6)
        self.base_force_kps[:] = max(base_kp, 1e-6)
        self.base_force_kds[:] = max(base_kd, 1e-6)

    def _create_envs(self):
        if not self._use_physical_door():
            return super()._create_envs()

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]
        self.robot_base_idx = self.body_names_to_idx[self.cfg.asset.base_name]
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.mass_randomized_body_indices = [
            idx for idx in range(self.num_bodies) if idx not in {self.robot_base_idx, self.gripper_idx}
        ][: self.num_torques]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        door_options = gymapi.AssetOptions()
        door_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        door_options.collapse_fixed_joints = False
        door_options.fix_base_link = bool(getattr(self.cfg.door, "door_fix_base_link", True))
        door_options.disable_gravity = bool(getattr(self.cfg.door, "door_disable_gravity", True))
        door_options.use_mesh_materials = True
        door_options.override_com = True
        door_options.override_inertia = True
        door_options.thickness = 0.001

        self.door_asset_list = []
        self.door_asset_body_names = []
        self.door_asset_dof_names = []
        self.door_asset_body_counts = []
        self.door_asset_dof_counts = []
        for spec in self._load_door_asset_specs_once():
            door_asset = self.gym.load_asset(self.sim, spec.root_dir, os.path.basename(spec.urdf_path), door_options)
            shape_props = self.gym.get_asset_rigid_shape_properties(door_asset)
            for prop in shape_props:
                if hasattr(prop, "friction"):
                    prop.friction = float(getattr(self.cfg.door, "door_static_friction", 2.0))
                if hasattr(prop, "rolling_friction"):
                    prop.rolling_friction = float(getattr(self.cfg.door, "door_dynamic_friction", 1.5))
            self.gym.set_asset_rigid_shape_properties(door_asset, shape_props)
            self.door_asset_list.append(door_asset)
            self.door_asset_body_names.append(self.gym.get_asset_rigid_body_names(door_asset))
            self.door_asset_dof_names.append(self.gym.get_asset_dof_names(door_asset))
            self.door_asset_body_counts.append(self.gym.get_asset_rigid_body_count(door_asset))
            self.door_asset_dof_counts.append(self.gym.get_asset_dof_count(door_asset))

        if len(set(self.door_asset_body_counts)) != 1 or len(set(self.door_asset_dof_counts)) != 1:
            raise RuntimeError("Door assets must have identical body and DOF counts for batched Isaac Gym tensors.")
        self.num_door_bodies = int(self.door_asset_body_counts[0])
        self.num_door_dofs = int(self.door_asset_dof_counts[0])
        self.num_actors_per_env = 2
        self.num_bodies_per_env = self.num_bodies + self.num_door_bodies
        self.dof_per_env = self.num_dof + self.num_door_dofs

        base_init_state_list = (
            self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.door_handles = []
        self.door_actor_spec_ids = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.mass_params_tensor = torch.zeros(
            self.num_envs, self.mass_param_dim, dtype=torch.float, device=self.device, requires_grad=False
        )

        door_base_pos = torch.tensor(
            getattr(self.cfg.door, "asset_base_pos_env", [0.0, 0.0, 0.0]),
            dtype=torch.float,
            device=self.device,
        )
        closed_handle_pos = torch.tensor(
            getattr(self.cfg.door, "closed_handle_pos_env", [0.85, 0.0, 0.85]),
            dtype=torch.float,
            device=self.device,
        )
        handle_root_offsets_yaw0 = getattr(self.cfg.door, "closed_handle_root_offsets_yaw0", None)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

            spec_id = i % len(self.door_asset_list)
            door_asset = self.door_asset_list[spec_id]
            door_bounds = self.door_asset_specs[spec_id].bounding_box
            door_pose = gymapi.Transform()
            door_yaw = float(getattr(self.cfg.door, "door_yaw", np.pi))
            if handle_root_offsets_yaw0 is not None and spec_id < len(handle_root_offsets_yaw0):
                root_to_handle = torch.tensor(handle_root_offsets_yaw0[spec_id], dtype=torch.float, device=self.device)
                yaw_cos = np.cos(door_yaw)
                yaw_sin = np.sin(door_yaw)
                root_to_handle_world = torch.tensor(
                    [
                        yaw_cos * root_to_handle[0].item() - yaw_sin * root_to_handle[1].item(),
                        yaw_sin * root_to_handle[0].item() + yaw_cos * root_to_handle[1].item(),
                        root_to_handle[2].item(),
                    ],
                    dtype=torch.float,
                    device=self.device,
                )
                root_pos = self.env_origins[i] + closed_handle_pos - root_to_handle_world + door_base_pos
            else:
                root_pos = self.env_origins[i] + door_base_pos
                root_pos[2] = root_pos[2] - float(door_bounds["min"][2]) + 0.1
            door_pose.p = gymapi.Vec3(float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
            door_pose.r = gymapi.Quat(0.0, 0.0, float(np.sin(door_yaw / 2.0)), float(np.cos(door_yaw / 2.0)))
            door_handle = self.gym.create_actor(env_handle, door_asset, door_pose, "door", i, 0, 1)
            door_dof_props = self._configure_door_dof_props(self.gym.get_asset_dof_properties(door_asset))
            self.gym.set_actor_dof_properties(env_handle, door_handle, door_dof_props)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.door_handles.append(door_handle)
            self.door_actor_spec_ids.append(spec_id)

        self.robot_actor_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device) * self.num_actors_per_env
        self.door_actor_ids = self.robot_actor_ids + 1

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        first_door_body_names = self.door_asset_body_names[0]
        self.door_body_name = first_door_body_names[-2] if len(first_door_body_names) >= 2 else first_door_body_names[-1]
        self.handle_body_name = first_door_body_names[-1]
        self.door_body_idx = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.door_handles[0], self.door_body_name, gymapi.DOMAIN_ENV
        )
        self.handle_body_idx = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.door_handles[0], self.handle_body_name, gymapi.DOMAIN_ENV
        )

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat(
                [
                    torch_rand_float(
                        self.cfg.domain_rand.leg_motor_strength_range[0],
                        self.cfg.domain_rand.leg_motor_strength_range[1],
                        (self.num_envs, self.num_leg_dofs),
                        device=self.device,
                    ),
                    torch_rand_float(
                        self.cfg.domain_rand.arm_motor_strength_range[0],
                        self.cfg.domain_rand.arm_motor_strength_range[1],
                        (self.num_envs, self.num_arm_dofs),
                        device=self.device,
                    ),
                ],
                dim=1,
            )
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

    def _configure_door_dof_props(self, door_dof_props):
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        self._assign_door_dof_prop(door_dof_props, "stiffness", 0.0)
        self._assign_door_dof_prop(door_dof_props, "damping", getattr(self.cfg.door, "door_joint_damping", 0.1))
        self._assign_door_dof_prop(door_dof_props, "friction", getattr(self.cfg.door, "door_joint_friction", 0.02))
        self._assign_door_dof_prop(door_dof_props, "effort", getattr(self.cfg.door, "door_joint_effort", 80.0))
        if len(door_dof_props["lower"]) >= 1 and len(door_dof_props["upper"]) >= 1:
            hinge_range = max(abs(float(door_dof_props["lower"][0])), abs(float(door_dof_props["upper"][0])))
            hinge_range = max(hinge_range, 1e-3)
            if self._door_hinge_open_sign() < 0.0:
                door_dof_props["lower"][0] = -hinge_range
                door_dof_props["upper"][0] = 0.0
            else:
                door_dof_props["lower"][0] = 0.0
                door_dof_props["upper"][0] = hinge_range
        if len(door_dof_props["upper"]) >= 2:
            door_dof_props["upper"][1] = min(
                float(door_dof_props["upper"][1]), float(getattr(self.cfg.door, "handle_upper_limit", np.pi / 4))
            )
        return door_dof_props

    def _door_hinge_open_sign(self):
        return -1.0 if float(getattr(self.cfg.door, "door_hinge_open_sign", 1.0)) < 0.0 else 1.0

    def _assign_door_dof_prop(self, props, name, values):
        if name not in props.dtype.names:
            return
        if isinstance(values, (list, tuple)):
            count = min(len(values), len(props[name]))
            props[name][:count] = np.asarray(values[:count], dtype=props[name].dtype)
            if count < len(props[name]):
                props[name][count:] = float(values[-1])
        else:
            props[name][:] = float(values)

    def _init_buffers(self):
        if not self._use_physical_door():
            return super()._init_buffers()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._all_root_states_tensor = gymtorch.wrap_tensor(actor_root_state)
        self._all_root_states = self._all_root_states_tensor.view(self.num_envs, self.num_actors_per_env, 13)
        self.root_states = self._all_root_states[:, 0, :]
        self.door_root_states = self._all_root_states[:, 1, :]

        self._all_dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)
        self._all_dof_state = self._all_dof_state_tensor.view(self.num_envs, self.dof_per_env, 2)
        self.dof_state = self._all_dof_state[:, : self.num_dof, :]
        self.dof_pos = self.dof_state[..., 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self.dof_state[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self._door_dof_pos = self._all_dof_state[:, self.num_dof : self.dof_per_env, 0]
        self._door_dof_vel = self._all_dof_state[:, self.num_dof : self.dof_per_env, 1]

        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat(
            [torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1
        )
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.arm_base_offset = torch.tensor(self.cfg.arm.base_offset, device=self.device, dtype=torch.float).repeat(
            self.num_envs, 1
        )
        self._full_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, self.num_bodies_per_env, 3
        )
        self.contact_forces = self._full_contact_forces[:, : self.num_bodies, :]
        self._full_rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies_per_env, 13
        )
        self.rigid_state = self._full_rigid_state[:, : self.num_bodies, :]
        self.initial_door_root_states = self.door_root_states.clone()

        self.gripper_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 0:3]
        self.gripper_velocity = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 7:10]
        self.ee_pos = self.rigid_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_state[:, self.gripper_idx, 7:]

        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)
        self.traj_timesteps = (
            torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device)
            .squeeze(1)
            / self.dt
        )
        self.traj_total_timesteps = self.traj_timesteps + (
            torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device)
            .squeeze(1)
            / self.dt
        )
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2]
        )
        self.curr_ee_goal_orn_rpy = self.ee_goal_orn_euler.clone()
        self.ee_start_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_pos_sphe_arm = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert self.cfg.goal_ee.command_mode in ["cart", "sphere"]
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor(
            [
                self.cfg.goal_ee.sphere_center.x_offset,
                self.cfg.goal_ee.sphere_center.y_offset,
                self.cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + quat_apply(
            self.base_yaw_quat, self.curr_ee_goal_cart
        )

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.dof_per_env, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.control_actions = torch.zeros_like(self.actions)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros_like(self.torques)
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.ee_sphe_radius_cmd,
                self.obs_scales.ee_sphe_pitch_cmd,
                self.obs_scales.ee_sphe_yaw_cmd,
                self.obs_scales.end_effector_roll_cmd,
                self.obs_scales.end_effector_pitch_cmd,
                self.obs_scales.end_effector_yaw_cmd,
                self.obs_scales.ee_force,
                self.obs_scales.ee_force,
                self.obs_scales.ee_force,
                self.obs_scales.base_force,
                self.obs_scales.base_force,
                self.obs_scales.base_force,
            ],
            device=self.device,
            requires_grad=False,
        )

        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device)
            )

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]

        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.gripper_p_gains = torch.zeros(
            self.cfg.env.num_gripper_joints, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.gripper_d_gains = torch.zeros(
            self.cfg.env.num_gripper_joints, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.cfg.env.num_gripper_joints):
            name = self.dof_names[self.num_torques + i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.gripper_p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.gripper_d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found and self.cfg.control.control_type in ["P", "V"]:
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:, :-self.cfg.env.num_gripper_joints]

        self.enable_arm_pd_equivalent_action_remap = bool(
            getattr(self.cfg.control, "enable_arm_pd_equivalent_action_remap", False)
        )
        self.arm_pd_remap_old_p_gains = self.p_gains[self.arm_dof_slice].clone()
        self.arm_pd_remap_old_d_gains = self.d_gains[self.arm_dof_slice].clone()
        self.arm_pd_remap_old_gripper_p_gains = self.gripper_p_gains.clone()
        self.arm_pd_remap_old_gripper_d_gains = self.gripper_d_gains.clone()
        if self.enable_arm_pd_equivalent_action_remap:
            old_stiffness = getattr(self.cfg.control, "arm_pd_remap_old_stiffness", {})
            old_damping = getattr(self.cfg.control, "arm_pd_remap_old_damping", {})
            for local_idx, dof_idx in enumerate(range(self.num_leg_dofs, self.num_torques)):
                dof_name = self.dof_names[dof_idx]
                self.arm_pd_remap_old_p_gains[local_idx] = self._resolve_pd_remap_gain(
                    old_stiffness, dof_name, self.p_gains[dof_idx].item()
                )
                self.arm_pd_remap_old_d_gains[local_idx] = self._resolve_pd_remap_gain(
                    old_damping, dof_name, self.d_gains[dof_idx].item()
                )
            for local_idx, dof_idx in enumerate(range(self.num_torques, self.num_torques + self.cfg.env.num_gripper_joints)):
                dof_name = self.dof_names[dof_idx]
                self.arm_pd_remap_old_gripper_p_gains[local_idx] = self._resolve_pd_remap_gain(
                    old_stiffness, dof_name, self.gripper_p_gains[local_idx].item()
                )
                self.arm_pd_remap_old_gripper_d_gains[local_idx] = self._resolve_pd_remap_gain(
                    old_damping, dof_name, self.gripper_d_gains[local_idx].item()
                )

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.freed_envs_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_gripper_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_gripper_cmd = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False
        )
        self.selected_env_ids_gripper_ext = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False
        )
        self.push_interval_gripper_cmd = torch.randint(
            int(self.push_interval_gripper_cmd_min),
            int(self.push_interval_gripper_cmd_max),
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        )
        self.push_interval_gripper_ext = torch.randint(
            int(self.push_interval_gripper_ext_min),
            int(self.push_interval_gripper_ext_max),
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        )
        self.push_end_time_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_cmd = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.settling_time_force_gripper_s = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.push_end_time_gripper_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_ext = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.force_target_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_gripper_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.freed_envs_base_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_base_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_base_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_base_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval_base_cmd = torch.randint(
            int(self.push_interval_base_cmd_min),
            int(self.push_interval_base_cmd_max),
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        )
        self.push_interval_base_ext = torch.randint(
            int(self.push_interval_base_ext_min),
            int(self.push_interval_base_ext_max),
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        )
        self.push_end_time_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_base_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_end_time_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.forces = torch.zeros(
            self.num_envs, self.num_bodies_per_env, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.forces_local = torch.zeros(
            self.num_envs, self.num_bodies_per_env, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.key_lin_vel_step = 0.05
        self.key_ang_vel_step = 0.08
        self.key_ee_cart_step = 0.02
        self.key_ee_orn_step = 0.05
        self.key_ee_orn_limit = 1.2
        self.key_ee_force_step = 5.0
        self.key_base_force_step = 5.0
        self.key_command_ee_local_cart = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.key_command_ee_orn_delta_rpy = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.key_command_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.global_steps = 0

    def _reset_dofs(self, env_ids):
        if not self._use_physical_door():
            return super()._reset_dofs(env_ids)

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids, : self.num_leg_dofs] = self.default_dof_pos[:, : self.num_leg_dofs] * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_leg_dofs), device=self.device
        )
        if self.num_torques > self.num_leg_dofs:
            self.dof_pos[env_ids, self.num_leg_dofs : self.num_torques] += torch_rand_float(
                -0.5, 0.5, (len(env_ids), self.num_torques - self.num_leg_dofs), device=self.device
            )
        self.dof_vel[env_ids] = 0.0
        self._door_dof_pos[env_ids] = 0.0
        self._door_dof_vel[env_ids] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._all_dof_state_tensor))
        self.gym.refresh_dof_state_tensor(self.sim)

    def _reset_root_states(self, env_ids):
        if not self._use_physical_door():
            return super()._reset_root_states(env_ids)

        self.root_states[env_ids] = self.base_init_state
        robot_start_pos = torch.tensor(
            getattr(self.cfg.door, "robot_start_pos_env", self.cfg.init_state.pos),
            dtype=torch.float,
            device=self.device,
        )
        self.root_states[env_ids, :3] = self.env_origins[env_ids] + robot_start_pos
        xy_noise = float(getattr(self.cfg.door, "robot_start_xy_noise", 0.0))
        if xy_noise > 0.0:
            self.root_states[env_ids, :2] += torch_rand_float(
                -xy_noise, xy_noise, (len(env_ids), 2), device=self.device
            )

        yaw = float(getattr(self.cfg.door, "robot_start_yaw", 0.0))
        yaw_noise = float(getattr(self.cfg.door, "robot_start_yaw_noise", 0.0))
        rand_yaw = yaw + yaw_noise * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(0 * rand_yaw, 0 * rand_yaw, rand_yaw)
        vel_noise = float(getattr(self.cfg.door, "robot_start_vel_noise", 0.0))
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -vel_noise, vel_noise, (len(env_ids), 6), device=self.device
        )
        self.door_root_states[env_ids] = self.initial_door_root_states[env_ids]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._all_root_states_tensor))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _compute_torques(self, actions):
        robot_torques = super()._compute_torques(actions)
        if not self._use_physical_door():
            return robot_torques

        full_torques = torch.zeros(self.num_envs, self.dof_per_env, dtype=torch.float, device=self.device)
        full_torques[:, : self.num_dof] = robot_torques
        full_torques[:, self.num_dof : self.dof_per_env] = self._compute_door_torques()
        return full_torques

    def _compute_door_torques(self):
        door_torques = torch.zeros(self.num_envs, self.num_door_dofs, dtype=torch.float, device=self.device)
        if self.num_door_dofs < 2:
            return door_torques

        hinge_sign = self._door_hinge_open_sign()
        door_angle = self._door_dof_pos[:, 0]
        opening_angle = torch.clamp(hinge_sign * door_angle, min=0.0)
        opening_vel = hinge_sign * self._door_dof_vel[:, 0]
        handle_angle_from_lower = self._door_dof_pos[:, 1] - self.door_dof_lower[:, 1]
        handle_threshold = float(self.cfg.door.handle_press_threshold_ratio) * torch.clamp(
            self.door_dof_upper[:, 1] - self.door_dof_lower[:, 1], min=1e-3
        )
        unlock_now = handle_angle_from_lower >= handle_threshold
        if not bool(getattr(self.cfg.door, "hard_lock_before_handle_threshold", True)):
            unlock_now |= opening_angle > 0.01
        self.open_door_stage[:] = self.open_door_stage | unlock_now
        door_torques[:, 0] = torch.where(
            self.open_door_stage,
            -hinge_sign
            * (
                float(getattr(self.cfg.door, "door_open_resistance", 3.0)) * opening_angle
                + float(getattr(self.cfg.door, "door_open_damping", 0.5)) * opening_vel
            ),
            -hinge_sign * float(getattr(self.cfg.door, "door_lock_force", 150.0)) * opening_angle,
        )
        door_torques[:, 1] = (
            -float(getattr(self.cfg.door, "handle_spring_stiffness", 40.0)) * self._door_dof_pos[:, 1]
            - float(getattr(self.cfg.door, "handle_spring_damping", 2.0)) * self._door_dof_vel[:, 1]
        )
        return door_torques

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self._use_physical_door() and getattr(self, "_door_high_level_ready", False):
            self._update_door_derived_state()

    def step(self, actions):
        if actions.shape[1] != self.num_actions:
            raise ValueError(f"b2z1_door_open expects {self.num_actions} high-level actions, got {actions.shape[1]}")

        action_clip = float(getattr(self.cfg.high_level, "action_clip", self.cfg.normalization.clip_actions))
        self.high_level_actions[:] = torch.clamp(actions.to(self.device), -action_clip, action_clip)

        command = self.low_level_command_adapter.command_from_high_level_action(self.high_level_actions)
        self.low_level_command_adapter.apply_command(command)
        low_level_actions = self._compute_low_level_actions()

        obs_dict, rewards, dones, infos = super().step(low_level_actions)

        done_mask = dones.to(dtype=torch.bool)
        self.high_level_last_actions[~done_mask] = self.high_level_actions[~done_mask]
        self.high_level_last_actions[done_mask] = 0.0
        return obs_dict, rewards, dones, infos

    def _compute_low_level_actions(self):
        if self.low_level_policy is None:
            return torch.zeros(
                self.num_envs, self._low_level_action_dim, dtype=torch.float, device=self.device
            )

        policy_obs = self.low_level_command_adapter.patch_latest_command_observation(self._low_level_obs_buf)
        policy_input = {
            "obs": policy_obs,
            "privileged_obs": self._low_level_privileged_obs_buf,
            "obs_pred": self._low_level_obs_pred,
        }
        with torch.no_grad():
            return self.low_level_policy(policy_input).to(self.device)

    def compute_observations(self):
        if not getattr(self, "_door_high_level_ready", False):
            return super().compute_observations()

        super().compute_observations()
        self._low_level_obs_buf = self.obs_buf.clone()
        self._low_level_privileged_obs_buf = self.privileged_obs_buf.clone()
        self._low_level_obs_pred = self.obs_pred.clone()

        high_level_obs = self._compute_high_level_observations()
        self.obs_buf = high_level_obs
        self.privileged_obs_buf = high_level_obs.clone()
        self.obs_pred = torch.zeros(self.num_envs, self.num_pred_obs, dtype=torch.float, device=self.device)

    def check_termination(self):
        super().check_termination()
        if not getattr(self, "_door_high_level_ready", False):
            return
        if bool(getattr(self.cfg.door, "terminate_on_reach", False)):
            self.reset_buf |= self._compute_ee_handle_distance() < float(self.cfg.door.success_distance)
        self.reset_buf |= self.door_open_hold_counter >= int(self.cfg.door.door_open_hold_steps)
        ee_far = (self.episode_length_buf > int(getattr(self.cfg.door, "ee_far_grace_steps", 40))) & (
            self.curr_dist > float(getattr(self.cfg.door, "ee_far_threshold", 1.1))
        )
        base_far = (self.episode_length_buf > int(getattr(self.cfg.door, "base_far_grace_steps", 80))) & (
            self.base_door_dis > float(getattr(self.cfg.door, "base_far_threshold", 2.0))
        )
        self.reset_buf |= ee_far | base_far

    def reset_idx(self, env_ids):
        door_episode_metrics = None
        if getattr(self, "_door_high_level_ready", False) and len(env_ids) > 0:
            door_episode_metrics = self._compute_door_episode_metrics(env_ids)

        super().reset_idx(env_ids)
        if not getattr(self, "_door_high_level_ready", False) or len(env_ids) == 0:
            return

        if door_episode_metrics is not None:
            self.extras.setdefault("episode", {}).update(door_episode_metrics)
        self.high_level_actions[env_ids] = 0.0
        self.high_level_last_actions[env_ids] = 0.0
        self._reset_door_task_buffers(env_ids)
        self.low_level_command_adapter.reset_command(env_ids)

    def _compute_door_episode_metrics(self, env_ids):
        closest_dist = torch.where(
            self.closest_dist[env_ids] >= 0.0,
            self.closest_dist[env_ids],
            self.curr_dist[env_ids],
        )
        return {
            "door_success_rate": self.door_open_success[env_ids].to(dtype=torch.float).mean(),
            "door_open_ratio": self.door_open_ratio[env_ids].mean(),
            "handle_open_ratio": self.handle_open_ratio[env_ids].mean(),
            "open_stage_rate": self.open_door_stage[env_ids].to(dtype=torch.float).mean(),
            "door_hold_steps": self.door_open_hold_counter[env_ids].to(dtype=torch.float).mean(),
            "ee_handle_dist": self.curr_dist[env_ids].mean(),
            "closest_ee_handle_dist": closest_dist.mean(),
            "base_door_dist": self.base_door_dis[env_ids].mean(),
        }

    def _init_door_task_buffers(self):
        self._load_door_asset_specs_once()
        if self._use_physical_door() and hasattr(self, "door_actor_spec_ids"):
            self.door_asset_indices = torch.tensor(self.door_actor_spec_ids, dtype=torch.long, device=self.device)
        else:
            self.door_asset_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device) % len(self.door_asset_specs)
        self.door_asset_names = [spec.name for spec in self.door_asset_specs]

        goal_offsets = [spec.handle_bounding["goal_pos"] for spec in self.door_asset_specs]
        self.goal_pos_offset_tensor_all = torch.tensor(goal_offsets, dtype=torch.float, device=self.device)
        self.goal_pos_offset_tensor = self.goal_pos_offset_tensor_all[self.door_asset_indices]

        dof_lower = [spec.dof_lower[:2] for spec in self.door_asset_specs]
        dof_upper = [spec.dof_upper[:2] for spec in self.door_asset_specs]
        self.door_dof_lower_all = torch.tensor(dof_lower, dtype=torch.float, device=self.device)
        self.door_dof_upper_all = torch.tensor(dof_upper, dtype=torch.float, device=self.device)
        hinge_range = torch.clamp(
            torch.maximum(torch.abs(self.door_dof_lower_all[:, 0]), torch.abs(self.door_dof_upper_all[:, 0])),
            min=1e-3,
        )
        if self._door_hinge_open_sign() < 0.0:
            self.door_dof_lower_all[:, 0] = -hinge_range
            self.door_dof_upper_all[:, 0] = 0.0
        else:
            self.door_dof_lower_all[:, 0] = 0.0
            self.door_dof_upper_all[:, 0] = hinge_range
        if self._use_physical_door() and self.door_dof_upper_all.shape[1] >= 2:
            handle_upper = float(getattr(self.cfg.door, "handle_upper_limit", np.pi / 4))
            self.door_dof_upper_all[:, 1] = torch.clamp(self.door_dof_upper_all[:, 1], max=handle_upper)
        self.door_dof_lower = self.door_dof_lower_all[self.door_asset_indices]
        self.door_dof_upper = self.door_dof_upper_all[self.door_asset_indices]

        self._door_handle_closed_pos_env = torch.tensor(
            self.cfg.door.closed_handle_pos_env, dtype=torch.float, device=self.device
        ).view(1, 3)
        yaw = torch.full((self.num_envs,), float(self.cfg.door.door_yaw), dtype=torch.float, device=self.device)
        zeros = torch.zeros_like(yaw)
        self.door_yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
        self._handle_approach_dir_local = self._door_vec(self.cfg.door.handle_approach_dir_local)
        self._handle_rotate_dir_local = self._door_vec(self.cfg.door.handle_rotate_dir_local)
        self._door_open_dir_local = self._door_vec(self.cfg.door.door_open_dir_local)

        self.door_handle_closed_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.door_handle_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.grasp_goal_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.pregrasp_goal_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.handle_approach_dir_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.handle_rotate_dir_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.door_open_dir_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.handle_target_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.handle_open_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.door_open_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.best_handle_open_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.best_door_open_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.open_door_stage = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.door_open_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.door_open_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_door_dis = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not self._use_physical_door():
            self._door_dof_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
            self._door_dof_vel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.prev_ee_handle_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.curr_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.closest_dist = torch.full((self.num_envs,), -1.0, dtype=torch.float, device=self.device)
        self._reset_door_task_buffers(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def _door_vec(self, values):
        return torch.tensor(values, dtype=torch.float, device=self.device).view(1, 3).repeat(self.num_envs, 1)

    def _reset_door_task_buffers(self, env_ids):
        self.door_handle_closed_pos_world[env_ids] = self.env_origins[env_ids] + self._door_handle_closed_pos_env
        self.handle_approach_dir_world[env_ids] = quat_apply(self.door_yaw_quat[env_ids], self._handle_approach_dir_local[env_ids])
        self.handle_rotate_dir_world[env_ids] = quat_apply(self.door_yaw_quat[env_ids], self._handle_rotate_dir_local[env_ids])
        self.door_open_dir_world[env_ids] = quat_apply(self.door_yaw_quat[env_ids], self._door_open_dir_local[env_ids])
        self.handle_open_ratio[env_ids] = 0.0
        self.door_open_ratio[env_ids] = 0.0
        self.best_handle_open_ratio[env_ids] = 0.0
        self.best_door_open_ratio[env_ids] = 0.0
        self.open_door_stage[env_ids] = False
        self.door_open_success[env_ids] = False
        self.success_recorded[env_ids] = False
        self.door_open_hold_counter[env_ids] = 0
        self._door_dof_pos[env_ids] = 0.0
        self._door_dof_vel[env_ids] = 0.0
        self._update_door_derived_state(env_ids)
        if self._use_physical_door():
            self.door_handle_closed_pos_world[env_ids] = self.door_handle_pos_world[env_ids]
        self.prev_ee_handle_dist[env_ids] = self._compute_ee_handle_distance(env_ids)
        self.curr_dist[env_ids] = self.prev_ee_handle_dist[env_ids]
        self.closest_dist[env_ids] = self.curr_dist[env_ids]

    def _compute_high_level_observations(self):
        low_state = self.low_level_command_adapter.get_state()
        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        handle_rot = (
            self._full_rigid_state[:, self.handle_body_idx, 3:7]
            if self._use_physical_door()
            else self.door_yaw_quat
        )
        door_root_pos = self.door_root_states[:, :3] if self._use_physical_door() else self.door_handle_closed_pos_world

        goal_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.grasp_goal_world - arm_base_pos)
        handle_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.door_handle_pos_world - arm_base_pos)
        handle_rot_local = quat_mul(quat_conjugate(self.base_yaw_quat), handle_rot)
        handle_rot_local_rpy = torch.stack(euler_from_quat(handle_rot_local), dim=1)
        approach_dir_local = quat_rotate_inverse(self.base_yaw_quat, self.handle_approach_dir_world)
        rotate_dir_local = quat_rotate_inverse(self.base_yaw_quat, self.handle_rotate_dir_world)
        open_dir_local = quat_rotate_inverse(self.base_yaw_quat, self.door_open_dir_world)
        ee_pos_local = quat_rotate_inverse(self.base_quat, self.ee_pos - arm_base_pos)
        ee_rot_local = quat_mul(quat_conjugate(self.base_quat), self.ee_orn)
        ee_rot_local_rpy = torch.stack(euler_from_quat(ee_rot_local), dim=1)
        ee_to_handle_local = quat_rotate_inverse(self.base_quat, self.grasp_goal_world - self.ee_pos)
        door_root_local = quat_rotate_inverse(self.base_yaw_quat, door_root_pos - arm_base_pos)
        ee_handle_dist = self.curr_dist.unsqueeze(1)
        robot_dof_pos = ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[
            :, :-self.cfg.env.num_gripper_joints
        ]
        robot_dof_vel = (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints]
        low_level_position_command = low_state.command_buffer[:, :9] * self.commands_scale[:9]

        obs_parts = [
            goal_pos_local,
            handle_pos_local,
            handle_rot_local_rpy,
            approach_dir_local,
            rotate_dir_local,
            open_dir_local,
            ee_pos_local,
            ee_rot_local_rpy,
            ee_to_handle_local,
            door_root_local,
            self._door_dof_pos[:, 0:1],
            self._door_dof_pos[:, 1:2],
            self.door_open_ratio.unsqueeze(1),
            self.handle_open_ratio.unsqueeze(1),
            self.open_door_stage.to(dtype=torch.float).unsqueeze(1),
            self.base_door_dis.unsqueeze(1),
            ee_handle_dist,
            low_state.base_lin_vel_local,
            low_state.base_ang_vel_local,
            self.projected_gravity,
            robot_dof_pos,
            robot_dof_vel,
            low_level_position_command,
            low_state.ee_goal_local_cart,
            low_state.ee_goal_local_rpy,
            self.high_level_actions,
            self.high_level_last_actions,
        ]
        return self._pack_high_level_obs(torch.cat(obs_parts, dim=1))

    def _pack_high_level_obs(self, obs):
        if obs.shape[1] != self._high_level_current_obs_dim:
            raise RuntimeError(
                "b2z1_door_open current observation dim mismatch: "
                f"got {obs.shape[1]}, expected {self._high_level_current_obs_dim}. "
                "Update cfg.high_level.num_current_obs together with the explicit obs contract."
            )
        if self._high_level_reserved_obs_dim == 0:
            return obs

        reserved = torch.zeros(
            self.num_envs, self._high_level_reserved_obs_dim, dtype=torch.float, device=self.device
        )
        packed = torch.cat([obs, reserved], dim=1)
        if packed.shape[1] != self.num_obs:
            raise RuntimeError(
                f"b2z1_door_open packed observation dim mismatch: got {packed.shape[1]}, expected {self.num_obs}"
            )
        return packed

    def _prepare_door_reward_function(self):
        self.reward_scales = class_to_dict(self.cfg.door_rewards.scales)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def compute_reward(self):
        return super().compute_reward()

    def _compute_ee_handle_distance(self, env_ids=None):
        if env_ids is None:
            return torch.norm(self.grasp_goal_world - self.ee_pos, dim=1)
        return torch.norm(self.grasp_goal_world[env_ids] - self.ee_pos[env_ids], dim=1)

    def _update_door_derived_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if self._use_physical_door():
            handle_state = self._full_rigid_state[env_ids, self.handle_body_idx, :]
            handle_pos = handle_state[:, :3]
            handle_rot = handle_state[:, 3:7]
            down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(len(env_ids), 1)
            self.door_handle_pos_world[env_ids] = handle_pos
            self.grasp_goal_world[env_ids] = handle_pos + quat_apply(handle_rot, self.goal_pos_offset_tensor[env_ids])
            self.handle_target_rot[env_ids] = quat_mul(handle_rot, down_quat)
            self.handle_approach_dir_world[env_ids] = quat_apply(
                self.door_yaw_quat[env_ids], self._handle_approach_dir_local[env_ids]
            )
            self.handle_rotate_dir_world[env_ids] = quat_apply(handle_rot, self._handle_rotate_dir_local[env_ids])
            self.door_open_dir_world[env_ids] = quat_apply(self.door_yaw_quat[env_ids], self._door_open_dir_local[env_ids])
            self.pregrasp_goal_world[env_ids] = (
                self.grasp_goal_world[env_ids] + self.handle_approach_dir_world[env_ids] * 0.18
            )

            dof_range = torch.clamp(self.door_dof_upper[env_ids] - self.door_dof_lower[env_ids], min=1e-3)
            hinge_range = torch.clamp(
                torch.maximum(torch.abs(self.door_dof_upper[env_ids, 0]), torch.abs(self.door_dof_lower[env_ids, 0])),
                min=1e-3,
            )
            door_angle = self._door_dof_pos[env_ids, 0]
            handle_from_lower = self._door_dof_pos[env_ids, 1] - self.door_dof_lower[env_ids, 1]
            opening_angle = self._door_hinge_open_sign() * door_angle
            self.door_open_ratio[env_ids] = torch.clamp(opening_angle / hinge_range, 0.0, 1.5)
            self.handle_open_ratio[env_ids] = torch.clamp(handle_from_lower / dof_range[:, 1], 0.0, 1.5)
            threshold = float(self.cfg.door.handle_press_threshold_ratio)
            self.open_door_stage[env_ids] |= self.handle_open_ratio[env_ids] >= threshold
            self.door_open_success[env_ids] = (
                self.door_open_ratio[env_ids] >= float(self.cfg.door.door_open_success_threshold)
            )
            self.door_open_hold_counter[env_ids] = torch.where(
                self.door_open_success[env_ids],
                self.door_open_hold_counter[env_ids] + 1,
                torch.zeros_like(self.door_open_hold_counter[env_ids]),
            )
            self.base_door_dis[env_ids] = torch.norm(
                self.grasp_goal_world[env_ids, :2] - self.base_pos[env_ids, :2], dim=1
            )
            self._update_curr_dist(env_ids)
            return

        handle_motion = self.handle_rotate_dir_world[env_ids] * self.handle_open_ratio[env_ids].unsqueeze(1) * 0.05
        door_motion = self.door_open_dir_world[env_ids] * self.door_open_ratio[env_ids].unsqueeze(1) * 0.35
        goal_offset = quat_apply(self.door_yaw_quat[env_ids], self.goal_pos_offset_tensor[env_ids])
        self.door_handle_pos_world[env_ids] = self.door_handle_closed_pos_world[env_ids] + handle_motion + door_motion
        self.grasp_goal_world[env_ids] = self.door_handle_pos_world[env_ids] + goal_offset
        down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(len(env_ids), 1)
        self.handle_target_rot[env_ids] = quat_mul(self.door_yaw_quat[env_ids], down_quat)
        self.pregrasp_goal_world[env_ids] = self.grasp_goal_world[env_ids] + self.handle_approach_dir_world[env_ids] * 0.18

        dof_range = torch.clamp(self.door_dof_upper[env_ids] - self.door_dof_lower[env_ids], min=1e-3)
        hinge_range = torch.clamp(
            torch.maximum(torch.abs(self.door_dof_upper[env_ids, 0]), torch.abs(self.door_dof_lower[env_ids, 0])),
            min=1e-3,
        )
        self._door_dof_pos[env_ids, 0] = self._door_hinge_open_sign() * self.door_open_ratio[env_ids] * hinge_range
        self._door_dof_pos[env_ids, 1] = self.door_dof_lower[env_ids, 1] + self.handle_open_ratio[env_ids] * dof_range[:, 1]

        self.door_open_success[env_ids] = self.door_open_ratio[env_ids] >= float(self.cfg.door.door_open_success_threshold)
        self.door_open_hold_counter[env_ids] = torch.where(
            self.door_open_success[env_ids],
            self.door_open_hold_counter[env_ids] + 1,
            torch.zeros_like(self.door_open_hold_counter[env_ids]),
        )
        self.base_door_dis[env_ids] = torch.norm(self.grasp_goal_world[env_ids, :2] - self.base_pos[env_ids, :2], dim=1)
        self._update_curr_dist(env_ids)

    def _update_curr_dist(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        dist = self._compute_ee_handle_distance(env_ids)
        self.curr_dist[env_ids] = dist
        self.closest_dist[env_ids] = torch.where(
            self.closest_dist[env_ids] < 0.0,
            dist,
            torch.minimum(self.closest_dist[env_ids], dist),
        )

    def get_grasp_goal_world(self):
        return self.grasp_goal_world.clone()

    def get_pregrasp_goal_world(self, offset=0.18):
        return self.grasp_goal_world + self.handle_approach_dir_world * float(offset)

    def scripted_actions_from_world_targets(self, target_pos_world, target_rot_world, gripper_open, base_x_cmd, yaw_cmd):
        actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        target_local = quat_rotate_inverse(
            self.base_yaw_quat,
            target_pos_world.to(self.device) - self.get_ee_goal_spherical_center(),
        )
        ee_scale = torch.clamp(self.low_level_command_adapter.ee_delta_scale, min=1e-6)
        actions[:, 0:3] = torch.clamp((target_local - self.curr_ee_goal_cart) / ee_scale, -1.0, 1.0)

        target_local_rot = quat_mul(quat_conjugate(self.base_yaw_quat), target_rot_world.to(self.device))
        target_rpy = torch.stack(euler_from_quat(target_local_rot), dim=1)
        rpy_scale = torch.clamp(self.low_level_command_adapter.ee_rpy_delta_scale, min=1e-6)
        actions[:, 3:6] = torch.clamp(_wrap_to_pi(target_rpy - self.curr_ee_goal_orn_delta_rpy) / rpy_scale, -1.0, 1.0)

        if isinstance(gripper_open, torch.Tensor):
            gripper_open_tensor = gripper_open.to(device=self.device, dtype=torch.bool)
        else:
            gripper_open_tensor = torch.full((self.num_envs,), bool(gripper_open), dtype=torch.bool, device=self.device)
        actions[:, 6] = torch.where(
            gripper_open_tensor,
            torch.ones(self.num_envs, dtype=torch.float, device=self.device),
            -torch.ones(self.num_envs, dtype=torch.float, device=self.device),
        )

        if not isinstance(base_x_cmd, torch.Tensor):
            base_x_cmd = torch.full((self.num_envs,), float(base_x_cmd), dtype=torch.float, device=self.device)
        if not isinstance(yaw_cmd, torch.Tensor):
            yaw_cmd = torch.full((self.num_envs,), float(yaw_cmd), dtype=torch.float, device=self.device)
        base_forward_scale = max(float(getattr(self.cfg.high_level, "base_forward_scale", 0.35)), 1e-6)
        base_yaw_scale = max(float(getattr(self.cfg.high_level, "base_yaw_scale", 0.5)), 1e-6)
        actions[:, 7] = torch.clamp(base_x_cmd.to(self.device) / base_forward_scale, -1.0, 1.0)
        actions[:, 8] = torch.clamp(yaw_cmd.to(self.device) / base_yaw_scale, -1.0, 1.0)
        return actions

    def _reward_approach_handle(self):
        dist_delta = torch.clamp(self.prev_ee_handle_dist - self.curr_dist, 0.0, 10.0)
        self.prev_ee_handle_dist[:] = self.curr_dist
        self.closest_dist[:] = torch.minimum(self.closest_dist, self.curr_dist)
        reward = torch.tanh(10.0 * dist_delta)
        reward *= (~self.door_open_success).to(dtype=torch.float)
        return reward

    def _reward_ee_align_handle(self):
        ee_orn = self.ee_orn / torch.clamp(torch.norm(self.ee_orn, dim=-1, keepdim=True), min=1e-6)
        metric = torch.norm(_orientation_error(self.handle_target_rot, ee_orn), dim=-1)
        reward = torch.exp(-3.0 * metric)
        reward *= (self.curr_dist < 0.25).to(dtype=torch.float)
        return reward

    def _reward_lever_press(self):
        progress = self.handle_open_ratio - self.best_handle_open_ratio
        self.best_handle_open_ratio[:] = torch.maximum(self.best_handle_open_ratio, self.handle_open_ratio)
        reward = torch.tanh(5.0 * torch.clamp(progress, min=0.0, max=1.0))
        reward *= (self.curr_dist < 0.18).to(dtype=torch.float)
        reward *= (self.handle_open_ratio < float(self.cfg.door.handle_press_threshold_ratio) + 0.05).to(dtype=torch.float)
        return reward

    def _reward_door_open_progress(self):
        progress = self.door_open_ratio - self.best_door_open_ratio
        self.best_door_open_ratio[:] = torch.maximum(self.best_door_open_ratio, self.door_open_ratio)
        reward = torch.tanh(5.0 * torch.clamp(progress, min=0.0, max=1.0))
        reward *= (self.handle_open_ratio >= float(self.cfg.door.handle_press_threshold_ratio)).to(dtype=torch.float)
        return reward

    def _reward_door_open_success(self):
        newly_successful = self.door_open_success & (~self.success_recorded)
        self.success_recorded |= self.door_open_success
        return newly_successful.to(dtype=torch.float)

    def _reward_base_command_penalty(self):
        penalty = torch.where(
            self.base_door_dis < float(getattr(self.cfg.door, "base_door_distance_threshold", 0.85)),
            torch.norm(self.commands[:, :3], dim=1),
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        )
        return penalty

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.high_level_actions - self.high_level_last_actions), dim=1)

    def _reward_gripper_rate(self):
        return torch.square(self.high_level_actions[:, 6] - self.high_level_last_actions[:, 6])

    def _reward_base_height(self):
        target = float(getattr(self.cfg.door, "base_height_target", 0.55))
        return torch.exp(-20.0 * torch.abs(self.base_pos[:, 2] - target))
