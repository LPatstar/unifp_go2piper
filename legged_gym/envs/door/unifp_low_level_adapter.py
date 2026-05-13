from dataclasses import dataclass
from typing import Optional

import torch

from legged_gym.envs.b2.legged_robot_b2z1_pos_force import (
    INDEX_BASE_FORCE_X,
    INDEX_BASE_FORCE_Z,
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Z,
    INDEX_EE_ROLL_CMD,
    INDEX_EE_YAW_CMD,
)
from legged_gym.utils.isaacgym_utils import sphere2cart


@dataclass
class LowLevelCommand:
    """Named command contract consumed by the UniFP low-level controller."""

    base_vel_local: torch.Tensor
    ee_goal_local_cart: torch.Tensor
    ee_goal_local_rpy: torch.Tensor
    gripper_cmd: Optional[torch.Tensor] = None
    ee_force_cmd_local: Optional[torch.Tensor] = None
    base_force_cmd_local: Optional[torch.Tensor] = None


@dataclass
class LowLevelState:
    """Named low-level state exposed upward without raw command indices."""

    base_pos_world: torch.Tensor
    base_lin_vel_local: torch.Tensor
    base_ang_vel_local: torch.Tensor
    ee_pos_world: torch.Tensor
    ee_vel_world: torch.Tensor
    ee_goal_local_cart: torch.Tensor
    ee_goal_local_rpy: torch.Tensor
    command_buffer: torch.Tensor
    ee_force_cmd_local: torch.Tensor
    base_force_cmd_local: torch.Tensor
    ee_force_external_local: torch.Tensor
    base_force_external_local: torch.Tensor


class UniFPLowLevelCommandAdapter:
    """Adapter between door-level actions and UniFP's B2+Z1 command buffers.

    High-level code should use this adapter instead of writing raw
    ``env.commands`` slices directly. The index knowledge stays here, so the
    high-level door task can be kept close to a robot-independent task policy.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.ee_delta_scale = self._tensor3(getattr(cfg, "ee_delta_scale", [0.02, 0.02, 0.02]))
        self.ee_rpy_delta_scale = self._tensor3(getattr(cfg, "ee_rpy_delta_scale", [0.06, 0.06, 0.06]))
        self.ee_rpy_limit = self._tensor3(getattr(cfg, "ee_rpy_limit", [0.7, 0.7, 0.7]))
        self.ee_force_scale = self._tensor3(getattr(cfg, "ee_force_scale", [30.0, 30.0, 30.0]))
        self.base_force_scale = self._tensor3(getattr(cfg, "base_force_scale", [0.0, 0.0, 0.0]))

    def _tensor3(self, values):
        return torch.tensor(values, dtype=torch.float, device=self.env.device).view(1, 3)

    def configure_external_control(self):
        """Disable low-level autonomous command randomization paths for high-level control."""
        self.env.cfg.env.teleop_mode = True
        self.env.key_command_mode = True
        self.env.enable_random_force_events = False

    def command_from_high_level_action(self, actions: torch.Tensor) -> LowLevelCommand:
        """Convert normalized high-level action into a named low-level command."""
        action_clip = float(getattr(self.cfg, "action_clip", 1.0))
        actions = torch.clamp(actions.to(self.env.device), -action_clip, action_clip)

        ee_goal_local_cart = self.env.curr_ee_goal_cart + actions[:, 0:3] * self.ee_delta_scale
        ee_goal_local_rpy = self.env.curr_ee_goal_orn_delta_rpy + actions[:, 3:6] * self.ee_rpy_delta_scale
        ee_goal_local_rpy = torch.max(torch.min(ee_goal_local_rpy, self.ee_rpy_limit), -self.ee_rpy_limit)

        base_vel_local = torch.zeros(self.env.num_envs, 3, dtype=torch.float, device=self.env.device)
        base_vel_local[:, 0] = actions[:, 7] * float(getattr(self.cfg, "base_forward_scale", 0.4))
        base_vel_local[:, 2] = actions[:, 8] * float(getattr(self.cfg, "base_yaw_scale", 0.6))

        ee_force_cmd_local = None
        if bool(getattr(self.cfg, "use_force_actions", False)) and actions.shape[1] >= 12:
            ee_force_cmd_local = actions[:, 9:12] * self.ee_force_scale

        base_force_cmd_local = None
        if bool(getattr(self.cfg, "use_base_force_actions", False)) and actions.shape[1] >= 15:
            base_force_cmd_local = actions[:, 12:15] * self.base_force_scale

        return LowLevelCommand(
            base_vel_local=base_vel_local,
            ee_goal_local_cart=ee_goal_local_cart,
            ee_goal_local_rpy=ee_goal_local_rpy,
            gripper_cmd=actions[:, 6:7] if bool(getattr(self.cfg, "use_gripper_action", False)) else None,
            ee_force_cmd_local=ee_force_cmd_local,
            base_force_cmd_local=base_force_cmd_local,
        )

    def apply_command(self, command: LowLevelCommand, env_ids: Optional[torch.Tensor] = None):
        """Write a named command into UniFP buffers for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
        if len(env_ids) == 0:
            return

        self.configure_external_control()

        base_cmd = command.base_vel_local[env_ids]
        self.env.commands[env_ids, 0] = torch.clamp(
            base_cmd[:, 0],
            self.env.command_ranges["lin_vel_x"][0],
            self.env.command_ranges["lin_vel_x"][1],
        )
        self.env.commands[env_ids, 1] = torch.clamp(
            base_cmd[:, 1],
            self.env.command_ranges["lin_vel_y"][0],
            self.env.command_ranges["lin_vel_y"][1],
        )
        self.env.commands[env_ids, 2] = torch.clamp(
            base_cmd[:, 2],
            self.env.command_ranges["ang_vel_yaw"][0],
            self.env.command_ranges["ang_vel_yaw"][1],
        )

        self.env._set_key_command_ee_goal_local_cart(env_ids, command.ee_goal_local_cart[env_ids])
        self.env.key_command_ee_orn_delta_rpy[env_ids] = command.ee_goal_local_rpy[env_ids]
        self.env.curr_ee_goal_orn_delta_rpy[env_ids] = command.ee_goal_local_rpy[env_ids]
        self.env.ee_goal_orn_delta_rpy[env_ids] = command.ee_goal_local_rpy[env_ids]
        self.env.ee_start_orn_delta_rpy[env_ids] = command.ee_goal_local_rpy[env_ids]
        self.env.commands[env_ids, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1] = command.ee_goal_local_rpy[env_ids]
        if command.gripper_cmd is not None:
            if not hasattr(self.env, "set_high_level_gripper_command"):
                raise RuntimeError("High-level gripper action is enabled, but the environment has no gripper adapter.")
            self.env.set_high_level_gripper_command(command.gripper_cmd[env_ids], env_ids)

        self.env.update_curr_ee_goal()
        self._write_force_commands(command, env_ids)

    def reset_command(self, env_ids: torch.Tensor):
        """Reset high-level-owned command fields after an environment reset."""
        if len(env_ids) == 0:
            return

        initial_cart = sphere2cart(self.env.init_start_ee_sphere).repeat(len(env_ids), 1)
        zero3 = torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.env.device)
        full_zero3 = torch.zeros(self.env.num_envs, 3, dtype=torch.float, device=self.env.device)
        full_cart = self.env.curr_ee_goal_cart.clone()
        full_rpy = self.env.curr_ee_goal_orn_delta_rpy.clone()
        full_cart[env_ids] = initial_cart
        full_rpy[env_ids] = zero3

        command = LowLevelCommand(
            base_vel_local=full_zero3,
            ee_goal_local_cart=full_cart,
            ee_goal_local_rpy=full_rpy,
            gripper_cmd=torch.ones(self.env.num_envs, 1, dtype=torch.float, device=self.env.device)
            if bool(getattr(self.cfg, "use_gripper_action", False))
            else None,
            ee_force_cmd_local=full_zero3,
            base_force_cmd_local=full_zero3,
        )
        self.apply_command(command, env_ids=env_ids)

    def get_state(self) -> LowLevelState:
        """Return named low-level state fields for high-level observation code."""
        ee_force_external_local = self.env.forces_local[:, self.env.gripper_idx, :3]
        base_force_external_local = self.env.forces_local[:, self.env.robot_base_idx, :3]
        return LowLevelState(
            base_pos_world=self.env.base_pos,
            base_lin_vel_local=self.env.base_lin_vel,
            base_ang_vel_local=self.env.base_ang_vel,
            ee_pos_world=self.env.ee_pos,
            ee_vel_world=self.env.ee_vel[:, :3],
            ee_goal_local_cart=self.env.curr_ee_goal_cart,
            ee_goal_local_rpy=self.env.curr_ee_goal_orn_delta_rpy,
            command_buffer=self.env.commands,
            ee_force_cmd_local=self.env.current_Fxyz_gripper_cmd,
            base_force_cmd_local=self.env.current_Fxyz_base_cmd,
            ee_force_external_local=ee_force_external_local,
            base_force_external_local=base_force_external_local,
        )

    def patch_latest_command_observation(self, low_level_obs: torch.Tensor) -> torch.Tensor:
        """Patch the newest low-level observation frame with the latest command buffer.

        The frozen low-level policy observes a stacked history. High-level
        commands are written just before low-level inference, so this patches
        only the newest frame's command segment and leaves the older history
        intact.
        """
        if low_level_obs is None:
            return low_level_obs

        patched = low_level_obs.clone()
        frame_dim = int(self.env.cfg.env.num_single_obs)
        command_dim = int(self.env.cfg.commands.num_commands)
        command_start = patched.shape[1] - frame_dim + frame_dim - command_dim
        command_end = command_start + command_dim
        if command_start >= 0 and command_end <= patched.shape[1]:
            patched[:, command_start:command_end] = self.env.commands * self.env.commands_scale
        return patched

    def _write_force_commands(self, command: LowLevelCommand, env_ids: torch.Tensor):
        if command.ee_force_cmd_local is not None:
            ee_force = command.ee_force_cmd_local[env_ids]
            min_force, max_force = self.env.cfg.commands.max_push_force_xyz_gripper_cmd
            ee_force = torch.clamp(ee_force, min_force, max_force)
            self.env.current_Fxyz_gripper_cmd[env_ids, :3] = ee_force
            self.env.commands[env_ids, INDEX_EE_FORCE_X : INDEX_EE_FORCE_Z + 1] = ee_force

        if command.base_force_cmd_local is not None:
            base_force = command.base_force_cmd_local[env_ids]
            min_force, max_force = self.env.cfg.commands.max_push_force_xyz_base_cmd
            base_force = torch.clamp(base_force, min_force, max_force)
            self.env.current_Fxyz_base_cmd[env_ids, :3] = base_force
            self.env.commands[env_ids, INDEX_BASE_FORCE_X : INDEX_BASE_FORCE_Z + 1] = base_force
