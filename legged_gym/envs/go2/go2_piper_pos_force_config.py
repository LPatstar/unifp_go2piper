import numpy as np

from legged_gym.envs.b2.b2z1_pos_force_config import (
    B2Z1PosForceRoughCfg,
    B2Z1PosForceRoughCfgPPO,
)


class Go2PiperPosForceRoughCfg(B2Z1PosForceRoughCfg):
    class goal_ee(B2Z1PosForceRoughCfg.goal_ee):
        # Collision box around the Go2 torso in the spherical-center frame.
        # This is intentionally tighter than the original B2 box so the debug
        # red frame and the actual EE collision rejection volume better match
        # the smaller Go2 body.
        collision_upper_limits = [0.30, 0.16, 0.03]
        collision_lower_limits = [-0.32, -0.16, -0.32]
        underground_limit = -0.32
        arm_induced_pitch = 0.12

        class sphere_center(B2Z1PosForceRoughCfg.goal_ee.sphere_center):
            x_offset = -0.03
            y_offset = 0.0
            z_invariant_offset = 0.35

        class ranges(B2Z1PosForceRoughCfg.goal_ee.ranges):
            init_pos_start = [0.42, np.pi / 8, 0.0]
            init_pos_end = [0.42, 0.0, 0.0]
            pos_l = [0.30, 0.77]
            pos_p = [-np.pi / 2.7, np.pi / 2.7]
            pos_y = [-3 * np.pi / 5, 3 * np.pi / 5]
            delta_orn_r = [-0.35, 0.35]
            delta_orn_p = [-0.35, 0.35]
            delta_orn_y = [-0.35, 0.35]

    class init_state(B2Z1PosForceRoughCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
            "piper_joint1": 0.0,
            "piper_joint2": 1.20,
            "piper_joint3": -2.0,
            "piper_joint4": 0.0,
            "piper_joint5": 0.75,
            "piper_joint6": 0.0,
            "piper_joint7": 0.01,
            "piper_joint8": -0.01,
        }

    class domain_rand(B2Z1PosForceRoughCfg.domain_rand):
        added_mass_range = [0.0, 5.0]
        randomize_base_com = True
        added_com_range_x = [-0.05, 0.05]
        added_com_range_y = [-0.05, 0.05]
        added_com_range_z = [-0.05, 0.05]

    class env(B2Z1PosForceRoughCfg.env):
        num_leg_dofs = 12
        # Preserve the original B2Z1 learning layout:
        # 12 leg DoFs + the first 5 arm joints are policy-controlled,
        # while the final Piper wrist/gripper joints are held by fixed PD.
        num_gripper_joints = 3
        num_actions = 17
        num_torques = 17
        num_single_obs = 73
        num_pred_obs = 12
        num_observations = int(B2Z1PosForceRoughCfg.env.frame_stack * num_single_obs)
        single_num_privileged_obs = 149
        num_privileged_obs = int(B2Z1PosForceRoughCfg.env.c_frame_stack * single_num_privileged_obs)

    class commands(B2Z1PosForceRoughCfg.commands):
        # Go2 + Piper is notably smaller/lighter than B2 + Z1, so use a
        # narrower EE force range for both commanded-force visualization and
        # externally applied disturbances.
        max_push_force_xyz_gripper_cmd = [-30, 30]
        max_push_force_xyz_gripper_ext = [-30, 30]
        max_push_force_xyz_base_cmd = [-10, 10]
        max_push_force_xyz_base_ext = [-10, 10]
        gripper_force_kp_range = [200., 200.]
        base_force_kd_range = [30.0, 30.0]

    class control(B2Z1PosForceRoughCfg.control):
        stiffness = {
            "hip": 40.0,
            "thigh": 40.0,
            "calf": 40.0,
            "piper_joint1": 40.0,
            "piper_joint2": 80.0,
            "piper_joint3": 80.0,
            "piper_joint4": 40.0,
            "piper_joint5": 80.0,
            "piper_joint6": 40.0,
            "piper_joint7": 40.0,
            "piper_joint8": 40.0,
        }
        damping = {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 1.0,
            "piper_joint1": 3.0,
            "piper_joint2": 6.0,
            "piper_joint3": 6.0,
            "piper_joint4": 3.0,
            "piper_joint5": 6.0,
            "piper_joint6": 3.0,
            "piper_joint7": 3.0,
            "piper_joint8": 3.0,
        }
        # Remap all Piper joints so current stiff PD produces torque close to
        # the soft-PD plant that made tuned7 successful. Joints 1-5 remap via
        # policy actions; fixed-PD joints 6-8 remap via their fixed targets.
        enable_arm_pd_equivalent_action_remap = True
        arm_pd_remap_old_stiffness = {
            "piper_joint1": 10.0,
            "piper_joint2": 10.0,
            "piper_joint3": 10.0,
            "piper_joint4": 10.0,
            "piper_joint5": 10.0,
            "piper_joint6": 10.0,
            "piper_joint7": 10.0,
            "piper_joint8": 10.0,
        }
        arm_pd_remap_old_damping = {
            "piper_joint1": 1.5,
            "piper_joint2": 1.5,
            "piper_joint3": 1.5,
            "piper_joint4": 1.5,
            "piper_joint5": 1.5,
            "piper_joint6": 1.5,
            "piper_joint7": 1.5,
            "piper_joint8": 1.5,
        }

    class arm(B2Z1PosForceRoughCfg.arm):
        base_offset = [-0.01, 0.0, 0.208]
        init_target_ee_base = [0.34, 0.0, 0.08]
        grasp_offset = 0.10

    class asset(B2Z1PosForceRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2_piper/go2/urdf/go2piper.urdf"
        name = "go2_piper"
        base_name = "base"
        foot_name = "foot"
        thigh_name = "thigh"
        gripper_name = "tcp_link"
        flip_visual_attachments = False
        penalize_contacts_on = ["thigh", "calf", "trunk", "piper_base_link", "piper_link"]
        terminate_after_contacts_on = []

    class viewer(B2Z1PosForceRoughCfg.viewer):
        follow_offset_scale = 0.55
        follow_target_height = 0.18

    class rewards(B2Z1PosForceRoughCfg.rewards):
        base_height_target = 0.35

        class scales(B2Z1PosForceRoughCfg.rewards.scales):
            # Keep the proven base/yaw and hybrid updates, but recover the
            # stronger leg/action regularization after tuned6 regressed.
            tracking_lin_vel_force_world = 2.5
            tracking_ang_vel = 1.3
            base_height = -1.2
            ang_vel_xy = -0.012
            tracking_ee_force_world = 2.3
            ref_dof_leg = 1.0
            action_rate = -0.02


class Go2PiperPosForceRoughCfgPPO(B2Z1PosForceRoughCfgPPO):
    class runner(B2Z1PosForceRoughCfgPPO.runner):
        run_name = ""
        experiment_name = "go2_piper_pos_force"
