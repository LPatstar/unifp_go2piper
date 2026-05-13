from legged_gym.envs.b2.b2z1_pos_force_config import (
    B2Z1PosForceRoughCfg,
    B2Z1PosForceRoughCfgPPO,
)


class B2Z1DoorOpenRoughCfg(B2Z1PosForceRoughCfg):
    """High-level door-opening task config.

    The inherited ``env`` dimensions remain the low-level UniFP dimensions
    while the simulator is constructed. The public high-level action and
    observation dimensions are declared in ``high_level`` and installed by the
    high-level environment after the low-level buffers exist.
    """

    class env(B2Z1PosForceRoughCfg.env):
        teleop_mode = True
        key_command_mode = True
        episode_length_s = 12

    class terrain(B2Z1PosForceRoughCfg.terrain):
        height = [0.0, 0.0]
        curriculum = False

    class domain_rand(B2Z1PosForceRoughCfg.domain_rand):
        push_robots = False
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor = False
        randomize_gripper_mass = False

    class commands(B2Z1PosForceRoughCfg.commands):
        zero_vel_cmd_prob = 0.0
        push_gripper_stators = False
        push_robot_base = False
        randomize_gripper_force_gains = False
        randomize_base_force_gains = False

    class high_level:
        num_actions = 9
        num_current_obs = 113
        num_reserved_obs = 15
        num_single_obs = num_current_obs + num_reserved_obs
        num_observations = num_single_obs
        num_privileged_obs = num_single_obs
        num_pred_obs = 12

        action_clip = 1.0
        ee_delta_scale = [0.02, 0.02, 0.02]
        ee_rpy_delta_scale = [0.06, 0.06, 0.06]
        ee_rpy_limit = [0.7, 0.7, 0.7]
        base_forward_scale = 0.35
        base_yaw_scale = 0.5

        use_force_actions = False
        use_base_force_actions = False
        ee_force_scale = [30.0, 30.0, 30.0]
        base_force_scale = [0.0, 0.0, 0.0]

    class low_level:
        policy_mode = "checkpoint"  # "checkpoint" or "zero"
        experiment_name = "b2z1_pos_force"
        load_run = -1
        checkpoint = -1

    class door:
        asset_root = "{LEGGED_GYM_ROOT_DIR}/resources/objects/door_set"
        asset_names = [
            "99650089960001",
            "99650089960006",
            "99655039960001",
            "99655039960006",
        ]
        use_physical_actor = True
        closed_handle_pos_env = [0.85, 0.0, 0.85]
        closed_handle_root_offsets_yaw0 = [
            [-0.03097153, -0.19897342, -0.07206905],
            [-0.03123808, -0.21056366, -0.10809296],
            [-0.02076721, -0.35829926, -0.06140488],
            [-0.01972580, -0.38145447, -0.10678279],
        ]
        asset_base_pos_env = [0.0, 0.0, 0.0]
        door_yaw = 3.141592653589793
        door_hinge_open_sign = -1.0
        handle_approach_dir_local = [-1.0, 0.0, 0.0]
        handle_rotate_dir_local = [0.0, -1.0, 0.0]
        door_open_dir_local = [1.0, 0.0, 0.0]
        handle_radius = 0.08
        handle_press_threshold_ratio = 0.65
        door_open_success_threshold = 0.45
        door_open_hold_steps = 6
        success_distance = 0.10
        terminate_on_reach = False
        base_door_distance_threshold = 0.85
        base_far_threshold = 2.0
        base_far_grace_steps = 80
        ee_far_threshold = 1.1
        ee_far_grace_steps = 40
        robot_start_pos_env = [1.85, 0.0, 0.62]
        robot_start_yaw = 3.141592653589793
        robot_start_xy_noise = 0.04
        robot_start_yaw_noise = 0.08
        robot_start_vel_noise = 0.05
        door_disable_gravity = True
        door_fix_base_link = True
        door_static_friction = 2.0
        door_dynamic_friction = 1.5
        door_joint_damping = [3.0, 10.0]
        door_joint_friction = [6.0, 18.0]
        door_joint_effort = [200.0, 200.0]
        handle_upper_limit = 0.7853981633974483
        hard_lock_before_handle_threshold = True
        door_open_resistance = 3.0
        door_open_damping = 0.5
        door_lock_force = 150.0
        door_close_torque_sign = 1.0
        handle_spring_stiffness = 40.0
        handle_spring_damping = 2.0
        base_height_target = 0.55

    class door_rewards:
        class scales:
            approach_handle = 1.0
            ee_align_handle = 0.15
            lever_press = 1.0
            door_open_progress = 2.0
            door_open_success = 5.0
            base_command_penalty = -0.05
            action_rate = -0.001
            gripper_rate = -0.001
            base_height = 0.2


class B2Z1DoorOpenRoughCfgPPO(B2Z1PosForceRoughCfgPPO):
    class policy(B2Z1PosForceRoughCfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(B2Z1PosForceRoughCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(B2Z1PosForceRoughCfgPPO.runner):
        policy_class_name = "StateTeacherActorCritic"
        experiment_name = "b2z1_door_open"
        run_name = "phase1_adapter_state_teacher"
        max_iterations = 20000
