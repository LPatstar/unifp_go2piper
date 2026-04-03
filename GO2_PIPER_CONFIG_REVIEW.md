# Go2 + Piper Config Review Checklist

这份清单用于记录两类东西：

- 哪些关键配置已经从 `b2z1` 改到了 `go2_piper`，但很可能还要继续调
- 哪些配置目前仍然继承自 `b2z1`，还没有专门按 `Go2 + Piper` 复查

文件主入口：

- [go2_piper_pos_force_config.py](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py)

上游参考：

- [b2z1_pos_force_config.py](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py)

## 一、已经改过，但建议继续调

这部分不是“完全没动过”的 inherited 参数，而是已经针对 Go2+Piper 做过一轮修改，但从现在的训练/评测目标来看，仍然值得继续调试。

### 1. EE 目标采样空间与碰撞约束

来源：

- [go2_piper_pos_force_config.py:9](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L9)
- [b2z1_pos_force_config.py:6](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L6)

相关参数：

- `goal_ee.collision_upper_limits`
- `goal_ee.collision_lower_limits`
- `goal_ee.underground_limit`
- `goal_ee.arm_induced_pitch`
- `goal_ee.sphere_center.x_offset / y_offset / z_invariant_offset`
- `goal_ee.ranges.init_pos_start / init_pos_end`
- `goal_ee.ranges.pos_l / pos_p / pos_y`
- `goal_ee.ranges.delta_orn_r / delta_orn_p / delta_orn_y`

为什么值得继续调：

- 这部分你已经从 B2 改成了 Go2+Piper 版本，但它仍然直接决定训练时目标分布
- 如果目标多数落在“纯手臂就能解决”的区域，策略就不太会学 whole-body reaching
- 如果目标空间过大、过高或过偏，又会让 reach 精度和成功率明显下降
- 它还要和红色 collision box、球心 `sphere_center`、以及实际 arm base 几何关系匹配

相对 B2 现在改了什么：

- 球心前后位置：`x_offset`
  - B2: `0.2`
  - Go2: `-0.03`
- 球心高度：`z_invariant_offset`
  - B2: `0.8`
  - Go2: `0.35`
- 半径范围：`pos_l`
  - B2: `[0.35, 0.95]`
  - Go2: `[0.30, 0.77]`
- pitch 范围：`pos_p`
  - B2: `[-2π/5, 2π/5]`
  - Go2: `[-π/2.7, π/2.7]`
- yaw 范围：`pos_y`
  - 两者当前都还是 `[-3π/5, 3π/5]`
- 初始目标：
  - B2: `init_pos_start=[0.66, π/4, 0]`, `init_pos_end=[0.66, 0, 0]`
  - Go2: `init_pos_start=[0.42, π/8, 0]`, `init_pos_end=[0.42, 0, 0]`
- 姿态扰动范围：
  - B2: `delta_orn_r/p/y = [-0.5, 0.5]`
  - Go2: `delta_orn_r/p/y = [-0.35, 0.35]`
- arm induced pitch：
  - B2: `0.38`
  - Go2: `0.12`
- collision box：
  - B2 更大、更偏前、更低
  - Go2 现在整体明显收紧，且跟小机体更匹配

建议重点观察：

- `position_only` 的 EE RMSE 是否仍偏大
- 策略是否明显不愿意用身体去辅助 reach
- 某些采样点是否经常接近 collision box 边界

### 2. EE / Base 力相关范围与补偿增益

来源：

- [go2_piper_pos_force_config.py:78](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L78)
- [b2z1_pos_force_config.py:142](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L142)

相关参数：

- `commands.max_push_force_xyz_gripper_cmd`
- `commands.max_push_force_xyz_gripper_ext`
- `commands.max_push_force_xyz_base_cmd`
- `commands.max_push_force_xyz_base_ext`
- `commands.gripper_force_kp_range`
- `commands.base_force_kd_range`

为什么值得继续调：

- 这部分你已经做过 Go2+Piper 一轮缩放
- 但它们会直接影响绿色/蓝色力箭头的量级、补偿偏移幅度和 base velocity compensation 的强弱
- 现在 `base_disturbance` 评测偏弱，说明这部分很可能还没到位

当前相对 B2 的主要变化：

- `gripper force range`
  - B2: `[-60, 60]`
  - Go2: `[-30, 30]`
- `base force range`
  - B2: `[-50, 50]`
  - Go2: `[-10, 10]`
- `base_force_kd_range`
  - B2: `[200, 200]`
  - Go2: `[30, 30]`
- `gripper_force_kp_range`
  - 当前仍保留 `[200, 200]`

### 3. Base 质量与质心随机化

来源：

- [go2_piper_pos_force_config.py:60](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L60)
- [b2z1_pos_force_config.py:74](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L74)

相关参数：

- `domain_rand.added_mass_range`
- `domain_rand.added_com_range_x`
- `domain_rand.added_com_range_y`
- `domain_rand.added_com_range_z`

为什么值得继续调：

- 这部分现在已经从 B2 的大机体量级缩到了更像 Go2 的范围
- 但它仍然直接影响底盘受扰后的姿态恢复、速度补偿难度和 sim-to-real 鲁棒性
- 如果训练后 still 偏保守、抗扰弱，或者一开随机化就明显退化，这组范围还值得继续微调

当前相对 B2 的主要变化：

- `added_mass_range`
  - B2: `[0., 15.]`
  - Go2: `[0., 5.]`
- `added_com_range_x/y/z`
  - B2: `[-0.15, 0.15]`
  - Go2: `[-0.05, 0.05]`

### 4. 本体与机械臂 PD 增益

来源：

- [go2_piper_pos_force_config.py:89](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L89)

相关参数：

- `control.stiffness`
- `control.damping`

为什么值得继续调：

- 这部分已经明显从 B2+Z1 改成了 Go2+Piper
- 但它仍然会强烈影响 reach 精度、动作硬度、whole-body 协调和仿真稳定性
- 如果后面出现“够不准”“动作发僵”“手臂抖动”“身体不愿意协同”，这里通常都值得再回头看

### 5. Arm 安装偏移与初始 EE 参考位

来源：

- [go2_piper_pos_force_config.py:117](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L117)

相关参数：

- `arm.base_offset`
- `arm.init_target_ee_base`
- `arm.grasp_offset`

为什么值得继续调：

- 这部分虽然已经改成 Piper 的几何关系
- 但它和 `goal_ee.sphere_center`、初始化姿态、目标分布是一整套耦合的
- 如果以后发现初始 pose 不自然、reach 常偏某一侧、或者 keyplay 下 home pose 不顺手，这里值得一起看

### 6. Base 高度目标

来源：

- [go2_piper_pos_force_config.py:136](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py#L136)

相关参数：

- `rewards.base_height_target = 0.35`

为什么值得继续调：

- 你已经从 B2 的高度目标改到了 Go2
- 但如果 whole-body reach 时身体仍然过保守，或者站姿显得别扭，它仍然可能要配合 `rewards.scales.base_height` 一起再调

## 二、目前还没专门按 Go2+Piper 复查，仍继承自 B2

这部分是更典型的“尚未调试、仍然继承 B2”的 inherited 参数。

### 7. 地形高度采样窗口

来源：

- [b2z1_pos_force_config.py:200](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L200)

相关参数：

- `terrain.measured_points_x`
- `terrain.measured_points_y`

为什么值得看：

- Go2 足迹更紧凑
- 这个感知窗口未必还需要 B2 那么宽

### 8. 步态节律先验

来源：

- [b2z1_pos_force_config.py:268](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L268)

相关参数：

- `rewards.cycle_time = 0.64`
- `rewards.target_joint_pos_scale = 0.17`
- `rewards.target_joint_pos_thd = 0.5`

为什么值得看：

- 这些隐式定义了腿部参考步态的节律和摆幅
- Go2 的自然节奏不一定与 B2 一样

### 9. Whole-body reaching 相关奖励约束

来源：

- [b2z1_pos_force_config.py:278](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L278)

相关参数：

- `rewards.scales.base_height = -2.0`
- `rewards.scales.ang_vel_xy = -0.02`

为什么值得看：

- 这两个会抑制机身俯仰/侧倾
- 如果目标是让 Go2 更积极用身体帮助机械臂够点，这两个项非常关键

### 10. Base 命令范围

来源：

- [b2z1_pos_force_config.py:128](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L128)

相关参数：

- `commands.ranges.lin_vel_x = [-0.6, 0.6]`
- `commands.ranges.lin_vel_y = [-0.4, 0.4]`
- `commands.ranges.ang_vel_yaw = [-0.6, 0.6]`

为什么值得看：

- 这组速度范围是按 B2 任务经验定的
- `Go2 + Piper` 的可用范围不一定相同

### 11. 扰动持续时间与频率

来源：

- [b2z1_pos_force_config.py:142](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L142)

相关参数：

- `push_gripper_interval_s_cmd/ext`
- `push_gripper_duration_s_cmd/ext`
- `push_base_interval_s_cmd/ext`
- `push_base_duration_s_cmd/ext`

为什么值得看：

- 同样的力值，持续更久会让小机体更难受

### 12. 电机强度随机化

来源：

- [b2z1_pos_force_config.py:82](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L82)

相关参数：

- `domain_rand.leg_motor_strength_range = [0.85, 1.15]`
- `domain_rand.arm_motor_strength_range = [0.85, 1.15]`

为什么值得看：

- Go2 腿和 Piper 臂的驱动特性与 B2+Z1 不同

### 13. 末端负载随机化

来源：

- [b2z1_pos_force_config.py:90](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L90)

相关参数：

- `domain_rand.gripper_added_mass_range = [0, 0.2]`

为什么值得看：

- Piper 的末端结构和 Z1 不一样

### 14. 初始状态随机化

来源：

- [b2z1_pos_force_config.py:66](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L66)

相关参数：

- `rand_yaw_range = pi/2`
- `origin_perturb_range = 0.5`
- `init_vel_perturb_range = 0.1`

为什么值得看：

- 这套初始扰动大小还是按 B2 习惯设定的

## 三、奖励里可以后看但别忘记

来源：

- [b2z1_pos_force_config.py:281](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L281)

值得留意的项：

- `torques`
- `dof_vel`
- `dof_acc`
- `dof_vel_arm`
- `dof_acc_arm`
- `action_rate`
- `action_rate_arm`
- `feet_height`
- `feet_height_high`
- `feet_drag`
- `feet_pos_xy`
- `hip_pos`
- `tracking_ee_sigma`

为什么值得看：

- 这些不是“尺寸变化后一定立刻出问题”的参数
- 但如果后面出现“动作发僵”“腿太保守”“手臂抖动”或“whole-body 不积极”，这些项通常会参与原因

## 四、建议的调参顺序

如果只想先看最重要的，建议按这个顺序：

1. `goal_ee` 采样空间与 `sphere_center`
2. `max_push_force_xyz_base_cmd/ext`
3. `gripper_force_kp_range / base_force_kd_range`
4. `rewards.scales.base_height / ang_vel_xy`
5. `added_mass_range`
6. `added_com_range_x/y/z`
7. `terrain.measured_points_x / y`
8. `cycle_time / target_joint_pos_scale / target_joint_pos_thd`
