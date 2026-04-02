# Go2 + Piper Config Review Checklist

这份清单用于记录：

- `go2_piper_pos_force_config.py` 当前已经覆盖了哪些配置
- 哪些参数仍然继承自 `b2z1_pos_force_config.py`
- 这些“还没改”的参数里，哪些会因为 `Go2` 体型更小、`Piper` 机械臂不同，而值得优先重新评估

## 当前已经覆盖的配置

文件: [legged_gym/envs/go2/go2_piper_pos_force_config.py](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/go2/go2_piper_pos_force_config.py)

目前 `Go2PiperPosForceRoughCfg` 已经显式覆盖了这些模块：

- `goal_ee`
- `init_state`
- `env`
- `commands`
  - 目前只改了 `gripper` 力范围
- `control`
- `arm`
- `asset`
- `viewer`
- `rewards`
  - 目前只改了 `base_height_target`

因此，下面列出的参数如果没有在 `go2_piper_pos_force_config.py` 里重新定义，就仍然沿用 `b2z1_pos_force_config.py` 的值。

## 优先复查

### 1. Base 质量与质心随机化

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:74](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L74)

相关参数：

- `domain_rand.added_mass_range = [0., 15.]`
- `domain_rand.added_com_range_x = [-0.15, 0.15]`
- `domain_rand.added_com_range_y = [-0.15, 0.15]`
- `domain_rand.added_com_range_z = [-0.15, 0.15]`

为什么值得看：

- 这些范围是按 B2 的体量设的
- 对 Go2 这种更小的底盘来说，可能偏大
- 容易让训练一开始就面临过强的动力学扰动

### 2. Base force 扰动范围

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:159](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L159)

相关参数：

- `commands.max_push_force_xyz_base_cmd = [-50, 50]`
- `commands.max_push_force_xyz_base_ext = [-50, 50]`

为什么值得看：

- 当前只把 `gripper` 的力范围从 `60N` 收到了 `30N`
- `base` 的扰动幅度仍然是 B2 的量级
- 对更轻、更短的 Go2 来说，可能偏激进

### 3. EE / Base 力补偿增益

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:149](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L149)
- [legged_gym/envs/b2/b2z1_pos_force_config.py:167](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L167)

相关参数：

- `commands.gripper_force_kp_range`
- `commands.gripper_force_kd_range`
- `commands.base_force_kp_range`
- `commands.base_force_kd_range`

为什么值得看：

- 你已经换了机械臂、也缩了末端力范围
- 但力补偿/偏移用的增益还在沿用 B2+Z1
- 这会直接影响末端受力后的目标补偿幅度

### 4. 地形高度采样窗口

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:200](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L200)

相关参数：

- `terrain.measured_points_x`
- `terrain.measured_points_y`

为什么值得看：

- 这是地形感知窗口
- B2 的身体更大，采样覆盖范围也更大
- Go2 的机身和足迹更紧凑，未必需要这么宽的窗口

### 5. 步态节律先验

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:268](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L268)

相关参数：

- `rewards.cycle_time = 0.64`
- `rewards.target_joint_pos_scale = 0.17`
- `rewards.target_joint_pos_thd = 0.5`

为什么值得看：

- 这些在隐式定义腿部参考步态的时序和摆幅
- Go2 的自然步态节奏不一定和 B2 一样

### 6. Whole-body reaching 相关奖励约束

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:278](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L278)

相关参数：

- `rewards.scales.base_height = -2.0`
- `rewards.scales.ang_vel_xy = -0.02`

为什么值得看：

- 这两个会抑制机身俯仰/侧倾
- 如果目标是让 Go2 更积极用身体帮助机械臂够点，这两个项非常关键

## 也很可能要看

### 7. Base 命令范围

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:128](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L128)

相关参数：

- `commands.ranges.lin_vel_x = [-0.6, 0.6]`
- `commands.ranges.lin_vel_y = [-0.4, 0.4]`
- `commands.ranges.ang_vel_yaw = [-0.6, 0.6]`

为什么值得看：

- 这组范围是按 B2 任务经验定的
- `Go2 + Piper` 的最稳妥速度范围不一定相同

### 8. 扰动持续时间与频率

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:142](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L142)

相关参数：

- `push_gripper_interval_s_cmd/ext`
- `push_gripper_duration_s_cmd/ext`
- `push_base_interval_s_cmd/ext`
- `push_base_duration_s_cmd/ext`

为什么值得看：

- 力值大小不是唯一因素
- 同样的力如果持续更久，小机体感受到的扰动也会更明显

### 9. 电机强度随机化

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:82](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L82)

相关参数：

- `domain_rand.leg_motor_strength_range = [0.85, 1.15]`
- `domain_rand.arm_motor_strength_range = [0.85, 1.15]`

为什么值得看：

- Go2 腿和 Piper 臂的驱动特性与 B2+Z1 不同
- 这两个区间未必还是最合理的随机化范围

### 10. 末端负载随机化

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:90](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L90)

相关参数：

- `domain_rand.gripper_added_mass_range = [0, 0.2]`

为什么值得看：

- Piper 的末端结构和 Z1 不一样
- 若要考虑抓取负载泛化，这个范围最好按 Piper 重新定

### 11. 初始状态随机化

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:66](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L66)

相关参数：

- `rand_yaw_range = pi/2`
- `origin_perturb_range = 0.5`
- `init_vel_perturb_range = 0.1`

为什么值得看：

- 这套初始扰动大小是按 B2 习惯设定的
- 换成 Go2 后未必仍是最合适的训练起点分布

## 奖励里可以后看但别忘记

来源:

- [legged_gym/envs/b2/b2z1_pos_force_config.py:281](/home/robodog/loco-manipulation/UniFP/legged_gym/envs/b2/b2z1_pos_force_config.py#L281)

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
- 但如果后面出现 “动作发僵”、“腿太保守”、“手臂抖动” 或 “whole-body 不积极” 的问题，这些项通常会参与原因

## 建议的调参顺序

如果只想先看最重要的，建议按这个顺序排查：

1. `added_mass_range`
2. `added_com_range_x/y/z`
3. `max_push_force_xyz_base_cmd/ext`
4. `gripper_force_kp_range / kd_range`
5. `base_force_kp_range / kd_range`
6. `terrain.measured_points_x / y`
7. `rewards.cycle_time`
8. `rewards.scales.base_height`
9. `rewards.scales.ang_vel_xy`

## 备注

- 以上清单只列“目前没改，但值得重新评估”的项
- 不代表这些参数一定错
- 只是它们目前仍继承自 B2+Z1，而 `Go2 + Piper` 的形态、质量分布、末端工作空间和动态特性都明显不同
