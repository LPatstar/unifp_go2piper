# RoboDuet Piper 机械臂适配计划 (Go2 平台)

> **✅ 适配已完成并验证可跑通 (2026-03-31)**
> 已验证：URDF 正确加载 (26 rigid bodies)，ee_idx=23=piper_link6，DOF props 正确，训练循环正常启动。

本文档详细列出将 RoboDuet 从 ARX5 机械臂适配到 Piper 机械臂所需的全部修改。
目标平台为 **Go2 + Piper**。

## Go1 vs Go2 关键差异（影响适配的部分）

在开始前必须了解，Go1 和 Go2 在 RoboDuet 中有以下结构性差异：

| 项目 | Go1 (arx5p2Go1.urdf) | Go2 (arx5go2.urdf) |
|------|----------------------|---------------------|
| rigid body 总数 | 32 | 37 |
| 臂安装父 link | `trunk` | `base` |
| 安装位置 | `xyz="0.30 0.0 0.04"` | `xyz="0.0 0 0.057"` |
| 额外 link | 无 | Head_upper, Head_lower, 8个 calflower |
| ee_idx (end-effector body) | 23（已验证=zarx_body6） | **23（已验证=piper_link6 ✓）** |
| trunk 质量 | 5.204 kg | 6.921 kg |
| thigh 关节限位 | [-0.663, 2.967] | [-1.571, 3.491] |
| 腿关节连接到 | `trunk` | `base` |

**重要**: 代码中 **没有 Go2 专用的 config 文件**，Go1 和 Go2 共用 `go1_gym/envs/go1/` 下的所有配置。区别仅在 `auto_train.py` 里根据 `--robot go2` 切换 URDF 路径。

---

## 前置准备：从 Piper URDF 中提取的信息

在开始修改前，你需要从 Piper URDF 中确认以下信息（标记为 `[PIPER_TODO]` 的地方需要填入实际值）：

```
1. 关节数量（可控 DOF）:        [PIPER_TODO]  (ARX5 是 6 个可控 + 2 个夹爪)
2. 关节名称列表:                [PIPER_TODO]  (ARX5 是 zarx_j1 ~ zarx_j8)
3. 各关节类型 (revolute/prismatic): [PIPER_TODO]
4. 各关节限位 (lower/upper):     [PIPER_TODO]
5. 末端执行器 link 名称:        [PIPER_TODO]  (ARX5 是 zarx_body6)
6. 安装方式 (固定关节连接到狗身上的位置/姿态): [PIPER_TODO]
7. 各关节推荐 PD 增益 (Kp/Kd):  [PIPER_TODO]
8. 默认初始关节角度 (折叠/收起姿态): [PIPER_TODO]
9. link 名称列表 (用于碰撞惩罚):   [PIPER_TODO]
10. 是否有夹爪及其 DOF 数:       [PIPER_TODO]
```

---

## 修改总览

共需修改 **6 个文件** + 创建 **1 个 URDF**，按依赖顺序排列：

| 步骤 | 文件 | 修改类型 | 难度 |
|------|------|----------|------|
| 1 | `resources/robots/piperGo2/urdf/piperGo2.urdf` | **新建** | 中 |
| 2 | `go1_gym/envs/go1/asset_config.py` | 修改 | 低 |
| 3 | `go1_gym/envs/automatic/legged_robot_config.py` | 修改 | 低 |
| 4 | `go1_gym/envs/automatic/legged_robot.py` | 修改 | **高** |
| 5 | `go1_gym/envs/automatic/__init__.py` | 修改 | 中 |
| 6 | `scripts/auto_train.py` | 修改 | 低 |
| 7 | (后续) `RoboDuet_Deployment/go2_deployment/` | 修改 | 高 |

---

## 步骤 1: 创建合并 URDF

**目标**: 创建 `resources/robots/piperGo2/urdf/piperGo2.urdf`

### 参考结构

Go2 版 ARX5 URDF (`arx5go2.urdf`) 的结构：

```
base
├── [fixed: floating_base] → trunk
│   ├── Head_upper_joint(fixed) → Head_upper → Head_lower
│   ├── FL_hip_joint → FL_hip → FL_thigh → FL_calf → FL_calflower → FL_foot
│   ├── FR_hip_joint → FR_hip → FR_thigh → FR_calf → FR_calflower → FR_foot
│   ├── RL_hip_joint → RL_hip → RL_thigh → RL_calf → RL_calflower → RL_foot
│   └── RR_hip_joint → RR_hip → RR_thigh → RR_calf → RR_calflower → RR_foot
└── [fixed: zarx5p2_mount] → base_link (臂安装基座)
    └── zarx_j1 → zarx_body1
        └── zarx_j2 → zarx_body2
            └── ... → zarx_body6 (末端执行器)
                └── zarx_j7 → zarx_body7 (夹爪1)
                    └── zarx_j8 → zarx_body8 (夹爪2)
```

### 需要做的

1. **复制 Go2 部分**: 从 `arx5go2.urdf` 中提取 Go2 的全部 link/joint 定义（从 `<link name="base">` 到最后一个 `RR_foot` 相关的 joint/link），包括 Head 和 calflower 部分
2. **替换 ARX5 部分**: 删除所有 `zarx_*` 和 `base_link` 的 link/joint，替换为 Piper 的 link/joint
3. **创建安装固定关节**: 参考 Go2 的 `zarx5p2_mount`:
   ```xml
   <!-- Go2 的 ARX5 安装方式（注意：挂在 base 上，不是 trunk） -->
   <joint name="zarx5p2_mount" type="fixed">
       <origin rpy="0 0 0" xyz="0.0 0 0.057"/>
       <parent link="base"/>    <!-- Go2 挂在 base 上 -->
       <child link="base_link"/>
   </joint>
   ```
   - Piper 的安装位置/姿态需根据实际硬件确定，可能与 ARX5 不同
   - **parent link 必须是 `base`**（Go2 的特点，Go1 是 `trunk`）
4. **复制 mesh 文件**: 将 Piper 的 mesh 文件放入 `resources/robots/piperGo2/meshes/`
   - Go2 的 mesh 文件在 `resources/robots/go2/meshes/` 下，URDF 中用相对路径引用
   - ARX5 的 mesh 被引用为 `../../arx5p2Go1/meshes/arx5p2_meshes/linkN.STL`
   - Piper 的 mesh 路径需要相应调整
5. **确保关节顺序**: Isaac Gym 按 URDF 中 revolute/prismatic joint 出现顺序分配 DOF index。Go2 的 12 个腿关节必须在前，Piper 臂关节在后

### 关键注意事项

- Go2 的 `dont_collapse="true"` 属性阻止 Isaac Gym 合并 fixed joint 连接的 link，这会影响 rigid body 总数和索引
- `ee_idx` 取决于 Isaac Gym 加载后的 rigid body 索引（会 collapse 无 `dont_collapse` 的 fixed joint），不等于 URDF 中 link 出现顺序。Go2+ARX5 已验证 ee_idx=23 正确
- **验证方法**: 加载后打印确认
  ```python
  print("Body names:", self.gym.get_asset_rigid_body_names(robot_asset))
  print("DOF names:", self.gym.get_asset_dof_names(robot_asset))
  print("Body count:", self.gym.get_asset_rigid_body_count(robot_asset))
  print("DOF count:", self.gym.get_asset_dof_count(robot_asset))
  ```

---

## 步骤 2: 修改 asset_config.py

**文件**: `go1_gym/envs/go1/asset_config.py`

**注意**: Go1 和 Go2 共用这个文件。URDF 路径在 `auto_train.py` 中按 `--robot` 参数覆盖。此处的修改对 Go1 和 Go2 都生效，但由于我们只用 Go2，可以直接改。

### 2.1 URDF 路径 (默认值，会被 auto_train.py 覆盖)

```python
# 原始 (line 9):
Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'

# 改为 (设为 Go2 版本作为默认):
Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/piperGo2/urdf/piperGo2.urdf'
```

### 2.2 碰撞惩罚 body 列表

```python
# 原始 (line 11-16):
Cnfg.asset.penalize_contacts_on = [
    'base', 'trunk',
    "arm", "wrist", 'zarx',      # ← ARX5 特有名称
    "gripper", "thigh", "calf",
    "Head"                         # Go2 有 Head_upper/Head_lower, Go1 没有但不影响
]

# 改为: 替换 ARX5 link 名称为 Piper 的 link 名称
Cnfg.asset.penalize_contacts_on = [
    'base', 'trunk',
    "[PIPER_TODO: piper link 名称前缀或关键词]",
    "thigh", "calf",
    "Head"   # 保留，Go2 有 Head links
]
```

### 2.3 全局 PD 增益 (stiffness)

```python
# 原始 (line 22):
Cnfg.control.stiffness = {'joint': 35., 'widow': 5., "zarx": 5., "zarx_j3": 20}

# 改为: 去掉 zarx/widow，添加 Piper 关节名前缀
# 'joint' 是通配符，匹配所有包含 "joint" 的关节名（Go2 腿关节是 FL_hip_joint 等）
Cnfg.control.stiffness = {'joint': 35., '[PIPER_TODO: 前缀]': [PIPER_TODO: 值]}
```

### 2.4 臂部 PD 增益 (per-joint)

```python
# 原始 (line 25-45): 每个 zarx_j1~j8 单独设置 stiffness 和 damping
# 改为: 用 Piper 的关节名和推荐增益

Cnfg.arm.control.stiffness_arm = {
    "[PIPER_TODO: 关节名前缀]": [PIPER_TODO],  # 通配符
    "[PIPER_TODO: joint1_name]": [PIPER_TODO],
    "[PIPER_TODO: joint2_name]": [PIPER_TODO],
    # ... 每个关节
}

Cnfg.arm.control.damping_arm = {
    # 同上结构
}
```

**调参建议**: 先用保守值 (stiffness 30-50, damping 3-10)，观察仿真中是否抖动再调整。

### 2.5 默认关节角度

```python
# 原始 (line 54-89): 包含 Go1 腿关节 + widow(历史遗留) + zarx 臂关节
# 改为:
Cnfg.init_state.default_joint_angles = {
    # Go2 腿部关节（名字与 Go1 相同: FL/FR/RL/RR_hip/thigh/calf_joint）
    'FL_hip_joint': 0.1,
    'RL_hip_joint': 0.1,
    'FR_hip_joint': -0.1,
    'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.8,
    'RL_thigh_joint': 1.,
    'FR_thigh_joint': 0.8,
    'RR_thigh_joint': 1.,
    'FL_calf_joint': -1.5,
    'RL_calf_joint': -1.5,
    'FR_calf_joint': -1.5,
    'RR_calf_joint': -1.5,

    # Piper 臂关节（折叠/收起姿态）
    "[PIPER_TODO: joint1_name]": [PIPER_TODO],
    "[PIPER_TODO: joint2_name]": [PIPER_TODO],
    # ... 每个关节
}
# 删除所有 widow_* 和 gripper 相关旧条目（line 70-79 的历史遗留 WidowX 臂定义）
```

---

## 步骤 3: 修改 legged_robot_config.py

**文件**: `go1_gym/envs/automatic/legged_robot_config.py`

### 3.1 ARM 维度参数

```python
class arm(PrefixProto, cli=False):
    num_actions_arm = 6          # ← 改为 Piper 的可控关节数 [PIPER_TODO]
    arm_num_privileged_obs = 9   # 保持不变
    arm_num_observation_history = 30  # 保持不变
    arm_num_observations = 26 - 6    # = 20，即 arm obs 维度
    # 拆解: get_arm_observations() 返回 cat(dof_pos(6), actions(6)) = 12 维
    # 加上后续拼接的 dxyz(3) + dabg(3) + rpy(2) = 8 维
    # 共 20 维
    # 若 Piper DOF ≠ 6:
    #   arm_num_observations = 2 * PIPER_DOF + 8  [PIPER_TODO]

    arm_num_obs_history = arm_num_observations * arm_num_observation_history
    arm_num_commands = 6         # 末端 6D 位姿命令，与臂 DOF 无关，保持不变
    num_actions_arm_cd = 8       # = num_actions_arm + 2 (planning actions)
                                  # ← 改为 [PIPER_DOF] + 2  [PIPER_TODO]
```

### 3.2 HYBRID 参数

```python
class hybrid(PrefixProto, cli=False):
    num_actions = 18    # = 12 (legs) + 6 (arm) ← 改为 12 + [PIPER_DOF]  [PIPER_TODO]
```

---

## 步骤 4: 修改 legged_robot.py (核心，最复杂)

**文件**: `go1_gym/envs/automatic/legged_robot.py`

### 4.1 ee_idx — 末端执行器 body 索引

> **✅ 已验证 (2026-03-30): `ee_idx=23` 在 Go2+ARX5 上是正确的。**
>
> **Isaac Gym collapse 规则**：没有 `dont_collapse="true"` 属性的 fixed joint，其 child link 会被合并到 parent。
>
> Go2+ARX5 URDF 有 37 个 link，其中 11 个被 collapse：
> - Head_upper, Head_lower（fixed，无 dont_collapse）
> - 4×calflower, 4×calflower1（fixed，无 dont_collapse）
> - base_link / 臂底座（fixed `zarx5p2_mount`，无 dont_collapse）
>
> 保留的 fixed joint child（有 `dont_collapse="true"`）：trunk, 4×foot
>
> 37 - 11 = **26 个 rigid body**，与运行时输出一致。index 23 = `zarx_body6`（末端执行器）。
>
> **适配 Piper 时**：用同样的方法计算——数 Piper URDF 中的 link 总数，减去会被 collapse 的 fixed joint child link 数，得到预期 rigid body 数。然后首次运行时打印验证：
> ```python
> body_names = self.gym.get_asset_rigid_body_names(robot_asset)
> print(f"All bodies: {list(enumerate(body_names))}")
> ```
> 找到 Piper 末端 link 的实际索引，更新 `self.ee_idx`。

### 4.2 DOF 属性设置中的 range(8)

```python
# 原始 (约 line 728-732):
if self.cfg.control.control_type == 'M':
    for i in range(8):  # ← 硬编码 8 (ARX5 总 DOF = 6可控 + 2夹爪)
        joint_name = f"zarx_j{i+1}"  # ← 硬编码 ARX5 关节名格式
        joint_idx = self.num_actions_loco + i
        props[joint_idx]['stiffness'] = self.cfg.arm.control.stiffness_arm[joint_name]
        props[joint_idx]['damping'] = self.cfg.arm.control.damping_arm[joint_name]

# 改为:
if self.cfg.control.control_type == 'M':
    num_arm_dofs = self.num_dof - self.num_actions_loco  # 总DOF - 腿DOF
    dof_names = self.gym.get_asset_dof_names(robot_asset)
    for i in range(num_arm_dofs):
        joint_name = dof_names[self.num_actions_loco + i]
        joint_idx = self.num_actions_loco + i
        props[joint_idx]['stiffness'] = self.cfg.arm.control.stiffness_arm.get(
            joint_name, self.cfg.arm.control.stiffness_arm.get("[PIPER_TODO: 前缀]", 30.0))
        props[joint_idx]['damping'] = self.cfg.arm.control.damping_arm.get(
            joint_name, self.cfg.arm.control.damping_arm.get("[PIPER_TODO: 前缀]", 5.0))
```

### 4.3 _keep_arm_fixed 方法

```python
# 原始 (line 234-244):
def _keep_arm_fixed(self):
    if global_switch.switch_open:
        idx = self.num_actions_loco + self.num_actions_arm  # 12 + 6 = 18
    else:
        idx = self.num_actions_loco  # 12
    self.dof_pos[:, idx:] = self.default_dof_pos[:, idx:]
    self.dof_vel[:, idx:] = 0.
```

**无需修改** — 已使用变量，会自动适配。如果 Piper 夹爪 DOF 不是 2，`idx:` 的 slice 范围会自动调整（因为 `self.dof_pos` 维度从 URDF 确定）。

### 4.4 action scaling 和 padding

```python
# 原始 (line 173-175):
actions_scaled = actions[:, :self.num_actions] * self.cfg.control.action_scale
actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction
actions_scaled = torch.nn.functional.pad(actions_scaled, (0, self.num_dof - self.num_actions), "constant", 0.0)
```

**无需修改** — `self.num_actions` 和 `self.num_dof` 会从配置和 URDF 自动确定。pad 的维度 = 总DOF - 可控DOF (即夹爪DOF 数)，会自动适配。

### 4.5 ee mass 添加

```python
# 原始 (line 806):
props[self.ee_idx].mass += 100./1000  # camera mass = 0.1 kg

# ee_idx 换 Piper 后需更新为正确索引（见步骤 4.1）
# 但数值可能需要调整:
props[self.ee_idx].mass += [PIPER_TODO: 末端附加质量 kg]
```

### 4.6 PD 增益分配

```python
# 原始 (line 1339-1341):
self.p_gains[i] = self.cfg.dog.control.stiffness_leg[dof_name] \
    if i < self.num_actions_loco else self.cfg.arm.control.stiffness_arm[dof_name]
```

**无需修改代码** — 使用 `dof_name` 做字典查找。只要 `asset_config.py` 中的 `stiffness_arm` 和 `damping_arm` 字典包含 Piper 的关节名即可。但必须确保 URDF 中的关节名与字典 key **精确匹配**（大小写敏感）。

---

## 步骤 5: 修改 __init__.py

**文件**: `go1_gym/envs/automatic/__init__.py`

### 5.1 硬编码的观测切片

```python
# 原始 (line 641):
obs[:, 12:18] = pose_in_ee

# 这里 12:18 = num_actions_loco : num_actions_loco + 6
# 它把 pose_in_ee (6D 末端位姿) 写入 arm obs 的前 6 维（覆盖 dof_pos）
# 改为:
obs[:, self.env.num_actions_loco : self.env.num_actions_loco + 6] = pose_in_ee
# pose_in_ee 始终是 6D (x,y,z,r,p,y)，与臂 DOF 无关
```

**如果 Piper DOF ≠ 6**: 这个索引的语义会变。arm obs 的前 N 维是 dof_pos (N维)，6维 的 pose_in_ee 会覆盖前 6 维。如果 N > 6，则后面 N-6 维的 dof_pos 不会被覆盖，导致 obs 混乱。如果 N < 6，则 pose_in_ee 会溢出到 actions 区域。需要重新设计 `get_arm_observations_hand` 的观测拼接逻辑。

### 5.2 arm 观测构建

```python
# 原始 (line 56-61):
obs_buf = torch.cat((
    (self.dof_pos[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]
     - self.default_dof_pos[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm])
     * self.obs_scales.dof_pos,
    self.actions[:, self.num_actions_loco:self.num_actions_loco+self.num_actions_arm]
), dim=-1)
```

**无需修改** — 已使用 `self.num_actions_loco` 和 `self.num_actions_arm` 变量。

---

## 步骤 6: 修改 auto_train.py

**文件**: `scripts/auto_train.py`

### 6.1 URDF 路径

```python
# 原始 (line 151-154):
if args.robot == "go1":
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'
elif args.robot == "go2":
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf'

# 改为:
if args.robot == "go1":
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'  # Go1 保持不变
elif args.robot == "go2":
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/piperGo2/urdf/piperGo2.urdf'
```

### 6.2 维度参数

```python
# 原始 (line 79-80):
Cfg.env.num_actions = 18        # = 12 (legs) + 6 (arm)
Cfg.env.num_observations = 63   # = dog_obs + arm_obs

# 若 Piper DOF = 6: 保持不变
# 若 Piper DOF ≠ 6:
#   num_actions = 12 + [PIPER_DOF]  [PIPER_TODO]
#   num_observations = 需重新计算  [PIPER_TODO]
#     dog_obs 不变 (它包含 dxyz/dabg/rpy 但不直接包含臂的 dof)
#     arm_obs = 2 * PIPER_DOF + 8
#     num_observations = dog_num_observations + arm_num_observations
```

### 6.3 WandB entity

已确认 `entity="1309519635"` 是当前用户的账号，**无需修改**。

### 6.4 训练启动

```bash
# 使用 --robot go2 启动训练
python scripts/auto_train.py --num_envs 4096 --run_name piper_go2 --sim_device cuda:0 --robot go2 --headless
```

---

## 修改检查清单

完成适配后，用以下检查确认无遗漏：

### 编译/加载检查
- [ ] `piperGo2.urdf` 能被 Isaac Gym 正确加载（无解析错误）
- [ ] mesh 文件路径正确、文件存在
- [ ] 打印 `gym.get_asset_dof_names()` 确认关节顺序：前 12 个是 Go2 腿关节，之后是 Piper
- [ ] 打印 `gym.get_asset_rigid_body_names()` 确认 Piper 末端 link 的实际索引，更新 `ee_idx`（见步骤 4.1）
- [ ] 打印 `gym.get_asset_dof_count()` 确认 DOF 总数 = 12 + Piper总DOF(含夹爪)

### 运行检查
- [ ] `python scripts/auto_train.py --debug --robot go2` 能启动且不报错
- [ ] 仿真中 Go2 站立稳定（腿部不受影响）
- [ ] `keep_arm_fixed=True` 时 Piper 臂保持默认姿态不飘移
- [ ] 臂的 PD 控制正常（不飞走、不剧烈抖动）
- [ ] 末端位姿追踪 reward 正常计算（值不是 nan/inf）
- [ ] Head link 碰撞惩罚正常工作

### 搜索遗漏
```bash
# 在 RoboDuet 目录下运行，确认无残留的 ARX5 硬编码
grep -rn "zarx" go1_gym/ scripts/ --include="*.py"
grep -rn "arx5" go1_gym/ scripts/ --include="*.py"
grep -rn "= 23" go1_gym/ scripts/ --include="*.py"  # ee_idx（换 Piper 后需更新）
grep -rn "range(8)" go1_gym/ scripts/ --include="*.py"  # 旧臂 DOF 循环
grep -rn "12:18" go1_gym/ scripts/ --include="*.py"  # 硬编码的臂 obs 索引
```

---

## 关于 Piper DOF = 6 的简化情况

如果 Piper 的可控关节数恰好也是 **6 DOF**（不含夹爪），则大多数维度参数无需修改，只需要：

1. 创建合并 URDF (`piperGo2.urdf`)
2. 修改关节名称映射 (stiffness/damping/default_angles 字典)
3. 根据 Piper collapse 后的 body 列表更新 `ee_idx`（见步骤 4.1 的计算方法）
4. 修改碰撞 body 名称列表
5. 修改 URDF 路径 (`auto_train.py` 和 `asset_config.py`)
6. 修复 `__init__.py` 中 `12:18` 硬编码

如果 Piper 是 **7 DOF**（很多工业臂是 7 DOF），则还需要调整：
- `num_actions_arm` (6→7)
- `num_actions` (18→19)
- `num_actions_arm_cd` (8→9)
- `arm_num_observations` (20→22)
- `num_observations` (63→65)
- `__init__.py` 中的观测拼接索引
- `rewards.py` 中 `:-2` 的 smoothness 切片逻辑（夹爪 DOF 数可能不同）

---

## 暂不修改的部分 (后续计划)

### RoboDuet_Deployment/go2_deployment 适配

部署代码需要等训练完成后再适配，主要修改点：

1. **`go2_arx_deploy/envs/lcm_agent.py`**: 臂关节索引、观测构建中的 `12:18` 索引、PD 增益
2. **`go2_arx_deploy/envs/arm_ac.py`**: 网络输入/输出维度（若 DOF 变化）
3. **`go2_arx_deploy/unitree_legged_sdk_bin/lcm_position_vr_go2.cpp`**: CAN 通信协议需改为 Piper 的通信接口（Piper 可能不用 CAN，需查明）
4. **`go2_arx_deploy/remote_pub.py`**: VR 命令映射（不变，仍是 6D 末端位姿）
5. **`go2_arx_deploy/scripts/deploy_policy_vr.py`**: checkpoint 路径、可能需要的模型维度调整

### 训练策略建议

1. 先用 `--debug` 跑几个 iteration 确认环境正常加载
2. 第一阶段: `keep_arm_fixed=True` 训练腿部策略 (~10000 iterations)
3. 第二阶段: 解冻臂部联合训练
4. 建议先用 Go2+ARX5 (`arx5go2.urdf`) 跑通一次完整训练，理解训练曲线和正常行为后再切换到 Piper
