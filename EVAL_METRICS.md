# Position-Force 自动评测说明

本文档说明 [eval_posforce.py](legged_gym/scripts/eval_posforce.py) 的设计目标、评测场景、指标含义和结果解读方式，供使用者在查看自动评测报告时参考。

## 1. 评测目标

这个自动评测不是简单地把 `play` 跑一遍然后取平均 reward，而是按照 UniFP 论文的核心目标，把当前策略拆成几类更容易解释的能力来测：

- 位置跟踪能力
- 混合力位控制下的补偿能力
- 机身受扰后的速度补偿能力
- force estimator 质量
- whole-body 稳定性、顺滑性和鲁棒性

当前脚本的重点是“低层控制能力 benchmark”。它适合做：

- 不同 checkpoint 之间的横向对比
- 调参前后的快速回归测试
- position-force 任务的阶段性质量检查

它不直接替代真实任务成功率评测，例如开门、擦黑板、抽屉等接触任务。

## 2. 运行方式

```bash
cd legged_gym/scripts
python eval_posforce.py --load_run=<run_name> --headless
```

当前脚本默认评测 `b2z1_pos_force`。如果需要回到 Go2+Piper，需要显式加 `--task=go2_piper_pos_force`。

常用参数：

- `--eval_case all`
  运行完整 benchmark。也可选 `position_only`、`hybrid_force_position`、`arm_force_estimation`、`base_force_estimation`、`base_disturbance`、`mixed_whole_body`
- `--eval_repeats <N>`
  每个 scripted scenario 重复运行 `N` 次，再做汇总。`N=1` 时使用正常的 `--seed` / config seed；`N>1` 时每个 case repeat 使用随机 seed，并在 `summary.json` 的 metadata 里记录实际 seed
- `--num_envs <N>`
  并行评测环境数
- `--output_dir <dir>`
  评测报告输出目录，默认是项目根目录下的 `eval_reports/`
- `--no_report`
  只在终端打印评测摘要，不导出 `summary.json` 和 `summary.md`

输出文件：

- `summary.json`
  机器可读结果，适合做版本对比和后处理，也会记录这次实际评测解析到的 run 和 checkpoint
- `summary.md`
  人类可读摘要，也会记录这次实际使用的模型文件、run 名和 checkpoint

默认情况下，评测输出文件夹名也会直接带上解析到的 run 名和 checkpoint，方便后续在调参记录里引用和回溯。

如果加了 `--no_report`，脚本仍会完整跑完 benchmark，并在终端打印总分和各 case 摘要，但不会创建输出目录，也不会写任何报告文件。

## 3. Benchmark 场景

## 3.1 Position-Only

目的：

- 测纯位置控制下的 EE 跟踪能力
- 测 reach 时的稳定性和姿态质量

特点：

- 不注入 force command
- 不注入外部 force disturbance
- 仅给一组固定 EE 笛卡尔目标点

当前脚本内包含：

- 前上方 reach
- 前下方 reach
- 左侧 reach
- 右侧 reach

## 3.2 Hybrid Force-Position

目的：

- 测试在有 EE force command 和外部力时，策略能否跟踪“补偿后的目标”
- 评估 force-aware compensation 是否工作正常

特点：

- 给固定 EE 目标点
- 同时给 EE force command
- 同时注入 EE 外部力

注意：

- 当前 benchmark 并没有构造显式接触物体任务
- 因此这里更准确地说是在测“力补偿下的目标跟踪”和“force estimator 质量”
- 它不是现实接触任务里的最终力控制成功率
- 主 tracking、success 和 settling 指标只统计真正的 `force_track` 段，不把前面的 `pre_force` 准备段混进去

## 3.3 Arm Force Estimation

目的：

- 专门测试机械臂 / 末端外力估计是否真的学会
- 避免全局 estimator 指标被大量 0 力样本稀释
- 直接观察真实 EE 外力存在时，预测力是否有幅值响应

特点：

- 不给 EE force command，只注入 EE external force
- 分别测试 `x`、`z` 和 `xy` 方向外力
- 只把 `force_probe` 段用于核心力估计评分

核心输出：

- nonzero-force relative accuracy
- all-sample MAE
- nonzero-force-only MAE
- 非零外力段的真实力范数均值与预测力范数均值

## 3.4 Base Force Estimation

目的：

- 专门测试狗本体 / 底座外力估计是否真的学会
- 检查 base external force 存在时，estimator 是否仍然塌缩为接近 0 的预测
- 与 `base_disturbance` 的运动补偿评分分离，避免“速度没坏”被误读成“力估计正确”

特点：

- 不给 base force command，只注入 base external force
- 分别测试 `x`、`y` 和 `xy` 方向外力
- 只把 `force_probe` 段用于核心力估计评分

核心输出：

- nonzero-force relative accuracy
- all-sample MAE
- nonzero-force-only MAE
- 非零外力段的真实力范数均值与预测力范数均值

## 3.5 Base Disturbance

目的：

- 测机身速度跟踪
- 测 base force compensation
- 测受扰后能否仍按补偿速度运动

特点：

- 给固定 `vx / vy / yaw`
- 再加 base force command 和 base 外部力
- 重点看底盘速度是否接近“补偿后的目标速度”
- 平移受扰 tracking 只统计 `disturbance` 段
- `base_yaw_tracking` 只用于 yaw 维度，不再混入平移速度 RMSE

注意：

- `base_disturbance` 更偏“受扰后速度控制是否还能工作”
- `base_force_estimation` 才是专门的“底座力估计是否准确”测试

## 3.6 Mixed Whole-Body

目的：

- 给一个更接近 whole-body 工作状态的综合序列
- 同时考察 EE reach、base move、force compensation、姿态和顺滑性

特点：

- 多段指令连续切换
- 包含移动、reach、force、受扰

这个场景更适合作为“综合健康度检查”，而不是单一能力上限测试。

## 4. 指标分层

报告里的指标分成两类：

- 百分数指标
  好处是直观，适合快速比较
- 物理单位指标
  好处是可解释，便于调参

建议不要只看总分。最好的阅读方式是：

1. 先看各 case score
2. 再看关键原始量，例如 `cm / m/s / N / s`
3. 再看 `Runtime Quality`
4. 最后看 overall score

这里特别说明一下：

- `Runtime Quality`
  表示运行品质，主要总结模型是否稳、是否顺、是否乱撞、是否滑步
- `Overall`
  表示整套 benchmark 的总评，既包含运行品质，也包含 position / hybrid / disturbance / estimator 等任务能力

所以 `Runtime Quality` 高但 `Overall` 不高是完全可能的。这通常意味着模型“运行得稳”，但“任务能力还不够强”。

## 5. 位置相关指标

### 5.1 EE Reach Success Rate

含义：

- 某个 reach 场景结束前的最后一个时间窗内
- 若 EE 误差低于阈值，则记为成功

默认主要阈值：

- `5 cm` 左右用于纯位置场景
- `6 cm` 左右用于 hybrid 补偿场景

解读：

- 更适合看“最终是否到位”
- 对最终是否能用很敏感

### 5.2 EE Tracking Score

含义：

- 由 EE 跟踪误差映射成 `0-100%`
- 误差越小，分数越高

解读：

- 比 success rate 更平滑
- 更适合比较“两个模型都能到，但谁更准”
- 当前脚本中，tracking score 使用的 RMSE 与 success 一样，只看每个 scripted scenario 结束前最后 `0.25 s` 的 tracking 窗口，不把刚切目标后的过渡误差混进去

### 5.3 EE RMSE

单位：

- `cm`

包括两种口径：

- `Nominal EE RMSE`
  相对原始位置目标
- `Compensated EE RMSE`
  相对力补偿后的目标

统计窗口：

- 只统计每个 scripted scenario 的最后 `0.25 s` tracking 窗口
- `warmup`、`pre_force` 等准备段不参与
- 目标刚切换后的到达过程不再计入 RMSE，但仍会体现在 `Settling Time` 里

解读：

- `Position-Only` 主要看 nominal
- `Hybrid Force-Position` 主要看 compensated

### 5.4 EE RPY RMSE

单位：

- `deg`，同时在 `summary.json` 中保留 `rad` 字段

含义：

- 把当前末端姿态和目标末端姿态都按 RPY 表示
- 对 roll / pitch / yaw 分别计算包裹到 `[-pi, pi]` 的角度误差
- 报告里单独显示一行 `EE RPY RMSE`，不和 XYZ 位置 RMSE 混在一起

统计窗口：

- 与 EE XYZ RMSE 一样，只统计每个 scripted scenario 的最后 `0.25 s` tracking 窗口
- 这个指标只是已有 tracking case 的附加读数，不是一个新的 task，也不改变现有 case score 权重

解读：

- XYZ RMSE 低但 RPY RMSE 高，说明末端到点了但姿态没有跟上
- RPY RMSE 低但 XYZ RMSE 高，说明姿态对齐了但位置还没到位

### 5.5 Settling Time

单位：

- `s`

含义：

- 目标变化后，误差首次进入容差带并维持一小段时间所需的时间

解读：

- 看“收敛速度”
- 同样 final error 很小的两个模型，settling time 更短者通常更好用

## 6. 速度与受扰指标

### 6.1 Base Nominal Velocity RMSE

单位：

- `m/s`

含义：

- 机身实际速度与原始速度命令之间的误差
- 报告中的 RMSE 使用对应 tracking scenario 最后 `0.25 s` 的窗口

### 6.2 Base Compensated Velocity RMSE

单位：

- `m/s`

含义：

- 机身实际速度与“受力补偿后的目标速度”之间的误差
- 报告中的 RMSE 使用对应 tracking scenario 最后 `0.25 s` 的窗口

解读：

- 在 `base_disturbance` 场景里，这个指标比 nominal velocity 更重要

### 6.3 Disturbance Band Accuracy

单位：

- `%`

含义：

- 在受扰阶段，有多少时间样本满足速度误差仍在容差带内

解读：

- 更像“受扰时能保持可用”的时间比例

### 6.4 Yaw Rate RMSE

单位：

- `rad/s`

含义：

- 偏航角速度跟踪误差
- 报告中的 RMSE 使用对应 yaw tracking scenario 最后 `0.25 s` 的窗口

解读：

- 防止只顾平移，不看转向控制

## 7. Estimator 指标

论文里 estimator 是 UniFP 非常关键的一层，因此这里单独拉出来测。

当前脚本直接比较：

- 策略 latent 解码输出
- 环境提供的 `obs_pred` 真值

这里的时序口径与训练保持一致：

- 用当前 observation 送入 policy，得到当前时刻的 latent 预测
- 再与同一时刻的 `obs_pred` 真值比较

它不是 next-step prediction 指标。

评测 4 项：

### 7.1 Base Velocity Estimation MAE

单位：

- `m/s`

### 7.2 EE Position Estimation MAE

单位：

- `cm`

说明：

- 环境内部预测的是 EE 球坐标量
- 报告中会先反算成局部笛卡尔坐标，再报告 `cm`
- 这样更直观

### 7.3 EE Force Estimation MAE

单位：

- `N`

报告同时包含：

- all-sample MAE
- nonzero-force-only MAE
- nonzero-force relative accuracy
- target force norm mean
- predicted force norm mean
- zero-force false-positive prediction norm

原因：

- 默认 benchmark 中很多阶段的外部力为 0
- 如果只看全局 MAE，一个永远预测 0 的 force estimator 也可能看起来不错
- 因此需要单独看非零 force 段，确认 estimator 在真正受力时是否有响应
- nonzero-force relative accuracy 使用 `1 - nonzero_mae / nonzero_target_norm` 的直观口径；如果真实存在明显外力但预测为 `0N`，该项会接近 `0%`

### 7.4 Base Force Estimation MAE

单位：

- `N`

报告同样包含 all-sample 与 nonzero-force-only 诊断。尤其要关注：

- `base_disturbance / disturbance`
- `mixed_whole_body / move_disturbance`

这些 tag 下的 base force target norm、predicted norm 和 nonzero MAE 能直接暴露“真实 base 外力存在时 estimator 是否仍然预测 0”。其中 `base_force_estimation` 是更干净的专项测试，因为它只注入外部力，不叠加 base force command。

### 7.5 Estimator Overall Score

单位：

- `%`

由 4 项 estimator 分数平均得到。force estimator 分支现在使用 nonzero-aware score：

- 如果该分支存在非零 force 样本，则综合 all-sample score 和 nonzero-force relative accuracy
- 如果没有非零 force 样本，则退回 all-sample score

解读：

- 如果 estimator score 很低，但 tracking 还可以，说明策略可能主要靠动作记忆或直接映射在硬撑
- 如果 estimator score 高但控制分不高，说明低层估计不错，但策略使用得还不够好
- 如果 all-sample force MAE 很低但 nonzero-force MAE 很高，说明报告被零力样本稀释，不能认为 force estimator 已经学好

报告还会在每个 case 的非零 force tag 下输出 estimator diagnostics，便于直接定位是哪类受力阶段出了问题。

## 8. Runtime Quality：稳定性与运行品质

### 8.1 Survival Rate

单位：

- `%`

含义：

- 每个 scripted scenario 中，机器人没有提前 reset 的比例

这是一个很硬的可靠性指标。

### 8.2 Posture Quality Score

单位：

- `%`

含义：

- 根据 base roll / pitch 偏差计算

解读：

- 分数高代表姿态整体更稳
- 但它不是要求机器人完全不倾斜
- 合理的 whole-body 倾身是允许的，重点是避免过大姿态误差

### 8.3 Foot Slip Ratio

单位：

- `%`

含义：

- 足端接触地面时，横向速度过大的比例

解读：

- 比较能反映底盘稳定性和地面交互质量

### 8.4 Contact Cleanliness Score

单位：

- `%`

含义：

- 统计 penalized contact bodies 的碰撞事件比例

解读：

- 看机身、腿、臂是否经常有非期望碰撞

### 8.5 Stability Score

单位：

- `%`

当前由以下部分综合：

- survival
- posture quality
- contact cleanliness
- foot slip

在报告中，这一整组指标会被汇总到 `Runtime Quality` 小节，而不是总评小节。

## 9. 控制代价指标

### 9.1 Smoothness Score

单位：

- `%`

依据：

- action delta
- torque delta

解读：

- 更高代表控制更平滑
- 对 sim2real 友好度通常也更有参考价值

### 9.2 Energy Proxy Mean

单位：

- 无严格物理标定的 proxy

含义：

- 使用 `|torque * joint velocity|` 的平均量作为控制代价代理

解读：

- 更适合做相对比较
- 不建议把它单独当成唯一优劣依据

## 10. 分数汇总方式

当前 overall score 采用加权汇总：

- `16%` Position score
- `20%` Hybrid force-position score
- `10%` Arm force estimation score
- `10%` Base force estimation score
- `16%` Base disturbance score
- `8%` Mixed whole-body score
- `12%` Estimator overall score
- `6%` Stability score
- `2%` Smoothness score

这样设计的原因是：

- 这个项目不是纯 locomotion 项目
- UniFP 的核心卖点是统一的 force-position control 和 estimator
- 现在把臂端和底座的 force estimator 拆成独立大项，避免 0 力样本或速度控制结果掩盖力估计问题
- 但稳定性与控制质量仍然必须保留权重

如果只运行单个 `eval_case`，脚本会只对本次实际参与的分项重新归一化加权，不会把没有运行的 case 直接按 0 分算入总分。

另外：

- `Runtime Quality` 不是 `Overall`
- `Runtime Quality` 更像“运行品质小结”
- `Overall` 才是最终总评

## 11. 如何读结果

推荐顺序：

### 11.1 先看 case score

例如：

- `position_only` 很高，但 `hybrid_force_position` 很低  
  说明纯 reach 没问题，但 force-aware 行为不足

- `hybrid_force_position` 很高，但 `base_disturbance` 很低  
  说明末端补偿好，但 locomotion 受扰性能弱

### 11.2 再看原始物理量

重点建议看：

- `Compensated EE RMSE`
- `Base Compensated Velocity RMSE`
- `EE Position Estimation MAE`
- `EE Force Estimation MAE`
- `Settling Time`

### 11.3 再看 Runtime Quality 和总分

- 如果 `Runtime Quality` 高，说明模型整体运行风格不错
- 如果 `Overall` 低，说明真正的任务能力项仍有短板
- 总分适合做 checkpoint 排序，但不适合单独替代诊断

## 12. 当前 benchmark 的边界

这套自动评测是“低层统一控制能力 benchmark”，它目前不直接包含：

- 接触物体后的真实任务成功率
- 开门、擦拭、抽屉等任务级成功判定
- 真实世界的传感器噪声和硬件约束

因此它最适合回答的问题是：

- 这个 checkpoint 是否比上一个更稳
- 位置 / force-aware / estimator 哪一块退化了
- 改配置后整体质量有没有上升

它不适合直接回答：

- 这个策略是否已经能稳定完成真实接触任务

如果后面你要往真实 task benchmark 走，可以在这套脚本之外，再补一层 object/task-level 评测。
