# Go2W 步态开发整理

整理日期：2026-09-20。任务：`OmniBot-Go2W-Gait-Flat-MLP-v0`。

当前实现是一套共享 MLP 策略，通过连续 command 控制 rolling、trot、bound 和机身高度，
不是三个独立网络，也不向网络输入离散 gait ID。详细公式、参考来源和历史训练记录见
[WTW 技术说明](go2w_gait_wtw.md)。

本功能定位为步态与高度控制，不是跳跃、恢复等跨技能策略。环境配置为 `gait_env_cfg.py`，
新训练的日志分组为 `omnibot_go2w_gait_flat_mlp_v0`。历史训练目录、checkpoint 和产物路径保留原名；
任务更名不改变网络、观测、动作或奖励，旧步态 checkpoint 仍可通过显式路径回放。

## 已开发的内容

| 模块 | 内容 | 主要文件 |
|---|---|---|
| 任务与训练入口 | 继承 Go2W Flat 的训练/Play 环境；注册步态任务；独立 MLP PPO 配置 | [环境配置](../locolab/source/locomotion/tasks/Go2W/gait_env_cfg.py)、[任务注册](../locolab/source/locomotion/tasks/__init__.py)、[PPO 配置](../locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py) |
| 连续指令 | 10 维 command、三类步态采样、heading 控制、静止采样、外部手动设定 | [指令实现](../locolab/source/locomotion/mdp/command/wtw_command.py)、[指令配置](../locolab/source/locomotion/mdp/command/wtw_command_cfg.py) |
| 相位与接触计划 | WTW 附录 D 的四轮相位映射、支撑占空比、平滑接触概率、相位邻域采样 | [wtw_math.py](../locolab/source/locomotion/mdp/wtw_math.py) |
| Go2W 奖励适配 | 轮底高度、轮地接触点速度、摆动轮速、落脚位置及机身高度追踪 | [wtw_rewards.py](../locolab/source/locomotion/mdp/wtw_rewards.py) |
| 高度参考姿态 | Go2W 解析 IK，为姿态奖励提供目标高度对应的关节参考 | [wtw_kinematics.py](../locolab/source/locomotion/mdp/wtw_kinematics.py) |
| 回放与评估 | 9 段带指令/实际状态标注的视频、逐帧 CSV、汇总 JSON、预览帧 | [录制脚本](../locolab/scripts/locomotion/record_wtw_showcase.py) |
| 回归验证 | 指令、相位、接触几何、IK、奖励及动作/观测隔离契约 | [test_go2w_wtw.py](../tests/test_go2w_wtw.py) |

功能代码范围共 **8 个新增 Python 文件、2 个已有 Python 文件中的接入改动**。
不包含同一工作区中的通用视频修复，以及已有的 rough/jump/高台/倒立任务。

## 当前指令与策略接口

| command 列 | 含义 | 当前训练采样 |
|---|---|---|
| 0 | 前后速度 | [-1, 1] m/s |
| 1 | 侧向速度 | [-0.5, 0.5] m/s |
| 2 | 转向角速度 | [-π, π] rad/s；80% 环境由 heading 误差生成 |
| 3 | 机身离地高度 | [0.22, 0.46] m |
| 4 | 步态频率 | 三类步态均为 [1.5, 4] Hz |
| 5–7 | 三个相位参数 | 步态中心附近 ±5°，任意轮对相对相位偏差不超过 10° |
| 8 | 支撑占空比 | trot/bound 为 0.5；rolling 为 1 |
| 9 | 轮底摆动高度 | trot/bound 为 [0, 0.1] m；rolling 为 0 |

- 相位中心：trot `(0.5, 0, 0)`，bound `(0, 0, 0.5)`，rolling `(0, 0, 0)`。
- 轮序 FR、FL、RR、RL；trot 对角轮同步，bound 前轮成对/后轮成对。
- rolling 的内部相位持续推进，但 actor/critic 的 4 维时钟观测均置零，避免周期输入诱发腿部抖动。
- 10% 环境静止，其余 rolling/trot/bound 等概率；每 10 s 重采样，episode 为 20 s。
- 保留 Flat 域随机化；关闭不适用于该 10 维 command 的旧速度课程。
- Actor 64 维、critic 318 维、action 16 维；均包含 10 维 command 和 4 维时钟观测。
- 腿部动作仍为 `q_default + 0.25 × action`，轮速 scale 为 10。
  Actor 关节观测仍为 `q_actual - q_default`，critic 关节位置仍为绝对角度。

## 奖励和高度 IK 的最终边界

| 奖励/代价 | 当前权重 | 说明 |
|---|---|---|
| 机身高度误差平方 | -50 | 从 -10 增强 5 倍，追踪 command[3] |
| 摆动接触力 / 支撑接触点速度 | -4 / -4 | 使用相位对应的软接触目标；接触点速度包含轮子滚动 |
| 轮底摆动高度误差 | -30 | 按 command[9] 和摆动相位生成目标 |
| Raibert 落脚位置 | -10 | rolling 关闭 |
| 摆动轮速 | -0.002 | 惩罚摆动期轮子转动，rolling 不受罚 |
| 静止姿态 / hip 默认姿态 | -0.5 / -0.5 | 静止项要求速度范数 <0.1 且全支撑 |
| 竖直速度 / roll-pitch 角速度 / 水平姿态 | -0.02 / -0.001 / -5 | 与当前步态配置一致 |

其余速度追踪、力矩、限位、动作平滑和碰撞项沿用当前分支的 Flat 配置。
`thigh_calf_default`、`penalize_lin_vel_xy` 和 `stay_alive` 在本任务关闭。

高度 IK 保持机身水平、hip=0 和原始轮心 x/y，求解 thigh/calf。
**它只改变奖励参考，不改变动作零点、关节观测或物理 reset。**
当前运动中的 thigh/calf 默认姿态奖励仍未开启，因此 IK 的主要有效作用在静止姿态项。
超出可达范围或软限位的高度请求会被拒绝，不裁剪 action。

已撤回的高度相关动作适配器和观测函数不在有效实现中。
外部全支撑 `frequency=0` 的兼容输入、关闭 reward IK 的配置开关仍有测试覆盖，
属于保留的接口能力，不是废弃训练路径。

## 已完成的训练和效果

最新训练：`2026-09-20_01-10-58_multi_skill_wtw_height50_rewardik_phase5_rollclock0_4096_remote_gpu4_5k_video`。
服务器 GPU 4，4096 环境，从零训练 5000 次，最终模型 `model_4999.pt`；W&B 和训练视频已启用。

- [最新 W&B](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/nhfgmyxj)
- [最终 checkpoint、ONNX / TorchScript 网络与使用说明](../deploy/policy/go2w/gait_wtw_height50_20260920/README.md)
- [最新 72 秒回放及完整数据](../artifacts/go2w_multi_skill/20260920_height50_showcase/README.md)
- [此前高度权重 -10 的对比回放](../artifacts/go2w_multi_skill/20260919_phase5_showcase/README.md)
- [保留产物与清理记录](../artifacts/go2w_multi_skill/README.md)

两轮采用同一展示协议：0.6 m/s，频率 2.5 Hz，标准相位，三种步态各展示 34/22/46 cm。
最新视频 72 秒、3600 帧，九段均无自动重置。目标 22 cm 时的实际平均高度：

| 步态 | 前一轮 | 最新 |
|---|---|---|
| trot | 35.7 cm | 26.3 cm |
| bound | 34.6 cm | 24.8 cm |
| rolling | 28.8 cm | 22.2 cm |

低、中高度误差减小，但低高度 trot/bound 仍偏高。这是有限回放，不是全范围鲁棒性或实机验证。

## 验证与尚未开发的部分

任务更名前已完成 73 项 CPU/回归测试、服务器短程 GPU 验证和上述最终模型回放。
测试覆盖 IK 与独立 URDF 正运动学一致性（1001 个高度）、软限位/可达性、相位邻域与轮对约束、
rolling 时钟屏蔽、指令切换、奖励索引，以及 IK 不进入动作/关节观测。
任务更名只调整注册 ID、配置类/文件、日志分组、回放入口及文档，不改变训练行为。
保留视频和数据的完整性已检查；更名未启动新的 GPU 仿真或训练。

尚未实现：WTW 原始 RMA/步态网格课程、该 10 维接口的 sim2sim/实机部署适配、全速度/相位范围评估、
连续步态/高度切换稳定性专项评估。本任务并非原始 Go1 方法的完整复现。

## 分支和清理范围

开发分支为 `codex/multi-skill`，步态功能基于 `bdc66cd` 开发；本次任务更名不修改分支名。
步态功能与文档单独提交；工作区中的通用视频修复、开发规范以及 rough 模型/评估产物不属于该功能。

本次只删除明确废弃的本地中间产物；最终/基线视频、逐帧数据、来源记录、当前启动脚本和全部服务器模型保留。
