# M20 配方参考（已停用整套迁移）

当前 Go2W Rough 采用原 Go2W 环境加选择性迁移的 M20 腿姿态、对角支撑和对角关节一致性奖励。
`gait_level` 已从活动课程和奖励中移除；力矩、action rate 和接触力使用 Isaac Lab 原始函数，
action smoothness 使用本仓库的 reset-aware 实现，不再乘 gait_level。

当前配置为：自碰撞开启、orientation=-1、stay_alive=None、velocity_slack=0.25，
保留原线/角速度精确跟踪。thigh/calf 使用局部地形高度跨度及命令方向预瞄放宽，
不再以地块 origin 的绝对高度判断；hip/thigh/calf 权重为 -1.5/-1/-0.5。
转向腾空、转向滑动、接触占空比和机身高度奖励保持关闭。

本目录带 gait_level 的函数及 `tasks/Go2W/m20_env_cfg.py` 保留用于历史配方参考，
不注册 `-M20-` 任务。下文整包配方、PPO 修改和启动命令均为历史记录。

当前使用 `OmniBot-Go2W-Rough-MLP-v0`、`OmniBot-Go2W-Rough-Rnn-v0`、
`OmniBot-Go2W-Rough-DreamWaQ-v0`；详细边界与实验次序见
[逐项迁移方案](../../../../../docs/m20_incremental_migration.md)。

历史任务：`OmniBot-Go2W-M20-Rough-v0` / `OmniBot-Go2W-M20-Rough-Rnn-v0`（原 Rough RNN）、
`OmniBot-Go2W-M20-Rough-MLP-v0`（原 Rough MLP）和 `OmniBot-Go2W-M20-Rough-DreamWaQ-v0`（原 Rough DreamWaQ）。

本移植将 DeepRoboticsLab/rl_training 的 M20 Rough 训练配方适配到 Go2W，
单独注册任务，不覆盖 `OmniBot-Go2W-Rough-v0`，也不向公共 `mdp` 命名空间导出同名奖励。

## 来源与范围

固定来源提交：[`6d317dfb33060226139e38600510e1751372eb5d`](https://github.com/DeepRoboticsLab/rl_training/tree/6d317dfb33060226139e38600510e1751372eb5d)。

使用的上游文件位于 `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/`：

- `velocity_env_cfg.py`
- `config/wheeled/deeprobotics_m20/rough_env_cfg.py`
- `mdp/commands.py`、`mdp/curriculums.py`、`mdp/rewards.py`、`mdp/events.py`
- 地形默认值按上游声明的 Isaac Lab `v2.3.2` 展开到本地配置，避免随依赖版本改变。

保留上游 Deep Robotics 和 Ziqi Fan 的版权声明；许可证见本目录的 `LICENSE`（BSD 3-Clause）
及 `LICENSE-robot_lab`（Apache 2.0）。本目录实现和 `tasks/Go2W/m20_env_cfg.py` 是对上述代码的修改。

## 移植内容

| 部分 | 配置 |
| --- | --- |
| Rewards | 22 个非零奖励项及原始权重；保留强水平姿态惩罚、转向腾空/支撑/占空比奖励和姿态放宽条件 |
| Terminations | 20 秒超时、地图边界截断和重力投影姿态判据；不使用 base 接触终止 |
| Curriculum | 位移驱动的地形升降级；平均地形等级映射到 `gait_level`；不启用速度范围课程 |
| Terrain | 六类 8×8 m 地形；台阶/方块/随机粗糙各 20%，两种斜坡各 10%；10×20 网格 |
| Commands | x ±2 m/s、y ±1 m/s、配置 yaw ±2 rad/s；每 10 秒采样；20% 静止、20% 原地转向、各 2% 纯前后/侧移 |
| Events | 摩擦/恢复系数、base 与非 base 质量、惯量、base 质心、reset PD 增益及状态、外力/力矩、周期推扰 |

环境配方迁移包括以上六部分。Actions 直接继承原 Go2W 配置：全部腿关节位置动作 scale=0.25，
轮速度动作 scale=10，不添加动作 clip。RNN 任务继承原 `OmniBot-Go2W-Rough-v0` 使用的
`Go2WRoughRnnDecoderPPORunnerCfg`，保留其 GRU 256、decoder、归一化和训练调度；
实验目录为 `omnibot_go2w_m20_rough_v0`。
MLP 任务继承本仓库的 `Go2WRoughMLPPPORunnerCfg`，使用独立目录
`omnibot_go2w_m20_rough_mlp_v0`，保留原 MLP 网络结构。
DreamWaQ 任务继承 `Go2WRoughDreamWaQPPORunnerCfg` 和 `Go2WDreamWaQObservationsCfg`，
沿用其 5 帧历史、速度监督、重建掩码和网络结构，使用独立实验目录
`omnibot_go2w_m20_rough_dreamwaq_v0`，六部分环境配方与 MLP/RNN 共用。

2026-09-12 按用户要求，三个 M20 runner 统一覆盖以下原版 M20 训练参数：
PPO 初始学习率 `1e-3`、熵系数 `0.003`、动作噪声参数化 `log`（每维学习 log_std）。
其余 PPO 参数及网络结构保持原 Go2W 设置；DreamWaQ 的 VAE 学习率和 RNN 辅助 decoder
学习率仍为 `3e-4`。原 Go2W Rough runner 不受影响。新参数训练从头开始，避免沿用旧优化器状态
或把直接学习 std 的旧 MLP/RNN 检查点加载到 log_std 参数化中；保持 10000 次迭代和录像配置。

## Go2W 适配与明确差异

- 保留本仓库 Go2W USD、默认站姿、执行器类型、PD 参数和部署一致的 23.5 N·m 力矩上限。
  calf 仿真速度单独对齐 URDF/USD 的 20.07 rad/s，hip/thigh 和轮仍为 30 rad/s。
  这些是仓库模型/部署限值核对，不代表新的实机额定参数测量。
- Reset 的 xy 收窄到地形中心 ±0.25 m。根据默认关节姿态下全部 27 个 URDF 碰撞基元和
  新 root 姿态计算最低点，必要时上抬到至少 3 cm 几何余量；不改 default joint 和 action。
  当前六类地形的中心有至少 2 m 平台，或以中央 2×2 m 高度上界作为 origin。完整机器人
  被限制在内侧 1.6×1.6 m 区域中，不使用 reset 前的陈旧扫描值。
  未知地形、过小平台或超出安全出生区的配置会明确报错；改变关节 reset 随机化时也需重新适配。
- 质量/惯量明确采用耦合语义：`I = I_default × (m/m_default) × s`，每个刚体独立采样
  `s∈[0.85,1.15]`，整体缩放完整惯量张量。质量变化先作用于惯量，s 表示额外建模误差，
  保持主轴和物理惯量不等式，不再覆盖质量重算结果，也不叠乘上一次噪声。
- 保留现有 Go2W policy/critic 观测接口及关节顺序。这里是训练配方迁移，不是 M20 checkpoint 的兼容运行器。
  动作尺度、控制模式、默认偏移及动作裁剪均与原 Go2W Rough 一致。
- `base_link` 映射为 `base`，hipx/hipy/knee 映射为 hip/thigh/calf，wheel 映射为 foot；
  两组对角轮显式按 `[FL_foot, RR_foot, FR_foot, RL_foot]` 排序，传感器和刚体选择器保持一致。
- 地形扫描路径使用本仓库生成地形的 `/World/ground/terrain`；新增 0.1×0.1 m 的机身下方扫描器，
  用于腿姿态惩罚在高地形上减弱的判断，按控制周期更新。
- `gait_level` 从上游模块全局变量改为环境实例内状态，首次使用按平均地形等级初始化；
  地形升降级复用本仓库逻辑，首次 reset 不将出生偏移误判为行走成绩。
- 动作二阶差分复用本仓库 `ActionSmoothnessPenalty` 的 reset/步号管理；
  不再用“动作值是否等于零”判断历史是否有效。
- 腾空高度历史在局部 reset 时清理；5 秒占空比缓冲的预热计数按环境单独重置，
  避免新 episode 继承旧 episode 的有效历史状态。保留上游零填充窗口的均值和预热乘数。
- 增加命令混合比例和 heading 依赖校验，兼容命令管理器传入 `None`/`slice` 的全体或局部 reset。
  省略上游仅用于日志的 `base_z`/`knee_pos` 命令指标。
- Play 配置单独关闭参数随机化、外力和推扰以及地形升降级，保留状态 reset 和命令混合。
  原始 Rough 任务的训练/Play 配置不受影响。

## 保留的上游行为

- Heading 增益为 0.5，误差 wrap 到 ±π；正常混合命令和原地转向实际目标角速度最大约 1.57 rad/s。
- `stand_still` 只检查平移命令，因此纯转向时仍生效；没有擅自解除这一姿态约束。
- “转向滑动”计算的是接触轮中心相对机身平移的速度，并非基于轮半径的真实接触点滑移率。
- 占空比奖励不显式规定对角相位；支撑状态奖励不要求高处一组必须完全离地。
- 地面高低按扫描命中点相对环境原点的 z 计算，不是地形粗糙度。
- 机身高度、通用终止奖励、轮力矩、功率及位置限位惩罚在源配置权重为零，本任务不启用它们。

## 历史使用命令（仅旧服务器快照支持）

在包含 Isaac Lab 的 Python 环境中，从本工作区的 `locolab/` 运行：

```bash
python scripts/locomotion/train.py --task OmniBot-Go2W-M20-Rough-v0 --num_envs 4096 --headless
python scripts/locomotion/play.py --task OmniBot-Go2W-M20-Rough-v0 --num_envs 16
```

检查点保存目录为 `logs/rsl_rl/omnibot_go2w_m20_rough_v0/`，与原有 Rough 实验分离。

MLP Rough 训练并启用机器人跟随录像（每 500 次迭代录制约 20 秒）：

```bash
python scripts/locomotion/train.py --task OmniBot-Go2W-M20-Rough-MLP-v0 --num_envs 4096 --headless --video --video_interval 500 --video_length 1000
```

MLP 检查点及视频保存于 `logs/rsl_rl/omnibot_go2w_m20_rough_mlp_v0/<run>/`，视频子目录为 `videos/train/`。

DreamWaQ 和 RNN 使用相同录像参数：

```bash
python scripts/locomotion/train.py --task OmniBot-Go2W-M20-Rough-DreamWaQ-v0 --num_envs 4096 --headless --video
python scripts/locomotion/train.py --task OmniBot-Go2W-M20-Rough-Rnn-v0 --num_envs 4096 --headless --video
```

## 验证

CPU 单元与配置声明检查，从仓库根目录运行：

```bash
python -m pytest tests/test_go2w_m20_config.py tests/test_go2w_m20_mdp.py tests/test_go2w_action_smoothness.py tests/test_terrain_curriculum.py tests/test_go2w_training_configs.py tests/test_runner_experiment_paths.py -q
```

配置测试检查源配方的有效权重、真实 Go2W URDF 的关节/轮名和排列、命令配额、地形比例、
终止与事件接线、参考配方不覆盖 action，以及整套 M20 配方不得重新进入当前训练注册的约束；tensor 测试检查命令特殊分支、无效参数、地形更新、课程状态隔离、
奖励门控、无效射线及奖励历史的局部 reset。测试通过不代表已运行物理仿真。

运行时 smoke，从 `locolab/` 执行：

```bash
python scripts/locomotion/smoke_env.py --validate-registered-go2w --headless --device cpu
python scripts/locomotion/smoke_env.py --task OmniBot-Go2W-M20-Rough-v0 --num_envs 4 --steps 320 --headless
python scripts/locomotion/train.py --task OmniBot-Go2W-M20-Rough-v0 --num_envs 16 --max_iterations 2 --logger tensorboard --smoke-marker --headless
```

分别确认日志含 `CONFIG_SMOKE_OK`、`ISAAC_SMOKE_OK`、`TRAINING_SMOKE_OK`，不能仅看进程退出码。
320 步 rollout 覆盖 5 秒占空比窗口；零动作 smoke 只验证管线和有限值，不验证策略能力。

本次迁移验证环境为 `/home/sshang/anaconda3/envs/env_isaaclab/bin/python`。
项目文档中的 `robot_lab` 路径在本机不存在。2026-09-11 的 Isaac 配置 smoke 在启动阶段
遇到 `NVML_ERROR_LIB_RM_VERSION_MISMATCH`、CUDA 804 和 `No CUDA devices found`，
未产生配置成功标记。

随后在服务器完成 `OmniBot-Go2W-M20-Rough-MLP-v0` 的 16 环境、4 次 PPO 更新和录像验证，
输出 `TRAINING_SMOKE_OK`；生成的 1280×720、50 fps MP4 可解码。
实际导出的参数核对为 ActorCritic MLP、学习率 3e-4、entropy 0.01，腿/轮 action scale=0.25/10，clip=None。
服务器通过独立用户目录中的 NVIDIA 580.173.02 共享库匹配已加载内核驱动，设置只作用于训练进程，
没有安装系统驱动或重启服务器。此处记录的是短时运行验证，不代表策略已收敛。

后续服务器验证：RNN（ActorCriticRecurrent，GRU 256）和 DreamWaQ（ActorCriticDreamWaQ）
分别完成 16 环境、7 次更新的 smoke，均输出成功标记，并生成可解码、非黑屏的 1280×720、50 fps 录像。
两者的 action scale 保持 0.25/10、clip=None；PPO 学习率为 3e-4，DreamWaQ 的 VAE 学习率也为 3e-4。
正式任务使用 4096 环境和原有 10000 次迭代上限，video 每 500 次迭代录制约 20 秒。
