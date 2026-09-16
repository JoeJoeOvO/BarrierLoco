# LocoLab

基于 Isaac Lab 的机器人运动控制训练框架。

## 项目结构

```
locolab/
├── config/                     # 配置文件
├── rsl_rl/                     # RSL-RL 强化学习库
│   ├── algorithms/             # 算法实现 (PPO, Distillation)
│   ├── modules/                # 网络模块 (Actor-Critic)
│   ├── networks/               # 网络结构 (MLP, CNN, Memory)
│   ├── runners/                # 训练运行器
│   ├── storage/                # 经验回放存储
│   └── utils/                  # 工具函数
├── scripts/                    # 运行脚本
│   ├── list_envs.py            # 列出可用环境
│   └── locomotion/
│       ├── train.py            # 训练脚本
│       ├── play.py             # 推理脚本
│       └── cli_args.py         # 命令行参数
├── source/                     # 核心源码
│   └── locomotion/
│       ├── assets/             # 机器人资产配置
│       │   ├── robot.py        # 机器人配置 (Go2W)
│       │   └── Go2W/           # Go2W 轮足机器人 USD 模型
│       ├── mdp/                # MDP 组件
│       │   ├── observations.py # 观测函数
│       │   ├── rewards.py      # 奖励函数
│       │   ├── curriculums.py  # 课程学习
│       │   ├── command/        # 命令生成
│       │   └── actions/        # 动作空间
│       ├── tasks/              # 任务定义
│       │   ├── velocity_env_cfg.py  # 基础速度控制环境
│       │   ├── Go2W/           # Go2W 专用配置
│       │   │   ├── flat_env_cfg.py   # 平地环境
│       │   │   └── rough_env_cfg.py  # 复杂地形环境
│       │   └── agents/         # RL 算法配置
│       └── utils/              # 工具函数
└── setup.py                    # 安装脚本
```

## 环境要求

- Isaac Sim 4.5+ / Isaac Lab 2.0+
- Python 3.10+
- PyTorch 2.x
- CUDA 12.x

## 安装

```bash
cd locolab
pip install -e .
```

## 使用方法

### 列出可用环境

```bash
python scripts/list_envs.py
```

### 训练

```bash
# Go2W 平地训练（推荐初始训练）
python scripts/locomotion/train.py --task=OmniBot-Go2W-Flat-v0 --num_envs=4096

# Go2W 复杂地形训练
python scripts/locomotion/train.py --task=OmniBot-Go2W-Rough-v0 --num_envs=4096

# 从检查点恢复训练
python scripts/locomotion/train.py --task=OmniBot-Go2W-Flat-v0 --num_envs=4096 --resume

# 每 500 个训练迭代录制 20 秒训练视频并自动上传到 W&B
python scripts/locomotion/train.py --task=OmniBot-Go2W-Rough-v0 --num_envs=4096 --logger wandb --video
```

当前 Go2W Rough 在迁移前配置上加入选定的腿力矩、机身水平和腿姿态奖励，任务为 `OmniBot-Go2W-Rough-MLP-v0`、
`OmniBot-Go2W-Rough-Rnn-v0` 和 `OmniBot-Go2W-Rough-DreamWaQ-v0`。
整套 M20 配方仍未注册；当前范围与验证方法见[逐项迁移方案](../docs/m20_incremental_migration.md)。

### 推理/测试

```bash
# 加载最新检查点
python scripts/locomotion/play.py --task=OmniBot-Go2W-Flat-v0 --num_envs=16

# 指定检查点
python scripts/locomotion/play.py --task=OmniBot-Go2W-Flat-v0 --checkpoint=logs/rsl_rl/xxx/model_1000.pt

# 录制 20 秒视频；相机自动跟随 0 号环境中的机器人
python scripts/locomotion/play.py --task=OmniBot-Go2W-Flat-v0 --video
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | - |
| `--num_envs` | 并行环境数量 | 4096 |
| `--max_iterations` | 最大训练迭代次数 | 1500 |
| `--seed` | 随机种子 | 42 |
| `--logger` | 日志类型 (tensorboard/wandb/neptune) | tensorboard |
| `--video` | 训练/回放时录制视频 | False |
| `--video_interval` | 训练录制间隔（iterations） | 500 |
| `--video_length` | 单段训练视频长度（env steps）；回放时可覆盖时长设置 | 训练 1000 / 回放未设置 |
| `--video_duration_s` | 单段回放视频时长（秒） | 20 |
| `--resume` | 从检查点恢复 | False |
| `--checkpoint` | 指定检查点路径 | - |

## 可用任务

| 任务 ID | 说明 |
|---------|------|
| `OmniBot-Go2W-Flat-v0` | Go2W 轮足机器人平地速度控制 |
| `OmniBot-Go2W-Rough-v0` | Go2W 轮足机器人复杂地形速度控制 |
| `OmniBot-Go2W-Velocity` | 旧版别名，等同于 Flat-v0 |

## Go2W 轮足机器人

Go2W 是基于 Unitree Go2 改装的轮足机器人：

- **腿部关节**: 12个（每条腿 hip/thigh/calf），位置控制
- **轮子关节**: 4个（foot），速度控制
- **动作空间**: 混合控制（腿部位置 + 轮子速度）

### Reward 设计

任务奖励（正向激励）：
- `track_lin_vel_xy`: 线速度跟踪 (weight=3.0)
- `track_ang_vel_z`: 角速度跟踪 (weight=1.5)
- `stay_alive`: 存活奖励 (weight=3.0)

正则化惩罚（负向约束）：
- `dof_acc_l2`: 关节加速度平滑 (weight=-5e-8)
- `dof_pos_limits`: 关节限位惩罚 (weight=-0.5)
- `action_rate_wheels`: 轮子动作变化率 (weight=-0.001)
- `undesired_contacts`: 非期望接触惩罚 (weight=-10.0)

### Termination 条件

- `time_out`: 超时（20秒）
- `base_contact`: 底座接触地面
- `bad_orientation`: 姿态角超过 1.0 rad（约57度）

## 训练日志

训练日志保存在 `logs/rsl_rl/<experiment_name>/` 目录下：

```
logs/rsl_rl/omnibot_go2w_flat/
└── 2024-01-01_12-00-00/
    ├── model_*.pt          # 模型检查点
    ├── params/
    │   ├── env.yaml        # 环境配置
    │   └── agent.yaml      # 算法配置
    └── exported/
        ├── policy.pt       # JIT 导出模型
        └── policy.onnx     # ONNX 导出模型
```

## 查看训练曲线

```bash
tensorboard --logdir=logs/rsl_rl
```

## 训练监控指标

| 指标 | 期望趋势 | 说明 |
|------|----------|------|
| `Train/mean_episode_length` | ↑ 稳定在 500+ | Episode 平均长度 |
| `Episode_Reward/track_lin_vel_xy` | ↑ | 速度跟踪奖励 |
| `Episode_Termination/bad_orientation` | ↓ < 0.5 | 姿态终止率 |
| `Episode_Termination/time_out` | ↑ | 超时终止率（正常结束） |
