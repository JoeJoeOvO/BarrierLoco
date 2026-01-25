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
│       │   ├── robot.py        # 机器人配置
│       │   ├── Go2W/           # Go2W 轮足机器人 USD 模型
│       │   └── unitree_actuators.py
│       ├── mdp/                # MDP 组件
│       │   ├── observations.py # 观测函数
│       │   ├── rewards.py      # 奖励函数
│       │   ├── curriculums.py  # 课程学习
│       │   ├── command/        # 命令生成
│       │   └── actions/        # 动作空间
│       ├── tasks/              # 任务定义
│       │   ├── velocity_env_cfg.py  # 速度控制环境配置
│       │   └── agents/         # RL 算法配置
│       └── utils/              # 工具函数
└── setup.py                    # 安装脚本
```

## 环境要求

- Isaac Sim 5.x / Isaac Lab
- Python 3.10+
- PyTorch 2.x
- CUDA 12.x

## 安装

```bash

# 安装包
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
# 基本训练
python scripts/locomotion/train.py --task=OmniBot-Go2W-Velocity --num_envs=4096

# 从检查点恢复训练
python scripts/locomotion/train.py --task=OmniBot-Go2W-Velocity --num_envs=4096 --resume
```

### 推理/测试

```bash
# 加载最新检查点
python scripts/locomotion/play.py --task=OmniBot-Go2W-Velocity --num_envs=16

# 指定检查点
python scripts/locomotion/play.py --task=OmniBot-Go2W-Velocity --checkpoint=logs/rsl_rl/xxx/model_1000.pt

# 录制视频
python scripts/locomotion/play.py --task=OmniBot-Go2W-Velocity --video --video_length=500
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | - |
| `--num_envs` | 并行环境数量 | 4096 |
| `--max_iterations` | 最大训练迭代次数 | 1500 |
| `--seed` | 随机种子 | 42 |
| `--logger` | 日志类型 (tensorboard/wandb/neptune) | tensorboard |
| `--resume` | 从检查点恢复 | False |
| `--checkpoint` | 指定检查点路径 | - |

## 可用任务

| 任务名称 | 说明 |
|----------|------|
| `OmniBot-Go2W-Velocity` | Go2W 轮足机器人速度控制 |

## 训练日志

训练日志保存在 `logs/rsl_rl/<experiment_name>/` 目录下：

```
logs/rsl_rl/omnibot_go2w_velocity/
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
