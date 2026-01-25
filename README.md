# OmniBot

轮足机器人全栈开发框架，涵盖训练、仿真验证、实机部署。

## 项目结构

```
OmniBot/
├── locolab/        # 基于 Isaac Lab 的强化学习训练
├── simulation/     # 基于 MuJoCo 的 Sim2Sim 仿真验证
└── deploy/         # Sim2Real 实机部署
```

## 模块说明

### locolab

基于 Isaac Lab 的机器人运动控制训练框架，使用 PPO 算法训练轮足机器人的运动策略。

- 支持 Go2W 轮足机器人
- 速度跟踪任务
- RSL-RL 强化学习库

详见 [locolab/README.md](locolab/README.md)

### simulation

基于 MuJoCo 的 Sim2Sim 仿真验证，用于在部署前验证训练好的策略。

- 轻量级仿真环境
- 策略迁移验证
- 性能分析

### deploy

Sim2Real 实机部署模块，将训练好的策略部署到真实机器人。

- 实时控制接口
- 硬件通信
- 安全保护

## 快速开始

### 1. 训练

```bash
cd locolab
pip install -e .
python scripts/locomotion/train.py --task=OmniBot-Go2W-Velocity --num_envs=4096
```

### 2. Sim2Sim 验证

```bash
cd simulation
# TODO: 添加仿真验证脚本
```

### 3. 实机部署

```bash
cd deploy
# TODO: 添加部署脚本
```

## 环境要求

- Isaac Sim 5.x / Isaac Lab
- MuJoCo 3.x
- Python 3.10+
- PyTorch 2.x
- CUDA 12.x
