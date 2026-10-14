# BarrierLoco

基于 Isaac Lab 的 Go2W 三技能统一仓库：多步态、高台、跳跃。
全部任务共用 `locolab/` 的训练与回放框架，通过 task 或技能名选择，各自保留训练配置和模型权重。

## 获取模型与环境

成功实验的权重通过 Git LFS 保存，克隆后运行 `git lfs pull`。模型内容和校验值见各模型目录的来源记录。
在其他机器使用时，将 `BARRIERLOCO_PYTHON` 设为自己的 Isaac Lab Python 路径，并运行 `"$BARRIERLOCO_PYTHON" -m pip install -e ./locolab`。

本地成功训练的高台模型位于 `deploy/policy/go2w/highplatform_rough15k_batch196608/`，同时附带可恢复训练的 Rough 父权重：

```bash
./scripts/skill.sh highplatform play --checkpoint deploy/policy/go2w/highplatform_rough15k_batch196608/model_19998.pt --real-time --follow-camera --live-plot
```

不指定 checkpoint 的高台回放仍使用保留的作者模型；上面的命令明确选择本地成功复现版本。

## 快速开始

已完成实验的配置、改动、权重与复现命令见 [实验记录索引](docs/experiments/README.md)。后续各技能实验统一在这里记录。

使用本机已有 `env_isaaclab`，无需手动激活环境：

```bash
cd /home/joey/BarrierLoco

# 每次启动一个技能的窗口回放，默认载入该技能已训练的最终模型
./scripts/skill.sh multiskill play
./scripts/skill.sh highplatform play
./scripts/skill.sh omnijump play

# 小规模无窗口训练；技能名可替换
./scripts/skill.sh multiskill train --headless --num_envs 64 --max_iterations 100

# 查看注册任务
./scripts/skill.sh multiskill list
```

| 技能 | Task ID | 模型目录（deploy/policy/go2w 下） |
|---|---|---|
| 多步态、高度控制（V1课程版） | `OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0` | `multiskill_scratch80k` |
| 高台 | `OmniBot-Go2W-HighPlatform-MLP-v0` | `highplatform_rough_mlp_20260922` |
| 跳跃 | `OmniBot-Go2W-Jump-MLP-v0` | `jump_smooth_mlp_20261009` |

`skill.sh` 默认使用 `/home/joey/miniconda3/envs/env_isaaclab/bin/python`，可通过 `BARRIERLOCO_PYTHON` 覆盖。
Multiskill 默认从零开始，1024 环境、30000 次课程训练；其余技能默认 64 环境。
训练使用本地 TensorBoard；回放默认 1 环境。Multiskill 的 play/sweep 使用 V1 连续 cmd 演示和右侧面板。
课程代码、最终80k模型和录屏程序均在当前仓库；旧V1/作者权重及中间结果已清理。
详细命令见 [V1整合说明](docs/multiskill_v1.md)。
Jump 训练默认 skill 第一阶段，最终模型回放默认 speed 第二阶段语义。

## 代码结构

- `locolab/scripts/locomotion/`：统一训练、回放和评估入口。
- `locolab/source/locomotion/tasks/Go2W/`：各技能环境、指令和奖励配置。
- `locolab/source/locomotion/mdp/`：奖励、观测、课程和指令实现。
- `locolab/rsl_rl/`：共享 PPO、网络及训练运行器。
- `deploy/policy/go2w/`：三份完整 checkpoint、ONNX/TorchScript 网络与来源参数。
- `tests/`：回归测试与仿真检查。

## 配置约定

三个已交付技能均显式保留训练时腿部 23.5 Nm / 30 rad/s 限值；Gait 自碰撞关闭，高台与 Jump 开启。
共享资产对其他任务仍使用按关节定义的限值。不要删除技能覆盖项或把三个网络的观测混用。
三个技能的 actor 输入维数分别为 64、53、54；输出都是 16 维动作。

详细启动、两阶段 Jump、动力学说明及验证记录见 [本地使用说明](docs/barrierloco_local.md)。
上游技能资料：[Gait](docs/go2w_gait_overview.md)、[高台](docs/go2w_highplatform_overview.md)、[Jump](docs/go2w_jump.md)。

本次统一仅验证 Isaac Lab，不包含新的 MuJoCo 或实机验证。
历史代码与独立目录已备份到工作区外，当前使用根目录这一份代码。

## Git

本地整合分支：`barrierloco-unified`。保留上游 Git 历史，成功实验记录位于 `docs/experiments/`。
开发时使用功能分支，不直接 push 到 main。
