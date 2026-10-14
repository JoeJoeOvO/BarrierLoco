# BarrierLoco 统一仓库使用说明

更新日期：2026-10-10。工作目录 `/home/joey/BarrierLoco`；本地分支 `barrierloco-unified`。

2026-10-11 更新：Multiskill 默认已切换为完整迁入的 V1 课程版，使用本地80k最终权重与连续指令演示。
当前启动、录屏与训练方式以 [V1整合说明](multiskill_v1.md) 为准；中间训练及测试结果已清理。

## 已整合来源

来源仓库 `shang13206/OmniBot`：

| 技能 | 上游分支 | 提交 |
|---|---|---|
| Gait | `codex/multi-skill` | `16107a4` |
| 高台 | `feature/go2w-highplatform-rough-mlp` | `9abf47e` |
| Jump | `feature/go2w-omnijump-mlp` | `de588a2` |

根项目原有 `3d45ffe` 已包含 Gait 和 Rough 基线，在其上合入高台和 Jump，保留 Git 合并历史。
现在只有根目录一份活动代码，`highplatform/`、`multiskill/`、`omnijump/` 不再作为独立仓库使用。

## 环境

- Python：`/home/joey/miniconda3/envs/env_isaaclab/bin/python`
- Isaac Lab：`/home/joey/isaaclab`，VERSION 2.3.2；Isaac Sim 4.5.0.0。
- PyTorch 2.7.0+cu128；RTX 3080，10 GB 显存。
- 根目录 `locolab` 已 editable 安装。训练与回放优先使用仓库内定制 RSL-RL。
- 回放保留新版 Isaac Lab 导入路径兼容；checkpoint 默认映射到当前 GPU。
- `scripts/skill.sh` 固定根目录源码路径，使用 env_isaaclab，不依赖旧子目录。

## 日常使用

Multiskill仅保留最终80k模型，默认回放/录屏/verify均指向`deploy/policy/go2w/multiskill_scratch80k/`。
默认训练从随机权重开始。完整命令见[Multiskill使用说明](multiskill_v1.md)，参数和两阶段复现见[正式实验记录](experiments/2026-10-12_multiskill_scratch80k/README.md)。

```bash
./scripts/skill.sh multiskill play
./scripts/skill.sh highplatform play
./scripts/skill.sh omnijump play
```

## Jump 两阶段

`skill.sh omnijump train` 默认 `--jump_stage skill`，从零初始化，禁止 resume。
第一阶段训练完成后，第二阶段显式覆盖阶段并恢复第一阶段 checkpoint：

```bash
./scripts/skill.sh omnijump train --headless --jump_stage speed --resume \
  --load_run '<第一阶段运行目录名>' --checkpoint model_4999.pt \
  --num_envs 64 --max_iterations 5000
```

上例模型名适用于完成 5000 次的第一阶段，短测应使用其真实文件名。
两阶段使用相同 Jump 实验目录；`--resume_experiment_name` 可更改来源实验目录。
最终模型回放默认 `--jump_stage speed`；回放第一阶段模型时显式改为 `--jump_stage skill`。
旧 `Jump-Flat-Pretrain` 任务已由上游删除，不再使用早期原地跳跃教程。
高台作者的完整训练是从 Rough MLP 续训；上面的从零短测不等同于该训练流程。

## 动力学与接口隔离

| 项目 | Gait | 高台 | Jump |
|---|---|---|---|
| Actor / Critic 维数 | 64 / 318 | 53 / 307 | 54 / 309 |
| 腿力矩 / 速度上限 | 23.5 Nm / 30 rad/s | 23.5 Nm / 30 rad/s | 23.5 Nm / 30 rad/s |
| 自碰撞 | 关闭 | 开启 | 开启 |
| 指令 | 10 维连续 command + 4 维时钟观测 | 3 维速度指令 | 3 维速度 + jump_active |

Gait 与高台的训练/回放配置显式覆盖共享资产限值；Jump 保留上游任务内的历史限值覆盖。
共享 `GO2W_CFG` 与普通 Rough 等其他任务仍使用新版小腿 35 Nm / 20 rad/s，未被技能配置污染。
三者腿 KP/KD=40/1、轮 KP/KD=0/0.5；周期 0.02 s；腿 action scale=0.25、轮速 scale=10。
修改一个技能的配置不会更改其他技能实例或全局资产。

## 验证

```bash
./scripts/skill.sh multiskill test -q
./scripts/skill.sh multiskill python tests/smoke_unified_skill_dynamics.py --headless
./scripts/skill.sh multiskill verify
./scripts/skill.sh highplatform verify
./scripts/skill.sh omnijump verify
```

统一后单元测试：249 passed、1 skipped。跳过项为可选 Python MuJoCo 绑定缺失。
真实 Isaac Lab 配置检查覆盖三个任务的训练/回放共六份配置，以及共享资产和对象隔离，输出 `UNIFIED_SKILL_DYNAMICS_OK`。
三个模型 CPU 哈希/严格加载/ONNX/TorchScript 校验均通过，每个包含 385 组输入。
合并前独立版本与统一版的三份训练 `env.yaml` 做了逐字段对比：仅归一化源码绝对路径后，三个任务均为零差异。
对比记录为 `logs/setup/unified/environment_comparison.json`，说明任务奖励、指令、观测、随机化及动力学配置均被保留。
本次完整验证日志在 `logs/setup/unified/`；短测只验证运行链路，不等同于重新测定技能成功率。
三任务各完成 16 环境、1 迭代的 GPU 训练；三份最终模型各完成单环境 GPU 回放与模型导出，
生成 6 秒、50 fps 的视频，位于各模型目录的 `videos/play/rl-video-step-0.mp4`。

## 旧版本清理与恢复

统一前完整工作区备份：
`/home/joey/BarrierLoco-backups/before-unify-20261010-125723/workspace.tar.gz`。
同目录 `SHA256SUMS` 已校验通过，包含原代码、独立仓库、本地改动、模型与日志。
旧子目录、临时 merge_probe、过时的独立部署说明及旧 Python 缓存已从活动工作区移除；Git 历史仍可追溯旧版本。
清理记录：`logs/setup/unified/cleanup.json`。原训练日志保留在根目录，历史模型和各子目录日志也保存在上述备份中。
需要恢复时先解压到其他目录，不要直接覆盖正在使用的统一仓库。
