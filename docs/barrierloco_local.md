# BarrierLoco 统一仓库使用说明

更新日期：2026-10-10。工作目录 `/home/joey/BarrierLoco`；本地分支 `barrierloco-unified`。

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

### 当前实验：仅扩展多步态前后速度

2026-10-10：Gait 任务的 `commands.base_velocity.lin_vel_x` 从 `(-1, 1)` 改为
`(-5, 5)` m/s，训练和 Play 配置均继承该覆盖。其余指令、地形、奖励、动力学和 PPO 参数保持不变。
这使用现有均匀采样，不增加速度课程，不承诺全范围跟踪；HighPlatform 仍为向前 `0.5–1.0` m/s
（Play 为 `1.0` m/s），Jump 不改动。

2026-10-10 18:11 已启动 Gait 修改版从零训练：4096 环境、5000 次迭代、seed 42、
TensorBoard、cuda:0，每 500 次保存模型。实验目录为 `locolab/logs/rsl_rl/barrierloco_gait_vx5/`。
启动记录、源码快照、控制台日志和退出码位于 `logs/setup/gait_vx5/train_20261010-181141/`；
`exit_code` 在进程退出时生成，0 表示正常退出。1024 和 4096 环境均已完成 10 次迭代短测。

随仓库附带的 Gait 权重仍是在 `(-1, 1)` m/s 上训练的，不能当作扩展版模型。
回放作者模型时显式恢复其指令范围：

```bash
./scripts/skill.sh multiskill play 'env.commands.base_velocity.lin_vel_x=[-1.0,1.0]'
```

后续训练扩展版时，可用 `--run_name gait_vx5` 标识实验，避免与原版记录混淆。
HighPlatform 的原流程需要 Rough 预训练 checkpoint，再续训高台；当前打包的是高台最终权重，
不能将它作为 Rough 基座。完整重新训练前需取得原 Rough checkpoint，或先重新训练 Rough 基座。
下文“统一后验证”的零差异结论描述的是本次速度修改之前的版本。

```bash
cd /home/joey/BarrierLoco
./scripts/skill.sh multiskill play
./scripts/skill.sh highplatform play
./scripts/skill.sh omnijump play

# 无窗口回放/录像，结束后自动退出
./scripts/skill.sh omnijump play --headless --video --video_duration_s 6

# 从零训练调试，日志使用 TensorBoard
./scripts/skill.sh highplatform train --headless --num_envs 64 --max_iterations 100
./scripts/skill.sh multiskill train --headless --num_envs 64 --max_iterations 100
./scripts/skill.sh omnijump train --headless --num_envs 64 --max_iterations 100

# 标准脚本方式；--task 可直接换成另外两个任务
source /home/joey/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python locolab/scripts/locomotion/train.py \
  --task OmniBot-Go2W-Gait-Flat-MLP-v0 \
  --headless --num_envs 64 --max_iterations 100 --logger tensorboard

tensorboard --logdir /home/joey/BarrierLoco/locolab/logs/rsl_rl
```

每次启动一个 GPU 任务。训练日志统一位于 `locolab/logs/rsl_rl/<experiment>/<run>/`。
自己的模型通过 `play --checkpoint /absolute/path/model.pt` 指定，并选择与训练一致的技能。
回放导出到 checkpoint 同目录 `exported/`，视频到 `videos/play/`。

## 技能与模型

### 当前模型追加 50000 次训练

用户选择从 `2026-10-10_18-11-47_scratch_4096_seed42/model_4999.pt` 继续，
不是从零开始，也不是只补到总计 50000 次。保持 4096 环境、seed 42、原 PPO 参数及 vx±5 范围，
恢复 actor、critic 和 optimizer；每 500 次保存 checkpoint。新实验目录与首轮分开。

```bash
./scripts/skill.sh multiskill train --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 50000 --seed 42 --resume \
  --resume_experiment_name barrierloco_gait_vx5 \
  --load_run 2026-10-10_18-11-47_scratch_4096_seed42 --checkpoint model_4999.pt \
  --experiment_name barrierloco_gait_vx5_continue50k --run_name resume4999_4096_seed42
```

当前 runner 从 checkpoint 的迭代标签 4999 开始循环，追加 50000 次更新后最终文件标签为 54998。
这是现有编号方式；两轮实际共执行 55000 次更新。首轮 5000 次耗时 1:20:57，
线性估算追加训练为 13:29:30，实际耗时取决于仿真运行速度。时间加长不保证收敛。

### 实时全向速度检查

```bash
./scripts/skill.sh multiskill play --checkpoint /absolute/path/model_4999.pt \
  --device cuda:0 --real-time --follow-camera --velocity-sweep --live-plot
```

相机跟随机器人；Isaac Sim 内部右侧 Velocity Tracking 面板显示机身坐标系下的 vx、vy（m/s）及 wz（rad/s）的曲线和实时数值，橙线为
实际发送给策略的 command，蓝线为仿真测量值。数据保存为 checkpoint 同目录的
`velocity_live_<时间>.csv`。CSV 在重置处留空；面板在工况切换或自动重置时清空曲线，并累计显示自动重置次数。

rolling、trot、bound 各检查 31 个工况，共 93 个：静止、vx=±0.5/1/2/3/4/5、
vy=±0.25/0.5、四个斜向组合、wz=±0.5/1.5/3，以及前后运动配合左右转向。
每工况默认 6 秒（`--sweep-seconds` 可调整），完整循环 558 秒；每个新工况重置环境，
固定高度 0.34 m、频率 2.5 Hz、运动步态摆动高度 0.06 m，不改变训练配置。
本回放用于观察响应，不是经过稳定时间筛选的成功率评测。

| 技能名 | Task ID | 默认 checkpoint |
|---|---|---|
| multiskill | `OmniBot-Go2W-Gait-Flat-MLP-v0` | `deploy/policy/go2w/gait_wtw_height50_20260920/model_4999.pt` |
| highplatform | `OmniBot-Go2W-HighPlatform-MLP-v0` | `deploy/policy/go2w/highplatform_rough_mlp_20260922/model_9998.pt` |
| omnijump | `OmniBot-Go2W-Jump-MLP-v0` | `deploy/policy/go2w/jump_smooth_mlp_20261009/model_9998.pt` |

三个模型均已完整下载，ONNX/TorchScript 与原始参数在同目录。
Jump 第一阶段 `model_4999.pt` 同时保留。
Gait 提供 rolling/trot/bound、机身高度、频率、相位、占空比、摆动高度控制；没有独立 roll/pitch command。
高台为直线爬台；新版 Jump 是行进中触发跳跃，采用 2.2 秒时钟和 speed 阶段航向锁定。
这三者仍为独立网络，通过启动参数切换任务，不涉及一次仿真过程中自动交接策略。

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
