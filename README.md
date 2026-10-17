# BarrierLoco

基于 Isaac Lab 的 Go2W 技能库与导航 Planner：多步态、高台、GoalGrid 跳跃，以及基于 OCC 的多技能调度。
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

# GoalGrid 从零训练
./scripts/skill.sh omnijump train --headless

# 小规模无窗口训练；技能名可替换
./scripts/skill.sh multiskill train --headless --num_envs 64 --max_iterations 100

# 查看注册任务
./scripts/skill.sh multiskill list
```

| 技能 | Task ID | 模型目录（deploy/policy/go2w 下） |
|---|---|---|
| 多步态、高度控制（V1课程版） | `OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0` | `multiskill_scratch80k` |
| 高台 | `OmniBot-Go2W-HighPlatform-MLP-v0` | `highplatform_rough_mlp_20260922` |
| 目标跑酷（`omnijump` / `goalparkour` / `parkour`） | `OmniBot-Go2W-Parkour-GoalGrid-MLP-v0` | `goalgrid_scratch5k`（待逐目标验收） |

`skill.sh` 默认使用 `/home/joey/miniconda3/envs/env_isaaclab/bin/python`，可通过 `BARRIERLOCO_PYTHON` 覆盖。
Multiskill 默认从零开始，1024 环境、30000 次课程训练；HighPlatform 默认 64 环境。
训练使用本地 TensorBoard；HighPlatform 回放默认 1 环境，GoalGrid 默认 40 环境。Multiskill 的 play/sweep 使用 V1 连续 cmd 演示和右侧面板。
课程代码、最终80k模型和录屏程序均在当前仓库；旧V1/作者权重及中间结果已清理。
详细命令见 [V1整合说明](docs/multiskill_v1.md)。
GoalGrid 默认从零、4096 环境、5000 次单阶段训练；详见 [新 Jump 整合说明](docs/goalparkour_integration.md)。
`omnijump play` 必须显式提供 GoalGrid checkpoint；目前已保留 [5000 次从零训练模型与回放命令](deploy/policy/go2w/goalgrid_scratch5k/README.md)，尚未完成逐目标验收。独立七轨迹版本及其模型已删除，30 条 GoalGrid 参考保持不变。

## 导航 Planner（开发与验证中）

Planner 输入本体状态（58 维）、局部双通道 OCC（占用与已观测区域）和相对目标位置；共享网络输出技能选择与各技能 cmd，调用冻结的三个底层策略。使用独立的混合动作 PPO 训练入口，每秒决策 10 次，底层控制 50 Hz。详见 [导航接口与训练说明](docs/planner_navigation.md) 和 [地形说明](docs/planner_terrain.md)。

支持 10 类混合地形（包含 Isaac Lab Rough 的 6 类）以及 5 类纯 Gap 地形：单沟、双沟、斜沟、折线沟、变宽沟。连续导航目标距离为 8–16 m，带方向变化并允许跨地块；目标位于实际地形表面。新版到达条件为水平距离小于 1.2 m、水平速度低于 1.0 m/s，并检查机身与目标面的高度关系，无需完全静止。

```bash
# 混合地形，从零训练 Planner；底层技能保持冻结
./scripts/skill.sh omnijump python locolab/scripts/locomotion/train_planner.py \
  --headless --continuous-mixed --jump-always-active \
  --num-envs 160 --horizon 32 --max-iterations 20000 --episode-seconds 180 \
  --progress-reward 3 --arrival-reward 20 --step-cost 0.02 \
  --output logs/planner/mixed_new_run

# 五类纯 Gap：沿用连续导航设置，用 --diverse-gaps 替换地形
./scripts/skill.sh omnijump python locolab/scripts/locomotion/train_planner.py \
  --headless --continuous-mixed --diverse-gaps --jump-always-active \
  --num-envs 160 --horizon 32 --max-iterations 20000 --episode-seconds 180 \
  --progress-reward 3 --arrival-reward 20 --step-cost 0.02 \
  --output logs/planner/diverse_gaps_new_run

# 回放自动读取 checkpoint 的地形和指令配置；相机跟随，右侧停靠曲线
./scripts/skill.sh omnijump python locolab/scripts/locomotion/play_planner.py \
  --checkpoint <planner_checkpoint.pt>

# 按训练的随机采样方式回放，并记录真实起跳而非只统计技能选择
./scripts/skill.sh omnijump python locolab/scripts/locomotion/play_planner.py \
  --checkpoint <planner_checkpoint.pt> --stochastic --follow-jumps

# 单独测试底层 Jump：右侧手动切换 0/1，保持到下一次按键
./scripts/skill.sh omnijump python locolab/scripts/locomotion/play_jump_switch.py --initial-flag 0
```

每次训练使用新的输出目录；`--resume <checkpoint>` 可恢复 Planner 网络与优化器，仿真环境重新初始化。`--verify` 会执行目标刷新、终止、模型载入及对应场景检查。日志、源码快照和中间 checkpoint 保留在本地 `logs/planner/`，不随 Git 推送，需要单独备份。

**当前限制与验证边界：**

- Planner 尚未完成成功验收。到达次数不是跨沟通过率，Jump 执行时间也不等于实际腾空时间；已观察到选中 Jump 却没有起跳的情况。
- 纯 Gap 沟宽为 0.15–0.9 m、深度为 0.6–1.4 m；较宽沟可能超出当前冻结技能的稳定能力。
- `--jump-always-active` 表示选中 Jump 期间输入开关恒为 1，并锁定 1.8 s；它与原始 GoalGrid 的 2.2 s 完整动作序列不同。默认旧接口也不等同于原始环境的触发条件。接口时序仍待对照验证。
- 手动开关窗口绕过 Planner 和自动计时，速度、高度与落点描述在开关为 0 时仍提供给网络，不能将其直接当作完整技能执行流程。
- 回放仅因跌落、倾斜或持续塌低/机身触地重置，不因到达、超时或跨地块重置。新增塌低检测晚于现有 Planner checkpoint 的训练，历史失败统计未包含此判定。
- Periodic / Event Skill 的通用接口、内部 clock/phase、完成检测及相应底层重训是后续设计，**尚未实现完整 Event Skill 框架**。目前没有 BPO、部署 safety filter 或持续学习训练。

成功实验才进入 [正式实验记录](docs/experiments/README.md)；上述 Planner 开发过程不视为成功复现。

## 代码结构

- `locolab/scripts/locomotion/`：统一训练、回放和评估入口。
- `locolab/source/locomotion/tasks/Go2W/`：各技能环境、指令和奖励配置。
- `locolab/source/locomotion/mdp/`：奖励、观测、课程和指令实现。
- `locolab/rsl_rl/`：底层技能 PPO、网络及训练运行器。
- `locolab/planner/`：OCC、冻结技能接口、导航网络、PPO 与跳跃诊断。
- `deploy/policy/go2w/`：已发布的完整 checkpoint、ONNX/TorchScript 网络与来源参数（GoalGrid 模型尚待逐目标验收）。
- `tests/`：回归测试与仿真检查。

## 配置约定

Multiskill 和 HighPlatform 保留训练时腿部 23.5 Nm / 30 rad/s 限值；Gait 自碰撞关闭，高台开启。
新 Parkour/GoalGrid 使用分关节 23.5/23.5/35 Nm、30/30/20 rad/s 限值及自碰撞，不能套用旧 Jump 的统一限值。
共享资产对其他任务仍使用按关节定义的限值。不要删除技能覆盖项或把三个网络的观测混用。
三个技能的 actor 输入维数分别为 64、53、57；输出都是 16 维动作。

历史启动、旧两阶段 Jump、动力学说明及验证记录见 [本地使用说明](docs/barrierloco_local.md)。
上游技能资料：[Gait](docs/go2w_gait_overview.md)、[高台](docs/go2w_highplatform_overview.md)、[Parkour](docs/go2w_parkour.md)、[GoalGrid](docs/go2w_parkour_goal_grid.md)。

本次统一仅验证 Isaac Lab，不包含新的 MuJoCo 或实机验证。
历史代码与独立目录已备份到工作区外，当前使用根目录这一份代码。

## Git

当前统一代码维护在 `main`，保留上游 Git 历史。成功实验记录位于 `docs/experiments/`；日志、视频和未验收的 Planner 中间模型不纳入版本控制。
