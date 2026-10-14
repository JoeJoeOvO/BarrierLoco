# Multiskill代码导航与维护边界

整理日期：2026-10-12。此次只整理入口参数、注释/格式、回归检查与文档，没有调整奖励、网络、课程采样或动力学。

## 任务层

| 功能 | Task / 文件 |
|---|---|
| 原始±1基线 | `OmniBot-Go2W-Gait-Flat-MLP-v0` / `locolab/source/locomotion/tasks/Go2W/gait_env_cfg.py` |
| V1扩速课程 | `OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0` / 同目录`gait_curriculum_env_cfg.py` |
| 注册与PPO配置 | `tasks/__init__.py`、`tasks/agents/rsl_rl_ppo_cfg.py` |

课程训练配置打开课程；课程Play配置只扩展到最终命令范围，不从第0步重跑课程。
原始任务承担观测、奖励及任务内动力学覆盖；课程任务只覆盖命令范围和课程开关，避免复制奖励配置。

## 指令与学习

以下路径相对`locolab/source/locomotion/`：

- `mdp/command/wtw_command_cfg.py`：10维cmd默认范围与采样参数。
- `mdp/command/wtw_command.py`：采样、手动cmd、相位、接触目标、课程状态。
- `mdp/wtw_curriculum.py`：迭代课程与混合采样的纯计算实现。
- `mdp/wtw_math.py`、`wtw_kinematics.py`、`wtw_rewards.py`：步态数学、参考姿态与奖励。

`locolab/rsl_rl/runners/gait_curriculum_runner.py`仅负责课程状态存取与恢复迭代，不复制PPO学习循环。
共享`train.py`处理随机初始化、权重初始化、完整恢复和配置来源记录。

## 稳定入口

`scripts/skill.sh multiskill ...`选择课程任务；以下三种训练语义不可混用：

| 命令 | 初始化语义 |
|---|---|
| `train --from-scratch` | 随机权重，禁止checkpoint初始化/恢复 |
| `train`（无初始化选项） | 默认随机权重，复现本地成功pipeline |
| `train --resume --checkpoint <path>` | 恢复策略、优化器和课程，额外训练指定次数 |

`play/sweep/record --checkpoint <path>`使用指定权重；省略时使用`multiskill_scratch80k/model_79999.pt`。
此次消除显式参数与默认参数重复注入的情况，使执行命令中的环境数、迭代数和checkpoint来源更清楚。
Multiskill默认权重统一为最终80k；其他技能默认值不变。

## 回放与记录

- `scripts/multiskill/cmd_sweep.py`：76组扩展cmd、连续切换、跟随视角、状态与控制按钮。
- `scripts/multiskill/swing_height.py`：完整摆动周期轮底峰值测量。
- `scripts/multiskill/record_cmd_sweep.py`：复用命令列表、带字幕视频和CSV。
- `locolab/source/locomotion/utils/velocity_panel.py`：可独立显示或嵌入V1面板的速度曲线。

保留旧`play.py`里的通用回放/历史速度扫描入口，避免破坏历史复现命令；日常多cmd演示使用上述专用入口。
中间模型与测试输出已清理；仅保留最终80k权重、导出和必要配置。实验记录：[从零30k+续训50k](experiments/2026-10-12_multiskill_scratch80k/README.md)。

## 整理后检查

- 97项入口路由、课程、WTW、训练注册、实验路径、分布式入口与高台配置测试通过。
- `tests/smoke_unified_skill_dynamics.py --headless`通过：基线Gait、课程Gait、HighPlatform、Jump的训练/Play配置及隔离性均符合约定。
- 脚本语法检查与`git diff --check`通过。
- 新增`tests/test_skill_entrypoint.py`专门防止从零入口偷偷加载作者模型，以及自选模型被默认路径混淆。
- 测试代码保留；历史测试输出不作为正式产物。

这些检查证明入口和配置兼容，不代表高速姿态和摆腿高度问题已经解决。后续改动须继续更新实验记录。
