# Multiskill最终pipeline：从零30k＋续训50k

仅保留最终可用pipeline与80k模型。中间模型、失败实验、诊断/测试/回放输出已按用户要求清理。
高台和Jump结果不属于此次清理范围。

## 保留产物

最终目录：`deploy/policy/go2w/multiskill_scratch80k/`。

- `model_79999.pt`：本地随机初始化训练30k后，恢复Actor/Critic/优化器/课程并追加50k；累计80000次更新。
- `policy.pt` / `policy.onnx`：最终模型导出，`export_policy.py`可重建和校验。
- `params/`：最终训练实际配置；`provenance.json`、`export_manifest.json`：来源、校验值和导出验证。
- `training_summary.json`：成功pipeline两阶段末次训练窗口，保留指标依据而不保留完整中间日志。
- 本记录的`scratch30k/`、`continue50k/`：两阶段实际env/agent/initialization配置。
- `manifest.json`：最终模型路径、SHA256、父模型来源哈希、源码快照说明。

80k模型SHA256：`2133340f4cd89853140a17941f8033b8c432ac5c1738774337e40a10175d1d00`。
30k衔接模型只保留来源哈希与配置，不再保存权重。重新复现需从零生成自己的30k模型再续训。
旧作者4999、旧V1的29999不是本次训练起点，均不再保留。

## 来源与相对上游的修改

上游Gait基线提交`16107a4bc2e1c223d8a3f84c0aa636334c817e39`；采用原BarrierPlanner/V1工作区的课程扩展并完整迁入当前repo。
本地训练时HEAD为`29af35dbbeb8c069fd13668c34719a795dc39674`，另有未提交整合内容；只checkout该HEAD不能得到全部实现。
运行不依赖BarrierPlanner目录。当前代码入口和参数清理不改变学习数学。

| 模块 | 保留的改动 |
|---|---|
| `mdp/wtw_curriculum.py`与`mdp/command/wtw_command*.py` | 线性速度/摆腿高度课程、50/30/20混合采样与课程时钟 |
| `tasks/Go2W/gait_curriculum_env_cfg.py`及注册 | 独立课程任务，原±1基线仍作继承基础 |
| `tasks/agents/rsl_rl_ppo_cfg.py` | 课程Runner，初始lr3e-4 |
| `rsl_rl/runners/gait_curriculum_runner.py` | 保存/恢复课程，续训从上一编号之后继续 |
| `scripts/locomotion/train.py` | 明确从零/初始化/恢复，保存初始化来源，支持绝对checkpoint路径与有限值检查 |
| `tasks/Go2W/gait_env_cfg.py` | 任务内保留腿23.5Nm/30rad/s、关闭自碰撞，与V1有效动力学一致 |
| `scripts/multiskill/`、`utils/velocity_panel.py` | V1连续cmd/跟随视角/状态/摆动峰值测量/录屏字幕，加右侧cmd与实际速度曲线 |
| `scripts/skill.sh` | 默认训练从零，默认回放/录屏/校验用最终80k；不再隐式加载旧权重 |

训练模块路径相对`locolab/source/locomotion/`，RL模块相对`locolab/`。
没有为50k续训额外改奖励或网络。

## 关键配置

- Task：`OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0`；平地。
- ActorCritic MLP，无GRU：Actor64输入、Critic318输入，隐藏层512/256/128、ELU。
- 16动作：12腿位置目标（scale0.25）＋4轮速目标（scale10）。
- 1024环境，seed42，cuda:0；24步/迭代；仿真dt0.005，decimation4，控制50Hz，episode20秒。
- 腿23.5Nm/30rad/s，PD40/1；轮PD0/0.5；关闭自碰撞。
- PPO：5epochs、4minibatches、lr3e-4、adaptive KL0.01、clip0.2、gamma0.99、lambda0.95、entropy0.01、max_grad_norm1。
- Actor/Critic不做观测归一化；log探索噪声，初始std1；每500次保存。
- env_isaaclab，Python3.10，Isaac Sim4.5，Isaac Lab2.3.2，PyTorch2.7.0+cu128，RTX3080 10GB。

10维cmd依次为vx、vy、yaw rate、机身高度、步频、三个相位参数、占空比、摆动高度；另有4维时钟输入。
vy±0.5m/s、yaw±πrad/s、高度0.22–0.46m、频率1.5–4Hz，10秒重采样；80%环境使用航向反馈，静止概率10%。
rolling/trot/bound等权；周期占空比0.5、rolling占空比1且swing0；相位扰动±5度，成对误差上限10度。

| 累计迭代 | vx绝对上限m/s | swing上限m |
|---:|---:|---:|
| 0 | 1 | 0.10 |
| 2000 | 1.5 | 0.12 |
| 6000 | 2 | 0.15 |
| 10000 | 3 | 0.15 |
| 15000 | 4 | 0.15 |
| 20000以后 | 5 | 0.15 |

节点间线性插值。采样分支：50%原范围、30%中间范围（vx最多±2）、20%扩展范围；区间重叠。
续训保留课程进度，50k全程保持最终上限。
主要奖励：速度2、角速度1、机身高度-50、姿态-5、摆腿高度-30、触地相位力/速度各-4、Raibert-10。
swing约束的是轮底高度的相位轨迹；日志height_error是机身高度误差，不是摆腿误差。
完整随机化/噪声/奖励/终止细节以随附env.yaml为准。

## 最终结果与局限

30k于2026-10-11约21:24完成，50k续训于2026-10-12约10:08完成；日志训练耗时分别5:05:41和8:33:17。
最终80k末次训练窗口：reward3.90，平均episode20秒，正常超时99.02%，姿态超限0.98%，速度误差指标0.2961，机身高度误差0.0202。
这些不是全程平均值或全cmd成功率。

用户对30k回放反馈整体不错，指出|vx|>3时姿态扭曲、摆腿高度精度不足，因而续训50k；最终80k已GUI加载回放。
这里“成功”指训练、模型和回放pipeline可用，不代表上述两项问题已量化解决。
尚未完成30k/80k同cmd定量对照，也未做MuJoCo/实机验证。
最终导出经过385组CPU输入对照，TorchScript最大绝对误差0，ONNX约0.0006104；容差rtol1e-5、atol1e-3。

## 从零复现与使用

```bash
cd /home/joey/BarrierLoco
# 第一阶段从零30000次
./scripts/skill.sh multiskill train --headless --from-scratch \
  --num_envs 1024 --max_iterations 30000 --seed 42 --device cuda:0
# 第二阶段：将路径替换为刚生成的30k模型，追加50000次
./scripts/skill.sh multiskill train --headless --resume \
  --checkpoint /absolute/path/to/new_run/model_29999.pt \
  --num_envs 1024 --max_iterations 50000 --seed 42 --device cuda:0
# 默认回放、导出校验、录屏均使用保留的80k模型
./scripts/skill.sh multiskill play
./scripts/skill.sh multiskill verify
./scripts/skill.sh multiskill record --headless --seconds 6 --output-dir outputs/multiskill_final_record_01
```

默认76组cmd每组8秒，连续切换、不触发训练超时重置，但保留跌倒终止；右侧含cmd/actual与曲线。
保留源码快照`artifacts/experiment_snapshots/multiskill_scratch80k_20261012/source.tar.gz`用于训练版本追溯，
它是当时工作区代码快照（包含未提交源码），不是机器环境/资产/权重备份；其历史默认路径可能已被本次清理替换。
日常运行使用当前入口。最终模型、导出文件按Git规则被忽略，备份时须单独复制最终目录。

清理后验证：最终权重SHA256不变，87项回归检查、导出校验及无窗口加载回放通过；验证临时输出已删除。
