# 2026-10-14 / HighPlatform / Rough 15k → 大 batch 续训 5k

## 目标与状态
在本机复现高台技能，并检查扩大每次更新采样量后是否缓解策略退化。训练及固定高台评测均完成。
Rough：2026-10-13 12:32–18:57；高台：2026-10-13 22:54 开始，训练耗时 6:17:44；自动评测于 2026-10-14 05:15 完成。时区 Asia/Shanghai。

## 来源与代码变更
上游 shang13206/OmniBot 的 highplatform 分支，提交 9abf47ed67a317194515c99df7ccacd518a11e4a。
当前 HEAD 为 29af35dbbeb8c069fd13668c34719a795dc39674，包含未提交的统一任务改动；实际源码快照、patch 及校验值见 manifest.json。
Rough/HighPlatform 核心奖励、地形课程、PPO 与上游保持一致；共享机器人资产的较新腿限制由 task/训练入口覆盖为原版限制。
训练入口支持明确从零开始、绝对路径恢复、有限值检查；runner 增加 checkpoint 设备映射。
本轮关键参数为 4096 环境 × 48 步，而非本地旧设置 1024 × 24；每次更新 196608 条样本，与作者 8192 × 24 的样本数相同。
保存间隔改为 100；未修改奖励、课程或续训学习率恢复逻辑。跟随镜头、速度曲线及离线材质只用于回放。
原版与融合版同条件 20 次更新所得网络和优化器状态完全相同，审计依据位于 artifacts/go2w_highplatform/audit_20261013。

## 配置与模型关系
- Rough：OmniBot-Go2W-Rough-MLP-v0，从零 15000 次更新，1024 环境、24 步，seed 42，cuda:0。
- HighPlatform：OmniBot-Go2W-HighPlatform-MLP-v0，从 Rough model_14999.pt 恢复网络和优化器，额外 5000 次更新，4096 环境、48 步。
- 两阶段共实际执行 20000 次更新；原版 runner 恢复时复用迭代标签，最终编号为 19998。
- MLP：actor 53→512→256→128→16，critic 307→512→256→128→1，ELU、log_std、无观测归一化。
- 输入含机器人状态、速度命令、关节状态及上一动作；critic 另有特权信息。动作 12 腿位置（scale 0.25）+4 轮速度（scale 10）。完整观测顺序见保存的 env.yaml。
- 腿限制 23.5 Nm / 30 rad/s，腿 PD 40/1，轮 PD 0/0.5，自碰撞开启；物理 dt=0.005，decimation=4，控制 50 Hz。
- 高台 episode 10 s；36 个高度等级 0.10–0.80 m，平台长 1.5 m、宽 2 m；命令 vx=0.5–1.0 m/s、vy=0，目标世界航向 0；距离课程、Rough 奖励加 feet_stumble。
- PPO：5 epochs，4 minibatches，每个 minibatch 49152；初始 lr=3e-4、自适应 KL=0.01、entropy=0.008、gamma=0.99、lambda=0.95、clip=0.2。
- 环境版本：{"python": "3.10.20", "torch": "2.7.0+cu128", "isaaclab": "0.54.3", "isaacsim": "4.5.0.0"}。使用 /home/joey/miniconda3/envs/env_isaaclab/bin/python。
两阶段实际配置和初始化记录均保存在本目录 rough/、highplatform/；权重路径及 SHA256 见 manifest.json。

## 关键复现条件：num_envs 与每次更新的采样量

**HighPlatform 的 `num_envs` 不能只当作显存或运行速度选项。复现本次成功结果，应保持 `num_envs=4096`、`num_steps_per_env=48`、`num_mini_batches=4`；不能直接降为 1024 环境而仍沿用 24 步，并期待相同的训练行为。**

这里的 rollout batch 指每次 PPO 更新前收集的状态转移数：`num_envs × num_steps_per_env`。在 4 个 minibatch 下，对比如下：

| 高台阶段配置 | 并行环境数 | 每环境采样步数 | 每次更新样本数 | 每个 minibatch 样本数 |
|---|---:|---:|---:|---:|
| 作者原配置 | 8192 | 24 | 196608 | 49152 |
| 本地此前退化配置 | 1024 | 24 | 24576 | 6144 |
| 本次成功配置 | 4096 | 48 | 196608 | 49152 |

此前本地 batch 只有作者的 **1/8**。从同一个本地 Rough 15k 父模型出发，1024×24 的高台训练出现课程退化，后期 `model_20000.pt` 固定高台评测为 0/256；本次 4096×48 的早期和最终三个 checkpoint 均为 256/256。单纯延长小 batch 训练并没有解决此前观察到的退化。失败对照的本地审计依据见 `artifacts/go2w_highplatform/audit_20261013/audit_summary.json`（诊断数据未随仓库发布）。

影响不只在总 batch：更多并行环境可在同一轮更新中提供更多独立的重置、随机化、指令和地形课程状态；增加每环境步数则延长连续轨迹，改变时间相关性与回报估计的截断位置。因此，**4096×48 与 8192×24 只匹配采样总数，不是完全等价的训练配置**。本次结果支持“扩大并行环境数和每次更新采样量这一组合缓解了退化”，尚不能证明 `num_envs` 单独是根因，也未验证任意等 batch 配置都有效。

若以后只能使用 1024 环境，192 步可在算术上匹配 196608 条样本，但这仍是待验证的新实验，不能视作本次已成功配置。比较不同配置时应同时记录环境数、采样步数、batch、minibatch 和总交互样本数；相同 5000 次更新下，4096×48 的交互样本数为 983040000，是 1024×24 的 8 倍。本条经验针对高台阶段；本次 Rough 父策略仍由 1024×24 训练得到。

## 复现命令
工作目录 /home/joey/BarrierLoco，使用新的实验名称，避免覆盖已有日志。

```bash
scripts/skill.sh highplatform python locolab/scripts/locomotion/train.py --task OmniBot-Go2W-Rough-MLP-v0 --headless --num_envs 1024 --max_iterations 15000 --seed 42 --device cuda:0 --logger tensorboard --validate-finite --from-scratch --highplatform-pretrain --experiment_name reproduce_rough15k
```

以新训练输出的 model_14999.pt 替换下方 --checkpoint 可完整从零复现；直接用所保留父权重则复现高台阶段。

```bash
scripts/skill.sh highplatform train --headless --num_envs 4096 --max_iterations 5000 --seed 42 --device cuda:0 --validate-finite --experiment_name reproduce_highplatform_batch196608 --resume --checkpoint /home/joey/BarrierLoco/locolab/logs/rsl_rl/barrierloco_rough_two_stage_20261013-123245-15k/2026-10-13_12-32-51_rough_1024_seed42_15000iter/model_14999.pt agent.num_steps_per_env=48 agent.save_interval=100
scripts/skill.sh highplatform play --checkpoint /home/joey/BarrierLoco/locolab/logs/rsl_rl/barrierloco_highplatform_20261013-225424-env4096-batch196608/2026-10-13_22-54-31_highplatform_4096_steps48_5k/model_19998.pt --device cuda:0 --real-time --follow-camera --live-plot
scripts/skill.sh highplatform python locolab/scripts/locomotion/evaluate_highplatform.py --checkpoint /home/joey/BarrierLoco/locolab/logs/rsl_rl/barrierloco_highplatform_20261013-225424-env4096-batch196608/2026-10-13_22-54-31_highplatform_4096_steps48_5k/model_19998.pt --output artifacts/go2w_highplatform/reproduce_batch196608 --headless --device cuda:0
```

## 产物与验证

供克隆后使用的权重已打包至 `deploy/policy/go2w/highplatform_rough15k_batch196608/`，含父模型和最终模型（Git LFS）；下列绝对路径保留原训练来源。
最终模型：/home/joey/BarrierLoco/locolab/logs/rsl_rl/barrierloco_highplatform_20261013-225424-env4096-batch196608/2026-10-13_22-54-31_highplatform_4096_steps48_5k/model_19998.pt
训练及评测日志：/home/joey/BarrierLoco/logs/setup/highplatform_pipeline/20261013-225424-env4096-batch196608
父模型和最终模型已随 Git LFS 发布；原训练源码快照和完整日志仍需单独备份。

| checkpoint | 50 cm | 60 cm | 70 cm | 80 cm | 总通过 |
|---|---:|---:|---:|---:|---:|
| 15500 | 64/64 | 64/64 | 64/64 | 64/64 | 256/256 |
| 16000 | 64/64 | 64/64 | 64/64 | 64/64 | 256/256 |
| 19998 | 64/64 | 64/64 | 64/64 | 64/64 | 256/256 |

每个 checkpoint 在速度 0.6/1.0 m/s、seed 42/43、每高度每组合 16 次位姿扰动下评估，共 256 次。关闭观测噪声和域随机化，保留自碰撞。
通过要求四轮登上台面、有接触支撑，之后越过平台并稳定落地；绕台不计成功。超时上限 10 s。详细判定和逐次结果见 evaluation_*.json。
末次训练窗口：平均等级 23.2364，reward 38.63，速度误差指标 0.0519，base_contact 0.22%；不是全程统计。

## 最终验证与局限
本次固定高台验收达到目标，早期和最终 checkpoint 均通过，观察范围内没有复现此前的性能退化。
这支持扩大采样规模这一组合改动有效，但同时改变了环境数、轨迹长度和总采样量，不能单独证明某一个变量是唯一根因。
仅一个训练 seed；不代表跨 seed 保证、连续高台回放验收或实机成功。没有修改原版课程判定和学习率恢复问题。
