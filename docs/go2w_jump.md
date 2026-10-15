> 历史 Jump 说明：默认 `omnijump` 入口已改为 GoalGrid。旧权重已归档到工作区外；当前使用方式见 [GoalGrid 整合说明](goalparkour_integration.md)。底层 Jump 代码仍供 Parkour 复用。

# Go2W Jump：平滑 TO 两阶段训练

## 仓库交付

旧两阶段权重及测试报告已从当前工作树移除。下文仅保留历史方法说明，当前 Jump 请使用 [GoalGrid](goalparkour_integration.md)。

在已有训练环境、`locolab/` 目录可用正式入口重现两阶段：

```sh
python scripts/locomotion/train.py --task OmniBot-Go2W-Jump-MLP-v0 \
  --jump_stage skill --num_envs 4096 --max_iterations 5000 --seed 42 \
  --headless --logger wandb --video --video_interval 500 --video_length 1000
python scripts/locomotion/train.py --task OmniBot-Go2W-Jump-MLP-v0 \
  --jump_stage speed --resume --load_run '<new skill run>' --checkpoint 'model_4999.pt' \
  --num_envs 4096 --max_iterations 5000 --seed 42 \
  --headless --logger wandb --video --video_interval 500 --video_length 1000
```

第二条命令应在第一阶段完成后运行。已有第一阶段权重也随网络包保留，
重新训练第二阶段时需要用实际 checkpoint 所在目录配置 resume 路径。

## 最终交付与产物索引

两阶段各 5000 次更新已正常完成，第一阶段完全随机初始化，第二阶段完整续接第一阶段。
主工作区下的最终产物包为 `artifacts/go2w/jump_smooth_training_20261009/`；
完整文件清单和校验和见其中的 `manifest.json`。

| 位置 | 保留内容 |
| --- | --- |
| `models/model_4999.pt` | 第一阶段最终 checkpoint，可重新训练第二阶段 |
| `models/model_9998.pt` | 第二阶段最终 checkpoint，含模型和优化器状态 |
| `params/skill/`、`params/speed/` | 两阶段实际训练参数 |
| `reference/` | 训练 TO、被采纳的控制序列、独立回放、验证记录和 TO 视频 |
| `source_snapshot.tar.gz` | 正式运行的源码、机器人资产、测试与依赖声明快照 |
| `final_review/` | 最终策略评估、四档速度视频和带噪声评估结果 |
| `run_records/` | 完整训练日志压缩件、事件曲线、提交与验证记录 |
| `pipeline.py` | 两阶段启动与 checkpoint 校验流程 |
| `maintenance/` | 清理清单、执行回执和历史清理记录 |

远端仅保留正式运行目录
`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-jump-smooth-two-stage-20261009-r1`，
其中两个最终 checkpoint 的原路径不变；每阶段还保留最后一段训练录像。
旧跳跃实验、失败启动、TO 搜索中间版本、中间 checkpoint、重复 W&B 本地缓存已清理；
非跳跃任务、运行中的其他任务和 W&B 在线历史未改动。

第一阶段权重 SHA256：
`173b0d9e22e2b451c317b7892b326a341aa85358a1c280e6f39cd36ec50f71fb`。
最终权重 SHA256：
`64dc3fe4396e79afd0bc3e8a2496f6cc262ffa78466d409ff7c1e454d74dc550`。

### 最终策略验证

普通地面起步、无 TO 空中重置、确定性策略，四个速度测试均起跳并稳定落地：

| vx 指令 (m/s) | 机身抬升 (cm) | 最低轮底净空峰值 (cm) | 腾空前移 (m) |
| --- | --- | --- | --- |
| 0.35 | 42.0 | 50.1 | 0.48 |
| 0.5 | 41.7 | 53.0 | 0.53 |
| 1.0 | 38.6 | 48.7 | 0.82 |
| 1.5 | 37.8 | 47.7 | 1.04 |

这些是平地确定性演示，不是障碍跨越或实机认证。起跳前 vx 比指令快约 0.26–0.36 m/s，
腾空偏航约 5–9°，记录到最高轮速 37.5 rad/s，超过任务配置的 30 rad/s。
20 次带训练动作噪声的独立测试全部起跳，但均未满足连续 0.1 s 四轮稳定着地判据。
详细证据保留在 `final_review/summary.json`，不因清理而删除这些限制记录。

在线记录使用服务器已有的 loopback 代理转发；若再次运行两阶段脚本，
须保持对应本机代理和 SSH 转发在线。不要把历史完成记录当成新运行的验证结果。

## 当前训练参考（2026-10-09）

正式训练使用平滑起跳→收腿 TO，从零执行了下述两个阶段。
当前默认参考为 `go2w_jump_smooth_transition_full_dynamics_to.json`，
由 `scripts/locomotion/export_jump_reference.py` 从独立动力学回放的真实状态导出，
不是 PD action target；没有缩放时间或改写姿态，旧 20 cm 参考仍保留用于历史复现。

- 起跳 / 顶点 / 首次触地：0.38 / 0.62 / 0.88 s；完整参考 1.8 s，加 0.4 s 恢复，jump 时钟共 2.2 s。
- 位姿以触发时 XY 和 yaw 为坐标基准，Z 保留真实离地高度。随机参考重置恢复完整四元数、
  世界角速度、12 腿关节和 4 轮状态，并保留真实关节限位内的参考姿态；普通 flat 重置不变。
- critic 继续接收归一化相位；actor 接口不变。stage2 的固定方向覆盖完整 2.2 s。
- 自碰撞开启，与 TO 验证一致；其余奖励权重、MLP、动作尺度和该任务执行器限值沿用原配方。
- TO 的固定起点回放通过；批量回放 195/256 通过，仍有超速和接触失败，不能视为鲁棒控制器认证。

流程脚本位于 `artifacts/go2w/jump_smooth_training_20261009/pipeline.py`。
新第一阶段不加载任何 checkpoint；第二阶段仅在新第一阶段完成并验证 model_4999 后自动续接。
以下旧基线说明仅供对照，新训练不加载该权重。
旧 checkpoint 的行为复现应使用对应的旧运行快照，不能把当前的新时序配置当作旧配置。

## 保留的历史版本

采用 `2026-09-28_10-15-22_moving_jump_grounded_ascent_from_posture4999_4096_1gpu_5k_video/model_8500.pt`
对应的奖励、网络、动作尺度与历史仿真执行器参数。默认任务为 `OmniBot-Go2W-Jump-MLP-v0`，
配置收敛到 `locolab/source/locomotion/tasks/Go2W/jump_env_cfg.py`，不再通过额外包装类覆盖。

保留权重 SHA256：`3a7aff517293e683f06d8cad98cd3fbdc2b07c6e950c09a78c4adf76a32783db`。
该旧基线按用户此前的保留要求单独留档；权重、参数、视频及历史源码快照集中于
`artifacts/go2w/jump_baseline_8500/`，worktree 中的重复 checkpoint 已删除。
保留的 20 cm TO 文件为 `go2w_jump_20cm_fast_full_dynamics_to.json`，
SHA256：`b6503f202e0cbc0c70ca7653629d2b00d427139267f6ad3912d94a5d88ca3f39`。

OmniJump、ForwardJump、独立高度课程、Flat→Jump 预训练的废弃入口、专用代码、
求解器与实验测试已移除。非跳跃任务及共享/部署执行器修正保留。
历史删除清单已归并到最终产物包的 `maintenance/history/`；W&B 在线历史不删除。

## 两阶段重训

用户明确要求第一阶段完全随机初始化，**不加载 model_8500，也不初始化自 Rough**。
model_8500 仅作为保留的能力基线。

| 阶段 | 初始化 | PPO 更新次数 | vx 范围 (m/s) | TO 随机相位 reset / 普通地面 reset |
| --- | --- | --- | --- | --- |
| skill | actor、critic、优化器全新 | 5000 | [0.2, 0.5] | 80% / 20% |
| speed | 完整加载 skill 最终 actor、critic、优化器 | 再 5000 | [0.2, 1.5] | 20% / 80% |

两段使用同一任务和基础跳跃奖励，外部 vy/yaw 指令为零；不启用速度课程。
第一段通过较多 TO 相位初始化学习起跳/腾空/落地，第二段以地面行进触发为主。
`train.py --jump_stage skill` 禁止 resume/warm-start；`--jump_stage speed` 必须显式 resume。
阶段参数在 `apply_jump_training_stage` 中统一设置并保存进 env.yaml。

### 第二阶段向前跳修订（2026-10-08，历史）

第一次两阶段训练已完成。确定性地面回放证明最终网络可在 vx=0.35/0.5/1.0/1.5 下
跳起并落地，但高速回放中 yaw 转偏，触发后 2.5 s 的侧向位移达到 0.89–1.10 m。
原水平速度奖励以当前 yaw 为基准，因此转偏后的运动仍可能被算作向前。

当时仅在 `speed` 阶段增加固定方向约束（当前锁定时长已随新 TO 延长至 2.2 s）：

- 接受 jump 时记录实际 yaw，锁定该方向 2.0 s，覆盖准备、腾空和实际落地恢复。
  该锁定独立于原 1.2 s jump 时序，不依赖接触锁存；失败请求不重置参考方向。
- 水平速度奖励在固定的触发 yaw 坐标系追踪 vx/vy，转身不能重新定义前进方向。
- 新增 yaw 偏差平方惩罚 −20、世界 z 轴角速度平方惩罚 −5、参考方向侧向速度误差平方惩罚 −10。
  旧姿态、三轴角速度、TO 关节轨迹及起跳奖励继续使用原权重。
- 保持 actor/critic=54/309、四维外部接口。方向锁定期间，将原 vx/vy 目标换算到
  当前机身 yaw 坐标系后填入已有的两列速度指令观测，使 MLP 能感知转偏。
  不改写 command manager 中的外部速度目标；actor 与 critic 使用同一换算。
- `skill` 和保留基线不启用这些方向约束。重跑第二阶段从原第一阶段最终
  `2026-10-07_21-43-20_jump_skill_scratch_4096_gpu3_5k_video/model_4999.pt`
  完整 resume，再训练 5000 次。

回放新第二阶段网络时使用 `play.py --jump_stage speed`，保留普通地面 reset 和固定
Play 速度，同时加载匹配的方向换算。后续部署也需复现同一指令换算和方向锁定。

正式训练：服务器单卡、4096 环境、seed=42，每段 5000 次，W&B online；
每 500 次迭代录制 1000 步（20 s）视频。流程控制脚本
`artifacts/go2w/jump_smooth_training_20261009/pipeline.py`
会在 skill 正常完成且 checkpoint/参数校验通过后自动启动 speed。
第一段失败则第二段不启动；不会静默重试或误加载保留的 model_8500。
旧 runner 从保存的迭代索引继续，因此两个阶段各执行 5000 次更新时最终文件编号为 9998，
实际总更新次数仍为 10000。

## 固定接口和奖励

- 纯 MLP，actor/critic=54/309，隐藏层 [512,256,128]，ELU；actor command 为
  `[vx, vy, yaw_rate, jump_trigger]`，跳跃相位只进入 critic。
- 腿位置/轮速度 action scale=0.25/10；不裁剪 action target；episode=10 s。
- 当前 TO 时序为 0.38/0.62/0.88/1.8 s，追加 0.4 s 恢复段，固定 2.2 s 后清零。
  触发只启动相位时钟，不改写机器人位置或速度；普通地面 reset 后先走 1–2 s，
  再每 2–4 s 以 80% 概率请求 jump。
- 触发要求四轮接触、机身倾角 <0.5 rad、|vz|≤0.2 m/s、实际 vx≥0.1 m/s。
- TO 关节位置 +80、起跳前/之后关节速度各 +20；固定上升段向上速度 +100、
  高度误差平方 −200、高度 tanh +20，三者不以四轮离地为前提。
- 姿态 −5、三轴角速度 −2、实际离地轮速 −0.002、触地后竖直速度 −2。
- hip L1 默认偏差 −0.5；thigh/calf 合并的非跳跃 L1 偏差 −0.25；
  保留限位、力矩、动作变化等原有约束，不加入后续实验的额外姿态正则。
- 原 PPO：学习率 1e-3、自适应 schedule、entropy=0.01、初始 std=1.0；
  不使用已废弃高度课程的 bounded-std、KL 回滚或 Huber critic 配方。
- 该任务历史仿真限制：12 腿关节统一 23.5 Nm / 30 rad/s。
  **这不是硬件规格修订**；共享资产和部署的小腿 35 Nm / 20 rad/s 保持不变。

## 历史基线验证

恢复基线后，单环境确定性回放从正常地面起步，vx=0.35：
机身相对触发点抬升 0.5164 m、连续四轮腾空 0.62 s，
1.76 s 确认稳定落地，无提前终止、无 PPO 更新。
机身抬升不是轮底净空，单次演示不是随机条件成功率。
旧 1.2 s 截止早于实际稳定落地，因此旧 jump_success 可为零；旧记录不追溯修改。

运行验证使用现有 env_isaaclab；指定的 robot_lab 不可用，不安装或重建依赖。
