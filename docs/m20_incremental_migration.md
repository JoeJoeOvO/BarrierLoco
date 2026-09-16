# Go2W 基线与 M20 逐项迁移

## 当前状态：恢复局部地形预瞄版本，关闭占空比（准备从原检查点接续）

2026-09-12 因台阶/越障表现退化，撤销整套 M20 配方作为活动训练任务。
基线为分支起点 `bdc66cdb8488e82393d7c256ffe86246e34fd417` 的 Go2W Rough；
回退时的环境、资产与 PPO 配置已逐文件对照 2026-07-22 feet-baseline 工作区。
随后按用户明确选择，在该基线上修改下列奖励；没有启用整套 M20 配方。
在阶段 1 上加入对角支撑及 gait_level、对角关节一致性，并将三项正则接入 g；轮端接触力超限惩罚已替换 feet_stumble。
24 项奖励的旧三组训练已按用户要求停止并保留结果；26 项奖励的新 MLP、DreamWaQ 已从头启动，RNN 未重新启动。
阶段 3 按用户选择增加 Velocity-slack，并提高线/角速度跟踪权重，当时配置为 27 项奖励。
26 项奖励的两组训练已按用户要求停止并保留检查点；27 项奖励的新 MLP、DreamWaQ 已从头启动，未热更新旧任务。
随后按用户要求，将本地 Rough MLP 的噪声参数化改为 log_std，并按最终选择将 hip/thigh/calf 权重设为 -1.5/-1.0/-0.5。
服务器阶段 3 对照继续保留，阶段 4 两组已停止并保留结果；用户随后要求恢复轮力矩平方惩罚，本地已将 torques 扩展到全部 16 个关节。
随后用户要求将水平姿态权重调为 -10、环境数增为 8192，并重新提交 MLP / DreamWaQ。
阶段 6 独立快照已启动并替换阶段 4 两组训练；阶段 3 对照保留。
用户随后要求关闭 Velocity-slack 与机身高度奖励；本地通过 None 显式覆盖，阶段 7 活动奖励为 25 项。
随后加入转向滑动、转向腾空和接触占空比，活动奖励变为 28 项；thigh/calf 保持纯转向/侧移豁免，四个阈值显式设为 0.1。
阶段 8 的 MLP / DreamWaQ 熵系数为 0.008，已替换阶段 6。用户于 2026-09-13 要求关闭转向滑动、腾空、占空比三项并重启；本地活动奖励恢复为 25 项，其余参数保持阶段 8。

当前本地已删除 gait_level 课程项；六项奖励改为固定权重，直接保留原有距离地形课程。阶段 10 的新两组训练已使用固定权重，阶段 9 的旧训练保留作为对照。

活动任务：

| 网络 | 任务 | PPO 初始学习率 | 熵系数 | 动作噪声 |
| --- | --- | ---: | ---: | --- |
| MLP | `OmniBot-Go2W-Rough-MLP-v0` | 3e-4 | 0.008 | log |
| RNN | `OmniBot-Go2W-Rough-Rnn-v0` | 3e-4 | 0.01 | scalar |
| DreamWaQ | `OmniBot-Go2W-Rough-DreamWaQ-v0` | 3e-4 | 0.008 | log |

当前奖励变化：

当前腿姿态奖励名称与 Go2W 关节对应：`hip_joint_pos_penalty`、`thigh_joint_pos_penalty`、`calf_joint_pos_penalty`。
分别替代此前的 `hipx_joint_pos_penalty`、`hipy_joint_pos_penalty`、`knee_joint_pos_penalty`；仅改名称，权重与计算逻辑不变。
历史日志和未注册的 M20 参考配方保留原名；阶段 12 新 DreamWaQ 已使用 hip/thigh/calf 名称。

| 项目 | 阶段 0 | 当前 |
| --- | --- | --- |
| 线速度跟踪 | 基础 2.0、精跟踪 0.5 | 基础 3.0、精跟踪 0.5；核宽度保持 0.5 / 0.2 |
| 角速度跟踪 | 基础 1.0、精跟踪 0.25 | 基础 1.5、精跟踪 0.25；核宽度保持 0.5 / 0.2 |
| Velocity-slack | 无 | 已关闭；阶段 3–6 为区间内 +1 |
| 机身高度 | 0.42 m 附近的容差惩罚，权重 -5 | 已关闭 |
| 力矩平方 | 全部关节，-5e-4 | 全部 16 个关节（12 腿 + 4 轮），-5e-4；阶段 1–4 排除轮关节 |
| 动作一阶变化 | -0.01 | -0.01 |
| 动作二阶平滑 | -0.002 | -0.002，保留 reset-aware 历史 |
| 水平姿态 | -0.5 | -10（阶段 1–4 为 -5） |
| hip 默认角偏离 | -0.25 × L1 | 替换为 -1.5 × L2 范数（阶段 3 为 -3） |
| thigh / calf 默认角偏离 | 无独立项 | -1.0 / -0.5 × L2 范数（阶段 3 为 -1.5 / -0.75） |
| stand_still | -0.5 × 腿关节 L1；完整命令范数 <0.1 | -1 × 腿关节 L1；仅平移命令范数 <0.06 |
| 对角支撑 | 无 | rotation_gait_status，+2 |
| 转向滑动 | 无 | 已关闭（阶段 8 曾启用） |
| 转向腾空 | 无 | 已关闭（阶段 8 曾启用） |
| 接触占空比 | 无 | 已关闭，恢复阶段 11 配置 |
| 对角关节一致性 | 无 | joint_mirror，-0.03；FL↔RR、FR↔RL 的 hip/thigh/calf |
| 轮端接触 | feet_stumble，水平/竖直力比 >5 时扣 -0.1 | contact_forces，每轮超出 56.6 N 的历史峰值部分求和，权重 -1.5e-4 |

以下是已关闭的 Velocity-slack 在阶段 3–6 的行为，函数保留供历史复现：使用与现有跟踪项相同的机身坐标系 x 轴速度。
当 vx_cmd 非零且实际速度沿命令方向满足 `0.3*abs(vx_cmd) <= sign(vx_cmd)*vx_actual <= 1.5*abs(vx_cmd)` 时给 +1，边界包含在内。
停住、运动方向相反、超出区间、纯侧移/纯转向等 vx_cmd=0 命令，以及非有限输入均不给该奖励。
倒车命令按相同方向比例处理；例如 vx_cmd=-0.6 时，vx_actual 在 [-0.9,-0.18] m/s 内得分。
本项不额外惩罚区间外速度，已有连续跟踪项负责误差引导；仅增加速度奖励，未撤销或调低腿姿态惩罚。

三组 L2 姿态项在完整命令范数 ≤0.1 且机身水平速度 ≤0.5 m/s 时乘 5。
thigh/calf 项在纯转向（abs(wz)>0.1 且 norm(vxy)<0.1）或纯侧移
（abs(vy)>0.1 且 norm(vx,wz)<0.1）时关闭。
当前 thigh/calf 改用已有 `height_scanner`（1.6×1.0 m）检测局部地形，移除对 env_origin.z 的依赖。
检测区域是机身支撑矩形（前后各 0.3 m、左右各 0.25 m），加上沿平移命令方向前伸至 0.5 m 的走廊。
即使实际速度为零，也按非零命令预瞄；倒车向后预瞄，纯静止仅检查支撑区域。
至少 10 条有效射线的 5%–95% 地面高度跨度严格大于 0.06 m 时，thigh/calf 惩罚乘 0.1。
过滤非有限射线和孤立极值；平坦高平台不触发，局部台阶、落差或足够倾斜的坡面可以触发。
机身小扫描器和旧高度奖励扫描器保留供诊断；不改变策略/critic 的观测维度。
历史 M20 参考配方仍保留原先的小扫描区域与原点相对高度判据，当前 Rough 使用新函数。

termination、terrain、commands、events 保持阶段 0；地形升降级规则保持原有实现，
当前不再计算、缓存或记录 gait_level。动作平滑权重固定为 -0.002。
关节力矩、一阶动作变化、二阶动作平滑、对角支撑、轮端接触力、对角关节一致性均使用固定权重。
hip/thigh/calf 默认姿态、stand_still、机身水平等其余奖励保持原倍率。
joint_mirror 比较两组对角腿对应关节的原始角度平方差，取两组平均，保留原版的直立系数，已移除 gait_level 缩放。
保留原版无命令门控的行为，直行也生效；不对 hip 角度额外取反，不包含轮关节。
接触力阈值按标称模型质量缩放：100×19.523/34.491≈56.6 N；保留当前 6 帧接触历史。
对角支撑使用 [FL, RR] 与 [FR, RL] 两组：平移命令模长 <0.5 m/s 且 abs(wz_cmd)>0.05 rad/s 时，
一组两轮接触、另一组平均世界高度高出超过 0.05 m 即得到 +2（乘 dt 前）。
保留原判据：较高组不要求离地，不检查实际 yaw 速度，也不要求两组随时间交替；
低速前进伴随转向命令时可以触发。历史阶段 1–9 使用的 g 倍率已从当前活动配置移除。
原 default joints、KP/KD、action scale、网络/观测、辅助解码器参数保持原版。
M20 配方专用的 reset 几何补偿、calf 速度覆盖和质量/惯量随机化不进入该基线。

四个 `-M20-` 注册入口及三个 M20 PPO runner 已移除。`mdp/m20/` 与 `tasks/Go2W/m20_env_cfg.py`
保留为未注册的参考代码；历史服务器快照、日志和检查点保留。阶段 1 已按用户要求提交三种网络训练。

## 后续实验次序

| 阶段 | 唯一改动 | 保持固定 | 当前状态 |
| --- | --- | --- | --- |
| 0 | 无：原 Rough 基线 | 全部参数 | git 起点和服务器基线快照保留作对照 |
| 1 | 用户指定的腿力矩选择、水平姿态及 M20 腿姿态奖励组 | PPO、动作、观测、地形、命令、终止和随机化 | 24 项奖励快照的三组训练已停止，结果保留 |
| 2 | 对角支撑、轮端接触力、对角关节一致性，并将三项动作/力矩正则接入 g | 其他奖励基础权重与控制配置 | 两组已停止，结果保留 |
| 3 | 线/角跟踪权重 3.0/1.5，新增 Velocity-slack +1 | 原精跟踪核、其余奖励、PPO/action/场景 | 已启动 MLP / DreamWaQ，RNN 不启动 |
| 4 | Rough MLP 改用 log_std；hip/thigh/calf 权重 -1.5/-1.0/-0.5 | 初始实际 std=1、学习率、熵系数、action、奖励门控和其余权重 | 已停止并保留结果，由阶段 6 替换 |
| 5 | 将 4 个轮关节重新纳入 torques 平方和 | 权重 -5e-4、gait_level、其余奖励和训练参数 | 已随阶段 6 启用 |
| 6 | 水平姿态 -10、8192 环境，并包含阶段 5 轮力矩恢复 | 其余奖励、PPO 参数、action 和台阶固定镜头 | 已停止并保留结果，由阶段 8 替换 |
| 7 | 关闭 Velocity-slack 与机身高度奖励 | 其余 25 项奖励、8192 环境、PPO/action 和镜头 | 已随阶段 8 启用 |
| 8 | 三项 M20 转向奖励、MLP / DreamWaQ 熵系数 0.008 | thigh/calf 原有门控、其余奖励与控制配置；slack/height 关闭 | 已停止并保留结果，由阶段 9 替换 |
| 9 | 关闭转向滑动、腾空和占空比三项 | 其余奖励、entropy 0.008、8192 环境、log_std、镜头 | 两组已从头启动，8192 环境 |
| 10 | 删除 gait_level 课程及六项奖励倍率 | 地形升降级、原基础权重、门控和历史重置 | 两组新训练已启动，原阶段 9 保留对照 |
| 11 | thigh/calf 改用局部高度跨度及命令方向预瞄 | 原权重、6 cm 阈值、10% 倍率、转向/侧移豁免；g 已移除 | 已停止 DreamWaQ 并保留结果，由阶段 12 替换 |
| 12 | 恢复无 gait 缩放的 rotation_gait_symmetry +15 | 局部地形预瞄、entropy 0.008、其余奖励/控制；滑动和腾空关闭 | DreamWaQ 已从头启动 |
| 后续 | commands、terrain、events 分开评估，每组只动一个参数或一种随机化 | 已通过验收的其他部分 | 候选，未启用 |

原 Rough 和 M20 的台阶高度范围均为 0.05–0.23 m、步宽均为 0.3 m，无需重复迁移。
不要把六类地形比例、速度范围、转向采样比例和域随机化一次替换。
转向腾空/支撑奖励需单独验证转向目标，不能用其分数代替台阶通过能力。
当前已解除六项奖励对 gait_level 的依赖，保留关节力矩 -5e-4、一阶 -0.01、二阶 -0.002 的固定权重。

本轮按用户要求直接更新当前三个 Rough 任务；原基线由 git 起点和服务器
`OmniBot-go2w-baseline-20260912` 快照保留。后续训练必须使用可区分的 run_name，
保留基线检查点和日志；其他候选优先使用独立配置和任务。
已确认的物理一致性修正也单独比较，不与 PPO/reward 改动混入同一组。
本表只定义候选与顺序，不表示已经提交这些训练或确认它们有效。

阶段 4 的姿态权重由 Rough 环境共享，当前新 MLP / DreamWaQ 已使用；RNN 下一次 Rough 启动也会生效。
HighPlatform MLP 继承 Rough MLP runner，也会使用 log_std，以保持两者暖启动时的噪声参数结构一致。
旧 scalar MLP 检查点含 `std`，新配置要求 `log_std`；当前加载器严格匹配参数，不会自动转换。
如需接续旧检查点，应显式转换为 `log_std=log(std)` 并处理对应优化器状态，不能直接按新配置恢复旧训练。

## 验收口径

先在同一组固定台阶高度、命令、出生姿态和仿真参数下比较确定性策略。
上台阶和下台阶分别记录通过率、前进距离、卡住比例、摔倒率，以及必要时 calf 限幅触发比例。
训练录像中的抽样噪声和不同出生位置会影响观感，不能单独据此判定策略优劣。

本轮按用户明确要求同时启动 MLP、DreamWaQ、RNN。后续新候选可先使用 DreamWaQ 初筛；
比较相同训练步数并保留基线，初筛有效后再检查其他种子和网络。总回报与平均 terrain level 作为辅助指标：
改变 reset 出生偏移后，基于 env_origin 计算的 terrain curriculum 不能直接当作等价通过率。

越障结果有可重复改善、其余基本能力无明显退化后，才将单项改动纳入下一阶段。
实际训练提交按明确选定的候选执行，不自动批量提交整套迁移。

## 阶段 1 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-posture-20260912`。
各组从头训练，4096 环境、10000 次迭代、seed=42；录像每 500 次迭代触发，1000 步/段（约 20 秒）。
实际验证时间：2026-09-12T10:08:14.520490+08:00。PPO 初始学习率 3e-4、熵系数 0.01；24 项奖励、模型有限值及录像开关均已核对。
以下迭代数为启动检查时刻，不表示训练已完成；首段正式录像计划在第 500 次迭代触发。

| 网络 | GPU | 检查时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 0 | 13 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/uxsqtl3h) |
| dreamwaq | 1 | 12 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/qouwy3z2) |
| rnn | 2 | 9 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/qrtj5g20) |

控制脚本、配置验证脚本和记录位于该快照的 `run_scripts/`；历史基线与旧训练目录未覆盖。

## 阶段 2 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-gait-20260912`。
旧三组已停止，检查点和日志保留。新 MLP、DreamWaQ 各 4096 环境、10000 次迭代、seed=42，从头训练；RNN 未重启。
录像每 500 次迭代触发，1000 步/段（约 20 秒）。实际验证时间：2026-09-12T12:36:13.275940+08:00。
已核对 26 项奖励、gait_level、轮端接触力 56.6 N 阈值、关节镜像、三项正则缩放，以及原 PPO/action 和初始检查点有限值。
以下迭代数仅为启动检查时刻；首段正式录像计划在第 500 次迭代触发。

| 网络 | GPU | 检查时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 0 | 28 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/210spdbk) |
| dreamwaq | 1 | 24 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/rwtw5n34) |

完整提交、启动脚本和验证记录在该快照 `run_scripts/` 下；旧实验目录未覆盖。

## 阶段 3 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-slack-20260912`。
旧 MLP、DreamWaQ 已停止并保留日志和 model_1500.pt；新两组从头训练，4096 环境、10000 次迭代、seed=42。
本次新增 Velocity-slack 权重 1.0，速度比范围 [0.3,1.5]；线/角跟踪权重 3.0/1.5，保留精跟踪 0.5/0.25。
PPO 初始学习率 3e-4、熵系数 0.01；MLP 使用 scalar，DreamWaQ 使用 log_std。其他奖励和控制配置保持阶段 2。
录像每 500 次迭代触发，1000 步/段（约 20 秒）。实际配置和初始检查点有限值已验证。
检查时间：2026-09-12T14:50:42.564897+08:00。下列迭代数是启动检查时刻，不代表训练完成；首段录像计划在第 500 次迭代触发。

| 网络 | GPU | 检查时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 0 | 45 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/l27rdt31) |
| dreamwaq | 1 | 41 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/deonv9l8) |

完整启动脚本、提交记录和验证结果位于该快照的 `run_scripts/`。

本轮视频启动使用 `--video_terrain pyramid_stairs_inv`，通过课程地形列映射选择凹台阶环境。
镜头相对地块固定；环境重置到其他难度地块时刷新镜头原点，避免继续拍摄空地。
train 自动上传标签保持 `Videos/train`；play 入口也支持同一选景参数，二者不修改训练地形分布。

## 阶段 4 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-logstd-posture-20260912`。
保留阶段 3 的 GPU 0/1 训练，新增 GPU 2/3 两组；均从头训练，4096 环境、10000 轮、seed=42。
两组动作噪声均为 log_std，初始实际 σ=1.0；hipx/hipy/knee 权重 -1.5/-1.0/-0.5。
跟踪权重 3.0/1.5、Velocity-slack +1、学习率 3e-4、熵系数 0.01、action 与 KP/KD 保持原值。
视频每 500 轮录制 1000 步，选择 `pyramid_stairs_inv`、env_index=3584，固定地块镜头并在地块改变时刷新。
110 个代码/资产文件已校验一致，57 项测试通过；实际启动核验时间 2026-09-12T17:02:14.133207+08:00。

| 网络 | GPU | 核验时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 2 | 172 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/gsrf9je4) |
| dreamwaq | 3 | 154 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/itt9e13v) |

核验时检查点均为有限值，实际保存的 27 项奖励、log_std、视频和镜头配置已通过检查。

## 阶段 6 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-8192-20260912`。
GPU 2/3 各 8192 环境、10000 轮、seed=42，均从头训练；MLP / DreamWaQ 均为 log_std。
水平姿态 -10；torques 恢复全部 16 个关节，权重 -5e-4×gait_level。
腿姿态 -1.5/-1.0/-0.5，跟踪 3.0/1.5、Velocity-slack +1，PPO 初始学习率 3e-4、熵系数 0.01 保持原值。
视频每 500 轮录制 1000 步（约 20 秒），固定凹台阶镜头，env_index 随环境数量更新为 7168。
每轮采样 8192×24=196608 步，是阶段 4 的两倍；按相同迭代比较时需注意样本量差异。
58 项相关测试通过；110 个代码/资产文件校验一致；实际启动核验时间 2026-09-12T20:50:28.646024+08:00。

| 网络 | GPU | 核验时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 2 | 7 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/kegd8ghk) |
| dreamwaq | 3 | 6 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/14galdhp) |

阶段 4 旧训练停止记录：

- mlp：停止前第 3624 轮，保留 `model_3500.pt`。
- dreamwaq：停止前第 3322 轮，保留 `model_3000.pt`。

实际保存的 27 项奖励、环境数、噪声参数化、控制参数、台阶镜头与检查点有限值已核验。

阶段 8 复用已有 reset-aware 的 FeetAirTime 与 RotationGaitSymmetry，实现每个环境独立重置历史。
转向腾空只要求 abs(wz_cmd)>0.1；滑动要求 norm(vxy_cmd)<0.1 且 abs(wz_cmd)>0.1；占空比要求 norm(vxy_cmd)<0.5 且 abs(wz_cmd)>0.05。
未改变 stand_still 的平移命令门控，也未增加 M20 的 20% 纯转向命令采样；yaw 跟踪权重继续为 1.5。

## 阶段 8 训练记录（2026-09-12）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-rough-turn-20260912`。
GPU 2/3 各 8192 环境、10000 轮、seed=42；MLP / DreamWaQ 均从头训练，log_std，初始实际 σ=1.0。
熵系数固定 0.008，学习率 3e-4；Velocity-slack 与 base_height 关闭。
新增 feet_slide_ang_z_cmd -2g、feet_air_time_ang_z_M20 权重 +50g、rotation_gait_symmetry +15g。
thigh/calf 纯转向和纯侧移豁免阈值显式设为 0.1；权重仍为 -1.0/-0.5；stand_still 和 commands 未改变。
其余保留水平姿态 -10、全部 16 个关节力矩 -5e-4g、线/角速度跟踪 3.0/1.5、原 action 与 KP/KD。
视频保持固定凹台阶视角、env_index=7168，每 500 轮录制 1000 步。
59 项相关测试通过，110 个代码/资产文件校验一致；8 环境 260 步实际仿真验证奖励有限值、选择顺序和局部历史重置。实际训练核验时间：2026-09-12T23:39:24.870049+08:00。

| 网络 | GPU | 核验时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 2 | 11 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/2y25li1r) |
| dreamwaq | 3 | 10 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/jw4fln9p) |

阶段 6 旧两组训练已停止并保留结果：

- mlp：停止前 1848 轮，保留 `model_1500.pt`。
- dreamwaq：停止前 1713 轮，保留 `model_1500.pt`。

## 阶段 9 训练记录（2026-09-13）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-turn-off-20260913`。
关闭 feet_slide_ang_z_cmd、feet_air_time_ang_z_M20、rotation_gait_symmetry；Velocity-slack 与 base_height 继续关闭，共 25 项活动奖励。
其余保留 thigh/calf 转向/侧移豁免、对角支撑 +2g、关节镜像 -0.03g、全部关节力矩 -5e-4g、水平姿态 -10。
GPU 2/3 各 8192 环境、10000 轮、seed=42，均从头训练；log_std、学习率 3e-4、熵系数 0.008。
录像继续使用固定凹台阶视角、env_index=7168，每 500 轮录制 1000 步。
59 项相关测试通过，110 个代码/资产文件校验一致；实际启动核验时间 2026-09-13T11:04:36.472899+08:00。

| 网络 | GPU | 核验时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 2 | 10 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/rhdisaax) |
| dreamwaq | 3 | 9 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/4w3878yc) |

阶段 8 旧训练停止并保留：

- mlp：停止前 8040 轮，最后检查点 `model_8000.pt`。
- dreamwaq：停止前 7481 轮，最后检查点 `model_7000.pt`。

实际配置确认三项奖励为 None，日志中不再包含对应 Episode_Reward 项；初始检查点均为有限值。

阶段 10：标准力矩、action rate、接触力直接使用 Isaac Lab 函数，smoothness 直接使用公共 reset-aware 类。
关节镜像与对角支撑使用无课程依赖的 unscaled 实现，保留原直立/命令/高度/接触判据。
未注册的 M20 历史配方仍保留原带 g 包装和课程代码供复现；当前 Rough 运行路径不调用这些包装，不产生 _m20_gait_level 或 Curriculum/gait_level。

## 阶段 10 训练记录（2026-09-13）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-no-gait-20260913`。
新增 GPU 0/1 上的 MLP / DreamWaQ，各 8192 环境、10000 轮、seed=42、从头训练；原 GPU 2/3 上阶段 9 训练保留作对照。
删除 gait_level 课程与日志，保留原距离地形课程；六项固定权重从第一步生效。
torques -5e-4、action_rate -0.01、action_smoothness -0.002、contact_forces -1.5e-4、joint_mirror -0.03、rotation_gait_status +2。
其余保持 entropy 0.008、log_std、学习率 3e-4、姿态权重 -1.5/-1/-0.5、水平姿态 -10；slack/height 与三个额外转向奖励关闭。
视频固定凹台阶区域，env_index=7168，每 500 轮录制 1000 步；action、KP/KD 和观测维度保持不变。
已复用 8 环境 32 步实际仿真验证，并核对验证源码一致；110 个代码/资产文件校验一致。生产训练核验时间 2026-09-13T11:51:03.739526+08:00。

| 网络 | GPU | 核验时迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 0 | 12 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/5dgbozty) |
| dreamwaq | 1 | 10 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/z877eb5p) |

保存的 env 配置确认无 gait_level 课程项、六项函数不再使用带倍率包装；日志无 Curriculum/gait_level，初始检查点均为有限值。

阶段 11 仅调整腿姿态放宽判据；已替换阶段 10 的 DreamWaQ，MLP 保留原配置继续运行。

## 阶段 11 DreamWaQ 训练记录（2026-09-13）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-local-terrain-20260913`。
仅替换 GPU 1 上的 DreamWaQ；8192 环境、10000 轮、seed=42、从头训练。MLP 及其他训练不变。
thigh/calf 使用局部 5%–95% 高度跨度门控：机身支撑区域 0.6×0.5 m，加命令方向 0.5 m 预瞄；至少 10 条有效射线，高差 >0.06 m 时乘 0.1。
保留 entropy 0.008、log_std、学习率 3e-4、25 项活动奖励、无 gait_level、原 action/KP/KD 和观测；高度/slack/三个额外转向奖励继续关闭。
录像保持固定凹台阶视角、env_index=7168，每 500 轮录制 1000 步。
65 项相关测试及真实台阶 RayCaster 验证已复用，且核对验证源码一致；110 个代码/资产文件校验一致。生产核验时间 2026-09-13T16:45:43.273027+08:00，当时 5 / 10000 轮，初始检查点为有限值。

W&B：[运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/wu0o1h5k)。

旧 DreamWaQ 停止前 3091 轮，保留 `model_3000.pt` 及全部日志。

阶段 12 使用 RotationGaitSymmetryUnscaled，复用原 5 秒接触历史、启动渐入及逐环境 reset；不计算 gait_level。
仅恢复占空比，活动奖励共 26 项；滑动、腾空、Velocity-slack 与高度奖励仍关闭。
此次启动同时使用 hip/thigh/calf 奖励新名称；对应计算和权重未改变。

## 阶段 12 DreamWaQ 训练记录（2026-09-13）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-duty-20260913`。
仅恢复 rotation_gait_symmetry 固定 +15，使用 RotationGaitSymmetryUnscaled；5 秒窗口、目标接触比例 0.5、核宽度 0.2，保留逐环境历史重置和启动渐入。
不恢复 gait_level，滑动、腾空、Velocity-slack 和高度奖励继续关闭，共 26 项活动奖励。
保留局部地形预瞄放宽、对角支撑 +2、全部关节力矩 -5e-4、水平姿态 -10；腿姿态日志统一为 hip/thigh/calf，权重 -1.5/-1/-0.5。
GPU 1、8192 环境、10000 轮、seed=42、从头训练；log_std、entropy 0.008、学习率 3e-4。固定台阶镜头录像每 500 轮、1000 步。
66 项相关测试通过，110 个运行代码/资产文件校验一致；生产核验时间 2026-09-13T22:06:44.917958+08:00，当时 7 / 10000 轮，初始检查点为有限值。

W&B：[运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/qta19gjy)。

旧 DreamWaQ 停止前 3160 轮，保留 `model_3000.pt` 和全部日志；其他训练未停止。

阶段 13：恢复 2026-09-13_16-32-59_rough_local_terrain_dreamwaq_video 的奖励配置，保留 hip/thigh/calf 新名称。
用户改为四卡从头训练：每卡 8192 环境，总计 32768，5000 次迭代；不加载 checkpoint 或优化器状态。
使用单机 rank 0–3，seed 42/43/44/45；共享策略和 VAE 梯度。PPO/action/KP/KD 保持原配置。
仅 rank 0 保存 checkpoint、配置和台阶录像；各 rank 使用同一运行目录。
修复 VAE 单个 rank 没有有效转移时的集合同步，避免部分 rank 跳过同步导致死锁。
录像完成后按当前迭代上传，避免视频文件名的历史步数造成 W&B 丢弃上传。

## 阶段 13 四卡 DreamWaQ 训练记录（2026-09-14）

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-four-gpu-20260914`。
运行：`2026-09-14_10-28-27_rough_local_terrain_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192 环境，总计 32768；从头训练 5000 次；seed 为 42/43/44/45。
每次 24 步，共 786432 条转移。仅 rank 0 保存模型、配置和固定台阶录像（env_index=7168）。
录像每 500 次录制 1000 步，1280×720、50 fps。

36 项相关测试通过；四卡无录像 2 次更新、带录像 3 次更新验证通过，四卡策略及 VAE 参数一致。
测试录像已用 OpenCV 成功解码。110 个运行源码/资产校验一致。
2026-09-14 10:41:18 核验正式训练已到 8/5000，初始 checkpoint 参数有限，PPO/VAE 优化器步数均为 20。
已确认从头训练；保存的 25 项奖励（映射 hip/thigh/calf 新名称）、动作、机器人、地形、事件、终止、课程及 PPO/policy 与阶段 11 一致。
旧占空比训练及本次替换的单卡恢复任务已停止，历史模型和日志保留。

W&B：[四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/9f3lsvv1)。
核验记录位于服务器快照 `run_scripts/verification.json`、`run_scripts/smoke_verification.json` 和 `run_scripts/source_reference.json`。

## 阶段 14：stay_alive 权重减半（2026-09-14）

仅将 Rough 的 `stay_alive` 权重从 5.0 改为 2.5，保留 `stay_alive_startup` 原有逻辑。
110 个运行源码/资产比对确认仅 rough_env_cfg.py 的这一项系数变化；19 项相关测试通过。
保留上一轮的奖励、课程、action、KP/KD、PPO 和网络设置。

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-half-alive-20260914`。
运行：`2026-09-14_20-10-51_rough_half_alive_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192 环境，总计 32768；从头训练 5000 次，seed 42/43/44/45。
仅 rank 0 录制固定凹台阶镜头，每 500 次录制 1000 步。
2026-09-14 20:24:05 核验已到 7/5000；初始 checkpoint 参数有限，PPO/VAE 优化器步数均为 20。
保存的配置与上一轮比对确认仅 stay_alive 权重不同。

W&B：[半权重 stay_alive 四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/my5luq5j)。

本轮自碰撞继续关闭。另行只读核对发现：GO2W_CFG 的 enabled_self_collisions=False，USD 的 /go2w_description/base 上 physxArticulation:enabledSelfCollisions 也为 False；四个 wheel collider 均启用，未发现轮子上的 filteredPairs。
因此既有 Rough 训练与台阶测试未约束机器人内部的碰撞；仅凭视频尚不能确认前后轮是否几何穿插。本轮未改自碰撞设置。

## 阶段 15：开启 Rough 自碰撞（2026-09-14）

在 Go2WRoughEnvCfg.__post_init__ 中设置 enabled_self_collisions=True，训练和继承的 Play 配置均生效。
保留 stay_alive=2.5 及全部其他奖励、课程、action、KP/KD、PPO 和网络设置。

19 项相关测试通过；真实仿真逐一核对 16 个 articulation root 的自碰撞属性均为 True、64 个轮子碰撞体启用、无轮子 filteredPairs。
16 环境两次 PPO 更新通过，状态和模型参数有限；110 个运行源码/资产文件校验一致。
旧 half-alive 任务在约 173 次迭代停止，仅停止该任务的进程，保留所有已有模型和日志。

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-self-collision-20260914`。
运行：`2026-09-14_20-39-13_rough_self_collision_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192 环境，总计 32768；从头训练 5000 次，seed 42/43/44/45。
仅 rank 0 录制固定凹台阶视角，每 500 次录制 1000 步。
核验时间 2026-09-14T20:53:33.668337+08:00，已到 10/5000；初始 checkpoint 参数有限，保存配置与上一轮仅自碰撞开关不同。

W&B：[开启自碰撞的四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/newxmvb6)。
核验文件：新快照 `run_scripts/self_collision_native.json`、`run_scripts/verification.json` 和 `run_scripts/source_reference.json`。

## 阶段 16：移除 stay_alive，恢复小权重 Velocity-slack（2026-09-15）

stay_alive=None；velocity_slack 权重 0.25，为原先 1.0 的四分之一。
保留原函数：命令方向上的实际 body-x 速度在命令绝对值的 0.3–1.5 倍内时奖励；零前向命令、停止、反向运动不奖励；反向命令使用符号对齐区间。
自碰撞继续开启，原线/角速度精确跟踪、腿姿态、action、KP/KD、地形课程和 PPO 均不变。
30 项相关测试通过；16 环境两次 PPO 更新及自碰撞属性检查通过，状态和模型参数有限。
110 个运行源码/资产核验一致，保存配置比对确认仅上述两项奖励变化。

快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-slack-low-20260915`。
运行：`2026-09-15_10-48-20_rough_no_alive_slack025_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192，共 32768 环境；从头 5000 次，seed 42/43/44/45。
仅 rank 0 固定凹台阶镜头，每 500 次录制 1000 步。
核验时间 2026-09-15T11:01:29.041709+08:00，已到 8/5000；初始 checkpoint 参数有限，确认没有加载旧模型。

W&B：[无 stay_alive、小权重 slack 四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/gwy30flg)。
证据：快照 `run_scripts/slack_low_native.json`、`run_scripts/verification.json`、`run_scripts/source_reference.json`。

## 阶段 17：orientation 减半（2026-09-15）

仅将 flat_orientation_l2 权重从 -10 改为 -5；roll/pitch 仍使用原始水平姿态计算，没有拆分奖励。
保留 stay_alive=None、velocity_slack=0.25（0.3–1.5 倍区间）、自碰撞及全部其他奖励、课程、action、KP/KD、PPO、网络设置。
15 项相关配置测试通过；110 个运行源码/资产文件核验一致，只有 Rough 配置中的该系数变化。
上一轮已正常完成，原模型和日志保留。

快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-ori5-20260915`。
运行：`2026-09-15_21-40-58_rough_ori5_slack025_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192，共 32768 环境；从头 5000 次，seed 42/43/44/45。
仅 rank 0 固定凹台阶镜头，每 500 次录制 1000 步。
核验时间 2026-09-15T21:55:48.478724+08:00，已到 20/5000；初始 checkpoint 参数有限，保存配置与上一轮仅 orientation 权重不同。

W&B：[orientation=-5 四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/lobcu91l)。
证据：快照 `run_scripts/verification.json`、`run_scripts/source_reference.json`。

## 阶段 18：orientation=-1（2026-09-16）

仅将 flat_orientation_l2 权重从 -5 改为 -1；原函数同时约束 roll/pitch，未拆分。
保留 stay_alive=None、velocity_slack=0.25（0.3–1.5 倍区间）、自碰撞和全部其他奖励、课程、action、KP/KD、PPO、网络设置。
15 项相关测试通过；110 个运行源码/资产核验一致，只有 Rough 配置中的该系数变化。
上一轮已正常结束，所有原模型和日志保留。

快照：`/home/sshang/omnibot_data/workspaces/OmniBot-go2w-ori1-20260916`。
运行：`2026-09-16_09-13-06_rough_ori1_slack025_4gpu_dreamwaq_video`。
GPU 0–3，每卡 8192，共 32768 环境；从头 5000 次，seed 42/43/44/45。
仅 rank 0 固定凹台阶镜头，每 500 次录制 1000 步。
核验时间 2026-09-16T09:26:06.409577+08:00，已到 8/5000；初始 checkpoint 参数有限，保存配置与上一轮仅 orientation 权重不同。

W&B：[orientation=-1 四卡 DreamWaQ](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/fpescb6c)。
证据：快照 `run_scripts/verification.json`、`run_scripts/source_reference.json`。
