# Go2W 平滑 TO 跳跃网络

任务 `OmniBot-Go2W-Jump-MLP-v0`；两阶段各 5000 次 PPO 更新，4096 环境、单卡、seed=42。
第一阶段随机初始化，第二阶段加载本目录第一阶段完整 checkpoint。最终编号为 9998，
是 runner 的恢复索引规则所致，实际共完成 10000 次更新。

- [技能与训练说明](../../../../docs/go2w_jump.md)
- [最终策略评估](../../../../artifacts/go2w_jump/20261009_final_9998/README.md)
- [最终 W&B run](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/tyqvcpfm)

## 文件

| 文件 | 用途 |
| --- | --- |
| `model_4999.pt` | 第一阶段完整 checkpoint，包含 actor、critic 和 optimizer |
| `model_9998.pt` | 第二阶段最终完整 checkpoint，用于回放或续训 |
| `policy.pt` | 最终确定性 actor 的 TorchScript |
| `policy.onnx` | 最终确定性 actor 的 ONNX，float32 `[1,54] → [1,16]` |
| `params/agent.yaml`、`params/env.yaml.gz` | 第二阶段原始参数，无损保留 |
| `params/skill/` | 第一阶段原始参数 |
| `params/source_files_sha256.json` | 正式训练时 105 个源码和参考文件的哈希 |
| `training_provenance.json` | 初始化方式、两阶段 run、参考、网络校验和及 W&B 来源 |
| `export_policy.py`、`export_manifest.json` | CPU 导出/验证工具及结果 |

网络由目录级 Git LFS 规则管理，不扩大整个仓库的 LFS 范围：

```sh
git lfs pull --include="deploy/policy/go2w/jump_smooth_mlp_20261009/*"
CUDA_VISIBLE_DEVICES='' python deploy/policy/go2w/jump_smooth_mlp_20261009/export_policy.py --verify-only
```

去掉 `--verify-only` 可重新导出。使用现有 `env_isaaclab` 完成 CPU 验证；
指定的 `robot_lab` 在本机不可用，未安装或升级训练依赖。
环境 YAML 含 Isaac 的 Python 类型标签，仅用于参数追溯，使用安全的 BaseLoader 阅读；
不要对不可信 YAML 使用可执行对象加载器。

## Actor 输入与动作

MLP：54 → 512 → 256 → 128 → 16，ELU；无 RNN、观测归一化或历史堆叠。
critic 的 309 维输入及跳跃相位只在训练时使用。

| 输入切片（左闭右开） | 含义 |
| --- | --- |
| `[0:3]` | 机身角速度，缩放 0.2 |
| `[3:6]` | 机身坐标系下的重力投影 |
| `[6:10]` | `vx, vy, yaw_rate, jump_active`，不额外缩放 |
| `[10:22]` | 12 个腿关节相对默认角的位置 |
| `[22:38]` | 16 个关节相对默认速度，缩放 0.05 |
| `[38:54]` | 上一步 16 维原始 action |

各项按训练规则先裁剪到 [-100,100] 再缩放，推理不加探索或观测噪声。
关节按 hip → thigh → calf → foot 分组，每组均为 FL、FR、RL、RR，
不是 SDK 的逐腿顺序。默认 hip=0、thigh=0.8、calf=-1.5。

控制周期 0.02 s：前 12 个动作对应 `q_default + 0.25 × action`，
后 4 个动作对应 `10 × action` 轮速。腿 KP/KD=40/1，轮 KP/KD=0/0.5；
未增加 action target 裁剪。

## 必须由调用端实现的时序

导出的网络仅为 actor，不包含指令生成器、计时器或物理控制器。
外部 jump 的上升沿在满足四轮接触、姿态/速度门限时启动固定时钟，
`jump_active` 保持 1 共 2.2 s 后清零；持续输入 1 不应不断重新触发。
参考起跳/顶点/首次触地为 0.38/0.62/0.88 s，参考结束 1.8 s，加 0.4 s 恢复。

最终网络使用 speed 阶段语义：触发时锁定 yaw，持续 2.2 s。
这段时间内需把锁定方向下的 vx/vy 转为当前机身 yaw 坐标系，填入原来的两列速度观测；
不要增加输入维度或把期望 heading 直接填进 yaw_rate。
实现见 `BinaryJumpCommand.heading_velocity_command` 和 `velocity_and_jump_commands`。
训练速度范围为 vx=[0.2,1.5]，vy/yaw_rate=0。

## 验证边界

完整严格加载 actor/critic，无参数丢弃；385 组 CPU 输入通过 TorchScript/ONNX 等价检查，
实际误差及阈值见 `export_manifest.json`。包括标准差 10 的高幅值压力输入；
绝对误差不能脱离共同使用的 rtol/atol 判断。该测试不代替动力学验证。

原生 checkpoint 已完成四档速度地面起步回放，4/4 跳起并稳定落地。
仍有速度偏快、5–9° 腾空偏航和最高 37.5 rad/s 轮速尖峰。
训练任务沿用腿部 23.5 Nm / 30 rad/s 的历史仿真限制；共享部署资产的小腿采用
35 Nm / 20 rad/s，两者并非相同动力学契约。

本目录不改默认策略、不接入新实机控制器；不能因为导出成功就直接用于硬件或宣称能跨障碍。
