# Go2W 高台网络：Rough MLP 续训最终版

对应任务 `OmniBot-Go2W-HighPlatform-MLP-v0`。从 Rough `model_4999.pt` 续训 5000 次，
8192 环境、服务器单卡、seed 42；最终 checkpoint 内部迭代编号为 9998，训练正常退出。

- [技能配置、开发改动与训练命令](../../../../docs/go2w_highplatform_overview.md)
- [最终模型固定高度评测](../../../../artifacts/go2w_highplatform/20260923_final_9998/README.md)
- [W&B](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/w7lto3un)

## 交付文件

| 文件 | 内容 |
|---|---|
| `model_9998.pt` | 原始完整 checkpoint，包含 actor、critic、optimizer，可回放和续训 |
| `policy.pt` | 确定性 actor 的 TorchScript，不包含 critic 或探索噪声 |
| `policy.onnx` | float32 `obs [1,53]` → `actions [1,16]`，固定 batch=1 |
| `params/agent.yaml` | 原始 PPO 配置，按服务器字节保留 |
| `params/env.yaml.gz` | 原始环境配置的无损 gzip 存档，解压后哈希与服务器一致 |
| `params/source_files_sha256.json` | 训练时 131 个源码／资产文件的哈希 |
| `training_provenance.json` | 分支、基线、训练 run、恢复权重哈希和 W&B 来源 |
| `export_manifest.json` | 文件校验值、推理接口、导出器版本和数值验证 |
| `export_policy.py` | CPU 严格加载、导出和校验脚本，复用 `play.py` 的导出器 |

网络由 Git LFS 管理。若克隆后得到文本指针，请在仓库根目录运行：

```sh
git lfs pull --include="deploy/policy/go2w/highplatform_rough_mlp_20260922/*"
```

`env.yaml.gz` 含 Isaac 保存的 Python 类型标签，只作原始参数追溯。
可用 `gzip -dc params/env.yaml.gz` 查看；不要对不可信 YAML 使用可执行对象加载器。
最终 checkpoint 的 SHA-256 为
`85a13a13d35bd2ec7ff1eb5a1a25aaa3640668a5918bdac4072774a7b097c46a`。

## 推理契约

Actor：`53 → 512 → 256 → 128 → 16`，隐藏层 ELU，无 RNN、历史输入或观测归一化。
Critic 为 307 维，仅训练使用；actor 不需要高度扫描、地形高度或机身线速度输入。
输入保持 Rough 格式，不包含 gait、jump 或 height command。

| 输入切片（左闭右开） | 内容 |
|---|---|
| `[0:3]` | 机身角速度，缩放 0.2 |
| `[3:6]` | 机身坐标系下的重力投影 |
| `[6:9]` | command `[vx, vy, yaw_rate]`，原始单位，不再缩放 |
| `[9:21]` | 12 个腿关节相对资产默认角的位置 |
| `[21:37]` | 16 个关节相对默认速度（默认均为 0），缩放 0.05 |
| `[37:53]` | 上一步 16 维原始 action |

各项按训练规则先裁剪到 `[-100,100]` 再缩放；推理不加入训练噪声。
注意：此任务 command **不是**步态网络的 `[2,2,0.25,...]` 缩放格式。
heading 反馈由控制端产生，网络接收反馈得到的 yaw_rate，不直接接收目标 heading。

控制周期 0.02s。前 12 维动作：`q_default + 0.25 × action`，默认 hip=0、thigh=0.8、calf=-1.5；
后 4 维：轮速 `10 × action`。没有新增 action 或 target 限位裁剪。
腿部 KP=40、KD=1；轮子 KP=0、KD=0.5。
训练顺序按关节类别分组，每组均为 FL、FR、RL、RR：hip → thigh → calf → foot，
**不是 SDK 的逐腿顺序**。实际运行时解析结果见
[runtime_contract.json](../../../../artifacts/go2w_highplatform/20260923_final_9998/runtime_contract.json)。

本目录仅交付网络与契约，不修改任何当前部署配置或默认策略。
即使同为 53 维，部署仍须核对顺序、缩放、默认角、PD、控制周期和机器人限位。
本次未新增 MuJoCo 或实机验证，不应直接据此宣称可安全实机爬 80cm。

## 校验和回放

Checkpoint 和原始参数已与服务器做 SHA-256 核对；actor／critic 完整严格加载，无参数丢弃。
385 组确定性 CPU 输入（零输入和标准差 0.1／1／10 的随机输入）通过数值检查：
TorchScript 最大绝对误差 0，ONNX 最大绝对误差 `1.686e-4`，标准差 1 的输入最大误差 `2.385e-5`。
高幅值压力输入触发一次超过旧 `1e-4` 绝对阈值的 FP32 累加误差，故记录实际阈值
`rtol=1e-5, atol=2e-4`，未修改权重。该验证不能代替动力学测试。

最终 native checkpoint 已在 Isaac Lab 完成 256 次固定高台测试：245 次通过；
50／60／70／80cm 分别为 64/64、59/64、59/64、63/64，详见评测报告。
这是 native PyTorch 回放，ONNX 的验证范围是 CPU 数值一致性。

```sh
# 仓库根目录；只读校验已提交的网络与 manifest。
CUDA_VISIBLE_DEVICES='' python deploy/policy/go2w/highplatform_rough_mlp_20260922/export_policy.py --verify-only
```

去掉 `--verify-only` 会重新生成推理网络和 manifest。指定的 `robot_lab` 环境本机不存在，
此次使用已有 `env_isaaclab` 完成 CPU 验证和服务器回放，未安装或升级依赖。
