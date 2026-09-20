# Go2W 步态网络：2026-09-20 height-50 / reward-only IK

对应任务：`OmniBot-Go2W-Gait-Flat-MLP-v0`。这是此前九段步态/高度展示使用的最终模型，
来自 `2026-09-20_01-10-58_multi_skill_wtw_height50_rewardik_phase5_rollclock0_4096_remote_gpu4_5k_video`，
4096 环境、5000 次训练迭代，checkpoint 内部迭代编号为 4999。

- [训练与步态说明](../../../../docs/go2w_gait_overview.md)
- [对应回放结果](../../../../artifacts/go2w_multi_skill/20260920_height50_showcase/README.md)
- [W&B 训练记录](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/nhfgmyxj)

## 文件

| 文件 | 用途 |
|---|---|
| `model_4999.pt` | 原始 checkpoint：actor、critic、PPO optimizer、迭代编号，可用于 Isaac Lab 回放/续训 |
| `policy.onnx` | 确定性 actor，float32 输入 `obs [1,64]`，输出 `actions [1,16]` |
| `policy.pt` | 同一 actor 的 TorchScript 导出，不含探索噪声或 critic |
| `params/agent.yaml` | 原始 PPO/网络配置，字节级保留 |
| `params/env.yaml.gz` | 原始 Isaac 环境配置的无损压缩存档，解压后 SHA-256 与服务器一致 |
| `export_manifest.json` | 来源、网络接口、文件校验值、导出工具和数值验证结果 |
| `export_policy.py` | CPU 导出及校验脚本，复用与 `play.py` 相同的 Isaac Lab exporter |

网络文件由 Git LFS 管理。克隆后若文件内容是 LFS 指针，请在仓库根目录执行：

```sh
git lfs pull --include="deploy/policy/go2w/gait_wtw_height50_20260920/*"
```

原始配置保留历史任务/日志名，当前代码入口已更名为 Gait，观测、动作和网络结构没有改变。
环境配置含 Isaac 保存的 Python 类型标签，仅作追溯；不要对不可信 YAML 使用可执行对象加载器。
可用 `gzip -dc params/env.yaml.gz` 只读查看解压后的配置。

## 推理边界

MLP 为 `64 → 512 → 256 → 128 → 16`，隐藏层 ELU，无观测归一化、无历史输入或 RNN 状态。
输入必须已经按训练规则处理；模型不负责生成 command、相位、时钟或关节重排。

| 输入切片（左闭右开） | 内容 |
|---|---|
| `[0:3]` | 机身坐标系角速度，缩放 0.2 |
| `[3:6]` | 重力在机身坐标系中的投影 |
| `[6:16]` | 10 维 command，逐维缩放 `[2,2,0.25,2,1,1,1,1,1,1]` |
| `[16:28]` | 12 个腿关节相对固定资产默认角的位置 |
| `[28:44]` | 全部 16 个关节相对默认速度，缩放 0.05 |
| `[44:60]` | 上一时刻的原始 16 维动作 |
| `[60:64]` | FR、FL、RR、RL 的四个步态时钟；全支撑 rolling 时全为 0 |

除 command/时钟外，各观测项先裁剪到 `[-100,100]` 再进行上述缩放；推理时不添加训练观测噪声。
控制周期 0.02 s；前 12 维输出用于 `q_asset_default + 0.25 × action`，后 4 维用于 `10 × action` 轮速。
输出不额外裁剪。高度 IK 只用于训练奖励，不进入输入关节参考或输出动作零点。

关节排列以训练中 Isaac 资产和选择器的解析顺序为准，不能直接套用 SDK 的逐腿顺序。
本目录不包含新的部署控制器或运行配置；现有 sim2sim/实机配置尚未适配 64 维步态输入，
**不要直接覆盖普通 Flat/Rough 网络进行实机运行**。

## 验证和回放

源 checkpoint 和配置均与服务器 SHA-256 一致，actor/critic 完整严格加载，无参数丢弃。
使用与 `play.py` 相同的导出器；385 组确定性 CPU 输入验证通过，TorchScript 最大绝对误差 0，
ONNX 最大绝对误差约 `6.10e-5`。包含零输入和不同尺度的随机输入，不代表新的动态回放评估。
本次没有启动 GPU 仿真、训练或实机验证，也没有修改现有部署配置。

在可用的 Isaac Lab Python 环境中，从仓库根目录检查已提交文件：

```sh
CUDA_VISIBLE_DEVICES='' python deploy/policy/go2w/gait_wtw_height50_20260920/export_policy.py --verify-only
```

去掉 `--verify-only` 会重新导出两种推理网络并更新 manifest。校验时只按受限的 `weights_only=True`
加载训练 checkpoint。指定的 `robot_lab` 环境在本机不存在，本次使用已有 `env_isaaclab` 环境完成 CPU 验证，
未安装或升级依赖。

使用仓库现有回放入口（以下命令会启动仿真，本次未执行）：

```sh
cd locolab
python scripts/locomotion/play.py --task OmniBot-Go2W-Gait-Flat-MLP-v0 \
  --checkpoint ../deploy/policy/go2w/gait_wtw_height50_20260920/model_4999.pt --num_envs 1
```
