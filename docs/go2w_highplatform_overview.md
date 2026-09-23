# Go2W 高台技能交付

任务为 `OmniBot-Go2W-HighPlatform-MLP-v0`，分支 `feature/go2w-highplatform-rough-mlp`。
在 main 的 Rough 基线 `3dda20b374a382cd01a7a7f7d45e3693767826d3` 上开发，
从最新 Rough MLP 的 `model_4999.pt` 继续训练，最终交付 `model_9998.pt`。
这是直线爬高台技能，不是步态、跳跃或多技能 command 任务。

- [最终网络与推理接口](../deploy/policy/go2w/highplatform_rough_mlp_20260922/README.md)
- [50／60／70／80cm 最终模型评测](../artifacts/go2w_highplatform/20260923_final_9998/README.md)
- [W&B 训练记录](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/w7lto3un)

## 开发内容与继承边界

`high_platform_env_cfg.py` 改为完整继承当前 `Go2WRoughRewardsCfg`：29 个 Rough 奖励项不改动，
仅加入 `feet_stumble=-0.1`。四轮任一轮接触力满足 `||Fxy|| > 5 × |Fz|` 时触发，严格大于阈值；
选择器只包含 `.*_foot`，不包含机身和腿。

移除旧高台配置独有的存活奖励、减半终止惩罚、旧关节默认角奖励、髋关节对称辅助函数，
以及平台专用轮阻随机化；重复的速度跟踪、关节限位、碰撞等覆盖也移除。
现在终止惩罚继承 Rough 的 `-200`，静止惩罚继承 Rough 的 `-1`，没有额外 `stay_alive`。
共享的轮阻实现仍被其他任务使用，因此保留，不作为废弃代码删除。

观测、动作、PD、机器人资产、自碰撞、终止条件、训练噪声和随机化均沿用 Rough；
有意保留的高台差异只有地形、直线 command、10s episode 和朝向固定的出生姿态。
训练器继承 `Go2WRoughMLPPPORunnerCfg`，只设置高台日志名、Rough 恢复来源和 5000 次增量迭代。
学习率 `3e-4`、entropy `0.008`、log 标准差参数化不变。

本次交付另加入独立固定高度评测脚本、可在 CPU 运行的成功判据及回归测试。
地形生成器、任务注册、六高台 Play 场景已经在基线上存在，保留复用，没有复制另一套。

## 地形、课程和 command

| 项目 | 最终配置 |
|---|---|
| 并行环境 / episode | 8192 / 10s |
| 单块地形 | 8m × 8m，36 行 × 20 列 |
| 高台高度 | 10–80cm，36 级，每级 2cm，`h=0.10+0.02×level` |
| 高台长 × 宽 | 沿前进方向 1.5m × 横向 2m；宽度占地形 25% |
| 起点 / 高台前沿 | 地形内 x=1.0m / x=2.5m，名义接近距离 1.5m |
| 初始课程 | 随机 level 0–15，即 10–40cm |
| 训练 reset | x、y 各 ±0.5m，yaw=0；其余继承 Rough |
| 线速度指令 | x 为 0.5–1.0m/s，y=0，不使用速度课程 |
| 朝向指令 | heading=true，全部环境世界 yaw 目标 0，增益 2 |
| 实际 yaw 速度指令 | 按朝向误差反馈生成，范围 `[-π,π]`，不是随机转弯 |
| 重采样 / 静止环境 | 20s / 0% |

课程保持原有距离规则：离出生原点的水平距离超过 4m 晋级；不足
`0.5 × 指令线速度模长 × episode 时长` 时降级（晋级优先）。超过第 35 级后随机回到 0–35 级。
这个规则本身不验证是否爬上台面，因此平均课程等级不能当作最大爬高能力；
两侧有绕行空间，最终能力使用独立的防绕行固定高度评测报告。

同一任务的 Play 配置使用 24m × 8m 的纵向场地（`play.py` 自动读取 `play_env_cfg_entry_point`），
依次放置 30／40／50／60／70／80cm 高台；每台长 1.5m、宽 2m，台与台之间平地间隔 1.5m。
单环境、30s、1m/s、固定朝向，关闭训练噪声和随机化。

## 最终训练和结果

训练 run：`2026-09-22_10-52-39_highplatform_rough_mlp_resume4999_stumble5_8192_1gpu_5k_video`。
来源为 `2026-09-21_21-39-38_rough_main_4gpu_mlp_scratch_5k_video/model_4999.pt`，
恢复 actor、critic 和 optimizer，服务器单卡 GPU 0、seed 42，额外执行 5000 次更新。
恢复后的循环标签是 4999–9998；最终文件编号不是“只训练了 4999 次”的含义。
训练退出码 0，W&B 同步完成，最终平均 terrain level 为 23.29832。

最终模型在关闭噪声与随机化、保留自碰撞的 256 次固定高台测试中通过 245 次（95.7%）：

| 高度 | 0.6m/s | 1.0m/s | 合计 |
|---|---:|---:|---:|
| 50cm | 32/32 | 32/32 | 64/64（100%） |
| 60cm | 32/32 | 27/32 | 59/64（92.2%） |
| 70cm | 32/32 | 27/32 | 59/64（92.2%） |
| 80cm | 32/32 | 31/32 | 63/64（98.4%） |

成功要求四轮先稳定上台，再越过平台并稳定落地；绕行、翻倒后的自动 reset 不计成功。
每组仅两个种子的小范围初始姿态扰动；不同高度成功率非单调不意味着 80cm 比 60cm 普遍容易。
以上不代表带全部训练随机化、连续六高台或实机成功率，也未验证 MuJoCo sim2sim。

## 复现入口

在已安装的 Isaac Lab Python 环境中执行；不需要改动或重装训练栈。
原始参数、源码哈希和恢复来源均在网络目录保存。
训练快照中除本次补充任务入口的 `locolab/README.md` 外，其余 130 个原有文件哈希均保持一致；
源码哈希清单保留训练时原值，不伪装成整理后的快照。

```sh
cd locolab
python scripts/locomotion/train.py --task OmniBot-Go2W-HighPlatform-MLP-v0 \
  --num_envs 8192 --max_iterations 5000 --seed 42 --device cuda:0 \
  --resume --resume_experiment_name omnibot_go2w_rough_mlp_v0 \
  --load_run 2026-09-21_21-39-38_rough_main_4gpu_mlp_scratch_5k_video \
  --checkpoint model_4999.pt --headless --logger wandb --log_project_name omnibot-go2w \
  --video --video_interval 500 --video_length 1000 --video_terrain high_platform
```

上面命令要求 Rough checkpoint 已位于对应 `logs/rsl_rl/omnibot_go2w_rough_mlp_v0/<run>/`，
并已配置自己的 W&B 身份。此次整理没有重启训练。

```sh
# 从仓库根目录运行固定高度评测；输出目录必须是新目录。
python locolab/scripts/locomotion/evaluate_highplatform.py \
  --checkpoint deploy/policy/go2w/highplatform_rough_mlp_20260922/model_9998.pt \
  --output artifacts/go2w_highplatform/new_evaluation --device cuda:0 --headless --video

# 六个连续高台的 Play 场景。
cd locolab
python scripts/locomotion/play.py --task OmniBot-Go2W-HighPlatform-MLP-v0 \
  --checkpoint ../deploy/policy/go2w/highplatform_rough_mlp_20260922/model_9998.pt \
  --num_envs 1 --video --video_duration_s 30
```

## 清理与验证

只清理本次 HighPlatform 专属目录的中间 checkpoint、旧版本评测、重复源码包、临时截图、
已同步 W&B 的重复媒体缓存；保留最终模型、原始参数、训练日志／事件和最终评测原始数据。
不清理 Rough 基座、不动其他技能或原步态工作区的未提交改动。

57 项针对性配置／奖励／课程／评测判据测试通过；网络严格加载并完成 385 组 CPU 等价性检查。
具体网络误差、原始配置 SHA-256 和验证边界见网络目录的 manifest 与说明。
