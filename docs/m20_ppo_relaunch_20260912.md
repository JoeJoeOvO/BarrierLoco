# M20 配方训练参数调整与重新提交（2026-09-12）

三个独立 M20 runner 的 PPO 初始学习率由 3e-4 调整为 1e-3，熵系数由 0.01 调整为 0.003，动作噪声统一使用每维 log_std。
原 Go2W runner、网络结构、归一化、KP/KD、动作 scale=0.25/10 及动作裁剪保持原值。DreamWaQ VAE 和 RNN 辅助 decoder 学习率仍为 3e-4。

新任务从头训练，不加载旧模型或优化器；每组 4096 环境、10000 次迭代，seed=42。录像每 500 次迭代触发，长度 1000 步（约 20 秒），720p/50 fps。
快照同时包含此前已验证的 reset 几何余量、calf 速度限幅 20.07 rad/s 和质量/惯量耦合修正，因此与旧任务的比较不只涉及 PPO 参数。旧 DreamWaQ、RNN 训练保留运行。

服务器快照：`/home/sshang/omnibot_data/workspaces/OmniBot-m20-ppo-aligned-20260912`。
任务控制、完整配置校验与提交记录在快照的 `run_scripts/` 下。

检查时间：2026-09-12T02:12:18.429633+08:00。以下迭代数为该时刻的启动确认，不是训练完成状态。

| 任务 | GPU | 已进入迭代 | W&B |
| --- | ---: | ---: | --- |
| mlp | 0 | 13 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/3d0guakl) |
| dreamwaq | 3 | 12 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/jayyp6b0) |
| rnn | 4 | 10 / 10000 | [运行](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/umjxs78w) |

## 验证

- 24 项配置、原网络、实验目录和 reset 安全检查通过。
- 修改的两个 Python 文件通过 Black API、isort、flake8；git diff --check 通过。Black CLI 进程池检查曾挂起，已终止并用直接 API 完成校验。
- 新快照 110 个训练源码及 Go2W 资产文件与本地 SHA-256 一致。
- 三种网络各完成 16 环境、7 次更新的 smoke，退出码 0，输出 TRAINING_SMOKE_OK；log_std 参数和模型权重均有限。
- 短训练录像可解码且抽检画面非黑屏。短测试每 24 步触发，产生约 0.48 秒片段；正式任务仍为每 500 次迭代录制 1000 步。
- 三个正式任务均进入 PPO 更新，保存参数确认为 learning_rate=0.001、entropy_coef=0.003、noise_std_type=log、resume=false；model_0.pt 权重有限。
- 正式任务首次计划录像在第 500 次迭代触发；本记录未声称该视频已生成。
