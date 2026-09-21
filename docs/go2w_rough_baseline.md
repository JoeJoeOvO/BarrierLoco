# Go2W Rough 训练基线

2026-09-21 确认：后续 Go2W Rough 训练以本配置为基线。新的调参实验从 main 新建分支，
显式记录相对基线的变化；本基线不包含外环 PI 或其他部署端积分补偿。

## 配置入口

- 环境与奖励：`locolab/source/locomotion/tasks/Go2W/rough_env_cfg.py`。
- 公共奖励：`locolab/source/locomotion/mdp/rewards.py`。
- 选择性迁移奖励：`locolab/source/locomotion/mdp/m20/rewards.py`。
- MLP、RNN、DreamWaQ 共用上述 Rough 环境；本次完整验证使用
  `OmniBot-Go2W-Rough-DreamWaQ-v0`。
- 历史 `m20_env_cfg.py` 及带 `gait_level` 的旧配方不作为活动训练入口。

## 已验证参数

| 项目 | 基线 |
| --- | --- |
| 运行规模 | 4 GPU，每卡 8192 环境，共 32768；从头训练 5000 次 |
| Seed | 基础 seed=42，各 rank 为 42、43、44、45 |
| PPO | 初始学习率 0.0003，保留 adaptive 调度；entropy 系数 0.008；log_std 参数化 |
| 动作 | 腿位置动作 scale=0.25，轮速度动作 scale=10；不新增动作裁剪 |
| 名义执行器增益 | 腿 KP=40、KD=1；轮 KP=0、KD=0.5；保留训练增益随机化 |
| 站姿 | hip=0、thigh=0.8、calf=-1.5 rad；root 初始高度 0.45 m |
| 自碰撞 | 开启 |
| 速度跟踪 | 线速度 3.0，角速度 1.5；保留原精跟踪项；角速度误差平方惩罚 -1 |
| Velocity slack | +0.25；前向实际速度处于命令的 0.3–1.5 倍且方向一致 |
| 姿态与存活 | orientation=-1；stay_alive、base_height 关闭 |
| 腿姿态 | hip/thigh/calf=-1.5/-1/-0.5；纯侧移 hip 减半；保留局部地形及转向豁免 |
| 静止 | stand_still=-1；平移和转向命令均小于 0.06 才生效 |
| 关节镜像 | -0.03；纯侧移关闭 |
| 转向支撑 | +2；同组摆动轮局部离地高度一致性，差值尺度 0.05 m，无绝对抬高目标 |
| 转向滑动 | -2 |
| 转向占空比 | +5；逐轮目标 0.5、5 秒窗口；同组腾空时间差尺度 0.1 s |
| 连续悬空 | -10 × 各轮超过 0.3 s 的超时量之和；沿用低速转向命令门控 |
| gait_level | 活动课程及奖励均不使用 |
| 录像 | rank 0，每 500 次录制 1000 步；锁定 pyramid_stairs_inv 地块 |

精确门控和参数以代码为准。环境数量和迭代数通过 CLI 显式指定，避免改变其他任务的 runner 默认值。

## 复现命令

在已配置好的 Isaac Lab Python 环境中，从 `locolab/` 执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/locomotion/train.py \
  --distributed --task OmniBot-Go2W-Rough-DreamWaQ-v0 \
  --num_envs 8192 --max_iterations 5000 --seed 42 \
  --run_name rough_baseline_4gpu_dreamwaq_video \
  --headless --video --video_interval 500 --video_length 1000 \
  --video_terrain pyramid_stairs_inv
```

此处 `--num_envs` 是每卡数量。服务器验证环境为 `env_isaaclab`；不需要安装或重建训练栈。

## 训练与评估依据

运行：`2026-09-21_10-45-39_rough_turn_consistency_4gpu_dreamwaq_video`。
[W&B 记录](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/6lk2xa0m)。
训练正常退出，最终 checkpoint 为 `model_4999.pt`，SHA256：
`8a5d1a924a5c90010e56de8228ff84d1b4276e0f50ca40627a64db9de7c8119e`。
合入前核对 110 个训练源码文件与本轮服务器快照一致。

- 固定台阶：8/12/16/20/23 cm，7 级、踏面 0.30 m，速度 0.6/1.0 m/s，
  seed 42/43，每组 16 个初始位姿；320/320 台阶、64/64 平地通过。
- 导航命令 `vx=0.30、yaw=-0.20`：平均实际 `vx=0.344、yaw=-0.191`。
- 仍有已知边界：全零后持续缓慢前移；纯侧移存在明显偏航；部分转向工况的占空比及位置保持不理想。
  后续导航跟踪改进在此基线上单独评估，不把固定台阶通过率解释为完整导航验收。
- 以上为原生 Isaac Lab 确定性评估，关闭观察噪声和域随机化，保留自碰撞；不代表实机通过率。

完整评估产物保存在工作区
`artifacts/go2w/2026-09-21_10-45-39_rough_turn_consistency_4gpu_dreamwaq_video/evaluation/model_4999/`，
包括报告、原始轨迹、训练标量与视频；这些产物未随配置提交。
