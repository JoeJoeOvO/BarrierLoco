# Go2W Jump 最终评估

对应 [网络包](../../../deploy/policy/go2w/jump_smooth_mlp_20261009/README.md) 中的 `model_9998.pt`，
SHA256 为 `64dc3fe4396e79afd0bc3e8a2496f6cc262ffa78466d409ff7c1e454d74dc550`。
两阶段各 5000 次更新均正常退出，未在评估中更新网络。

## 确定性地面起步

单环境平地、普通地面 reset、先行进再接受 grounded jump trigger；
关闭训练动作噪声和域随机化，开启与训练相同的自碰撞与 2.2 s 方向锁定。
没有随机空中重置。每档测试一次，不能据此推算随机扰动成功率。

| vx 指令 (m/s) | 起跳前实际 vx (m/s) | 机身抬升 (cm) | 最低轮底净空峰值 (cm) | 腾空前移 (m) |
| --- | --- | --- | --- | --- |
| 0.35 | 0.614 | 42.0 | 50.1 | 0.48 |
| 0.5 | 0.757 | 41.7 | 53.0 | 0.53 |
| 1.0 | 1.359 | 38.6 | 48.7 | 0.82 |
| 1.5 | 1.817 | 37.8 | 47.7 | 1.04 |

四次均稳定落地，无提前终止；连续腾空约 0.56–0.58 s。
偏航约 5–9°、腾空横移约 11–15 cm；记录到的轮速最高 37.5 rad/s，
超过配置的 30 rad/s。轮速每 20 ms 采样，可能漏掉更快的峰值。
平地净空与前移距离不等于已验证的障碍高度或沟隙宽度。

## 带探索噪声的补测

相同地面起步，每档 5 次；只在接受 jump 后采样策略动作。
20/20 次抬升超过 18 cm，但 0/20 满足连续 0.1 s 四轮稳定着地判据。
训练日志的平均高度混有非跳跃环境和参考相位重置；日志成功率 0
不能直接解释为确定性策略不会跳跃。该补测也不是域随机化或实机测试。

- `results.json`：最终策略四档原始结果。
- `noise_results.json`：20 次探索噪声测试。
- `stage1_results.json`：第一阶段两个低速测试。
- `summary.json`：汇总及额外轮速、速度跟踪诊断。

报告中的绝对 checkpoint 路径是原服务器的来源记录；本仓库网络包提供相同字节的权重。
完整录像/原始状态留在主工作区 `artifacts/go2w/jump_smooth_training_20261009/final_review/`，
不把训练缓存和重复录像纳入 Git。训练记录见
[第一阶段](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/9n03qoo7) 和
[第二阶段](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/tyqvcpfm)。

## 复验

在已有 Isaac Lab 环境、仓库根目录运行：

```sh
python locolab/scripts/locomotion/evaluate_jump.py \
  --checkpoint deploy/policy/go2w/jump_smooth_mlp_20261009/model_9998.pt \
  --case rolling_vx035 rolling_vx05 rolling_vx10 rolling_vx15 \
  --heading_lock --follow_xy --video_dir /tmp/go2w-jump-review/video \
  --output /tmp/go2w-jump-review/results.json --device cuda:0 --headless
```

不录视频的噪声复验可加 `--stochastic --repeats 5` 并省略 `--video_dir`。
GPU 执行需按运行环境的资源授权进行；本次打包仅做 CPU 导出等价验证。
