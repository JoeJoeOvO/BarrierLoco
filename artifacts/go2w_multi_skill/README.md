# Go2W 步态产物索引

代码与配置总览见 [开发整理](../../docs/go2w_gait_overview.md)。

当前任务已更名为 `OmniBot-Go2W-Gait-Flat-MLP-v0`。本目录和历史运行保留原名以追溯既有结果；
其中启动/回放脚本仅用于当时的服务器快照，不是当前分支的新训练入口。新运行方式见上述开发文档。

| 目录 | 用途 | 保留内容 |
|---|---|---|
| [20260920_height50_showcase](20260920_height50_showcase/README.md) | 当前最终模型展示 | 72 秒 MP4、9 张预览、逐帧 CSV、来源/统计 JSON、回放脚本 |
| [20260919_phase5_showcase](20260919_phase5_showcase/README.md) | 高度奖励调整前的对比基线 | 72 秒 MP4、9 张预览、逐帧 CSV、来源/统计 JSON |
| [20260920_height50_rewardik_restart](20260920_height50_rewardik_restart/train.sh) | 当前训练的可追溯启动参数 | 服务器 GPU 4、4096 环境、5000 次、W&B/video 启动脚本 |

最终模型位于服务器相应训练目录的 `model_4999.pt`；完整路径见各展示目录的 `summary.json.checkpoint`。
`summary.json.video` 保留服务器原始录制路径；本地观看文件为 `go2w_gaits_heights_72s.mp4`。
视频/预览是本地产物，由本目录 `.gitignore` 排除，未删除；不要将其混入代码提交。

## 2026-09-20 清理清单

仅清理以下本地文件，未对服务器执行删除：

- 两轮展示目录中的 `go2w_gaits_heights.mp4`：较高码率的原始录制本地副本，保留完整 72 秒播放版和服务器原始文件。
- `20260919_phase5_showcase/smoke/`：早期短录制检查的 4 个文件，不是最终演示。
- 仓库根目录下 `locolab/logs/rsl_rl/omnibot_go2w_multi_skill_flat_mlp_v0/2026-09-16_09-55-59_multi_skill_wtw_smoke/`：
  早期两次迭代的短测产物，共 7 个文件（含 `model_0.pt`、`model_1.pt`），不是训练完成的策略。
- `locolab/source/locomotion/mdp/__pycache__/wtw_actions.cpython-311.pyc`：已撤回动作适配器的孤立缓存。

共删除 14 个文件，文件内容合计 137,924,052 字节（约 131.5 MiB）。
rough、jump、高台等其他任务的代码、模型和产物不在清理范围内。

保留的视频在清理前均检查为 72 秒、1280×720、50 fps、3600 帧；SHA-256：

```text
2fbcdd8ae8fd18d2e45027d25203f12b49f154dd7e8cc1d68946bbb4a66e0d3e  20260919_phase5_showcase/go2w_gaits_heights_72s.mp4
5908bd3d891cc0ff073971848a7afd42b4c746b9267291cf75f6f0975edcf701  20260920_height50_showcase/go2w_gaits_heights_72s.mp4
```
