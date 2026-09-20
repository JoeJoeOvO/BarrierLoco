# Go2W 多步态 / 高度回放

- 最终视频：`go2w_gaits_heights_72s.mp4`（H.264，1280×720，50 fps，72 s）。
- 本地仅保留完整 72 秒播放版；未剪除失败片段或初始收敛过程。
  原始录制的本地副本和短测目录已清理，服务器原始录制路径保留在 `summary.json.video`。
- 训练：`2026-09-19_17-00-23_multi_skill_wtw_phase5_pair10_rollclock0_4096_remote_gpu4_5k_video`。
- Checkpoint：`model_4999.pt`；完整来源与每段原始 command 见 `summary.json`。
- 回放在服务器 GPU 4、单环境中执行，不修改训练参数，不启动新训练。
- 各段采用标准相位中心、频率 2.5 Hz、前进速度 0.6 m/s、侧向/转向速度 0。
- trot/bound：占空比 0.5、轮底摆动高度 7.5 cm；rolling：全支撑、摆动高度 0。
- Play 配置关闭观测噪声和域随机化，各段从同一中性姿态独立 reset 开始；正常终止条件保留。
- `telemetry.csv` 含全部 3600 个控制帧；九段各 400 帧，自动重置次数均为 0。

| 时间 | 模式 | 目标高度 | 实际平均高度（每段 2 s 后） |
|---|---|---|---|
| 0–8 s | trot | 34 cm | 43.0 cm |
| 8–16 s | trot | 22 cm | 35.7 cm |
| 16–24 s | trot | 46 cm | 46.1 cm |
| 24–32 s | bound | 34 cm | 41.1 cm |
| 32–40 s | bound | 22 cm | 34.6 cm |
| 40–48 s | bound | 46 cm | 44.5 cm |
| 48–56 s | rolling | 34 cm | 37.0 cm |
| 56–64 s | rolling | 22 cm | 28.8 cm |
| 64–72 s | rolling | 46 cm | 41.0 cm |

视频标注的是目标高度，并同时显示实时实际高度；不能将目标标签解读为已准确达到。
这次仅是单种速度、标准步态相位、单种无随机化场景的展示，不代表全范围鲁棒性验证。

接触阈值为合力大于 1 N，顺序 FR/FL/RR/RL。各模式在三段 2 s 后共 900 帧：

- trot：`1001` 405 帧、`0110` 390 帧、`0000` 75 帧、`0100` 30 帧。
- bound：`0011` 473 帧、`1100` 405 帧、`0000` 15 帧、`0010` 7 帧。
- rolling：`1111` 900 帧。

录制脚本：`locolab/scripts/locomotion/record_wtw_showcase.py`。
