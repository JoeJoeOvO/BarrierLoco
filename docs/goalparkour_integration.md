# GoalGrid / Parkour 本地整合

来源：`shang13206/OmniBot` 的 `codex/go2w-parkour-goal-grid`，提交 `730ee7e2cc02b61e6dde27ea4c05bab63e2b813f`。
独立源码副本在 `/home/joey/barrierloco_sources/omnibot-goal-parkour`；运行时只使用 BarrierLoco 内的代码和参考库，不依赖该副本或 BarrierPlanner。
使用本机 `env_isaaclab`，没有创建新 Conda 环境。

## 替换范围与模型状态

- `omnijump`、`goalparkour` 与 `parkour` 均进入 `OmniBot-Go2W-Parkour-GoalGrid-MLP-v0`，替代原来的两阶段 Jump 工作流。
- 新 GoalGrid 包含 30 条参考动作：原 7 条，加 23 条独立验收的平地目标动作。上游此提交没有发布它训练完成的模型。
- 独立七轨迹任务、专用评估脚本和 `parkour_mlp_20261012` 模型包已删除。`parkour` 仅作为 GoalGrid 的兼容别名，不再选择旧模型。
- 保留原始 30 条参考及其采样比例，不修改轨迹、动力学或奖励。七条基础参考库仍是 GoalGrid 的来源完整性校验依赖，不是另一个训练任务。
- 旧 54 维 Jump 权重已归档到 `/home/joey/barrierloco_sources/barrierloco-legacy-jump-20261014`。旧 Jump task 已取消注册；Parkour 复用的底层指令/终止/奖励工具仍保留，不再提供旧 Jump task 训练入口。
- Multiskill/HighPlatform 的环境、奖励、网络和权重未替换。

## 策略接口

GoalGrid 使用纯 MLP（非 GRU）：actor `57 → 512 → 256 → 128 → 16`，ELU；critic 输入 312 维。

| Actor 输入（按实际拼接顺序） | 维数 |
|---|---:|
| 机身角速度 | 3 |
| 机身坐标系重力投影 | 3 |
| 指令 `[vx_yaw, vy_yaw, yaw_rate, jump, height, target_dx, target_dy]` | 7 |
| 12 个腿关节相对默认位置 | 12 |
| 16 个腿/轮关节相对默认速度 | 16 |
| 上一步动作 | 16 |

角速度缩放 0.2，关节速度缩放 0.05；完整定义见 observation 配置。Actor 不接收地形高度图、机身线速度或显式跳跃相位；critic 额外使用速度、地形和接触等特权信息，并得到动态剩余落点位移与相位。
`height/dx/dy` 是参考动作起始坐标系下的固定目标描述，不是机器人当前位置到目标的实时误差；高度指机身相对抬升，不等于轮底净离地或障碍高度。物理仿真记录为 1.8 秒，另加 0.4 秒恢复延伸；完整参考序列为 2.2 秒，不随速度伸缩。
输出为 16 个动作：12 个腿关节位置偏移（乘 0.25 后加默认关节位置）和 4 个轮关节速度（乘 10）。
Parkour 使用分关节腿力矩 23.5/23.5/35 Nm、速度 30/30/20 rad/s，自碰撞开启，控制频率 50 Hz。

## 任务与地形

原 7 条包含 3 个平地目标、40/60 cm 沟壑、左右 20° 横坡斜台，并混合普通滚动。
GoalGrid 的新增目标来自机身抬升 15/20/25 cm、落点 X=40/60/80 cm、Y=-15/0/15 cm 的组合；27 个候选中 23 个通过，4 个短距离斜向目标未纳入。
训练保留 30% 普通滚动通道，其余按参考库分配跳跃；物理沟壑/斜台与参考绑定。速度采样范围为 0.2–1.5 m/s。
此处目标范围描述实际离散参考库，不能据此宣称策略已覆盖任意连续三维目标。

## 使用

```bash
# 新 GoalGrid 从零训练；未传参数时采用 4096 环境、5000 更新、TensorBoard
scripts/skill.sh omnijump train --headless --device cuda:0

# 训练出新 GoalGrid checkpoint 后再回放；没有权重时入口会明确报错
scripts/skill.sh goalparkour play --checkpoint /absolute/path/to/goalgrid_model.pt --real-time --follow-camera --live-plot
```

下载/打包必须包含 Git LFS 实体文件；两个 `.json.gz` 参考库也是运行依赖，已纳入 Python 包数据。
共享训练入口兼容上游 `--from_scratch` 和本地 `--from-scratch`，保留 Parkour 的明确初始化约束与初始化哈希记录；回放曲线读取 Parkour 的 `parkour` 指令项。
本次只部署整合和短程验证，没有启动正式 5000 次训练，也没有适配 C++/实机部署。原始接口/算法说明见 [历史参考来源说明](go2w_parkour.md) 和 [GoalGrid 上游说明](go2w_parkour_goal_grid.md)。
