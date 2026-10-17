# Planner 多地形环境

任务：`BarrierLoco-Go2W-PlannerTerrain-v0`。当前是环境基础设施：地形、机器人、目标、地形观测、重置和终止条件；尚未连接冻结技能、多头 Planner 或 PPO 奖励。现有三个技能及其动力学配置不变。

## 地形组成

使用本机 Isaac Lab 的 `isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`，深拷贝其中全部六类配置，避免修改全局预设。

| 类别 | 参数 |
| --- | --- |
| flat | 平地 |
| platform | 高台 0.20–0.60 m，长 2 m |
| gap | 沟宽 0.20–0.60 m、深 1 m |
| mixed | 高台后接沟 |
| pyramid_stairs | 金字塔楼梯，级高 0.05–0.23 m、级宽 0.30 m |
| pyramid_stairs_inv | 倒金字塔楼梯，相同级高、级宽 |
| boxes | 随机方块，网格 0.45 m、高度范围 0.05–0.20 m |
| random_rough | 随机粗糙面，噪声范围 0.02–0.10 m、量化步长 0.02 m |
| hf_pyramid_slope | 金字塔斜坡，坡度比 0–0.4 |
| hf_pyramid_slope_inv | 下凹斜坡，坡度比 0–0.4 |

与 Rough 预设的差别：地块由 8×8 m 改为 16×16 m（兼容随机方块的正方形要求和长距离技能场景）；10 类各占一列，权重均为 0.1；6 行对应六个难度区间。每行内随机采样难度，非六个固定数值。粗糙面的噪声范围按上游实现采样，不保证所有类别都随行号变难。当前没有按成功率升级的课程。

训练配置默认 256 环境，初始难度行 0–2；预览 60 环境覆盖全部 10×6 地块。地形类别与难度行保存在 `scene.terrain.terrain_types` / `terrain_levels`，后续用于分地形指标，不作为策略显式技能标签。

自定义场景在起点前方 13 m 设置目标；Rough 使用生成器中心起点，目标位于前方 6 m。金字塔类从中心向外走，因此普通楼梯主要是下行，倒金字塔主要是上行；后续训练可再扩展双向起终点采样。自定义场景横向偏离起点超过 2 m 终止（Rough 放宽到 7 m，支持二维导航），防止从旁边绕过自定义高台/沟。只对沟和混合场景启用掉沟高度判断，避免正常下坡被误判。其他终止包括到达目标、越界、姿态超限和 30 秒超时。

观测含本体状态、上次动作、机体坐标系下目标 XY、前向偏置 1.5 m 的 6×4 m 高度扫描（0.2 m 分辨率）及射线有效性掩码。扫描缺失值会填充并裁剪；不能把无效射线当作已知安全地面。动作暂为 16 维原生关节诊断输入，尚不是技能选择或技能 cmd。

## 预览与验证

在仓库根目录：

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/preview_planner_terrain.py
```

`omnijump python` 在这里仅选择运行环境，不加载 Jump 模型。窗口中 Previous/Next 切换地块；绿色为起点、橙色为目标。机器人使用零动作保持默认关节目标，不代表已经会通过地形。

无界面检查所有类别和难度：

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/preview_planner_terrain.py --headless --steps 64 --report /tmp/planner-terrain-check.json
/home/joey/miniconda3/envs/env_isaaclab/bin/python -m pytest tests/test_planner_terrain.py -q
```

检查覆盖实际合并碰撞网格上的起终点射线、沟底深度、观测有限性及环境步进；不证明技能通过率或 Planner 效果。需要在接入技能时解决三套策略的动力学差别，并设计技能接口、切换约束及导航奖励。
