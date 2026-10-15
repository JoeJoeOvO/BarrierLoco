# GoalGrid 5000 次从零训练模型

状态：训练已完成，尚未完成逐目标独立验收；不能作为成功复现实验或可靠能力范围的证明。

- Task：`OmniBot-Go2W-Parkour-GoalGrid-MLP-v0`。
- 4096 环境，24 步/更新，5000 更新，seed 42；未加载旧模型。
- 30 条原始参考；57 维 actor、312 维 critic、16 维动作。
- checkpoint 为 `model_4999.pt`；`policy.pt` 和 `policy.onnx` 为其回放导出。
- 配置和初始化信息见 `params/`，训练版本与文件哈希见 `provenance.json`。
- 最后一轮训练日志地面起跳成功率约 42.4%，不是独立评测成功率。

在仓库根目录回放：

```bash
scripts/skill.sh omnijump play --checkpoint deploy/policy/go2w/goalgrid_scratch5k/model_4999.pt --real-time --follow-camera --live-plot --offline-visuals
```

连续目标泛化检查（平地，右侧 H/X/Y 曲线；每次跳跃内目标固定）：

```bash
scripts/skill.sh omnijump play --checkpoint deploy/policy/go2w/goalgrid_scratch5k/model_4999.pt --parkour-generalization --real-time --follow-camera --live-plot --offline-visuals
```

该模式只改变 actor 的目标输入，不修改参考轨迹。它保留物理失败检查，关闭与旧参考绑定的成功/失败终止及相应奖励，独立测量峰值和首次落点。高度目标是峰值、XY 目标是首次落点，不是逐帧轨迹；瞬时曲线差不等于最终跟踪误差。此模式不构成训练奖励或成功率评测。

离散参考全遍历可使用 `--parkour-sweep` 替换 `--parkour-generalization`。模型由 Git LFS 保存，克隆后需下载 LFS 实体。
