# Multiskill最终版使用说明

仅保留本地从零30k＋续训50k的最终80k结果。V1课程实现仍保留，但不再保存作者/V1旧权重或中间训练、测试产物。

- Task：`OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0`
- 模型：`deploy/policy/go2w/multiskill_scratch80k/model_79999.pt`
- MLP：64→512→256→128→16，无GRU；Critic输入318。
- 控制：10维cmd，包括速度、机身高度、步频、相位、占空比、摆动高度。
- 地形：平地；三种步态rolling/trot/bound；课程从±1扩到±5m/s，摆动高度上限0.15m。

## 回放、录屏和校验

```bash
cd /home/joey/BarrierLoco
./scripts/skill.sh multiskill play
./scripts/skill.sh multiskill sweep --seconds 8
./scripts/skill.sh multiskill record --headless --seconds 6 --output-dir outputs/multiskill_record_01
./scripts/skill.sh multiskill verify
```

全部默认使用最终80k模型，也可显式传`--checkpoint`给回放或录屏。
回放沿用V1的跟随镜头、76组连续cmd、右侧状态面板和速度曲线，保留Previous/Next/Hold-Auto。
切换cmd不重置物理状态，关闭训练超时但保留跌倒终止。录屏输出目录须尚不存在。
verify检查本地最终模型及其TorchScript/ONNX数值一致性，不是实机验证。

## 从零复现与续训

```bash
# 默认已是从零，显式写出便于审查。第一阶段30000次。
./scripts/skill.sh multiskill train --headless --from-scratch \
  --num_envs 1024 --max_iterations 30000 --seed 42 --device cuda:0

# 用新训练产生的30k模型续训50000次；替换实际路径。
./scripts/skill.sh multiskill train --headless --resume \
  --checkpoint /absolute/path/to/new_run/model_29999.pt \
  --num_envs 1024 --max_iterations 50000 --seed 42 --device cuda:0

# 若要在保留的80k模型上继续：
./scripts/skill.sh multiskill train --headless --resume \
  --checkpoint "$PWD/deploy/policy/go2w/multiskill_scratch80k/model_79999.pt" \
  --num_envs 1024 --max_iterations 5000 --seed 42 --device cuda:0
```

从零/续训都不再依赖旧作者权重。没有额外启动训练。
详细参数、结果局限和源码来源见[正式实验记录](experiments/2026-10-12_multiskill_scratch80k/README.md)。
代码分工见[代码导航](multiskill_code_map.md)。模型按仓库规则被Git忽略，迁移机器时须单独备份最终模型目录。
