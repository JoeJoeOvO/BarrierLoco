# Multiskill最终80k模型

本地随机初始化30k＋续训50k，仅保留此最终模型。

- Task：OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0
- checkpoint：model_79999.pt（含优化器与课程状态）
- 导出：policy.pt、policy.onnx
- 验证：`./scripts/skill.sh multiskill verify`
- 回放：`./scripts/skill.sh multiskill play`
- 配置/来源/摘要：params/、provenance.json、training_summary.json、export_manifest.json

正式复现说明见`docs/experiments/2026-10-12_multiskill_scratch80k/README.md`。
该目录内的模型是实际文件，不依赖旧训练目录或BarrierPlanner。模型被Git忽略，需单独备份。
