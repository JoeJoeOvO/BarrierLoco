# HighPlatform：本地成功复现权重

对应任务 `OmniBot-Go2W-HighPlatform-MLP-v0`。

- `model_14999.pt`：本地 Rough MLP 从零训练 15000 iter 的父权重。
- `model_19998.pt`：从上述父权重恢复，4096 环境 × 48 步，额外训练 5000 iter。
- 固定 50/60/70/80 cm 高台、速度 0.6/1.0 m/s、两组评估 seed，共 256/256 通过。
- `params/`、`rough_params/`：实际训练配置；`provenance.json`：来源、SHA256 和原始路径。
- 权重通过 Git LFS 保存。克隆后执行 `git lfs pull`。

在仓库根目录回放：

```bash
scripts/skill.sh highplatform play --checkpoint deploy/policy/go2w/highplatform_rough15k_batch196608/model_19998.pt --device cuda:0 --real-time --follow-camera --live-plot
```

仅验证 Isaac Lab；本目录未提供实机验收或部署导出模型。完整流程见 [实验记录](../../../../docs/experiments/2026-10-14_highplatform_rough15k_batch196608/README.md)。
