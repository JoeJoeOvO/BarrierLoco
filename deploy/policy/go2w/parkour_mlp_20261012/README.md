# Retained Go2W Parkour MLP — 2026-10-12

Final scratch checkpoint after 5000 updates: `model_4999.pt`.
The checkpoint and seven-motion TO bank are checksum-verified by `export_policy.py`.

- Task: `OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0`.
- Actor: 57 observations, 512/256/128 ELU MLP, 16 deterministic mean actions.
- Commands: velocity XY, yaw rate, jump, fixed TO-frame height/landing XY.
- Critic: 312 observations, with private phase and remaining-displacement feedback.
- Model, original training parameters, random-initialization proof, ONNX/JIT and export validation are retained.

See `docs/go2w_parkour.md` for the exact interface, training design, validation and limitations.
Run `python deploy/policy/go2w/parkour_mlp_20261012/export_policy.py --verify-only` from the repository root.

This 57-input policy is not plug-compatible with the existing 54-input Jump deployment.
Native inference was reviewed; the existing C++ runtime was not adapted or real-robot validated.
