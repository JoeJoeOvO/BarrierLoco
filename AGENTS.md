# Repository Guidelines

## Project Structure & Module Organization

`locolab/` contains the Isaac Lab training stack: RL algorithms in `locolab/rsl_rl/`, task and asset definitions in `locolab/source/locomotion/`, and entry scripts in `locolab/scripts/locomotion/`.
`deploy/` contains sim-to-real code: policy configs in `deploy/policy/`, C++ runtime and robot interfaces in `deploy/src/`, and build helpers in `deploy/build.sh`.
`simulation/` is reserved for lightweight sim tooling but is currently minimal; the active MuJoCo simulator entrypoint is `deploy/src/rl_deploy/src/rl_sim_mujoco.cpp`.

## Build, Test, and Development Commands

- `cd locolab && pip install -e .`: install the training package in editable mode.
- `cd locolab && python scripts/list_envs.py`: list registered Isaac Lab tasks.
- `cd locolab && python scripts/locomotion/train.py --task OmniBot-Go2W-Rough-v0 --num_envs 4096`: start training.
- `cd locolab && python scripts/locomotion/play.py --task OmniBot-Go2W-Rough-v0 --checkpoint <path>`: run inference and export `policy.pt` / `policy.onnx`.
- `cd deploy && ./build.sh -mj`: build MuJoCo simulator targets.
- `cd deploy && ./cmake_build/bin/rl_sim_mujoco go2w scene`: run sim-to-sim deployment.
- `pre-commit run --all-files`: run repository formatting and lint hooks before commit.

## Coding Style & Naming Conventions

Python uses Black and isort with a `120` character line length; target Python is 3.10. Prefer `snake_case` for functions, variables, and YAML keys; use `PascalCase` for config classes such as `Go2WRoughEnvCfg`. C++ follows the existing repository style: headers in `include/`, implementation in `src/`, concise comments, and no large unrelated refactors mixed into feature work.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `tests/` and `test_*.py` naming. Add new Python tests under `tests/` when practical; if no automated test fits, include a reproducible smoke-test command in the PR. For deploy changes, validate both build success and the relevant runtime path, for example `./build.sh -mj` plus a short `rl_sim_mujoco` run.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit style such as `feat(deploy): ...`, `refactor(Go2W): ...`, and `fix: ...`. Keep subjects imperative and scoped when possible. PRs should include: what changed, affected modules (`locolab` or `deploy`), exact validation commands, and screenshots or logs for UI / simulation changes. Do not modify `isaaclab_rl` unless explicitly requested.

## Environment Notes

Use the user's `env_isaaclab` Conda environment for this workspace: `/home/joey/miniconda3/envs/env_isaaclab/bin/python`. Use `scripts/skill.sh` to select multiskill, highplatform, or omnijump in the unified root repository. Keep generated policies under `deploy/policy/<robot>/<config>/policy.onnx`, and keep policy observation ordering aligned with the training task before testing deployment.

Gait, HighPlatform and Jump retain their bundled checkpoints' task-local leg limits (23.5 Nm / 30 rad/s). Gait disables self-collision; HighPlatform and Jump enable it. Do not remove these overrides when updating the shared robot asset. After changing skill dynamics, run `scripts/skill.sh multiskill python tests/smoke_unified_skill_dynamics.py --headless` in addition to relevant tests.

## Experiment Records

Create a formal Markdown experiment record under `docs/experiments/` only after a complete pipeline has successfully achieved the intended task outcome, using `TEMPLATE.md` and updating the index. Do not create or append formal records for intermediate training stages, preflight checks, failed attempts, or stops. A zero exit code alone is not task success. Keep raw logs, checkpoints, configs, and machine-readable job status for debugging throughout development. Once successful, document the final reproducible pipeline: code provenance and changes, exact task/config/seed, checkpoint lineage, iteration counts, reproduction commands, artifact hashes, validation scope, and remaining limitations. Do not claim a concern is fixed without evidence. Keep the pipeline independent of the original BarrierPlanner directory and note large artifacts requiring separate backup.
