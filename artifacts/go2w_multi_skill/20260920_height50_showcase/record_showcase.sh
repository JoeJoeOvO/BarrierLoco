#!/usr/bin/env bash
# Historical pre-rename snapshot replay; see docs/go2w_gait_wtw.md for the current task.
set -euo pipefail

root=/home/sshang/omnibot_data/workspaces/OmniBot-multi-skill-wtw-height50-rewardik-20260920
run="$root/locolab/logs/rsl_rl/omnibot_go2w_multi_skill_flat_mlp_v0/2026-09-20_01-10-58_multi_skill_wtw_height50_rewardik_phase5_rollclock0_4096_remote_gpu4_5k_video"
cd "$root/locolab"
exec 9>"$root/showcase.lock"
flock -n 9 || { echo "Showcase recording already running." >&2; exit 1; }
exec env -u CUDA_VISIBLE_DEVICES -u WANDB_RUN_ID \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    WANDB_MODE=disabled \
    LD_LIBRARY_PATH=/home/sshang/miniconda3/envs/env_isaaclab/lib \
    PYTHONPATH="$root:$root/locolab" \
    /home/sshang/miniconda3/envs/env_isaaclab/bin/python -u scripts/locomotion/record_wtw_showcase.py \
    --checkpoint "$run/model_4999.pt" \
    --output-dir "$run/videos/showcase_20260920" \
    --segment-seconds 8 --device cuda:4 --headless --logger tensorboard \
    --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=4"
