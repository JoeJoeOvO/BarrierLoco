#!/usr/bin/env bash
# Historical pre-rename snapshot launcher; see docs/go2w_gait_wtw.md for current commands.
set -euo pipefail

run_root=/home/sshang/omnibot_data/workspaces/OmniBot-multi-skill-wtw-height50-rewardik-20260920
cd "$run_root/locolab"
exec 9>"$run_root/train.lock"
flock -n 9 || { echo "This multi-skill training is already running." >&2; exit 1; }
runtime=(
    env -u CUDA_VISIBLE_DEVICES -u WANDB_RUN_ID
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY
    WANDB_MODE=online
    LD_LIBRARY_PATH=/home/sshang/miniconda3/envs/env_isaaclab/lib
    PYTHONPATH="$run_root:$run_root/locolab"
    /home/sshang/miniconda3/envs/env_isaaclab/bin/python -u scripts/locomotion/train.py
    --task OmniBot-Go2W-MultiSkill-Flat-MLP-v0 --seed 42 --device cuda:4 --headless
    --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=4"
)
"${runtime[@]}" --num_envs 12 --max_iterations 2 --logger tensorboard \
    --run_name multi_skill_wtw_height50_rewardik_smoke --smoke-marker > "$run_root/smoke.log" 2>&1
grep -q TRAINING_SMOKE_OK "$run_root/smoke.log"
echo "Height-50/reward-only-IK GPU smoke passed; starting full training."
exec "${runtime[@]}" \
    --num_envs 4096 --max_iterations 5000 \
    --logger wandb --log_project_name omnibot-go2w \
    --video --video_interval 500 --video_length 1000 \
    --run_name multi_skill_wtw_height50_rewardik_phase5_rollclock0_4096_remote_gpu4_5k_video
