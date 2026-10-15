#!/usr/bin/env bash
# Select a skill in the unified repository using the existing Isaac Lab environment.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BARRIERLOCO_PYTHON:-/home/joey/miniconda3/envs/env_isaaclab/bin/python}"
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {highplatform|multiskill|omnijump|goalparkour|parkour} {train|play|sweep|record|verify|test|list|python} [arguments]"
    exit 2
fi
skill="$1"
operation="$2"
shift 2
# Keep each option only once: an explicit CLI value always wins over a default.
has_option() {
    local option="$1" argument
    shift
    for argument in "$@"; do
        [[ "$argument" == "$option" || "$argument" == "$option="* ]] && return 0
    done
    return 1
}
train_args=()
play_args=()
case "$skill" in
    highplatform)
        task=OmniBot-Go2W-HighPlatform-MLP-v0
        bundle=deploy/policy/go2w/highplatform_rough_mlp_20260922
        model=model_9998.pt
        ;;
    multiskill)
        task=OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0
        bundle=deploy/policy/go2w/multiskill_scratch80k
        model=model_79999.pt
        ;;
    omnijump|goalparkour|parkour)
        # New jump slot: the 30-motion GoalGrid has no published final policy yet.
        task=OmniBot-Go2W-Parkour-GoalGrid-MLP-v0
        bundle=""
        model=""
        ;;
    *) echo "Unknown skill: $skill" >&2; exit 2 ;;
esac
skill_root="$root"
cd -- "$skill_root"
# Prefer this repository even when another editable installation exists.
export PYTHONPATH="$skill_root:$skill_root/locolab${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
case "$operation" in
    train)
        if [[ "$skill" == multiskill ]]; then
            # Reproduce the retained pipeline from random weights unless resume/init is explicit.
            train_args=(--validate-finite)
            has_option --num_envs "$@" || train_args+=(--num_envs 1024)
            has_option --max_iterations "$@" || train_args+=(--max_iterations 30000)
            supplied_init=false
            for argument in "$@"; do
                case "$argument" in --from-scratch|--resume|--resume=*|--init_checkpoint|--init_checkpoint=*) supplied_init=true ;; esac
            done
            if [[ "$supplied_init" == false ]]; then
                train_args+=(--from-scratch)
            fi
        elif [[ "$task" == *Parkour-* ]]; then
            train_args=(--validate-finite)
            has_option --num_envs "$@" || train_args+=(--num_envs 4096)
            has_option --max_iterations "$@" || train_args+=(--max_iterations 5000)
            if ! has_option --resume "$@" && ! has_option --from-scratch "$@" && ! has_option --from_scratch "$@"; then
                train_args+=(--from-scratch)
            fi
        else
            has_option --num_envs "$@" || train_args+=(--num_envs 64)
        fi
        exec "$python_bin" locolab/scripts/locomotion/train.py --task "$task" \
            --logger tensorboard "${train_args[@]}" "$@"
        ;;
    play)
        if [[ -z "$bundle" ]] && ! has_option --checkpoint "$@"; then
            echo "GoalGrid has no published trained checkpoint. Supply --checkpoint for your GoalGrid model." >&2
            exit 2
        fi
        checkpoint_args=()
        has_option --checkpoint "$@" || checkpoint_args=(--checkpoint "$root/$bundle/$model")
        if [[ "$skill" == multiskill ]]; then
            exec "$python_bin" scripts/multiskill/cmd_sweep.py --extended "${checkpoint_args[@]}" "$@"
        fi
        env_args=()
        if ! has_option --num_envs "$@"; then
            if [[ "$task" == *GoalGrid* ]]; then env_args=(--num_envs 40)
            else env_args=(--num_envs 1); fi
        fi
        exec "$python_bin" locolab/scripts/locomotion/play.py --task "$task" \
            "${env_args[@]}" "${checkpoint_args[@]}" "${play_args[@]}" "$@"
        ;;
    sweep|record)
        checkpoint_args=()
        has_option --checkpoint "$@" || checkpoint_args=(--checkpoint "$root/$bundle/$model")
        [[ "$skill" == multiskill ]] || { echo "sweep/record are multiskill operations" >&2; exit 2; }
        if [[ "$operation" == sweep ]]; then
            exec "$python_bin" scripts/multiskill/cmd_sweep.py --extended "${checkpoint_args[@]}" "$@"
        fi
        exec "$python_bin" scripts/multiskill/record_cmd_sweep.py "${checkpoint_args[@]}" "$@"
        ;;
    verify)
        [[ -n "$bundle" ]] || { echo "No published GoalGrid policy to verify." >&2; exit 2; }
        export CUDA_VISIBLE_DEVICES=''
        exec "$python_bin" "$bundle/export_policy.py" --verify-only "$@"
        ;;
    test) exec "$python_bin" -m pytest "$@" ;;
    list) exec "$python_bin" locolab/scripts/list_envs.py "$@" ;;
    python) exec "$python_bin" "$@" ;;
    *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
