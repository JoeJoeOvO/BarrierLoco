"""Check skill dispatch without starting Isaac Sim or loading a checkpoint."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def dispatch(skill, operation, *args):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/skill.sh"), skill, operation, *args],
        env={**os.environ, "BARRIERLOCO_PYTHON": "/bin/echo"},
        capture_output=True,
        text=True,
        check=True,
    )
    return shlex.split(result.stdout)


def test_scratch_does_not_inject_author_checkpoint():
    args = dispatch("multiskill", "train", "--from-scratch", "--num_envs", "32", "--max_iterations", "2")
    assert "--init_checkpoint" not in args
    assert args.count("--num_envs") == args.count("--max_iterations") == 1
    assert args[args.index("--num_envs") + 1] == "32"


def test_resume_preserves_explicit_checkpoint():
    args = dispatch("multiskill", "train", "--resume", "--checkpoint", "/tmp/model.pt")
    assert "--init_checkpoint" not in args
    assert args[args.index("--checkpoint") + 1] == "/tmp/model.pt"


def test_default_reproduces_successful_scratch_pipeline():
    args = dispatch("multiskill", "train")
    assert "--init_checkpoint" not in args
    assert "--from-scratch" in args
    assert args[args.index("--num_envs") + 1] == "1024"


@pytest.mark.parametrize("operation", ["play", "sweep", "record"])
def test_custom_playback_checkpoint_is_not_shadowed(operation):
    args = dispatch("multiskill", operation, "--checkpoint=/tmp/model_79999.pt")
    assert "--checkpoint" not in args
    assert "--checkpoint=/tmp/model_79999.pt" in args


@pytest.mark.parametrize("skill", ["highplatform"])
def test_other_skills_do_not_load_gait_weights(skill):
    args = dispatch(skill, "train")
    assert "--init_checkpoint" not in args
    assert args[args.index("--num_envs") + 1] == "64"
    args = dispatch(skill, "play")
    assert "gait" not in args[args.index("--checkpoint") + 1]


@pytest.mark.parametrize("skill", ["omnijump", "goalparkour", "parkour"])
def test_jump_slot_uses_goalgrid_scratch_without_legacy_weights(skill):
    args = dispatch(skill, "train")
    assert args[args.index("--task") + 1] == "OmniBot-Go2W-Parkour-GoalGrid-MLP-v0"
    assert "--from-scratch" in args and "--jump_stage" not in args
    assert args[args.index("--num_envs") + 1] == "4096"
    result = subprocess.run(["bash", str(ROOT / "scripts/skill.sh"), skill, "play"], capture_output=True, text=True)
    assert result.returncode == 2 and "no published trained checkpoint" in result.stderr
    args = dispatch(skill, "play", "--checkpoint", "/tmp/goalgrid.pt", "--num_envs", "2")
    assert args.count("--num_envs") == 1 and "--jump_stage" not in args


def test_goalgrid_resume_remains_explicit():
    args = dispatch("goalparkour", "train", "--resume", "--checkpoint", "/tmp/goalgrid.pt")
    assert "--from-scratch" not in args
