"""Explicit scratch mode cannot silently resume an existing network or optimizer."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parkour_scratch_cli", ROOT / "locolab/scripts/locomotion/cli_args.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def args(**changes):
    values = dict(
        task="OmniBot-Go2W-Parkour-GoalGrid-MLP-v0",
        from_scratch=True,
        resume=False,
        warm_start_rough=None,
        checkpoint=None,
        load_run=None,
        resume_experiment_name=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_explicit_scratch_without_checkpoint_is_allowed():
    CLI.validate_scratch_request(args())
    CLI.validate_scratch_request(args(task="OmniBot-Go2W-Parkour-GoalGrid-MLP-v0"))


@pytest.mark.parametrize("name", ["resume", "warm_start_rough", "checkpoint", "load_run", "resume_experiment_name"])
def test_scratch_rejects_every_checkpoint_source(name):
    with pytest.raises(ValueError, match=name):
        CLI.validate_scratch_request(args(**{name: True if name == "resume" else "old_model"}))


def test_other_tasks_and_legacy_resume_are_not_accidentally_changed():
    CLI.validate_scratch_request(args(from_scratch=False, resume=True))
    with pytest.raises(ValueError, match="currently supported"):
        CLI.validate_scratch_request(args(task="OmniBot-Go2W-Rough-v0"))


@pytest.mark.parametrize(
    "task",
    ("OmniBot-Go2W-Parkour-MLP-v0", "OmniBot-Go2W-Parkour-Speed-MLP-v0", "OmniBot-Go2W-Parkour-FixedTarget-MLP-v0"),
)
def test_discarded_parkour_variants_are_not_scratch_targets(task):
    with pytest.raises(ValueError, match="currently supported"):
        CLI.validate_scratch_request(args(task=task))


def runner():
    policy = torch.nn.Module()
    policy.actor = torch.nn.Sequential(torch.nn.Linear(57, 16))
    policy.critic = torch.nn.Sequential(torch.nn.Linear(312, 1))
    return SimpleNamespace(
        current_learning_iteration=0,
        alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters())),
    )


def test_fresh_actor_and_critic_report_auditable_fingerprints():
    r = runner()
    report = CLI.scratch_initialization_report(r)
    assert report["mode"] == "scratch" and not report["checkpoint_loaded"]
    assert report["initial_iteration"] == report["optimizer_state_entries"] == 0
    assert (report["actor_inputs"], report["critic_inputs"]) == (57, 312)
    assert len(report["actor_first_weight_sha256"]) == len(report["critic_first_weight_sha256"]) == 64
    assert report == CLI.scratch_initialization_report(r)


@pytest.mark.parametrize("changed", ["iteration", "optimizer"])
def test_nonfresh_runner_fails_the_startup_audit(changed):
    r = runner()
    if changed == "iteration":
        r.current_learning_iteration = 9998
    else:
        r.alg.policy.actor(torch.ones(1, 57)).sum().backward()
        r.alg.optimizer.step()
    with pytest.raises(ValueError, match="iteration zero"):
        CLI.scratch_initialization_report(r)
