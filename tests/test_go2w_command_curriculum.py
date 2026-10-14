"""Command coverage, physical-state invariance and restart-clock regression checks."""

import importlib.util
from pathlib import Path

import pytest
import torch

from test_go2w_wtw import command_term  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "curriculum_under_test", ROOT / "locolab/source/locomotion/mdp/wtw_curriculum.py"
)
curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum)


def test_curriculum_milestones_interpolate_and_hold():
    curriculum.validate_schedule(curriculum.DEFAULT_SCHEDULE)
    for iteration, speed, swing in curriculum.DEFAULT_SCHEDULE:
        assert curriculum.curriculum_limits(iteration) == pytest.approx((speed, swing))
    assert curriculum.curriculum_limits(1000) == pytest.approx((1.25, 0.11))
    assert curriculum.curriculum_limits(30000) == pytest.approx((5.0, 0.15))


@pytest.mark.parametrize("schedule", [(), ((1, 1, .1),), ((0, 1, .1), (0, 2, .15)), ((0, 2, .1), (10, 1, .15))])
def test_invalid_schedules_rejected(schedule):
    with pytest.raises(ValueError):
        curriculum.validate_schedule(schedule)


def test_sampling_retains_easy_commands_and_limits_extreme_exposure():
    torch.manual_seed(314)
    vx, swing = curriculum.sample_curriculum_commands(100000, "cpu", 30000)
    assert vx.min() >= -5 and vx.max() <= 5
    assert swing.min() >= 0 and swing.max() <= .15
    # 20% full-band exposure, of which 60% lies beyond |vx|=2.
    assert float((vx.abs() > 2).float().mean()) == pytest.approx(.12, abs=.006)
    assert float((swing > .1).float().mean()) == pytest.approx(1 / 6, abs=.006)
    assert float((vx < -4).float().mean()) > .015
    assert float((vx > 4).float().mean()) > .015
    vx, swing = curriculum.sample_curriculum_commands(10000, "cpu", 0)
    assert vx.abs().max() <= 1 and swing.max() <= .1


def test_command_curriculum_survives_resets_and_resumes_without_changing_robot(command_term):
    term = command_term
    term.cfg.curriculum_enabled = True
    term.cfg.curriculum_schedule = curriculum.DEFAULT_SCHEDULE
    term.cfg.curriculum_steps_per_iteration = 24
    term.cfg.lin_vel_x = (-5.0, 5.0)
    term.cfg.footswing_height = (0.0, .15)
    term._env.common_step_counter = 20000 * 24
    original_pose = term.robot.data.default_joint_pos.clone()
    original_limits = term.robot.data.soft_joint_pos_limits.clone()
    state = term.curriculum_state_dict()
    term.reset()
    assert term.curriculum_state_dict() == state
    assert (term.command[term.category == 0, 9] == 0).all()
    torch.testing.assert_close(term.robot.data.default_joint_pos, original_pose)
    torch.testing.assert_close(term.robot.data.soft_joint_pos_limits, original_limits)
    term._env.common_step_counter = 0
    term.load_curriculum_state_dict(state)
    assert term.curriculum_step == 480000
    term._env.common_step_counter = 24
    metrics = term.reset()
    assert metrics["curriculum_iteration"] == 20001
    assert metrics["curriculum_vx_limit"] == 5
    assert metrics["curriculum_swing_limit"] == .15
    with pytest.raises(ValueError):
        term.load_curriculum_state_dict({**state, "steps_per_iteration": 32})
