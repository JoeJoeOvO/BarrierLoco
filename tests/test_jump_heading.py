"""Directional contracts for stage-two jumps without changing policy dimensions."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_go2w_jump_stack import _load_jump_rewards
from test_moving_jump import make_command, step


def yaw_quat(angles):
    angles = torch.as_tensor(angles)
    zeros = torch.zeros_like(angles)
    return torch.stack(((angles / 2).cos(), zeros, zeros, (angles / 2).sin()), -1)


def heading_command(monkeypatch, count=1):
    term, env, robot, sensor = make_command(monkeypatch, count)
    term.cfg.heading_hold_duration = 2.0
    term.cfg.jump_probability = 0.0
    robot.data.root_quat_w = yaw_quat(torch.zeros(count))
    robot.data.root_ang_vel_w = torch.zeros(count, 3)
    return term, env, robot, sensor


def reward_env(env, term, target):
    env.command_manager = SimpleNamespace(get_term=lambda _: term, get_command=lambda _: target)
    return env


def test_heading_is_captured_only_on_accepted_trigger(monkeypatch):
    term, env, robot, sensor = heading_command(monkeypatch, 2)
    robot.data.root_quat_w = yaw_quat([math.pi / 2, 0.4])
    sensor.data.net_forces_w[1].zero_()
    before = robot.data.root_pos_w.clone()
    assert term.set_trigger(1).tolist() == [True, False]
    torch.testing.assert_close(term.trigger_yaw, torch.tensor([math.pi / 2, 0.0]))
    torch.testing.assert_close(term.heading_time_remaining, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(before, robot.data.root_pos_w)
    robot.data.root_quat_w = yaw_quat([1.9, 0.8])
    assert not term.set_trigger(1).any()
    torch.testing.assert_close(term.trigger_yaw, torch.tensor([math.pi / 2, 0.0]))


def test_heading_hold_covers_recovery_and_has_an_independent_fixed_deadline(monkeypatch):
    term, env, _, _ = heading_command(monkeypatch)
    term.set_trigger(1)
    step(term, env, 60)
    assert not term.active.any()
    assert term.heading_tracking_mask.item()
    assert term.heading_time_remaining.item() == pytest.approx(0.8, abs=1e-5)
    before = term.heading_time_remaining.clone()
    term.update_state()
    term.update_state()
    torch.testing.assert_close(term.heading_time_remaining, before)
    step(term, env, 40)
    assert not term.heading_tracking_mask.any()


def test_partial_reset_does_not_reanchor_other_robots(monkeypatch):
    term, _, robot, _ = heading_command(monkeypatch, 2)
    robot.data.root_quat_w = yaw_quat([0.3, 0.9])
    term.set_trigger(1)
    term.reset(torch.tensor([0]))
    assert term.heading_tracking_mask.tolist() == [False, True]
    assert term.trigger_yaw[1].item() == pytest.approx(0.9)


def test_wrapped_yaw_error_and_quaternion_sign_are_consistent(monkeypatch):
    term, _, robot, _ = heading_command(monkeypatch)
    robot.data.root_quat_w = yaw_quat([-math.pi + 0.01])
    term.set_trigger(1)
    robot.data.root_quat_w = yaw_quat([math.pi - 0.01])
    torch.testing.assert_close(term.heading_error(), torch.tensor([-0.02]), atol=1e-6, rtol=0)
    robot.data.root_quat_w *= -1
    torch.testing.assert_close(term.heading_error(), torch.tensor([-0.02]), atol=1e-6, rtol=0)


def test_skill_stage_retains_legacy_behavior_without_quaternion_input(monkeypatch):
    term, env, robot, _ = make_command(monkeypatch)
    assert not hasattr(robot.data, "root_quat_w")
    term.cfg.jump_probability = 0
    assert term.set_trigger(1).item()
    step(term, env, 3)
    assert not term.heading_tracking_mask.any()


def test_directional_penalties_use_trigger_frame_and_mask(monkeypatch):
    rewards = _load_jump_rewards(monkeypatch)
    term, env, robot, _ = heading_command(monkeypatch, 3)
    robot.data.root_quat_w = yaw_quat([0.0, math.pi / 2, 0.0])
    term.set_trigger(1)
    robot.data.root_quat_w = yaw_quat([0.2, math.pi / 2 + 0.2, 0.8])
    robot.data.root_ang_vel_w[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    robot.data.root_lin_vel_w = torch.tensor([[1.0, 0.5, 9.0], [2.0, 1.0, -9.0], [4.0, 9.0, 0.0]])
    term.heading_time_remaining[2] = 0
    reward_env(env, term, torch.zeros(3, 3))
    torch.testing.assert_close(rewards.jump_heading_error_l2(env, "jump"), torch.tensor([0.04, 0.04, 0.0]))
    torch.testing.assert_close(rewards.jump_heading_yaw_velocity_l2(env, "jump"), torch.tensor([1.0, 4.0, 0.0]))
    torch.testing.assert_close(
        rewards.jump_heading_lateral_velocity_l2(env, "jump", "velocity"), torch.tensor([0.25, 4.0, 0.0])
    )


def test_turning_no_longer_redefines_forward_velocity_reward(monkeypatch):
    rewards = _load_jump_rewards(monkeypatch)
    term, env, robot, _ = heading_command(monkeypatch)
    term.set_trigger(1)
    robot.data.root_quat_w = yaw_quat([math.pi / 2])
    robot.data.root_lin_vel_w = torch.tensor([[0.0, 1.5, 8.0]])
    target = torch.tensor([[1.5, 0.0, 0.0]])
    reward_env(env, term, target)
    args = (env, "velocity", "jump")
    torch.testing.assert_close(rewards.jump_track_lin_vel_xy_exp(*args, std=0.5), torch.tensor([2.0]))
    assert rewards.jump_track_lin_vel_xy_exp(*args, std=0.5, lock_jump_heading=True).item() < 1e-6
    robot.data.root_lin_vel_w[:, :2] = torch.tensor([[1.5, 0.0]])
    torch.testing.assert_close(
        rewards.jump_track_lin_vel_xy_exp(*args, std=0.5, lock_jump_heading=True), torch.tensor([2.0])
    )


def test_existing_command_observation_encodes_error_without_mutation(monkeypatch):
    term, env, robot, _ = heading_command(monkeypatch)
    term.set_trigger(1)
    robot.data.root_quat_w = yaw_quat([math.pi / 2])
    target = torch.tensor([[1.5, 0.0, 0.0]])
    before = target.clone()
    env.command_manager = SimpleNamespace(
        get_term=lambda _: term,
        get_command=lambda name: target if name == "base_velocity" else term.command,
    )
    source = Path(__file__).resolve().parents[1] / "locolab/source/locomotion/mdp/observations.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "velocity_and_jump_commands"
    )
    scope = {"torch": torch, "ManagerBasedEnv": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    observe = scope["velocity_and_jump_commands"]
    torch.testing.assert_close(observe(env), torch.tensor([[1.5, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(
        observe(env, lock_jump_heading=True), torch.tensor([[0.0, -1.5, 0.0, 1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(target, before)
    term.heading_time_remaining.zero_()
    torch.testing.assert_close(observe(env, lock_jump_heading=True), torch.tensor([[1.5, 0.0, 0.0, 1.0]]))
