"""Velocity-slack must reward progress without rewarding standstill or wrong-way motion."""

from types import SimpleNamespace

import pytest
import torch
from test_go2w_action_smoothness import _load_rewards_module


def make_env(command_x, velocity_x):
    command = torch.zeros(len(command_x), 3)
    velocity = torch.zeros_like(command)
    command[:, 0] = torch.tensor(command_x)
    velocity[:, 0] = torch.tensor(velocity_x)
    return SimpleNamespace(
        commands=command,
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=velocity))},
        command_manager=SimpleNamespace(get_command=lambda name: command),
        _m20_gait_level=torch.tensor(0.0),
    )


def test_forward_and_reverse_slack_include_boundaries_and_exclude_stopping(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = make_env(
        [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [0.2999, 0.3, 0.8, 1.5, 1.5001, 0, -0.5, -0.3, -1.5, -0.2, -2, 0, 0.5],
    )
    result = rewards.velocity_slack(env, "base_velocity")
    torch.testing.assert_close(result, torch.tensor([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32))


def test_zero_x_commands_and_lateral_motion_do_not_earn_forward_bonus(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = make_env([0, 0, 0, 0.6, 0.6, 0.6], [0, 0.4, -0.4, 0, 0.18, 0.9])
    env.commands[1, 2] = 1.0
    env.commands[2, 1] = 1.0
    env.scene["robot"].data.root_lin_vel_b[3, 1] = 10.0
    torch.testing.assert_close(
        rewards.velocity_slack(env, "base_velocity"), torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.float32)
    )


def test_nonfinite_inputs_never_receive_a_bonus(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = make_env(
        [float("nan"), float("inf"), -float("inf"), 1, 1, 1],
        [0.5, float("inf"), -float("inf"), float("nan"), float("inf"), -float("inf")],
    )
    result = rewards.velocity_slack(env, "base_velocity")
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros(6))


def test_tiny_commands_cannot_reward_zero_motion_and_threshold_is_configurable(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = make_env([1e-44, 1e-4, 0.1, 0.2], [0.0, 1e-4, 0.1, 0.2])
    torch.testing.assert_close(rewards.velocity_slack(env, "base_velocity"), torch.tensor([0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(
        rewards.velocity_slack(env, "base_velocity", command_threshold=0.1), torch.tensor([0.0, 0.0, 0.0, 1.0])
    )


@pytest.mark.parametrize(
    "params",
    [
        {"min_ratio": 0.0},
        {"min_ratio": -0.1},
        {"min_ratio": 2.0, "max_ratio": 1.5},
        {"max_ratio": float("inf")},
        {"min_ratio": float("nan")},
        {"command_threshold": -0.1},
        {"command_threshold": float("inf")},
    ],
)
def test_invalid_band_configuration_is_rejected(monkeypatch, params):
    rewards = _load_rewards_module(monkeypatch)
    with pytest.raises(ValueError):
        rewards.velocity_slack(make_env([1.0], [0.5]), "base_velocity", **params)
