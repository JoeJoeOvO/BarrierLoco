from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = ROOT / "locolab/source/locomotion/mdp/rewards.py"


def _load_rewards_module(monkeypatch: pytest.MonkeyPatch):
    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    class FakeSceneEntityCfg:
        def __init__(self, name, *args, **kwargs):
            self.name = name

    assets = ModuleType("isaaclab.assets")
    assets.Articulation = type("Articulation", (), {})
    assets.RigidObject = type("RigidObject", (), {})
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = FakeManagerTermBase
    managers.RewardTermCfg = type("RewardTermCfg", (), {})
    managers.SceneEntityCfg = FakeSceneEntityCfg
    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensor = type("ContactSensor", (), {})
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.quat_apply_inverse = lambda quaternion, vector: vector
    utils = ModuleType("isaaclab.utils")
    utils.math = math_utils
    isaaclab = ModuleType("isaaclab")
    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.sensors = sensors
    isaaclab.utils = utils
    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.assets": assets,
        "isaaclab.managers": managers,
        "isaaclab.sensors": sensors,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("go2w_rewards", REWARDS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_smoothness_is_stateful_idempotent_and_supports_partial_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewards = _load_rewards_module(monkeypatch)
    action_manager = SimpleNamespace(
        active_terms=["joint_pos", "joint_vel"],
        action_term_dim=[2, 1],
        total_action_dim=3,
        action=torch.zeros(2, 3),
        prev_action=torch.zeros(2, 3),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        action_manager=action_manager,
    )
    action_terms = ["joint_pos", "joint_vel"]
    term = rewards.ActionSmoothnessPenalty(SimpleNamespace(params={"action_term_names": action_terms}), env)
    history_ptr = term._prev_prev_action.data_ptr()

    def evaluate(step: int, current: list[float], previous: list[float], episode_length: int) -> torch.Tensor:
        env.common_step_counter = step
        env.episode_length_buf[:] = episode_length
        action_manager.action[:] = torch.tensor(current)
        action_manager.prev_action[:] = torch.tensor(previous)
        return term(env, action_term_names=action_terms).clone()

    torch.testing.assert_close(evaluate(0, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0), torch.zeros(2))
    torch.testing.assert_close(evaluate(1, [2.0, 4.0, 6.0], [1.0, 2.0, 3.0], 1), torch.zeros(2))
    torch.testing.assert_close(evaluate(2, [4.0, 8.0, 12.0], [2.0, 4.0, 6.0], 2), torch.zeros(2))
    expected = torch.full((2,), 14.0)
    torch.testing.assert_close(evaluate(3, [7.0, 14.0, 21.0], [4.0, 8.0, 12.0], 3), expected)

    action_manager.action.fill_(100.0)
    action_manager.prev_action.fill_(50.0)
    torch.testing.assert_close(term(env, action_term_names=action_terms), expected)
    torch.testing.assert_close(evaluate(4, [10.0, 20.0, 30.0], [7.0, 14.0, 21.0], 4), torch.zeros(2))
    assert term._prev_prev_action.data_ptr() == history_ptr

    term.reset(env_ids=[0])
    torch.testing.assert_close(term._prev_prev_action[0], torch.zeros(3))
    torch.testing.assert_close(term._prev_prev_action[1], torch.tensor([7.0, 14.0, 21.0]))

    masked_term = rewards.action_smoothness_penalty(SimpleNamespace(params={"mask": [2]}), env)
    env.common_step_counter = 5
    env.episode_length_buf.fill_(5)
    action_manager.action[:] = torch.tensor([100.0, 100.0, 7.0])
    action_manager.prev_action[:] = torch.tensor([0.0, 0.0, 4.0])
    torch.testing.assert_close(masked_term(env, mask=[2]), torch.ones(2))
