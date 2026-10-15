"""Off-grid playback must vary goals without silently rewriting the reference bank."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "locolab/source/locomotion/utils/parkour_generalization.py"
SPEC = importlib.util.spec_from_file_location("generalization", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_continuous_targets_stay_in_declared_envelope():
    for index in range(128):
        _, (height, x, y), speed = MODULE.target_for_attempt(index)
        assert 0.15 <= height <= 0.25
        assert 0.4 <= x <= 0.8
        assert -0.15 <= y <= 0.15
        assert 0.2 <= speed <= 1.5
    heights = [MODULE.target_for_attempt(i)[1][0] for i in range(32)]
    assert len({round(h, 5) for h in heights}) > 20


def test_actor_override_keeps_input_and_other_observations_unchanged():
    inspector = MODULE.GoalGridInspector.__new__(MODULE.GoalGridInspector)
    inspector.term = SimpleNamespace(finished=torch.tensor([False]))
    inspector.target = (0.213, 0.573, -0.081)
    obs = torch.arange(57, dtype=torch.float32)[None]
    original = obs.clone()
    result = inspector.actor_observation(obs)
    torch.testing.assert_close(obs, original)
    torch.testing.assert_close(result[:, :10], obs[:, :10])
    torch.testing.assert_close(result[:, 13:], obs[:, 13:])
    torch.testing.assert_close(result[0, 10:13], torch.tensor(inspector.target))
    inspector.term.finished[:] = True
    assert (inspector.actor_observation(obs)[0, 9:13] == 0).all()


def test_tensordict_changes_actor_only():
    TensorDict = pytest.importorskip("tensordict").TensorDict

    inspector = MODULE.GoalGridInspector.__new__(MODULE.GoalGridInspector)
    inspector.term = SimpleNamespace(finished=torch.tensor([False]))
    inspector.target = (0.217, 0.631, 0.071)
    obs = TensorDict({"policy": torch.zeros(1, 57), "critic": torch.zeros(1, 312)}, batch_size=[1])
    result = inspector.actor_observation(obs)
    torch.testing.assert_close(result["policy"][0, 10:13], torch.tensor(inspector.target))
    assert not obs["policy"].any()
    assert not result["critic"].any()
