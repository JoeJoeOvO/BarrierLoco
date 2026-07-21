from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_MODULE = ROOT / "locolab/source/locomotion/mdp/curriculums.py"


def _load_curriculum_module(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)

    spec = importlib.util.spec_from_file_location("_terrain_curriculum", CURRICULUM_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTerrain:
    def __init__(self):
        self.terrain_levels = torch.tensor([0, 3])
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0)))
        self.updates = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.updates.append((env_ids.clone(), move_up.clone(), move_down.clone()))


def _fake_env(common_step_counter: int):
    terrain = _FakeTerrain()
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])))

    class Scene:
        env_origins = torch.zeros(2, 3)

        def __init__(self):
            self.terrain = terrain

        def __getitem__(self, name):
            assert name == "robot"
            return asset

    return SimpleNamespace(
        common_step_counter=common_step_counter,
        scene=Scene(),
        command_manager=SimpleNamespace(get_command=lambda name: torch.ones(2, 3)),
        max_episode_length_s=10.0,
    )


def test_terrain_curriculum_skips_initial_reset(monkeypatch) -> None:
    curriculum = _load_curriculum_module(monkeypatch)
    env = _fake_env(common_step_counter=0)

    level = curriculum.terrain_levels_reward(env, torch.tensor([0, 1]))

    assert level.item() == 1.5
    assert env.scene.terrain.updates == []


def test_terrain_curriculum_updates_after_simulation_starts(monkeypatch) -> None:
    curriculum = _load_curriculum_module(monkeypatch)
    env = _fake_env(common_step_counter=1)

    curriculum.terrain_levels_reward(env, torch.tensor([0, 1]))

    _, move_up, move_down = env.scene.terrain.updates[0]
    torch.testing.assert_close(move_up, torch.tensor([True, False]))
    torch.testing.assert_close(move_down, torch.tensor([False, True]))
