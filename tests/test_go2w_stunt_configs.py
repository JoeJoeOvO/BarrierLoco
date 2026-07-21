from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GO2W_TASKS = ROOT / "locolab/source/locomotion/tasks/Go2W"
TERRAINS = ROOT / "locolab/source/locomotion/terrains"


def _load_high_platform_module():
    path = TERRAINS / "high_platform.py"
    spec = importlib.util.spec_from_file_location("_go2w_high_platform", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    if importlib.util.find_spec("trimesh") is not None:
        spec.loader.exec_module(module)
        return module

    trimesh = ModuleType("trimesh")

    class Mesh:
        def __init__(self, extents, center):
            half_extents = 0.5 * np.asarray(extents, dtype=float)
            center = np.asarray(center, dtype=float)
            self.bounds = np.stack((center - half_extents, center + half_extents))

    trimesh.Trimesh = Mesh
    trimesh.creation = SimpleNamespace(box=lambda extents, transform: Mesh(extents, transform))
    trimesh.transformations = SimpleNamespace(translation_matrix=lambda position: position)
    sys.modules["trimesh"] = trimesh
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("trimesh", None)
    return module


def test_discarded_duplicate_stunt_modules_are_absent() -> None:
    assert not (GO2W_TASKS / "handstand_env_cfg.py").exists()
    assert not (GO2W_TASKS / "upright_env_cfg.py").exists()


def test_high_platform_terrain_has_quantized_training_heights_and_fixed_play_sequence() -> None:
    terrain = _load_high_platform_module()
    training_cfg = SimpleNamespace(
        size=(8.0, 8.0),
        platform_height_range=(0.10, 0.80),
        num_height_levels=36,
        platform_start_x=2.5,
        platform_length=1.5,
        platform_width=2.0,
        spawn_x=1.0,
        ground_thickness=1.0,
    )
    heights = []
    for level in range(training_cfg.num_height_levels):
        difficulty = (level + 0.5) / training_cfg.num_height_levels
        meshes, origin = terrain.high_platform_terrain(difficulty, training_cfg)
        heights.append(meshes[1].bounds[1, 2])
        np.testing.assert_allclose(origin, [1.0, 4.0, 0.0])
    np.testing.assert_allclose(heights, np.arange(0.10, 0.801, 0.02), atol=1.0e-8)

    play_cfg = SimpleNamespace(
        size=(24.0, 8.0),
        platform_heights=(0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
        platform_start_x=2.5,
        platform_length=1.5,
        platform_gap=1.5,
        platform_width=2.0,
        spawn_x=1.0,
        ground_thickness=1.0,
    )
    meshes, _ = terrain.high_platform_sequence_terrain(0.0, play_cfg)
    assert len(meshes) == 7
    np.testing.assert_allclose([mesh.bounds[1, 2] for mesh in meshes[1:]], play_cfg.platform_heights)
