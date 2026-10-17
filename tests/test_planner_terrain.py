"""Geometry contracts: traversable spawn/goal, real gaps and deterministic courses."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "locolab/source/locomotion/terrains/planner_course.py"
SPEC = importlib.util.spec_from_file_location("planner_course_under_test", PATH)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


@pytest.mark.parametrize("width", [6.0, 16.0])
@pytest.mark.parametrize("family", CORE.FAMILIES)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_course_geometry(family, difficulty, width):
    layout = CORE.CourseLayout(family, difficulty, size=(16.0, width))
    meshes, origin = CORE.build_course(layout)
    np.testing.assert_allclose(origin, [1.5, width / 2, 0])
    assert layout.surface_height(layout.spawn_x) == layout.surface_height(layout.goal_x) == 0
    assert all(mesh.is_watertight and np.isfinite(mesh.vertices).all() for mesh in meshes)
    for left, right in layout.gaps:
        x = (left + right) / 2
        assert layout.surface_height(x) == -1.0
        # No floor or platform may bridge a gap at z=0. Verify actual boxes.
        covering = [m for m in meshes if m.bounds[0, 0] < x < m.bounds[1, 0]]
        assert max(m.bounds[1, 2] for m in covering) == pytest.approx(-1.0)
    for left, right, height in layout.platforms:
        assert 0.2 <= height <= 0.6 + 1e-9
        assert left - layout.spawn_x > 2.0
        assert layout.goal_x - right > 2.0
    again, _ = CORE.build_course(layout)
    for a, b in zip(meshes, again):
        np.testing.assert_array_equal(a.vertices, b.vertices)


def test_rejects_unsupported_course_contract():
    with pytest.raises(ValueError):
        CORE.CourseLayout("unknown", 0.5)
    with pytest.raises(ValueError):
        CORE.CourseLayout("flat", float("nan"))


def test_rough_descent_is_not_gap_failure():
    from types import SimpleNamespace

    import torch

    spec = importlib.util.spec_from_file_location("planner_mdp_under_test", PATH.parents[1] / "mdp/planner_terrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Scene(dict):
        pass

    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, -1.0]] * 10))))
    scene.env_origins = torch.zeros(10, 3)
    scene.terrain = SimpleNamespace(terrain_types=torch.arange(10))
    env = SimpleNamespace(scene=scene)
    assert module.fell_into_gap(env).tolist() == [False, False, True, True] + [False] * 6
    assert module.goal_distance(env).tolist() == [13.0] * 4 + [6.0] * 6
    scene["robot"].data.root_pos_w[:, 0] = 8.0
    assert module.left_course(env).tolist() == [False] * 4 + [True] * 6
