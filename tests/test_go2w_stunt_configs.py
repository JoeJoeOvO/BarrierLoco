from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GO2W_TASKS = ROOT / "locolab/source/locomotion/tasks/Go2W"
TERRAINS = ROOT / "locolab/source/locomotion/terrains"
TASK_REGISTRY = ROOT / "locolab/source/locomotion/tasks/__init__.py"
RUNNERS = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"


def _class_names(path: Path) -> set[str]:
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}


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


def test_two_wheel_balance_tasks_share_one_config_module() -> None:
    path = GO2W_TASKS / "two_wheel_balance_env_cfg.py"
    classes = _class_names(path)

    assert {
        "Go2WTwoWheelBalanceCommandsCfg",
        "Go2WTwoWheelBalanceRewardsCfg",
        "Go2WTwoWheelBalanceCurriculumCfg",
        "Go2WTwoWheelBalanceEnvCfg",
        "Go2WHandstandEnvCfg",
        "Go2WHandstandPlayEnvCfg",
        "Go2WUprightEnvCfg",
        "Go2WUprightPlayEnvCfg",
    } <= classes
    assert not (GO2W_TASKS / "handstand_env_cfg.py").exists()
    assert not (GO2W_TASKS / "upright_env_cfg.py").exists()

    source = path.read_text()
    assert "class Go2WTwoWheelBalanceCurriculumCfg" in source
    assert "lin_vel_cmd = None" in source
    assert "ang_vel_cmd = None" in source

    crawl = (GO2W_TASKS / "crawl_env_cfg.py").read_text()
    assert "class Go2WCrawlCurriculumCfg" in crawl
    assert "ang_vel_cmd = None" in crawl


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


def test_high_platform_declares_its_own_reward_profile() -> None:
    path = GO2W_TASKS / "high_platform_env_cfg.py"
    classes = _class_names(path)
    source = path.read_text()

    assert "Go2WHighPlatformRewardsCfg" in classes
    assert "rewards: Go2WHighPlatformRewardsCfg" in source
    assert "self.rewards.joint_default" not in source
    assert "weight=-100.0" in source
    assert "dof_power_l1 = None" not in source


def test_stunt_tasks_are_registered_with_dedicated_runners() -> None:
    registry = TASK_REGISTRY.read_text()
    runners = RUNNERS.read_text()

    expected = {
        "OmniBot-Go2W-Handstand-v0": ("Go2WHandstandEnvCfg", "Go2WHandstandPPORunnerCfg"),
        "OmniBot-Go2W-Upright-v0": ("Go2WUprightEnvCfg", "Go2WUprightPPORunnerCfg"),
        "OmniBot-Go2W-Crawl-v0": ("Go2WCrawlEnvCfg", "Go2WCrawlPPORunnerCfg"),
        "OmniBot-Go2W-HighPlatform-MLP-v0": (
            "Go2WHighPlatformEnvCfg",
            "Go2WHighPlatformMLPPPORunnerCfg",
        ),
    }
    for task_id, (env_cfg, runner_cfg) in expected.items():
        task_block = registry[registry.index(task_id) : registry.index(task_id) + 600]
        assert env_cfg in task_block
        assert runner_cfg in task_block
        assert f"class {runner_cfg}" in runners

    high_platform_runner = runners[runners.index("class Go2WHighPlatformMLPPPORunnerCfg") :]
    assert "class Go2WHighPlatformMLPPPORunnerCfg(Go2WRoughMLPPPORunnerCfg)" in high_platform_runner[:200]
    assert "self.max_iterations = 5000" in high_platform_runner[:800]
    assert 'self.run_name = "highplatform"' in high_platform_runner[:800]
