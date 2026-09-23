"""HighPlatform keeps the Rough baseline and adds only the stumble reward."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "locolab/source/locomotion/tasks/Go2W/high_platform_env_cfg.py"


class _Cfg(SimpleNamespace):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)


class _References:
    def __getattr__(self, name):
        return name


def _class(path: Path, name: str) -> ast.ClassDef:
    return next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name
    )


@pytest.fixture
def rewards():
    namespace = {
        "configclass": lambda cls: cls,
        "RewTerm": _Cfg,
        "SceneEntityCfg": _Cfg,
        "mdp": _References(),
        "m20_rewards": _References(),
        "math": math,
    }
    for path, name in (
        (ROOT / "locolab/source/locomotion/tasks/velocity_env_cfg.py", "LocomotionRewardsCfg"),
        (ROOT / "locolab/source/locomotion/tasks/Go2W/rough_env_cfg.py", "Go2WRoughRewardsCfg"),
        (PLATFORM, "Go2WHighPlatformRewardsCfg"),
    ):
        node = _class(path, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Go2WRoughRewardsCfg"](), namespace["Go2WHighPlatformRewardsCfg"]()


def test_high_platform_inherits_every_rough_reward_without_overrides(rewards):
    rough, platform = rewards
    rough_terms = {name: getattr(rough, name) for name in dir(rough) if isinstance(getattr(rough, name), _Cfg)}
    platform_terms = {
        name: getattr(platform, name) for name in dir(platform) if isinstance(getattr(platform, name), _Cfg)
    }
    assert len(rough_terms) == 29
    assert set(platform_terms) == set(rough_terms) | {"feet_stumble"}
    for name, term in rough_terms.items():
        assert vars(platform_terms[name]) == vars(term), name
    assert platform.stay_alive is None
    assert platform.base_height is None
    assert platform.termination_penalty.weight == -200.0
    assert platform.stand_still.weight == -1.0
    assert platform.stand_still.func == "stand_still_joint_deviation_se2"
    for legacy in ("hip_default", "joint_default", "hip_lr_symmetry"):
        assert not hasattr(platform, legacy)
    node = _class(PLATFORM, "Go2WHighPlatformRewardsCfg")
    assert [ast.unparse(base) for base in node.bases] == ["Go2WRoughRewardsCfg"]
    assert not any(isinstance(item, ast.FunctionDef) for item in node.body)


def test_stumble_uses_the_existing_four_wheel_force_ratio_test(rewards):
    _, platform = rewards
    term = platform.feet_stumble
    assert term.func == "feet_stumble"
    assert term.weight == -0.1
    assert term.params["ratio_threshold"] == 5.0
    assert term.params["sensor_cfg"].name == "contact_forces"
    assert term.params["sensor_cfg"].body_names == ".*_foot"

    path = ROOT / "locolab/source/locomotion/mdp/rewards.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == term.func
    )
    namespace = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": _Cfg, "ContactSensor": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    forces = torch.zeros(5, 8, 3)
    forces[0, 0, 0] = 100.0  # A non-wheel body must not trigger the term.
    forces[1, 1] = torch.tensor([30.0, 40.0, 10.0])  # Exactly the threshold, not above it.
    forces[2, 3] = torch.tensor([51.0, 0.0, 10.0])
    forces[3, 5] = torch.tensor([0.0, 0.0, -10.0])
    forces[4, 7] = torch.tensor([1.0, 0.0, 0.0])
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    result = namespace[term.func](
        env, _Cfg("contact_forces", body_ids=[1, 3, 5, 7]), ratio_threshold=term.params["ratio_threshold"]
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))


def test_training_keeps_rough_interfaces_and_randomization():
    node = _class(PLATFORM, "Go2WHighPlatformEnvCfg")
    assert [ast.unparse(base) for base in node.bases] == ["Go2WRoughEnvCfg"]
    fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
    assert fields == {"scene", "commands", "rewards"}
    post_init = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__")
    assert ast.unparse(post_init.body[0]) == "super().__post_init__()"
    overrides = {
        ast.unparse(target): ast.literal_eval(item.value)
        for item in post_init.body
        if isinstance(item, ast.Assign)
        for target in item.targets
    }
    assert overrides == {
        "self.episode_length_s": 10.0,
        "self.events.reset_base.params['pose_range']": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0.0, 0.0),
        },
    }
    assert "wheel_resistance" not in PLATFORM.read_text()


def test_mlp_runner_inherits_the_latest_rough_runner():
    path = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"
    node = _class(path, "Go2WHighPlatformMLPPPORunnerCfg")
    assert [ast.unparse(base) for base in node.bases] == ["Go2WRoughMLPPPORunnerCfg"]
    post_init = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__")
    assert ast.unparse(post_init.body[0]) == "super().__post_init__()"
    overrides = {
        ast.unparse(target): ast.literal_eval(item.value)
        for item in post_init.body
        if isinstance(item, ast.Assign)
        for target in item.targets
    }
    assert overrides == {
        "self.max_iterations": 5000,
        "self.experiment_name": "omnibot_go2w_highplatform_mlp_v0",
        "self.resume_experiment_name": "omnibot_go2w_rough_mlp_v0",
    }
