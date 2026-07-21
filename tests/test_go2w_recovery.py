from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "locolab/source/locomotion/mdp/recovery.py"


class FakeScene(dict):
    def __init__(self, *args, sensors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensors = {} if sensors is None else sensors


def _load_recovery_module():
    spec = importlib.util.spec_from_file_location("go2w_recovery", RECOVERY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_pose_probabilities_are_normalized_in_declared_order() -> None:
    recovery = _load_recovery_module()

    names, probabilities = recovery.normalized_pose_distribution(
        normal_pose_probability=0.5,
        special_poses={"supine": {"probability": 0.25}, "prone": {"probability": 0.25}},
    )

    assert names == ["normal", "supine", "prone"]
    torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.25, 0.25]))


@pytest.mark.parametrize(
    ("normal_probability", "special_poses"),
    [
        (-0.1, {"supine": {"probability": 1.1}}),
        (0.0, {"supine": {"probability": -0.1}}),
        (0.0, {"supine": {"probability": 0.0}}),
    ],
)
def test_recovery_pose_probabilities_reject_invalid_distributions(
    normal_probability: float,
    special_poses: dict[str, dict[str, float]],
) -> None:
    recovery = _load_recovery_module()

    with pytest.raises(ValueError):
        recovery.normalized_pose_distribution(normal_probability, special_poses)


def test_recovery_pose_indices_are_sampled_once_for_the_requested_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    recovery = _load_recovery_module()
    sampled = torch.tensor([2, 0, 1, 2])
    calls: list[tuple[torch.Tensor, int, bool]] = []

    def fake_multinomial(probabilities: torch.Tensor, count: int, replacement: bool) -> torch.Tensor:
        calls.append((probabilities, count, replacement))
        return sampled

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    result = recovery.sample_pose_indices(
        normal_pose_probability=0.5,
        special_poses={"supine": {"probability": 0.25}, "prone": {"probability": 0.25}},
        count=4,
        device="cpu",
    )

    assert result == (["normal", "supine", "prone"], sampled)
    assert len(calls) == 1
    assert calls[0][1:] == (4, True)


def test_normal_recovery_pose_uses_default_joints_while_fallen_pose_may_use_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery = _load_recovery_module()

    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda name: SimpleNamespace(name=name, joint_ids=slice(None))
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.sample_uniform = lambda lower, upper, shape, device: torch.broadcast_to(
        torch.as_tensor(upper, device=device), shape
    ).clone()
    math_utils.quat_from_euler_xyz = lambda roll, pitch, yaw: torch.stack(
        (torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)), dim=-1
    )
    math_utils.quat_mul = lambda orientation, delta: orientation
    utils = ModuleType("isaaclab.utils")
    utils.math = math_utils
    isaaclab = ModuleType("isaaclab")
    isaaclab.managers = managers
    isaaclab.utils = utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    monkeypatch.setattr(
        recovery,
        "sample_pose_indices",
        lambda *args, **kwargs: (["normal", "fallen"], torch.tensor([0, 1])),
    )
    root_state = torch.zeros(2, 13)
    root_state[:, 3] = 1.0
    data = SimpleNamespace(
        default_root_state=root_state,
        default_joint_pos=torch.tensor([[0.20, -0.30], [0.40, -0.50]]),
        default_joint_vel=torch.zeros(2, 2),
        soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]] * 2),
        soft_joint_vel_limits=torch.full((2, 2), 10.0),
    )
    written: dict[str, torch.Tensor] = {}
    asset = SimpleNamespace(
        device="cpu",
        data=data,
        write_root_pose_to_sim=lambda value, **kwargs: None,
        write_root_velocity_to_sim=lambda value, **kwargs: None,
        write_joint_state_to_sim=lambda positions, velocities, **kwargs: written.update(
            positions=positions.clone(), velocities=velocities.clone()
        ),
    )
    scene = FakeScene({"robot": asset})
    scene.env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(scene=scene)
    zero_root_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}

    recovery.reset_robot_state_multi_pose(
        env=env,
        env_ids=torch.tensor([0, 1]),
        normal_pose_probability=0.5,
        normal_pose_range=zero_root_range,
        normal_velocity_range=zero_root_range,
        normal_joint_position_range=(-0.05, 0.05),
        normal_joint_velocity_range=(0.0, 0.0),
        special_poses={
            "fallen": {
                "probability": 0.5,
                "pose_range": zero_root_range,
                "velocity_range": zero_root_range,
                "joint_velocity_range": (0.0, 0.0),
            }
        },
        sample_joint_positions_from_limits=True,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=[0, 1]),
    )

    torch.testing.assert_close(written["positions"][0], torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(written["positions"][1], torch.tensor([1.0, 1.0]))


def test_recovery_rewards_track_orientation_airborne_wheels_and_joint_speed() -> None:
    recovery = _load_recovery_module()
    robot_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
    wheel_cfg = SimpleNamespace(name="contacts", body_ids=[0, 1])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.5]]),
            joint_vel=torch.tensor([[3.0, -1.0], [4.0, -5.0]]),
        )
    )
    contact_history = torch.tensor(
        [
            [[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        ]
    )
    scene = FakeScene(
        {"robot": robot},
        sensors={"contacts": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=contact_history))},
    )
    env = SimpleNamespace(scene=scene)

    torch.testing.assert_close(
        recovery.upright_orientation_reward(env, robot_cfg),
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(recovery.wheel_air_count(env, wheel_cfg, 1.0), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        recovery.recovering_joint_velocity_penalty(env, wheel_cfg, robot_cfg, 2.5),
        torch.tensor([0.5, 0.0]),
    )
