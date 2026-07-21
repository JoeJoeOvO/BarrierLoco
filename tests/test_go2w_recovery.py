from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "locolab/source/locomotion/mdp/recovery.py"
RECOVER_ENV_PATH = ROOT / "locolab/source/locomotion/tasks/Go2W/recover_env_cfg.py"
TASKS_PATH = ROOT / "locolab/source/locomotion/tasks/__init__.py"
RUNNERS_PATH = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"


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


def test_recovery_task_uses_one_reset_event_and_task_local_self_collision() -> None:
    env_source = RECOVER_ENV_PATH.read_text()
    tasks_source = TASKS_PATH.read_text()
    runner_source = RUNNERS_PATH.read_text()

    assert 'id="OmniBot-Go2W-Recover-DreamWaQ-v0"' in tasks_source
    assert "Go2WRecoverDreamWaQPPORunnerCfg" in runner_source
    assert env_source.count("func=mdp.reset_robot_state_multi_pose") == 1
    assert "reset_base = None" in env_source
    assert "reset_robot_joints = None" in env_source
    assert "self.scene.robot.spawn.self_collision" not in env_source
    assert "self.scene.robot.spawn.articulation_props.enabled_self_collisions = True" in env_source

    runner_block = runner_source[runner_source.index("class Go2WRecoverDreamWaQPPORunnerCfg") :]
    assert "self.max_iterations = 5000" in runner_block[:800]
    assert "class Go2WRecoverDreamWaQPPORunnerCfg(Go2WRoughDreamWaQPPORunnerCfg)" in runner_block[:200]
