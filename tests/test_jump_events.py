from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "locolab/source/locomotion/mdp"


def _load_jump_events(monkeypatch):
    class FakeSceneEntityCfg:
        def __init__(self, name: str, **_kwargs):
            self.name = name

    math_module = ModuleType("isaaclab.utils.math")
    math_module.sample_uniform = (
        lambda lower, upper, size, device: (
            (torch.as_tensor(lower, device=device) + torch.as_tensor(upper, device=device)) * 0.5
        )
        .expand(size)
        .clone()
    )
    math_module.quat_from_euler_xyz = lambda roll, pitch, yaw: torch.stack(
        (torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)), dim=-1
    )
    math_module.quat_mul = lambda first, _second: first
    math_module.quat_apply = lambda _quat, vector: vector

    modules = {
        "isaaclab": ModuleType("isaaclab"),
        "isaaclab.utils": ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math_module,
        "isaaclab.assets": ModuleType("isaaclab.assets"),
        "isaaclab.managers": ModuleType("isaaclab.managers"),
    }
    modules["isaaclab.utils"].math = math_module
    modules["isaaclab.assets"].Articulation = object
    modules["isaaclab.managers"].SceneEntityCfg = FakeSceneEntityCfg
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    package = ModuleType("jump_events_under_test")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name, path in (
        (f"{package.__name__}.jump_reference", MDP / "jump_reference.py"),
        (f"{package.__name__}.jump_events", MDP / "jump_events.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.jump_events"]


class _Robot:
    def __init__(self, num_envs: int):
        self.device = "cpu"
        self.find_joints_calls = 0
        self.data = SimpleNamespace(
            default_root_state=torch.zeros((num_envs, 13)),
            default_joint_pos=torch.zeros((num_envs, 16)),
            default_joint_vel=torch.zeros((num_envs, 16)),
            soft_joint_pos_limits=torch.tensor([-3.0, 3.0]).expand(num_envs, 16, 2).clone(),
            soft_joint_vel_limits=torch.full((num_envs, 16), 20.0),
        )
        self.data.default_root_state[:, 2] = 0.42
        self.data.default_root_state[:, 3] = 1.0

    def find_joints(self, names, preserve_order=False):
        assert preserve_order
        self.find_joints_calls += 1
        if len(names) == 4 and all(name.endswith("_foot_joint") for name in names):
            return list(range(12, 16)), list(names)
        return list(range(12)), list(names)

    def write_root_pose_to_sim(self, value, env_ids):
        self.root_pose = value.clone()
        self.last_env_ids = env_ids.clone()

    def write_root_velocity_to_sim(self, value, env_ids):
        self.root_velocity = value.clone()

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        self.joint_position = position.clone()
        self.joint_velocity = velocity.clone()


class _Trajectory:
    duration = 0.8
    metadata = {"target_jump_height": 0.2}
    base_position = torch.zeros((1, 3))

    def __init__(self):
        self.sample_sizes = []

    def sample(self, query):
        self.sample_sizes.append(query.numel())
        count = query.numel()
        return SimpleNamespace(
            base_position=torch.zeros((count, 3)),
            base_linear_velocity=torch.zeros((count, 3)),
            base_pitch=torch.zeros(count),
            base_pitch_rate=torch.zeros(count),
            joint_position=torch.ones((count, 12)),
            joint_velocity=torch.ones((count, 12)),
        )


def _make_env(num_envs: int = 8):
    robot = _Robot(num_envs)

    class Scene:
        env_origins = torch.zeros((num_envs, 3))

        def __getitem__(self, _name):
            return robot

    scene = Scene()
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene), robot


def _reset(
    events,
    env,
    env_ids,
    probability,
    phase_time_range=(0.0, 0.8),
    align_reference_height=False,
    preserve_reference_joint_positions=False,
):
    events.reset_timed_jump_or_flat_state(
        env,
        env_ids,
        jump_probability=probability,
        reference_path="unused.json",
        leg_joint_names=[f"joint_{index}" for index in range(12)],
        pose_range={},
        velocity_range={},
        joint_position_range=(1.0, 1.0),
        joint_velocity_range=(0.0, 0.0),
        phase_time_range=phase_time_range,
        asset_cfg=events.SceneEntityCfg("robot"),
        align_reference_height=align_reference_height,
        preserve_reference_joint_positions=preserve_reference_joint_positions,
    )


def test_timed_jump_reset_interpolates_only_selected_envs_and_caches_joint_ids(monkeypatch) -> None:
    events = _load_jump_events(monkeypatch)
    env, robot = _make_env()
    trajectory = _Trajectory()
    monkeypatch.setattr(events, "load_jump_reference", lambda *_args, **_kwargs: trajectory)

    real_rand = torch.rand
    mask_sampled = False

    def controlled_rand(*size, **kwargs):
        nonlocal mask_sampled
        if not mask_sampled and size == (4,):
            mask_sampled = True
            return torch.tensor([0.1, 0.9, 0.2, 0.9], device=kwargs.get("device"))
        return real_rand(*size, **kwargs)

    monkeypatch.setattr(events.torch, "rand", controlled_rand)
    env_ids = torch.tensor([1, 3, 5, 7])
    _reset(events, env, env_ids, probability=0.5)

    assert trajectory.sample_sizes == [2]
    assert env._jump_episode_mask[env_ids].tolist() == [True, False, True, False]

    mask_sampled = True
    _reset(events, env, env_ids, probability=1.0)
    assert robot.find_joints_calls == 1


def test_timed_jump_buffer_validation_repairs_each_invalid_buffer(monkeypatch) -> None:
    events = _load_jump_events(monkeypatch)
    env, _ = _make_env()
    env._jump_episode_mask = torch.zeros(env.num_envs, dtype=torch.float32)
    env._jump_initial_phase_time = torch.zeros(env.num_envs - 1, dtype=torch.float32)
    env._jump_trigger_position = torch.zeros((env.num_envs, 2), dtype=torch.float64)
    env._jump_target_height = torch.zeros(env.num_envs, dtype=torch.float64)
    env._jump_reset_pending = torch.zeros(env.num_envs, dtype=torch.float32)

    events._ensure_timed_jump_buffers(env, torch.float32)

    assert env._jump_episode_mask.shape == (env.num_envs,)
    assert env._jump_episode_mask.dtype == torch.bool
    assert env._jump_initial_phase_time.shape == (env.num_envs,)
    assert env._jump_trigger_position.shape == (env.num_envs, 3)
    assert env._jump_trigger_position.dtype == torch.float32
    assert env._jump_target_height.dtype == torch.float32
    assert env._jump_reset_pending.dtype == torch.bool


def test_reference_phase_endpoint_accepts_float32_roundoff_not_real_overrun(monkeypatch):
    events = _load_jump_events(monkeypatch)
    env, _ = _make_env()
    trajectory = _Trajectory()
    trajectory.duration = float(torch.tensor(1.4, dtype=torch.float32))
    monkeypatch.setattr(events, "load_jump_reference", lambda *_args, **_kwargs: trajectory)
    ids = torch.arange(env.num_envs)
    _reset(events, env, ids, 1.0, phase_time_range=(0.0, 1.4))
    assert (env._jump_initial_phase_time <= trajectory.duration).all()
    _reset(events, env, ids, 1.0, phase_time_range=(1.4, 1.4))
    assert env._jump_initial_phase_time.eq(trajectory.duration).all()
    with pytest.raises(ValueError, match="phase_time_range"):
        _reset(events, env, ids, 1.0, phase_time_range=(0.0, 1.401))


def test_full_reference_reset_keeps_valid_physical_pose_without_changing_flat_limits(monkeypatch):
    events = _load_jump_events(monkeypatch)
    env, robot = _make_env()
    robot.data.soft_joint_pos_limits[:] = torch.tensor([-0.8, 0.8])
    robot.data.joint_pos_limits = torch.tensor([-1.2, 1.2]).expand(env.num_envs, 16, 2).clone()
    robot.data.default_joint_pos[:] = 1.0
    trajectory = _Trajectory()
    monkeypatch.setattr(events, "load_jump_reference", lambda *args, **kwargs: trajectory)
    ids = torch.arange(env.num_envs)
    _reset(events, env, ids, 1.0, preserve_reference_joint_positions=True)
    torch.testing.assert_close(robot.joint_position[:, :12], torch.ones(env.num_envs, 12))
    _reset(events, env, ids, 0.0, preserve_reference_joint_positions=True)
    torch.testing.assert_close(robot.joint_position, torch.full((env.num_envs, 16), 0.8))


def test_forward_reset_height_matches_actual_reference_and_legacy_reset_is_unchanged(monkeypatch):
    events = _load_jump_events(monkeypatch)
    env, robot = _make_env()
    env.scene.env_origins[:, 2] = 0.20
    trajectory = _Trajectory()
    trajectory.base_position[:, 2] = 0.36
    original_sample = trajectory.sample

    def sample(query):
        result = original_sample(query)
        result.base_position[:, 2] = 0.55
        return result

    trajectory.sample = sample
    monkeypatch.setattr(events, "load_jump_reference", lambda *args, **kwargs: trajectory)
    ids = torch.arange(env.num_envs)
    _reset(events, env, ids, 1.0, align_reference_height=True)
    torch.testing.assert_close(robot.root_pose[:, 2], torch.full((env.num_envs,), 0.75))
    torch.testing.assert_close(env._jump_trigger_position[:, 2], torch.full((env.num_envs,), 0.56))
    _reset(events, env, ids, 1.0)
    torch.testing.assert_close(robot.root_pose[:, 2], torch.full((env.num_envs,), 0.81))


@pytest.mark.parametrize("first_control_pending", [False, True])
def test_reference_wheel_velocity_is_restored_but_flat_reset_stays_zero(monkeypatch, first_control_pending):
    events = _load_jump_events(monkeypatch)
    env, robot = _make_env()
    if first_control_pending:
        robot.data.soft_joint_vel_limits.zero_()
        robot.data.joint_vel_limits = torch.full((env.num_envs, 16), 30.0)
    trajectory = _Trajectory()
    trajectory.metadata = {
        **trajectory.metadata,
        "wheel_joint_names": [p + "_foot_joint" for p in ("FR", "FL", "RR", "RL")],
    }
    original_sample = trajectory.sample

    def sample(query):
        result = original_sample(query)
        result.wheel_position = torch.tensor([[4.0, 5.0, 6.0, 7.0]]).repeat(len(query), 1)
        result.wheel_velocity = torch.tensor([[10.0, 11.0, 12.0, 13.0]]).repeat(len(query), 1)
        return result

    trajectory.sample = sample
    monkeypatch.setattr(events, "load_jump_reference", lambda *args, **kwargs: trajectory)
    ids = torch.arange(env.num_envs)
    _reset(events, env, ids, 1.0)
    torch.testing.assert_close(
        robot.joint_velocity[:, 12:], torch.tensor([[10.0, 11.0, 12.0, 13.0]]).repeat(env.num_envs, 1)
    )
    _reset(events, env, ids, 0.0)
    assert robot.joint_velocity[:, 12:].eq(0).all()


def test_full_base_state_reset_restores_all_angular_axes(monkeypatch):
    events = _load_jump_events(monkeypatch)
    env, robot = _make_env()
    trajectory = _Trajectory()
    original_sample = trajectory.sample
    quat = torch.tensor([[0.99, 0.01, 0.02, 0.03]])
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)

    def sample(query):
        result = original_sample(query)
        result.base_orientation = quat.repeat(len(query), 1)
        result.base_angular_velocity_world = torch.tensor([[0.1, 0.2, 0.3]]).repeat(len(query), 1)
        return result

    trajectory.sample = sample
    monkeypatch.setattr(events.math_utils, "quat_mul", lambda first, second: second)
    monkeypatch.setattr(events, "load_jump_reference", lambda *args, **kwargs: trajectory)
    _reset(events, env, torch.arange(env.num_envs), 1.0)
    torch.testing.assert_close(robot.root_pose[:, 3:7], quat.repeat(env.num_envs, 1))
    torch.testing.assert_close(robot.root_velocity[:, 3:], torch.tensor([[0.1, 0.2, 0.3]]).repeat(env.num_envs, 1))
