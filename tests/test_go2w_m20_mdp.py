from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_go2w_action_smoothness import _load_rewards_module

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "locolab/source/locomotion/mdp"


def _load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def port(monkeypatch):
    common = _load_rewards_module(monkeypatch)
    # Use tensor-backed adapters only for Isaac's runtime objects; load the real migrated functions.
    lab_mdp = ModuleType("isaaclab.envs.mdp")
    lab_mdp.joint_torques_l2 = (
        lambda env, asset_cfg: env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids].square().sum(1)
    )
    lab_mdp.action_rate_l2 = lambda env: (env.action_manager.action - env.action_manager.prev_action).square().sum(1)
    lab_mdp.contact_forces = lambda env, threshold, sensor_cfg: (
        env.scene.sensors[sensor_cfg.name]
        .data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        .norm(dim=-1)
        .amax(dim=1)
        .sub(threshold)
        .clamp(min=0.0)
        .sum(dim=1)
    )

    class UniformVelocityCommand:
        def __init__(self, cfg, env):
            self.cfg, self.device, self.num_envs = cfg, env.device, env.num_envs
            self.vel_command_b = torch.zeros(env.num_envs, 3)
            self.heading_target = torch.zeros(env.num_envs)
            self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool)
            self.is_heading_env = torch.zeros(env.num_envs, dtype=torch.bool)

        def reset(self, env_ids):
            assert not isinstance(env_ids, slice)
            self._resample_command(env_ids)
            return {}

        def _resample_command(self, env_ids):
            for axis, limits in enumerate(
                (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)
            ):
                self.vel_command_b[env_ids, axis] = torch.empty(len(env_ids)).uniform_(*limits)
            self.heading_target[env_ids] = torch.empty(len(env_ids)).uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = True
            self.is_standing_env[env_ids] = False

    lab_mdp.UniformVelocityCommand = UniformVelocityCommand
    lab_mdp.UniformVelocityCommandCfg = type("UniformVelocityCommandCfg", (), {})
    envs = ModuleType("isaaclab.envs")
    envs.mdp = lab_mdp
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", lab_mdp)
    sys.modules["isaaclab"].envs = envs
    sys.modules["isaaclab.utils"].configclass = lambda cls: cls

    parent = ModuleType("_m20_port")
    parent.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, "_m20_port", parent)
    monkeypatch.setitem(sys.modules, "_m20_port.rewards", common)
    _load_module(monkeypatch, "_m20_port.curriculums", MDP / "curriculums.py")
    package = ModuleType("_m20_port.m20")
    package.__path__ = [str(MDP / "m20")]
    monkeypatch.setitem(sys.modules, "_m20_port.m20", package)
    modules = {
        name: _load_module(monkeypatch, f"_m20_port.m20.{name}", MDP / "m20" / f"{name}.py")
        for name in ("curriculums", "rewards", "commands")
    }
    return SimpleNamespace(**modules)


def _env(n=4, dt=0.02):
    data = SimpleNamespace(
        joint_pos=torch.zeros(n, 4),
        default_joint_pos=torch.zeros(n, 4),
        applied_torque=torch.ones(n, 4),
        root_lin_vel_b=torch.zeros(n, 3),
        root_lin_vel_w=torch.zeros(n, 3),
        root_pos_w=torch.zeros(n, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
        body_pos_w=torch.zeros(n, 4, 3),
        body_lin_vel_w=torch.zeros(n, 4, 3),
    )
    contact_data = SimpleNamespace(
        current_contact_time=torch.zeros(n, 4),
        last_air_time=torch.zeros(n, 4),
        net_forces_w_history=torch.zeros(n, 3, 4, 3),
    )
    landed = torch.zeros(n, 4, dtype=torch.bool)
    sensor = SimpleNamespace(data=contact_data, compute_first_contact=lambda dt: landed)
    terrain = SimpleNamespace(
        terrain_levels=torch.full((n,), 3),
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0))),
    )
    updates = []

    def update_origins(env_ids, up, down):
        updates.append((env_ids.clone(), up.clone(), down.clone()))
        terrain.terrain_levels[env_ids] += up.long() - down.long()

    terrain.update_env_origins = update_origins

    class Scene(dict):
        pass

    scene = Scene(
        robot=SimpleNamespace(data=data),
        contact_forces=sensor,
        height_scanner_base=SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.zeros(n, 9, 3))),
    )
    scene.sensors = {"contact_forces": sensor}
    scene.env_origins = torch.zeros(n, 3)
    scene.terrain = terrain
    commands = torch.zeros(n, 3)
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=dt,
        scene=scene,
        common_step_counter=0,
        max_episode_length_s=20.0,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        commands=commands,
        landed=landed,
        updates=updates,
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )


def _entity(name, ids=None):
    return SimpleNamespace(name=name, joint_ids=ids, body_ids=ids)


def test_gait_scaling_and_terrain_curriculum_are_environment_local(port):
    levels = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 8.0, float("nan"), float("inf")])
    expected = torch.tensor([0.0, 0.0, math.exp(-2), math.exp(-1), 1.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(port.curriculums.gait_level_from_mean(levels), expected)
    first, second = _env(), _env()
    second.scene.terrain.terrain_levels.fill_(1)
    port.curriculums.terrain_levels(first, torch.arange(4))
    assert first.updates == []
    assert port.curriculums.gait_level(first).item() == 1.0
    assert port.curriculums.gait_level(second).item() == pytest.approx(math.exp(-2))
    first.common_step_counter = 1
    first.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([5.0, 1.0, 0.0, 0.0])
    first.commands[:, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    port.curriculums.terrain_levels(first, torch.arange(4))
    _, up, down = first.updates[0]
    assert up.tolist() == [True, False, False, False]
    assert down.tolist() == [False, True, False, False]
    assert port.curriculums.gait_level(second).item() == pytest.approx(math.exp(-2))


def _command_cfg(**overrides):
    cfg = SimpleNamespace(
        heading_command=True,
        rel_zero_vel_envs=0.2,
        rel_only_lin_y_envs=0.02,
        rel_only_lin_x_envs=0.02,
        rel_only_ang_z_envs=0.2,
        ranges=SimpleNamespace(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
        ),
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def test_command_slots_partial_resets_and_original_ranges(port, monkeypatch):
    cfg = _command_cfg()
    command = port.commands.M20VelocityCommand(cfg, _env(n=7))
    command.vel_command_b.fill_(9.0)
    cfg.ranges.lin_vel_x = (-0.01, 0.01)
    cfg.ranges.lin_vel_y = (-0.01, 0.01)
    # Zero, lateral, forward, yaw, normal. Keep environment 0 and 6 untouched.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.1, 0.21, 0.23, 0.3, 0.9]))
    torch.manual_seed(4)
    command._resample_command([1, 2, 3, 4, 5])
    assert command.is_standing_env.tolist() == [False, True, False, False, False, False, False]
    assert command.is_heading_env.tolist() == [False, False, False, False, True, True, False]
    torch.testing.assert_close(command.vel_command_b[[0, 6]], torch.full((2, 3), 9.0))
    torch.testing.assert_close(command.vel_command_b[1], torch.zeros(3))
    assert abs(command.vel_command_b[2, 1]) > 0.01
    assert abs(command.vel_command_b[3, 0]) > 0.01
    assert command.vel_command_b[2, [0, 2]].count_nonzero() == 0
    assert command.vel_command_b[3, [1, 2]].count_nonzero() == 0
    assert command.vel_command_b[4, :2].count_nonzero() == 0
    assert command.vel_command_b[5, :2].count_nonzero() == 0
    assert abs(command.heading_target[4]) <= math.pi
    command._resample_command([])


@pytest.mark.parametrize(
    "overrides",
    [
        {"rel_zero_vel_envs": -0.1},
        {"rel_zero_vel_envs": float("nan")},
        {"rel_zero_vel_envs": 0.9},
        {"heading_command": False},
    ],
)
def test_invalid_command_mixtures_fail_early(port, overrides):
    with pytest.raises(ValueError):
        port.commands.M20VelocityCommand(_command_cfg(**overrides), _env())


def test_command_reset_accepts_manager_all_and_slice_resets(port):
    command = port.commands.M20VelocityCommand(_command_cfg(), _env())
    command.vel_command_b.fill_(9.0)
    command.reset(slice(1, 3))
    torch.testing.assert_close(command.vel_command_b[[0, 3]], torch.full((2, 3), 9.0))
    command.reset()
    assert (command.vel_command_b.abs() <= 2.0).all()


def test_posture_gates_and_invalid_height_rays(port):
    env = _env(n=6)
    env.scene["robot"].data.joint_pos.fill_(1.0)
    env.commands[:] = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]])
    hits = env.scene["height_scanner_base"].data.ray_hits_w
    hits[2, :, 2] = 0.07
    hits[3, :, 2] = float("inf")
    value = port.rewards.joint_pos_penalty_except_turn_side_cmd(
        env,
        "base_velocity",
        _entity("robot", [0, 1, 2, 3]),
        _entity("height_scanner_base"),
    )
    torch.testing.assert_close(value, torch.tensor([0.0, 0.0, 0.2, 2.0, 2.0, 10.0]))


def test_bad_orientation_penalty_and_termination_agree_at_threshold(port):
    env = _env()
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor(
        [
            [0.7, 0.0, -0.714],
            [0.701, 0.0, -0.713],
            [0.0, 0.0, 1.0],
            [0.0, -0.8, -0.6],
        ]
    )
    expected = torch.tensor([False, True, True, True])
    torch.testing.assert_close(port.rewards.bad_orientation(env), expected)
    torch.testing.assert_close(port.rewards.bad_orientation_penalty(env), expected.float())


def test_air_time_history_does_not_leak_across_partial_resets(port):
    env = _env(n=2)
    env.commands[:, 2] = 1.0
    asset_cfg, sensor_cfg = _entity("robot", [0, 1, 2, 3]), _entity("contact_forces", [0, 1, 2, 3])
    term = port.rewards.FeetAirTime(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

    def call():
        return term(env, "base_velocity", sensor_cfg, asset_cfg)

    torch.testing.assert_close(call(), torch.zeros(2))
    env.scene["robot"].data.body_pos_w[:, :, 2] = 0.1
    call()
    term.reset([0])
    env.scene["robot"].data.body_pos_w[:, :, 2] = 0.0
    sensor = env.scene.sensors["contact_forces"].data
    sensor.current_contact_time.fill_(0.02)
    sensor.last_air_time.fill_(0.4)
    env.landed.fill_(True)
    torch.testing.assert_close(call(), torch.tensor([0.0, 0.8]))
    term.reset()
    torch.testing.assert_close(call(), torch.zeros(2))


def test_rotation_status_and_duty_cycle_reset(port):
    env = _env(n=2, dt=0.5)
    env.commands[:, 2] = 1.0
    sensor_cfg, asset_cfg = _entity("contact_forces", [0, 1, 2, 3]), _entity("robot", [0, 1, 2, 3])
    contacts = env.scene.sensors["contact_forces"].data.current_contact_time
    contacts[:, :2] = 1.0
    env.scene["robot"].data.body_pos_w[:, 2:, 2] = 0.1
    torch.testing.assert_close(
        port.rewards.rotation_gait_status(env, "base_velocity", sensor_cfg, asset_cfg), torch.ones(2)
    )
    env.commands[1, 0] = 1.0
    torch.testing.assert_close(
        port.rewards.rotation_gait_status(env, "base_velocity", sensor_cfg, asset_cfg), torch.tensor([1.0, 0.0])
    )
    env.commands[1, 0] = 0.0
    term = port.rewards.RotationGaitSymmetry(SimpleNamespace(params={}), env)
    for step in range(10):
        contacts[:] = torch.tensor([1, 1, 0, 0]) if step % 2 else torch.tensor([0, 0, 1, 1])
        value = term(env, "base_velocity", sensor_cfg)
    torch.testing.assert_close(value, torch.ones(2))
    term.reset([0])
    contacts[:] = torch.tensor([0, 0, 1, 1])
    value = term(env, "base_velocity", sensor_cfg)
    assert value[0] < 0.1
    assert value[1] == 1.0
    term.reset()
    assert term._count.count_nonzero() == 0
    assert term._history.count_nonzero() == 0


def test_contact_force_and_sliding_use_gait_and_command_gates(port):
    env = _env()
    env._m20_gait_level = torch.tensor(0.5)
    sensor_cfg, asset_cfg = _entity("contact_forces", [0, 1, 2, 3]), _entity("robot", [0, 1, 2, 3])
    forces = env.scene.sensors["contact_forces"].data.net_forces_w_history
    forces[:, :, :, 2] = 110.0
    torch.testing.assert_close(port.rewards.contact_forces(env, 100.0, sensor_cfg), torch.full((4,), 20.0))
    env.commands[:, 2] = 1.0
    env.commands[1, 0] = 1.0
    env.scene["robot"].data.body_lin_vel_w[:, :, 0] = 2.0
    env.scene["robot"].data.root_lin_vel_w[:, 0] = 1.0
    value = port.rewards.feet_slide_ang_z_cmd(env, "base_velocity", sensor_cfg, asset_cfg)
    torch.testing.assert_close(value, torch.tensor([2.0, 0.0, 2.0, 2.0]))
