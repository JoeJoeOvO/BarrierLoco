from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "locolab/source/locomotion/mdp"
GO2W = ROOT / "locolab/source/locomotion/tasks/Go2W"
REFERENCE = MDP / "reference_motions/go2w_jump_20cm_fast_full_dynamics_to.json"


def test_jump_forward_commands_are_not_masked_as_standing():
    tree = ast.parse((GO2W / "jump_env_cfg.py").read_text())
    command_cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WJumpCommandsCfg")
    velocity = next(
        n.value for n in command_cfg.body if isinstance(n, ast.Assign) and n.targets[0].id == "base_velocity"
    )
    kwargs = {k.arg: k.value for k in velocity.keywords}
    assert ast.literal_eval(kwargs["rel_standing_envs"]) == 0.0
    assert ast.literal_eval(kwargs["heading_command"]) is False
    assert ast.literal_eval(kwargs["limit_ranges"]) is None
    ranges = {k.arg: ast.literal_eval(k.value) for k in kwargs["ranges"].keywords}
    assert ranges == {"lin_vel_x": (0.2, 0.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)}


def _load_jump_rewards(monkeypatch):
    class FakeSceneEntityCfg:
        def __init__(self, name: str, **_kwargs):
            self.name = name

    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = FakeSceneEntityCfg
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)

    package = ModuleType("jump_mdp_under_test")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.jump_rewards",
        MDP / "jump_rewards.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_jump_command(monkeypatch):
    class FakeCommandTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self.metrics = {}
            self.time_left = torch.zeros(self.num_envs, device=self.device)
            self.command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        def reset(self, env_ids=None):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            extras = {}
            for metric_name, metric_value in self.metrics.items():
                extras[metric_name] = torch.mean(metric_value[ids]).item()
                metric_value[ids] = 0.0
            self.command_counter[ids] = 0
            self._resample(ids)
            return extras

        def _resample(self, ids):
            self.time_left[ids] = torch.empty(len(ids)).uniform_(*self.cfg.resampling_time_range)
            self._resample_command(ids)
            self.command_counter[ids] += 1

        def compute(self, dt):
            self._update_metrics()
            self.time_left -= dt
            self._resample((self.time_left <= 0).nonzero().flatten())
            self._update_command()

    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")
    managers.CommandTerm = FakeCommandTerm
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)

    spec = importlib.util.spec_from_file_location(
        "binary_jump_command_under_test",
        MDP / "command/jump_command.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jump_stack_has_no_discarded_experiments() -> None:
    discarded = (
        GO2W / "jump_olympus_env_cfg.py",
        MDP / "command/rsi_jump_command.py",
        MDP / "jump_initialization.py",
    )
    assert not any(path.exists() for path in discarded)
    assert list((MDP / "reference_motions").glob("*.cem_checkpoint.json")) == []
    assert set((MDP / "reference_motions").glob("go2w_jump*.json")) == {
        REFERENCE,
        MDP / "reference_motions/go2w_jump_smooth_transition_full_dynamics_to.json",
    }


def test_dense_ascent_reward_requires_active_scheduled_four_wheel_flight(monkeypatch) -> None:
    rewards = _load_jump_rewards(monkeypatch)
    term = SimpleNamespace(
        active=torch.tensor([False, True, True]),
        is_in_flight=torch.tensor([False, True, True]),
        reference_time=torch.tensor([0.20, 0.40, 0.40]),
        contact_state=torch.tensor(
            [
                [True, True, True, True],
                [False, False, False, False],
                [True, False, False, False],
            ]
        ),
        update_state=lambda: None,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_lin_vel_w=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]))
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda _name: term),
    )

    ascent = rewards.jump_ascent_vertical_velocity_reward(
        env,
        command_name="jump_command",
        ascent_end_time=0.54,
        require_all_wheels_airborne=True,
    )

    torch.testing.assert_close(ascent, torch.tensor([0.0, 2.0, 0.0]))


def test_ascent_shaping_reaches_grounded_robot_only_in_scheduled_ascent(monkeypatch) -> None:
    rewards = _load_jump_rewards(monkeypatch)
    times = torch.tensor([0.20, 0.40, 0.40, 0.54, 0.60, 0.40])
    trajectory = rewards.load_jump_reference(REFERENCE)
    sample = trajectory.sample(times)
    root_pos = sample.base_position.clone()
    root_pos[[1, 2], 2] += 0.10
    root_vel = torch.zeros(6, 3)
    root_vel[:, 2] = 1.0
    term = SimpleNamespace(
        active=torch.tensor([True, True, True, True, True, False]),
        is_in_flight=torch.tensor([False, True, True, True, True, True]),
        reference_time=times,
        trigger_height=torch.full((6,), float(trajectory.base_position[0, 2])),
        contact_state=torch.tensor([[True] * 4, [True] * 4, [False] * 4, [True] * 4, [True] * 4, [True] * 4]),
        update_state=lambda: None,
    )
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos, root_lin_vel_w=root_vel))
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_term=lambda _: term))
    common = dict(command_name="jump_command", ascent_end_time=0.54, require_all_wheels_airborne=False)

    upward = rewards.jump_ascent_vertical_velocity_reward(env, **common)
    height_l2 = rewards.jump_ascent_reference_height_l2(env, reference_path=str(REFERENCE), **common)
    height_tanh = rewards.jump_ascent_reference_height_tanh_reward(
        env, reference_path=str(REFERENCE), std=0.05, **common
    )

    torch.testing.assert_close(upward, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(height_l2, torch.tensor([0.0, 0.01, 0.01, 0.0, 0.0, 0.0]))
    near_target = 1.0 - torch.tanh(torch.tensor(2.0))
    torch.testing.assert_close(height_tanh, torch.tensor([0.0, near_target, near_target, 1.0, 0.0, 0.0]))


def test_to_pose_and_velocity_cover_partial_contact_and_landing_without_overlap(monkeypatch) -> None:
    rewards = _load_jump_rewards(monkeypatch)
    times = torch.tensor([0.0, 0.34, 0.36, 0.50, 0.72, 0.78, 0.80])
    trajectory = rewards.load_jump_reference(REFERENCE)
    sample = trajectory.sample(times)
    term = SimpleNamespace(
        active=torch.tensor([True, True, True, True, True, True, False]),
        reference_time=times,
        # Contact should not remove pose guidance, including delayed takeoff.
        contact_state=torch.tensor([[True, False, True, False]]).expand(7, -1),
        update_state=lambda: None,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=sample.base_position.clone(),
            joint_pos=sample.joint_position.clone(),
            joint_vel=sample.joint_velocity.clone(),
        )
    )
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_term=lambda _name: term))
    kwargs = dict(
        command_name="jump_command",
        reference_path=str(REFERENCE),
        asset_cfg=SimpleNamespace(name="robot", joint_ids=list(range(12))),
    )
    pose = rewards.jump_reference_joint_position_reward(env, std=0.261799, require_all_wheels_airborne=False, **kwargs)
    pre_velocity = rewards.jump_pre_takeoff_reference_joint_velocity_reward(env, takeoff_time=0.36, std=3.0, **kwargs)
    post_velocity = rewards.jump_reference_joint_velocity_reward(env, start_time=0.36, std=3.0, **kwargs)
    expected = term.active.float()
    torch.testing.assert_close(pose, expected)
    torch.testing.assert_close(pre_velocity + post_velocity, expected)
    assert not (pre_velocity.bool() & post_velocity.bool()).any()

    robot.data.joint_pos += 0.4
    robot.data.joint_vel += 4.0
    worse_pose = rewards.jump_reference_joint_position_reward(env, std=0.261799, **kwargs)
    worse_velocity = rewards.jump_reference_joint_velocity_reward(env, start_time=0.36, std=3.0, **kwargs)
    assert (worse_pose[term.active] < pose[term.active]).all()
    assert (worse_velocity[2:6] < post_velocity[2:6]).all()


def test_wheel_speed_penalty_uses_each_actual_contact_not_jump_clock(monkeypatch) -> None:
    rewards = _load_jump_rewards(monkeypatch)
    # Joint storage order differs from sensor storage order on purpose.
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_vel=torch.tensor(
                [
                    [2.0, 100.0, 4.0, 100.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0],
                ]
            )
        )
    )
    forces = torch.zeros(3, 4, 3)
    forces[0, [0, 2], 2] = 10.0
    forces[1, :, 2] = 10.0
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    # No jump term / phase is available: actual airborne wheels are constrained
    # even after the fixed clock ends, while grounded wheels can roll freely.
    env = SimpleNamespace(scene=Scene(robot=robot))
    penalty = rewards.jump_wheel_velocity_l2(
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=[1, 0, 3, 2]),
        sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3]),
    )
    torch.testing.assert_close(penalty, torch.tensor([5.0, 0.0, 13.5]))


def test_landing_cost_excludes_takeoff_and_ballistic_flight_but_covers_late_landing(monkeypatch) -> None:
    rewards = _load_jump_rewards(monkeypatch)
    term = SimpleNamespace(
        reference_time=torch.tensor([0.0, 0.30, 0.60, 0.70, 0.78, 0.80, 0.80]),
        contact_state=torch.tensor(
            [
                [True] * 4,
                [True] * 4,
                [False] * 4,
                [True, False, False, False],
                [True] * 4,
                [True] * 4,
                [True] * 4,
            ]
        ),
        update_state=lambda: None,
    )
    velocity = torch.zeros(7, 3)
    velocity[:, 2] = torch.tensor([-1.0, 2.0, -1.0, -1.0, 0.1, 0.5, -0.2])
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=velocity))},
        command_manager=SimpleNamespace(get_term=lambda _name: term),
    )
    penalty = rewards.jump_landing_vertical_velocity_l2(
        env, command_name="jump_command", descent_start_time=0.54, velocity_tolerance=0.2
    )
    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.0, 0.64, 0.0, 0.09, 0.0]))


def test_jump_posture_config_keeps_action_scale_and_continuous_reward_contract():
    tree = ast.parse((GO2W / "jump_env_cfg.py").read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WJumpRewardsCfg")
    terms = {n.targets[0].id: n.value for n in cfg.body if isinstance(n, ast.Assign)}

    def kwargs(name):
        return {k.arg: k.value for k in terms[name].keywords}

    def params(name):
        node = kwargs(name)["params"]
        return {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}

    assert ast.literal_eval(terms["pre_takeoff_reference_joint_position"]) is None
    assert ast.literal_eval(params("reference_joint_position")["require_all_wheels_airborne"]) is False
    assert ast.literal_eval(kwargs("reference_joint_position")["weight"]) == 80.0
    assert ast.literal_eval(kwargs("reference_joint_velocity")["weight"]) == 20.0
    assert kwargs("hip_default")["func"].attr == "joint_deviation_l1"
    assert ast.literal_eval(kwargs("hip_default")["weight"]) == -0.5
    assert kwargs("thigh_calf_default")["func"].attr == "jump_inactive_joint_deviation_l1"
    assert ast.literal_eval(kwargs("thigh_calf_default")["weight"]) == -0.25
    assert "thigh_default" not in terms and "calf_default" not in terms
    limit_cfg = {k.arg: k.value for k in params("dof_pos_limits")["asset_cfg"].keywords}
    assert limit_cfg["joint_names"].id == "GO2W_REFERENCE_LEG_JOINTS"
    wheel = params("jump_wheel_velocity")
    wheel_asset = {k.arg: ast.literal_eval(k.value) for k in wheel["asset_cfg"].keywords}
    wheel_sensor = {k.arg: ast.literal_eval(k.value) for k in wheel["sensor_cfg"].keywords}
    assert wheel_asset["preserve_order"] and wheel_sensor["preserve_order"]
    assert [name.removesuffix("_joint") for name in wheel_asset["joint_names"]] == wheel_sensor["body_names"]
    assert ast.literal_eval(kwargs("landing_vertical_velocity")["weight"]) < 0
    for name in ("takeoff_upward_velocity", "ascent_height_error", "ascent_height_tanh"):
        assert ast.literal_eval(params(name)["require_all_wheels_airborne"]) is False
    assert ast.literal_eval(kwargs("takeoff_upward_velocity")["weight"]) == 100.0


def test_binary_jump_command_uses_time_for_phase_and_contact_for_outcome(monkeypatch) -> None:
    jump_command = _load_jump_command(monkeypatch)

    class Sensor:
        def __init__(self):
            self.data = SimpleNamespace(net_forces_w=torch.full((1, 4, 3), 10.0))

        def find_bodies(self, _body_names):
            return [0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.42]]),
            root_lin_vel_w=torch.zeros((1, 3)),
        )
    )
    sensor = Sensor()

    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, _name):
            return robot

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        scene=Scene(),
    )
    cfg = SimpleNamespace(
        asset_name="robot",
        contact_sensor_name="contact_forces",
        foot_body_names=".*_foot",
        jump_probability=1.0,
        contact_force_threshold=1.0,
        minimum_jump_height=0.05,
        landing_velocity_threshold=0.2,
        landing_recovery_time=0.04,
        takeoff_time=0.04,
        apex_time=0.06,
        landing_time=0.08,
        sequence_duration=0.12,
        resampling_time_range=(1000.0, 1000.0),
    )
    term = jump_command.BinaryJumpCommand(cfg, env)
    term.reset(torch.tensor([0]))

    for _ in range(2):
        env.common_step_counter += 1
        term.update_state()
    assert term.phase.item() == jump_command.JumpPhase.IN_FLIGHT
    assert not term.takeoff_event.item()  # Scheduled flight is not physical takeoff.

    sensor.data.net_forces_w.zero_()
    robot.data.root_pos_w[:, 2] = 0.47
    env.common_step_counter += 1
    term.update_state()
    assert term.all_wheels_airborne_achieved.item()
    assert term.takeoff_event.item()

    sensor.data.net_forces_w.fill_(10.0)
    robot.data.root_lin_vel_w.zero_()
    for _ in range(3):
        env.common_step_counter += 1
        term.update_state()

    assert term.command.item() == 0.0
    assert term.reference_time.item() == pytest.approx(0.12)
    assert term.metrics["jump_success"].item() == pytest.approx(1.0)

    metrics = term.reset(torch.tensor([0]))
    assert metrics["jump_height"] == pytest.approx(0.05)
    assert metrics["jump_success"] == pytest.approx(1.0)

    term._reset_jump_state(torch.tensor([0]))
    term._activate_schedule(torch.tensor([0]), torch.tensor([0.10]))
    assert term.history_prior_airborne.item()
    assert term.history_prior_apex.item()
    assert term.history_prior_landed.item()
    assert not term.all_wheels_airborne_achieved.item()
    assert not term.stable_landing_achieved.item()


def test_bundled_reference_is_the_validated_fast_full_dynamics_motion() -> None:
    payload = json.loads(REFERENCE.read_text())
    metadata = payload["metadata"]

    assert set(payload) == {
        "metadata",
        "time",
        "base_position",
        "base_linear_velocity",
        "base_pitch",
        "base_pitch_rate",
        "joint_position",
        "joint_velocity",
    }

    assert metadata["task"] == "OmniBot-Go2W-Jump-MLP-v0"
    assert metadata["model"] == "isaac_lab_full_go2w_rigid_body_dynamics"
    assert metadata["validation"]["success"] is True
    assert metadata["sequence_duration"] == 0.8
    assert metadata["flight_start_time"] == 0.36
    assert metadata["apex_time"] == 0.54
    assert metadata["flight_end_time"] == 0.72
    assert max(position[2] for position in payload["base_position"]) - payload["base_position"][0][2] >= 0.18

    spec = importlib.util.spec_from_file_location("jump_reference_under_test", MDP / "jump_reference.py")
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    profile = reference.GO2W_LEGACY_JUMP_PROFILE
    trajectory = reference.JumpReferenceTrajectory.from_json(REFERENCE, profile=profile)
    sample = trajectory.sample(torch.tensor([-1.0, metadata["apex_time"], 2.0]))

    assert profile.reference_path == REFERENCE
    assert profile.takeoff_time == metadata["target_takeoff_time"]
    assert profile.apex_time == metadata["target_apex_time"]
    assert profile.landing_time == metadata["target_landing_time"]
    assert profile.sequence_duration == metadata["sequence_duration"]
    assert profile.leg_joint_names == tuple(metadata["leg_joint_names"])
    assert trajectory.duration == pytest.approx(0.8)
    assert sample._fields[:6] == (
        "base_position",
        "base_linear_velocity",
        "base_pitch",
        "base_pitch_rate",
        "joint_position",
        "joint_velocity",
    )
    assert sample.wheel_position is None and sample.wheel_velocity is None
    assert sample.base_orientation is None and sample.base_angular_velocity_world is None
    torch.testing.assert_close(sample.base_position[0], trajectory.base_position[0])
    torch.testing.assert_close(sample.base_position[-1], trajectory.base_position[-1])

    mismatched_payload = json.loads(REFERENCE.read_text())
    mismatched_payload["metadata"]["target_apex_time"] = profile.apex_time + 0.01
    with pytest.raises(ValueError, match="target_apex_time"):
        reference.JumpReferenceTrajectory(mismatched_payload, profile=profile)
