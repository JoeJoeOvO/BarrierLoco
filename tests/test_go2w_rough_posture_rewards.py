"""Selected Rough posture rewards keep baseline control settings and explicit gates."""

from __future__ import annotations

import ast
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_go2w_m20_mdp import _entity, _env
from test_go2w_m20_mdp import port as _port_fixture

ROOT = Path(__file__).resolve().parents[1]
port = _port_fixture


class _Cfg:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


class _References:
    def __getattr__(self, name):
        return f"isaac:{name}"


@pytest.fixture
def rough_rewards(port):
    mdp_refs = _References()
    mdp_refs.penalize_ang_vel_z = port.rewards.common_rewards.penalize_ang_vel_z
    mdp_refs.wheel_air_time_excess = port.rewards.common_rewards.wheel_air_time_excess
    mdp_refs.stand_still_joint_deviation_se2 = port.rewards.common_rewards.stand_still_joint_deviation_se2
    mdp_refs.joint_torques_l2 = port.rewards.lab_mdp.joint_torques_l2
    mdp_refs.action_rate_l2 = port.rewards.lab_mdp.action_rate_l2
    mdp_refs.contact_forces = port.rewards.lab_mdp.contact_forces
    mdp_refs.ActionSmoothnessPenalty = port.rewards.common_rewards.ActionSmoothnessPenalty
    namespace = {
        "configclass": lambda cls: cls,
        "math": math,
        "RewTerm": _Cfg,
        "SceneEntityCfg": _Cfg,
        "mdp": mdp_refs,
        "m20_rewards": port.rewards,
        "stand_still_joint_deviation_l1": "isaac:stand_still_joint_deviation_l1",
    }
    for relative_path, class_name in (
        ("locolab/source/locomotion/tasks/velocity_env_cfg.py", "LocomotionRewardsCfg"),
        ("locolab/source/locomotion/tasks/Go2W/rough_env_cfg.py", "Go2WRoughRewardsCfg"),
    ):
        path = ROOT / relative_path
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Go2WRoughRewardsCfg"]()


def test_selected_rewards_include_wheel_torque_and_replace_old_hip_penalty(rough_rewards, port):
    cfg = rough_rewards
    model = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    names = [joint.get("name") for joint in model.findall("joint") if joint.get("type") != "fixed"]
    selected = [
        name
        for name in names
        if any(re.fullmatch(pattern, name) for pattern in cfg.torques.params["asset_cfg"].joint_names)
    ]
    assert len(selected) == 16
    assert sum("foot" in name for name in selected) == 4
    assert cfg.torques.func is port.rewards.lab_mdp.joint_torques_l2
    assert cfg.torques.weight == -5.0e-4
    assert cfg.orientation.func == "isaac:flat_orientation_l2"
    assert cfg.orientation.weight == -1.0
    assert not hasattr(cfg, "hip_default")
    assert cfg.hip_joint_pos_penalty.func is port.rewards.joint_pos_penalty_relax_side_cmd
    assert cfg.hip_joint_pos_penalty.weight == -1.5
    for name, pattern, weight in (
        ("thigh_joint_pos_penalty", ".*_thigh_joint", -1.0),
        ("calf_joint_pos_penalty", ".*_calf_joint", -0.5),
    ):
        term = getattr(cfg, name)
        assert term.func is port.rewards.joint_pos_penalty_local_terrain
        assert term.weight == weight
        assert term.params["asset_cfg"].joint_names == pattern
        assert term.params["sensor_cfg"].name == "height_scanner"
        assert term.params["lookahead_distance"] == 0.5
        assert term.params["min_valid_rays"] == 10
        for key in ["ang_cmd_threshold", "y_cmd_threshold", "xy_norm_max", "xz_norm_max"]:
            assert term.params[key] == 0.1
    assert cfg.stand_still.func is port.rewards.common_rewards.stand_still_joint_deviation_se2
    assert cfg.stand_still.weight == -1.0
    assert cfg.stand_still.params["command_threshold"] == 0.06
    assert cfg.action_smoothness.weight == -0.002
    assert cfg.action_smoothness.func is port.rewards.common_rewards.ActionSmoothnessPenalty
    assert cfg.action_rate.weight == -0.01
    assert cfg.action_rate.func is port.rewards.lab_mdp.action_rate_l2
    assert cfg.base_height is None
    assert not hasattr(cfg, "feet_stumble")
    assert cfg.contact_forces.func is port.rewards.lab_mdp.contact_forces
    assert cfg.contact_forces.weight == -1.5e-4
    assert cfg.contact_forces.params["threshold"] == 56.6
    assert cfg.contact_forces.params["sensor_cfg"].body_names == ".*_foot"


def test_rough_uses_small_slack_without_alive_or_height_and_preserves_tracking_kernels(rough_rewards):
    cfg = rough_rewards
    assert cfg.track_lin_vel_xy.weight == 3.0
    assert cfg.track_ang_vel_z.weight == 1.5
    assert cfg.track_lin_vel_xy.params["std"] == 0.5
    assert cfg.track_ang_vel_z.params["std"] == 0.5
    assert cfg.track_lin_vel_xy_enhance.weight == 0.5
    assert cfg.track_ang_vel_z_enhance.weight == 0.25
    assert cfg.track_lin_vel_xy_enhance.params["std"] == 0.2
    assert cfg.track_ang_vel_z_enhance.params["std"] == 0.2
    assert cfg.stay_alive is None
    assert cfg.velocity_slack.func == "isaac:velocity_slack"
    assert cfg.velocity_slack.weight == 0.25
    assert cfg.velocity_slack.params == {
        "command_name": "base_velocity",
        "min_ratio": 0.3,
        "max_ratio": 1.5,
        "command_threshold": 0.0,
    }
    assert cfg.base_height is None
    assert sum(isinstance(getattr(cfg, name), _Cfg) for name in dir(cfg)) == 29


def test_rotation_slide_and_duty_enabled_without_airtime(rough_rewards, port):
    assert rough_rewards.feet_air_time_ang_z_M20 is None
    assert rough_rewards.feet_slide_ang_z_cmd.weight == -2.0
    assert rough_rewards.feet_slide_ang_z_cmd.func is port.rewards.feet_slide_ang_z_cmd_unscaled
    assert rough_rewards.rotation_gait_symmetry.weight == 5.0
    assert rough_rewards.rotation_gait_symmetry.func is port.rewards.RotationGaitPerWheelDuty
    assert rough_rewards.rotation_gait_symmetry.params["sensor_cfg"].preserve_order
    assert rough_rewards.rotation_gait_status.weight == 2.0
    assert rough_rewards.joint_mirror.weight == -0.03


def test_dormant_duty_cycle_has_no_gait_dependency_and_preserves_partial_reset(monkeypatch, port):
    def forbidden(*args, **kwargs):
        raise AssertionError("Active reward must not read gait_level")

    monkeypatch.setattr(port.rewards, "gait_level", forbidden)
    env = _env(n=4)
    env.commands[:] = torch.tensor([[0, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2], [0, 0, 0.05]])
    cfg = _Cfg(params={"command_name": "base_velocity"})
    term = port.rewards.RotationGaitSymmetryUnscaled(cfg, env)
    params = {**cfg.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    contact = env.scene.sensors["contact_forces"].data.current_contact_time
    for step in range(250):
        contact[:, :2] = float(step % 2 == 0)
        contact[:, 2:] = float(step % 2 == 1)
        contact[1] = torch.tensor([1, 1, 0, 0])
        result = term(env, **params)
    torch.testing.assert_close(result, torch.tensor([1.0, math.exp(-12.5), 0.0, 0.0]))
    assert not hasattr(env, "_m20_gait_level")
    term.reset([0, 2])
    assert term._count.tolist() == [0, 250, 0, 250]
    assert term._history[[0, 2]].count_nonzero() == 0


def test_wheel_contact_force_uses_peak_excess_instead_of_direction_ratio(rough_rewards):
    env = _env(n=4)
    env._m20_gait_level = torch.tensor(0.5)
    forces = env.scene.sensors["contact_forces"].data.net_forces_w_history
    forces[0, :, 0, 2] = 50.0
    forces[1, :, 0, 0] = 20.0  # Horizontal-only force below the threshold is not penalized.
    forces[2, 1, 0, 2] = 100.0  # A previous-frame peak still counts.
    forces[3, 0, 0, 2] = 80.0
    forces[3, 2, 1, 2] = 120.0  # Excess from separate wheels is added.
    term = rough_rewards.contact_forces
    params = {**term.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    expected = torch.tensor([0.0, 0.0, 43.4, 86.8])
    torch.testing.assert_close(term.func(env, **params), expected)
    env._m20_gait_level.fill_(0.0)
    torch.testing.assert_close(term.func(env, **params), expected)


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_torque_and_action_rate_include_wheels_and_ignore_stale_gait(rough_rewards, scale):
    env = _env(n=2)
    env._m20_gait_level = torch.tensor(scale)
    env.scene["robot"].data.applied_torque[:] = torch.tensor([1.0, 2.0, 3.0, -4.0])
    env.action_manager = SimpleNamespace(
        action=torch.tensor([[1.0, 2.0, 3.0, 4.0]]).repeat(2, 1),
        prev_action=torch.zeros(2, 4),
    )
    torque = rough_rewards.torques.func(env, asset_cfg=_entity("robot", [0, 1, 2, 3]))
    rate = rough_rewards.action_rate.func(env)
    torch.testing.assert_close(torque, torch.full((2,), 30.0))
    env.scene["robot"].data.applied_torque[:, 2:] = 0.0
    legs_only = rough_rewards.torques.func(env, asset_cfg=_entity("robot", [0, 1, 2, 3]))
    torch.testing.assert_close(torque - legs_only, torch.full((2,), 25.0))
    torch.testing.assert_close(rate, torch.full((2,), 30.0))


def test_fixed_smoothness_keeps_history_and_handles_partial_resets(rough_rewards):
    env = _env(n=2)
    env._m20_gait_level = torch.tensor(0.0)
    env.action_manager = SimpleNamespace(
        active_terms=["joint_pos", "joint_vel"],
        action_term_dim=[12, 4],
        action=torch.zeros(2, 16),
        prev_action=torch.zeros(2, 16),
    )
    term_cfg = rough_rewards.action_smoothness
    term = term_cfg.func(term_cfg, env)

    def step(number, current, previous):
        env.common_step_counter = number
        env.episode_length_buf.fill_(number + 1)
        env.action_manager.action.fill_(current)
        env.action_manager.prev_action.fill_(previous)
        return term(env, **term_cfg.params)

    for number, current, previous in [(0, 1.0, 0.0), (1, 3.0, 1.0), (2, 6.0, 3.0)]:
        torch.testing.assert_close(step(number, current, previous), torch.full((2,), 16.0 if number == 2 else 0.0))
    env._m20_gait_level.fill_(0.25)
    torch.testing.assert_close(step(3, 10.0, 6.0), torch.full((2,), 16.0))
    torch.testing.assert_close(term(env, **term_cfg.params), torch.full((2,), 16.0))
    env._m20_gait_level.fill_(0.5)
    torch.testing.assert_close(term(env, **term_cfg.params), torch.full((2,), 16.0))
    torch.testing.assert_close(step(4, 15.0, 10.0), torch.full((2,), 16.0))
    term.reset([0])
    env.common_step_counter = 5
    env.episode_length_buf[:] = torch.tensor([0, 6])
    env.action_manager.action.fill_(21.0)
    env.action_manager.prev_action.fill_(15.0)
    torch.testing.assert_close(term(env, **term_cfg.params), torch.tensor([0.0, 16.0]))


def test_joint_mirror_preserves_upright_factor_and_ignores_stale_gait(rough_rewards, port):
    term = rough_rewards.joint_mirror
    assert term.func is port.rewards.joint_mirror_except_side_cmd
    assert term.weight == -0.03
    model = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    names = [joint.get("name") for joint in model.findall("joint") if joint.get("type") != "fixed"]
    env = _env(n=4)
    env._m20_gait_level = torch.tensor(0.5)
    env.commands[:, 0] = 1.0  # Straight command: this penalty must still apply.
    robot = env.scene["robot"]
    robot.data.joint_pos = torch.zeros(4, len(names))

    def find_joints(pattern):
        ids = [index for index, name in enumerate(names) if re.fullmatch(pattern, name)]
        assert len(ids) == 3 and not any("foot" in names[index] for index in ids)
        return ids, [names[index] for index in ids]

    robot.find_joints = find_joints
    robot.data.joint_pos[:, [i for i, name in enumerate(names) if "foot" in name]] = 1000.0
    robot.data.joint_pos[0, names.index("FL_hip_joint")] = 0.2
    robot.data.joint_pos[0, names.index("RR_hip_joint")] = 0.2
    robot.data.joint_pos[1, names.index("FL_hip_joint")] = 0.2
    robot.data.joint_pos[1, names.index("RR_hip_joint")] = -0.2
    robot.data.joint_pos[2:, names.index("FR_thigh_joint")] = 0.3
    robot.data.joint_pos[2:, names.index("RL_thigh_joint")] = 0.1
    robot.data.projected_gravity_b[3, 2] = 1.0
    result = term.func(env, **term.params)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.08, 0.02, 0.0]))
    env._m20_gait_level.zero_()
    torch.testing.assert_close(term.func(env, **term.params), result)


def test_posture_norm_and_static_multiplier_do_not_depend_on_gait_level(port):
    env = _env(n=7)
    env.scene["robot"].data.joint_pos.fill_(1.0)
    env.scene.terrain.terrain_levels.zero_()
    env.commands[:] = torch.tensor(
        [[0, 0, 0], [0.2, 0, 0], [0, 0, 0.2], [0, 0.2, 0], [0, 0, 0], [0.1, 0, 0], [0.1001, 0, 0]]
    )
    env.scene["robot"].data.root_lin_vel_b[4:6, 0] = torch.tensor([0.6, 0.5])
    result = port.rewards.joint_pos_penalty(env, "base_velocity", _entity("robot", [0, 1, 2, 3]))
    # Four unit deviations have norm 2, not squared norm 4; only static rows get x5.
    torch.testing.assert_close(result, torch.tensor([10.0, 2.0, 2.0, 2.0, 2.0, 10.0, 2.0]))


def test_historical_height_relaxation_uses_origin_relative_height_and_keeps_turn_side_gates(port):
    env = _env(n=7)
    env.scene["robot"].data.joint_pos.fill_(1.0)
    env.commands[:, 0] = 1.0
    env.scene.env_origins[:, 2] = 5.0
    hits = env.scene["height_scanner_base"].data.ray_hits_w
    hits[:, :, 2] = 5.0
    hits[1, :, 2] = 5.07
    hits[2, :, 2] = float("inf")
    hits[3, :, 2] = float("nan")
    hits[3, 0, 2] = 5.07
    hits[4, :, 2] = 4.93
    env.commands[5] = torch.tensor([0.0, 0.0, 0.2])
    env.commands[6] = torch.tensor([0.0, 0.2, 0.0])
    result = port.rewards.joint_pos_penalty_except_turn_side_cmd(
        env, "base_velocity", _entity("robot", [0, 1, 2, 3]), _entity("height_scanner_base")
    )
    torch.testing.assert_close(result, torch.tensor([2.0, 0.2, 2.0, 0.2, 2.0, 0.0, 0.0]))


def test_diagonal_support_has_explicit_pair_order_and_curriculum_weight(rough_rewards, port):
    term = rough_rewards.rotation_gait_status
    assert term.func is port.rewards.rotation_gait_status_per_wheel
    assert term.weight == 2.0
    assert "use_gait_level" not in term.params
    assert term.params["height_difference_std"] == 0.05
    assert "target_height" not in term.params
    assert term.params["lin_vel_threshold"] == 0.5
    assert term.params["ang_vel_threshold"] == 0.05
    for key in ["asset_cfg", "sensor_cfg"]:
        selector = term.params[key]
        assert selector.body_names == ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
        assert selector.preserve_order


def test_diagonal_support_preserves_command_height_and_contact_boundaries(port):
    env = _env(n=10)
    sensor_cfg = _entity("contact_forces", [0, 1, 2, 3])
    asset_cfg = _entity("robot", [0, 1, 2, 3])
    env.commands[:] = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [0.3, 0.0, 0.2],
            [0.6, 0.0, 0.2],
            [0.5, 0.0, 0.2],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
        ]
    )
    contacts = env.scene.sensors["contact_forces"].data.current_contact_time
    contacts[:, :2] = 1.0
    positions = env.scene["robot"].data.body_pos_w
    positions[:, 2:, 2] = 0.06
    positions[6, 2:, 2] = 0.05
    contacts[7] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Preserve the discussed behavior: the higher pair need not be airborne.
    contacts[8] = 1.0
    contacts[9] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    positions[9, :2, 2] = 0.06
    positions[9, 2:, 2] = 0.0
    expected = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    result = port.rewards.rotation_gait_status_unscaled(env, "base_velocity", sensor_cfg, asset_cfg)
    torch.testing.assert_close(result, expected)
    assert not hasattr(env, "_m20_gait_level")
    env._m20_gait_level = torch.tensor(0.25)
    torch.testing.assert_close(
        port.rewards.rotation_gait_status_unscaled(env, "base_velocity", sensor_cfg, asset_cfg), expected
    )


def test_rough_terrain_progresses_without_gait_state_or_reward_scaling(port):
    path = ROOT / "locolab/source/locomotion/tasks/Go2W/rough_env_cfg.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Go2WRoughCurriculumCfg")
    namespace = {
        "configclass": lambda cls: cls,
        "LocomotionCurriculumCfg": object,
        "CurrTerm": _Cfg,
        "mdp": SimpleNamespace(terrain_levels_reward=port.curriculums.terrain_levels_reward),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    curriculum = namespace["Go2WRoughCurriculumCfg"]()
    assert not hasattr(curriculum, "gait_level")
    env = _env(n=2)
    env.scene.terrain.terrain_levels.fill_(1)
    env.commands[:, 2] = 1.0
    env.scene.sensors["contact_forces"].data.current_contact_time[:, :2] = 1.0
    env.scene["robot"].data.body_pos_w[:, 2:, 2] = 0.06
    ids = torch.arange(2)
    sensor_cfg = _entity("contact_forces", [0, 1, 2, 3])
    asset_cfg = _entity("robot", [0, 1, 2, 3])
    for level in [1, 2, 3]:
        assert curriculum.terrain_levels.func(env, ids).item() == level
        assert env.scene.terrain.terrain_levels.tolist() == [level, level]
        reward = port.rewards.rotation_gait_status_unscaled(env, "base_velocity", sensor_cfg, asset_cfg)
        torch.testing.assert_close(reward, torch.ones(2))
        assert not hasattr(env, "_m20_gait_level")
        env.common_step_counter += 1
        env.scene["robot"].data.root_pos_w[:, 0] = 5.0
    assert curriculum.lin_vel_cmd is None and curriculum.ang_vel_cmd is None


def test_stand_still_excludes_turns_translation_and_wheels(rough_rewards):
    commands = [
        [0, 0, 0],
        [0, 0, 0.8],
        [0, 0, -0.8],
        [0, 0.3, 0],
        [0.1, 0, 0],
        [0.03, 0.03, 0.03],
        [0, 0, 0.06],
        [0.06, 0, 0],
    ]
    env = _env(n=len(commands))
    env.commands[:] = torch.tensor(commands)
    env.scene["robot"].data.default_joint_pos[:] = torch.tensor([0.1, 0.8, -1.5, 0.0])
    env.scene["robot"].data.joint_pos[:] = torch.tensor([0.3, 0.5, -1.0, 100.0])
    result = rough_rewards.stand_still.func(env, "base_velocity", _entity("robot", [0, 1, 2]))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_diagonal_support_can_reward_sustained_lift_without_rotation(port):
    """Document the current shaping loophole; this test does not change its behavior."""
    env = _env(n=1)
    env.commands[:, 2] = 0.8
    env.scene.sensors["contact_forces"].data.current_contact_time[:, :2] = 1.0
    env.scene["robot"].data.body_pos_w[:, 2:, 2] = 0.06
    for _ in range(600):
        result = port.rewards.rotation_gait_status_unscaled(
            env, "base_velocity", _entity("contact_forces", [0, 1, 2, 3]), _entity("robot", [0, 1, 2, 3])
        )
        torch.testing.assert_close(result, torch.ones(1))
        env.common_step_counter += 1


def test_active_slide_ignores_gait_and_uses_command_gate(monkeypatch, port):
    def forbidden(*args, **kwargs):
        raise AssertionError("Active slide must not read gait_level")

    monkeypatch.setattr(port.rewards, "gait_level", forbidden)
    env = _env(n=3)
    env.commands[:] = torch.tensor([[0, 0, 0.8], [0, 0.3, 0], [0, 0, 0]])
    env.scene.sensors["contact_forces"].data.net_forces_w_history[:, :, :, 2] = 10
    env.scene["robot"].data.body_lin_vel_w[:, :, 0] = 1
    result = port.rewards.feet_slide_ang_z_cmd_unscaled(
        env, "base_velocity", _entity("contact_forces", [0, 1, 2, 3]), _entity("robot", [0, 1, 2, 3])
    )
    torch.testing.assert_close(result, torch.tensor([4.0, 0.0, 0.0]))


def test_wheel_air_time_excess_penalizes_individual_wheels_before_landing(rough_rewards):
    term = rough_rewards.wheel_air_time_excess
    assert term.weight == -10.0
    assert term.params["max_air_time"] == 0.3
    assert term.params["sensor_cfg"].body_names == ".*_foot"
    env = _env(n=4)
    env.commands[:, 2] = 0.8
    sensor = env.scene.sensors["contact_forces"].data
    # Selected wheels are columns 0..3; the last column is an unrelated airborne body.
    sensor.current_air_time = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 100.0],
            [0.0, 0.2, 0.0, 0.3, 100.0],
            [0.0, 1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 1.5, 100.0],
        ]
    )
    params = {**term.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    torch.testing.assert_close(term.func(env, **params), torch.tensor([0.0, 0.0, 0.7, 1.9]))
    # One grounded wheel cannot cancel its continuously airborne partner.
    sensor.current_air_time[2, 1] = 2.0
    torch.testing.assert_close(term.func(env, **params)[2], torch.tensor(1.7))
    # Emulate the sensor timer being cleared on landing or a partial environment reset.
    sensor.current_air_time[2, 1] = 0.0
    sensor.current_air_time[3] = 0.0
    torch.testing.assert_close(term.func(env, **params), torch.zeros(4))


def test_wheel_air_time_excess_shares_duty_gate_and_ignores_gait(rough_rewards):
    env = _env(n=7)
    env.commands[:] = torch.tensor(
        [
            [0.0, 0.0, 0.8],
            [0.0, 0.0, -0.8],
            [0.3, 0.0, 0.2],
            [0.5, 0.0, 0.2],
            [0.0, 0.0, 0.05],
            [0.0, 0.3, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    env.scene.sensors["contact_forces"].data.current_air_time = torch.ones(7, 4)
    env._m20_gait_level = torch.tensor(0.0)
    params = {**rough_rewards.wheel_air_time_excess.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    torch.testing.assert_close(
        rough_rewards.wheel_air_time_excess.func(env, **params), torch.tensor([2.8, 2.8, 2.8, 0.0, 0.0, 0.0, 0.0])
    )


def test_hip_posture_halved_only_for_pure_lateral_commands(rough_rewards, port):
    env = _env(n=8)
    env.commands[:] = torch.tensor(
        [
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [0.0, 0.0, 0.8],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.3, 0.0],
            [0.0, 0.3, 0.1],
        ]
    )
    env.scene["robot"].data.joint_pos[:] = torch.tensor([0.2, -0.2, 0.2, -0.2])
    selector = _entity("robot", [0, 1, 2, 3])
    term = rough_rewards.hip_joint_pos_penalty
    assert term.params["side_scale"] == 0.5
    original = port.rewards.joint_pos_penalty(env, "base_velocity", selector)
    actual = term.func(env, **{**term.params, "asset_cfg": selector})
    torch.testing.assert_close(actual, original * torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_yaw_tracking_penalty_covers_zero_commands_and_both_turn_directions(rough_rewards):
    term = rough_rewards.penalize_ang_vel_z
    assert term.weight == -1.0
    env = _env(n=8)
    env.commands[:] = torch.tensor(
        [
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, -0.8],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, -0.8],
        ]
    )
    env.scene["robot"].data.root_ang_vel_b = torch.zeros(8, 3)
    env.scene["robot"].data.root_ang_vel_b[:, 2] = torch.tensor([0.8, -0.8, 0.5, -0.5, 0.8, -0.8, 0.6, 0.8])
    # Unrelated roll/pitch angular rates must not affect yaw tracking.
    env.scene["robot"].data.root_ang_vel_b[:, :2] = 10.0
    result = term.weight * term.func(env, **term.params)
    torch.testing.assert_close(result, torch.tensor([-0.64, -0.64, -0.25, -0.25, 0.0, 0.0, -0.04, -2.56]))


def test_yaw_tracking_penalty_keeps_increasing_for_large_errors(rough_rewards):
    env = _env(n=4)
    env.scene["robot"].data.root_ang_vel_b = torch.tensor([[0.0, 0.0, x] for x in [0.0, 0.5, 1.0, 2.0]])
    term = rough_rewards.penalize_ang_vel_z
    torch.testing.assert_close(term.weight * term.func(env, **term.params), torch.tensor([0.0, -0.25, -1.0, -4.0]))


def test_diagonal_mirror_is_disabled_only_for_pure_side_commands(monkeypatch, rough_rewards, port):
    env = _env(n=8)
    env.commands[:] = torch.tensor(
        [[0, 0.3, 0], [0, -0.3, 0], [0, 0.1, 0], [0.1, 0.3, 0], [0, 0.3, 0.1], [0, 0, 0.8], [1, 0, 0], [0, 0, 0]]
    )
    monkeypatch.setattr(port.rewards, "joint_mirror_unscaled", lambda *args: torch.ones(8))
    term = rough_rewards.joint_mirror
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def _per_wheel_height_scene(n):
    env = _env(n=n)
    env.scene["robot"].data.body_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 4, 1)
    env.commands[:, 2] = 0.8
    hits = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.3], [0.0, 1.0, -0.2], [1.0, 1.0, 0.6]]).repeat(n, 1, 1)
    env.scene["height_scanner"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
    pos = env.scene["robot"].data.body_pos_w
    pos.copy_(hits)
    pos[:, :, 2] += 0.086
    pos[:, 2:, 2] += 0.06
    env.scene.sensors["contact_forces"].data.current_contact_time[:, :2] = 1
    return env


def _per_wheel_height_params(rough_rewards):
    return {
        **rough_rewards.rotation_gait_status.params,
        "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3]),
        "asset_cfg": _entity("robot", [0, 1, 2, 3]),
    }


def test_turn_height_rejects_compensation_contact_and_missing_local_ground(rough_rewards):
    env = _per_wheel_height_scene(8)
    pos = env.scene["robot"].data.body_pos_w
    hits = env.scene["height_scanner"].data.ray_hits_w
    contact = env.scene.sensors["contact_forces"].data.current_contact_time
    pos[1, 2:, 2] = hits[1, 2:, 2] + 0.086 + torch.tensor([0.01, 0.12])
    pos[2, 2:, 2] = hits[2, 2:, 2] + 0.086 + torch.tensor([0.06, 0.15])
    contact[3] = 1  # High terrain underneath contacting wheels must not count as a swing.
    env.commands[4] = torch.tensor([0.0, 0.3, 0.0])
    hits[5] = float("nan")
    hits[6, :, :2] += 10.0  # Finite but no nearby ray.
    hits[7, 2] = float("inf")  # One missing swing-wheel ground sample cannot be averaged away.
    result = rough_rewards.rotation_gait_status.func(env, **_per_wheel_height_params(rough_rewards))
    expected = torch.tensor([1.0, math.exp(-4.84), math.exp(-3.24), 0.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(result).all()


def test_turn_height_is_translation_invariant_and_supports_both_pairs_and_yaw_signs(rough_rewards):
    env = _per_wheel_height_scene(3)
    params = _per_wheel_height_params(rough_rewards)
    pos = env.scene["robot"].data.body_pos_w
    hits = env.scene["height_scanner"].data.ray_hits_w
    env.commands[1, 2] = -0.8
    pos[1, :, 2] = hits[1, :, 2] + 0.086
    pos[1, :2, 2] += 0.06
    env.scene.sensors["contact_forces"].data.current_contact_time[1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    pos[2] += torch.tensor([8.0, -6.0, 3.0])
    hits[2] += torch.tensor([8.0, -6.0, 3.0])
    torch.testing.assert_close(rough_rewards.rotation_gait_status.func(env, **params), torch.full((3,), 1.0))
    # Explicit selector order, not raw simulation body order, defines the diagonal pairs.
    permutation = [2, 0, 3, 1]
    pos[:] = pos[:, permutation].clone()
    contact = env.scene.sensors["contact_forces"].data.current_contact_time
    contact[:] = contact[:, permutation].clone()
    params["sensor_cfg"] = _entity("contact_forces", [1, 3, 0, 2])
    params["asset_cfg"] = _entity("robot", [1, 3, 0, 2])
    torch.testing.assert_close(rough_rewards.rotation_gait_status.func(env, **params), torch.full((3,), 1.0))


def test_turn_height_and_duty_keep_command_threshold_boundaries(rough_rewards):
    env = _per_wheel_height_scene(5)
    env.commands[:] = torch.tensor([[0.49, 0, 0.051], [0.5, 0, 0.8], [0, 0, 0.05], [0, 0, -0.051], [0, -0.3, 0]])
    sensor = env.scene.sensors["contact_forces"].data
    sensor.current_air_time = torch.zeros(5, 4)
    cfg = rough_rewards.rotation_gait_symmetry
    duty = cfg.func(cfg, env)
    params = {**cfg.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    for step in range(250):
        sensor.current_contact_time[:, :2] = float(step % 2 == 0)
        sensor.current_contact_time[:, 2:] = float(step % 2 == 1)
        result = duty(env, **params)
    expected_gate = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    torch.testing.assert_close(result, expected_gate)
    sensor.current_contact_time[:] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    height = rough_rewards.rotation_gait_status.func(env, **_per_wheel_height_params(rough_rewards))
    torch.testing.assert_close(height, expected_gate * 1.0)


def test_per_wheel_duty_cannot_hide_a_fixed_airborne_wheel_and_penalizes_timer_mismatch(rough_rewards, port):
    env = _env(n=4)
    env.commands[:, 2] = 0.8
    sensor = env.scene.sensors["contact_forces"].data
    sensor.current_air_time = torch.zeros(4, 4)
    cfg = rough_rewards.rotation_gait_symmetry
    term = cfg.func(cfg, env)
    original = port.rewards.RotationGaitSymmetryUnscaled(cfg, env)
    params = {**cfg.params, "sensor_cfg": _entity("contact_forces", [0, 1, 2, 3])}
    old_params = {k: v for k, v in params.items() if k != "swing_time_std"}
    for step in range(250):
        sensor.current_contact_time[:] = torch.tensor([1, 1, 0, 0] if step % 2 == 0 else [0, 0, 1, 1])
        sensor.current_contact_time[1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        sensor.current_contact_time[2] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sensor.current_air_time.zero_()
        sensor.current_air_time[1] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        sensor.current_air_time[2] = torch.tensor([0.0, 0.0, 0.4, 0.4])
        # Same marginal duty, but one member's swing has a 0.1-second lead.
        sensor.current_air_time[3, 0] = 0.1
        result = term(env, **params)
        old = original(env, **old_params)
    assert old[1].item() == 1.0  # Reproduce the previous group-average loophole.
    torch.testing.assert_close(result, torch.tensor([1.0, math.exp(-112.5), math.exp(-12.5), math.exp(-0.5)]))
    # Even without a timing discrepancy, the fixed per-wheel imbalance loses duty credit.
    sensor.current_air_time.zero_()
    sensor.current_contact_time[:] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = term(env, **params)
    assert result[1].item() == pytest.approx(math.exp(-12.5))
    term.reset([0, 2])
    assert term._count.tolist() == [0, 250, 0, 250]
    assert term._history[[0, 2]].count_nonzero() == 0
    term(env, **params)
    assert term._count.tolist() == [1, 250, 1, 250]
    term.reset()
    assert term._history.count_nonzero() == 0 and term._count.count_nonzero() == 0


def test_height_radius_matches_go2w_collision_and_preserves_weights(rough_rewards):
    model = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    wheels = [link for link in model.findall("link") if link.get("name").endswith("_foot")]
    assert len(wheels) == 4
    for link in wheels:
        radius = float(link.find("collision/geometry/cylinder").get("radius"))
        assert rough_rewards.rotation_gait_status.params["wheel_radius"] == radius
    assert rough_rewards.rotation_gait_status.weight == 2.0
    assert rough_rewards.rotation_gait_symmetry.weight == 5.0
    assert rough_rewards.rotation_gait_symmetry.params["swing_time_std"] == 0.1
    assert rough_rewards.wheel_air_time_excess.weight == -10.0
    assert rough_rewards.wheel_air_time_excess.params["max_air_time"] == 0.3


def test_clearance_accounts_for_left_right_offsets_and_tilted_cylinder(rough_rewards):
    env = _per_wheel_height_scene(1)
    data = env.scene["robot"].data
    hits = env.scene["height_scanner"].data.ray_hits_w
    theta = math.pi / 6
    # Same +30-degree rotation about X on both sides moves their collision
    # centers in opposite Z directions because the center offsets have opposite signs.
    data.body_quat_w[:] = torch.tensor([math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0])
    params = _per_wheel_height_params(rough_rewards)
    offsets = torch.tensor(params["wheel_center_offsets"])
    extent = 0.086 * math.cos(theta) + 0.045 / 2 * math.sin(theta)
    data.body_pos_w[..., 2] = hits[..., 2] + extent - offsets * math.sin(theta)
    data.body_pos_w[:, 2:, 2] += 0.06
    actual = rough_rewards.rotation_gait_status.func(env, **params)
    torch.testing.assert_close(actual, torch.tensor([1.0]))
    model = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    for index, name in enumerate(["FL_foot", "RR_foot", "FR_foot", "RL_foot"]):
        collision = model.find(f"link[@name='{name}']/collision")
        assert float(collision.find("geometry/cylinder").get("length")) == params["wheel_width"]
        xyz = [float(v) for v in collision.find("origin").get("xyz").split()]
        assert xyz == [0.0, params["wheel_center_offsets"][index], 0.0]


def test_swing_height_consistency_has_no_absolute_target_or_high_lift_cost(rough_rewards):
    env = _per_wheel_height_scene(7)
    pos = env.scene["robot"].data.body_pos_w
    ground = env.scene["height_scanner"].data.ray_hits_w[..., 2]
    # All equal positive clearances get full credit, even below 5 cm or well above it.
    heights = torch.tensor(
        [[0.01, 0.01], [0.03, 0.03], [0.05, 0.05], [0.15, 0.15], [0.30, 0.30], [0.02, 0.07], [0.17, 0.22]]
    )
    pos[:, 2:, 2] = ground[:, 2:] + 0.086 + heights
    result = rough_rewards.rotation_gait_status.func(env, **_per_wheel_height_params(rough_rewards))
    torch.testing.assert_close(result, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, math.exp(-1), math.exp(-1)]))
    # The same height difference gets the same score at different absolute heights.
    torch.testing.assert_close(result[5], result[6])
    pos[0, 2, 2] = ground[0, 2] + 0.086 - 0.01
    result = rough_rewards.rotation_gait_status.func(env, **_per_wheel_height_params(rough_rewards))
    assert result[0].item() == 0.0  # Scan-estimated penetration cannot count as a valid swing.
