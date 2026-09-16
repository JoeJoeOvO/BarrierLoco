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
    assert cfg.hip_joint_pos_penalty.func is port.rewards.joint_pos_penalty
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
    assert cfg.stand_still.func == "isaac:stand_still_joint_deviation_l1"
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
    assert sum(isinstance(getattr(cfg, name), _Cfg) for name in dir(cfg)) == 25


def test_rotation_extras_are_disabled_but_support_remains(rough_rewards):
    assert rough_rewards.feet_air_time_ang_z_M20 is None
    assert rough_rewards.feet_slide_ang_z_cmd is None
    assert rough_rewards.rotation_gait_symmetry is None
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
    assert term.func is port.rewards.joint_mirror_unscaled
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
    assert term.func is port.rewards.rotation_gait_status_unscaled
    assert term.weight == 2.0
    assert "use_gait_level" not in term.params
    assert term.params["target_height"] == 0.05
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
