"""Check the dormant recipe and keep bulk migration out of active task registration."""

from __future__ import annotations

import ast
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_go2w_m20_mdp import port as _port_fixture

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "locolab/source/locomotion/tasks/Go2W/m20_env_cfg.py"
TASKS = ROOT / "locolab/source/locomotion/tasks/__init__.py"
port = _port_fixture


class _Term:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


class _References:
    def __getattr__(self, name):
        return f"isaac:{name}"


class _TerrainTypes:
    def __getattr__(self, name):
        return _Term


@pytest.fixture
def recipe(port):
    tree = ast.parse(CFG.read_text())
    classes = {
        "Go2WM20CommandsCfg",
        "Go2WM20EventsCfg",
        "Go2WM20RewardsCfg",
        "Go2WM20TerminationsCfg",
        "Go2WM20CurriculumCfg",
    }
    declarations = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name in classes)
        or (
            isinstance(node, ast.Assign)
            and node.targets[0].id in {"LEG_JOINTS", "WHEEL_JOINTS", "DIAGONAL_FEET", "M20_TERRAIN_CFG"}
        )
    ]
    namespace = {
        "configclass": lambda cls: cls,
        "math": math,
        "RewTerm": _Term,
        "EventTerm": _Term,
        "DoneTerm": _Term,
        "CurrTerm": _Term,
        "SceneEntityCfg": _Term,
        "mdp": _References(),
        "m20_events": _References(),
        "lab_mdp": _References(),
        "terrain_gen": _TerrainTypes(),
        "rewards": port.rewards,
        "curriculums": port.curriculums,
        "stand_still_joint_deviation_l1": "isaac:stand_still_joint_deviation_l1",
        "terrain_out_of_bounds": "isaac:terrain_out_of_bounds",
        "M20VelocityCommandCfg": type("CommandCfg", (_Term,), {"Ranges": _Term}),
    }
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(CFG), "exec"), namespace)
    return SimpleNamespace(terrain=namespace["M20_TERRAIN_CFG"], **{name: namespace[name]() for name in classes})


def test_active_reward_set_and_weights_match_pinned_upstream(recipe):
    expected = {
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.02,
        "flat_orientation_l2": -50.0,
        "joint_torques_l2": -2.5e-5,
        "joint_acc_l2": -2e-7,
        "joint_acc_wheel_l2": -1e-7,
        "stand_still": -1.0,
        "hipx_joint_pos_penalty": -3.0,
        "hipy_joint_pos_penalty": -1.5,
        "knee_joint_pos_penalty": -0.75,
        "joint_mirror": -0.03,
        "action_rate_l2": -0.01,
        "action_smooth_l2": -0.025,
        "undesired_contacts": -1.0,
        "contact_forces": -1.5e-4,
        "track_lin_vel_xy_exp": 5.0,
        "track_ang_vel_z_exp": 3.0,
        "feet_air_time_ang_z_M20": 50.0,
        "feet_slide_ang_z_cmd": -2.0,
        "rotation_gait_status": 2.0,
        "rotation_gait_symmetry": 15.0,
        "bad_orientation_penalty": -1000.0,
    }
    actual = {
        name: term.weight for name, term in vars(type(recipe.Go2WM20RewardsCfg)).items() if isinstance(term, _Term)
    }
    assert actual == expected
    assert recipe.Go2WM20RewardsCfg.track_lin_vel_xy_exp.params["std"] ** 2 == pytest.approx(0.5)
    assert recipe.Go2WM20RewardsCfg.stand_still.func == "isaac:stand_still_joint_deviation_l1"


def test_selectors_match_real_go2w_urdf_and_preserve_diagonal_order(recipe):
    model = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    joints = [joint.attrib["name"] for joint in model.findall("joint") if joint.attrib["type"] != "fixed"]
    links = [link.attrib["name"] for link in model.findall("link")]
    cfg = recipe.Go2WM20RewardsCfg

    def matches(patterns, names):
        if isinstance(patterns, str):
            patterns = [patterns]
        return [name for pattern in patterns for name in names if re.fullmatch(pattern, name)]

    assert len(matches(cfg.joint_torques_l2.params["asset_cfg"].joint_names, joints)) == 12
    assert len(matches(cfg.joint_acc_wheel_l2.params["asset_cfg"].joint_names, joints)) == 4
    for name in ("rotation_gait_status", "rotation_gait_symmetry", "feet_air_time_ang_z_M20", "feet_slide_ang_z_cmd"):
        sensor_cfg = getattr(cfg, name).params["sensor_cfg"]
        assert matches(sensor_cfg.body_names, links) == ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]
        assert sensor_cfg.preserve_order
    collisions = matches(cfg.undesired_contacts.params["sensor_cfg"].body_names, links)
    assert "base" in collisions
    assert not any(name.endswith("_foot") for name in collisions)
    for pair in cfg.joint_mirror.params["mirror_joints"]:
        assert len(matches(pair[0], joints)) == len(matches(pair[1], joints)) == 3


def test_termination_curriculum_and_event_wiring(recipe, port):
    terminations = recipe.Go2WM20TerminationsCfg
    assert terminations.bad_orientation.func is port.rewards.bad_orientation
    assert not hasattr(terminations, "base_contact")
    assert terminations.terrain_out_of_bounds.time_out
    curriculum = recipe.Go2WM20CurriculumCfg
    assert curriculum.terrain_levels.func is port.curriculums.terrain_levels
    assert not hasattr(curriculum, "lin_vel_cmd")
    events = recipe.Go2WM20EventsCfg
    assert events.randomize_actuator_gains.mode == "reset"
    assert events.push_robot.mode == "interval"
    assert events.push_robot.interval_range_s == (10.0, 15.0)
    assert events.physics_material.params["num_buckets"] == 1024
    assert events.reset_robot_joints.params == {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
    assert not hasattr(events, "randomize_joint_friction")
    assert events.reset_base.func == "isaac:reset_root_state_with_clearance"
    assert events.reset_base.params["pose_range"]["x"] == (-0.25, 0.25)
    assert events.reset_base.params["minimum_clearance"] == 0.03
    assert events.randomize_inertia.func == "isaac:randomize_mass_consistent_inertia"


def test_terrain_mixture_and_special_command_budget(recipe):
    terrain = recipe.terrain
    assert terrain.size == (8.0, 8.0)
    assert (terrain.num_rows, terrain.num_cols) == (10, 20)
    assert {name: cfg.proportion for name, cfg in terrain.sub_terrains.items()} == {
        "pyramid_stairs": 0.2,
        "pyramid_stairs_inv": 0.2,
        "boxes": 0.2,
        "random_rough": 0.2,
        "hf_pyramid_slope": 0.1,
        "hf_pyramid_slope_inv": 0.1,
    }
    assert terrain.sub_terrains["random_rough"].noise_range == (0.01, 0.16)
    assert terrain.sub_terrains["boxes"].grid_height_range == (0.025, 0.2)
    command = recipe.Go2WM20CommandsCfg.base_velocity
    assert command.ranges.lin_vel_x == (-2.0, 2.0)
    assert command.ranges.lin_vel_y == (-1.0, 1.0)
    assert command.heading_control_stiffness == 0.5
    assert command.rel_heading_envs == 1.0
    assert (
        command.rel_zero_vel_envs,
        command.rel_only_lin_x_envs,
        command.rel_only_lin_y_envs,
        command.rel_only_ang_z_envs,
    ) == (
        0.2,
        0.02,
        0.02,
        0.2,
    )


def test_only_original_rough_tasks_are_registered_after_rollback():
    tree = ast.parse(TASKS.read_text())
    registrations = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "gym.register":
            keywords = {k.arg: k.value for k in node.keywords}
            registrations[ast.literal_eval(keywords["id"])] = ast.unparse(keywords["kwargs"])
    assert not any("-M20-" in name for name in registrations)
    for suffix, runner in (
        ("", "Go2WRoughRnnDecoderPPORunnerCfg"),
        ("-Rnn", "Go2WRoughRnnDecoderPPORunnerCfg"),
        ("-MLP", "Go2WRoughMLPPPORunnerCfg"),
    ):
        spec = registrations[f"OmniBot-Go2W-Rough{suffix}-v0"]
        assert "rough_env_cfg:Go2WRoughEnvCfg" in spec
        assert "rough_env_cfg:Go2WRoughPlayEnvCfg" in spec
        assert runner in spec
    dreamwaq = registrations["OmniBot-Go2W-Rough-DreamWaQ-v0"]
    assert "dreamwaq_env_cfg:Go2WRoughDreamWaQEnvCfg" in dreamwaq
    assert "dreamwaq_env_cfg:Go2WRoughDreamWaQPlayEnvCfg" in dreamwaq
    assert "Go2WRoughDreamWaQPPORunnerCfg" in dreamwaq


def test_migration_inherits_original_go2w_actions_without_overrides():
    tree = ast.parse(CFG.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Go2WM20RoughEnvCfg")
    assert [ast.unparse(base) for base in env_class.bases] == ["Go2WFlatEnvCfg"]
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            assert ast.unparse(node.target) != "actions"
        elif isinstance(node, ast.Assign):
            assert all(ast.unparse(target) != "actions" for target in node.targets)
        elif isinstance(node, ast.Attribute):
            assert ast.unparse(node) != "self.actions"
    flat = ast.parse((CFG.parent / "flat_env_cfg.py").read_text())
    base = next(node for node in flat.body if isinstance(node, ast.ClassDef) and node.name == "Go2WFlatEnvCfg")
    action = next(
        node for node in base.body if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "actions"
    )
    assert ast.unparse(action.value) == "Go2WActionsCfg()"


def test_active_runners_have_no_bulk_m20_overrides():
    path = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"
    tree = ast.parse(path.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert not any(name.startswith("Go2WM20") for name in classes)
    assert {
        "Go2WRoughMLPPPORunnerCfg",
        "Go2WRoughRnnDecoderPPORunnerCfg",
        "Go2WRoughDreamWaQPPORunnerCfg",
    } <= classes


def test_dreamwaq_adds_original_observations_without_changing_the_m20_recipe():
    tree = ast.parse(CFG.read_text())
    for class_name, parent_name in (
        ("Go2WM20RoughDreamWaQEnvCfg", "Go2WM20RoughEnvCfg"),
        ("Go2WM20RoughDreamWaQPlayEnvCfg", "Go2WM20RoughPlayEnvCfg"),
    ):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        assert [ast.unparse(base) for base in cls.bases] == [parent_name]
        declarations = [node for node in cls.body if isinstance(node, ast.AnnAssign)]
        assert [(ast.unparse(node.target), ast.unparse(node.value)) for node in declarations] == [
            ("observations", "Go2WDreamWaQObservationsCfg()")
        ]
    assert "self.observations.policy_history.enable_corruption = False" in ast.unparse(cls)
