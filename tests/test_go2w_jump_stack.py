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
TASK_REGISTRY = ROOT / "locolab/source/locomotion/tasks/__init__.py"
RUNNERS = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"


def _class_names(path: Path) -> set[str]:
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}


def _function_names(path: Path) -> set[str]:
    return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


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

        def reset(self, env_ids=None):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            extras = {}
            for metric_name, metric_value in self.metrics.items():
                extras[metric_name] = torch.mean(metric_value[ids]).item()
                metric_value[ids] = 0.0
            self._resample_command(ids)
            return extras

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


def test_jump_stack_has_one_canonical_profile_and_no_discarded_experiments() -> None:
    jump_cfg = GO2W / "jump_env_cfg.py"
    pretrain_cfg = GO2W / "jump_flat_pretrain_env_cfg.py"
    command_cfg = MDP / "command/jump_command_cfg.py"
    events = MDP / "jump_events.py"
    rewards = MDP / "jump_rewards.py"

    assert {
        "Go2WJumpEnvCfg",
        "Go2WJumpPlayEnvCfg",
        "Go2WJumpRewardsCfg",
    } <= _class_names(jump_cfg)
    assert {
        "Go2WJumpFlatPretrainCommandsCfg",
        "Go2WJumpFlatPretrainEnvCfg",
        "Go2WJumpFlatPretrainPlayEnvCfg",
    } <= _class_names(pretrain_cfg)

    cfg_source = jump_cfg.read_text()
    assert "Olympus" not in cfg_source
    assert "weight=100.0" in cfg_source
    assert "weight=-200.0" in cfg_source
    assert "penalize_lin_vel_xy = None" in cfg_source
    assert "class Go2WJumpCurriculumCfg" in cfg_source
    assert "lin_vel_cmd = None" in cfg_source
    assert "ang_vel_cmd = None" in cfg_source
    assert '"require_all_wheels_airborne": True' in cfg_source
    assert "jump_reference_base_position_reward" not in cfg_source
    assert "jump_reference_contact_reward" not in cfg_source
    assert "jump_apex_height_reward" not in cfg_source
    assert "jump_landing_quality_reward" not in cfg_source

    command_cfg_source = command_cfg.read_text()
    assert "takeoff_velocity_threshold" not in command_cfg_source
    assert "maximum_command_time" not in command_cfg_source
    assert _function_names(events) == {
        "_ensure_timed_jump_buffers",
        "_sample_ranges",
        "_stratified_phase_times",
        "reset_timed_jump_or_flat_state",
    }
    assert "rsi" not in events.read_text().lower()

    assert _function_names(rewards) == {
        "_gate_to_current_four_wheel_flight",
        "_jump_ascent_reference_height_error",
        "_jump_command_term",
        "_masked_gaussian",
        "_positive_std",
        "_reference_context",
        "jump_angular_velocity_l2",
        "jump_ascent_reference_height_l2",
        "jump_ascent_reference_height_tanh_reward",
        "jump_ascent_vertical_velocity_reward",
        "jump_attitude_l2",
        "jump_inactive_base_height_l2",
        "jump_inactive_joint_deviation_l1",
        "jump_inactive_lin_vel_z_l2",
        "jump_inactive_track_ang_vel_z_exp",
        "jump_pre_takeoff_reference_joint_position_reward",
        "jump_pre_takeoff_reference_joint_velocity_reward",
        "jump_reference_joint_position_reward",
        "jump_wheel_velocity_l2",
    }

    discarded = (
        GO2W / "jump_olympus_env_cfg.py",
        MDP / "command/rsi_jump_command.py",
        MDP / "jump_initialization.py",
    )
    assert not any(path.exists() for path in discarded)
    assert list((MDP / "reference_motions").glob("*.cem_checkpoint.json")) == []
    assert list((MDP / "reference_motions").glob("go2w_jump*.json")) == [REFERENCE]


def test_jump_tasks_use_one_canonical_runner_tree() -> None:
    registry = TASK_REGISTRY.read_text()
    runners = RUNNERS.read_text()

    jump_block = registry[registry.index("OmniBot-Go2W-Jump-MLP-v0") :]
    assert "Go2W.jump_env_cfg:Go2WJumpEnvCfg" in jump_block[:600]
    assert "Go2WJumpMLPPPORunnerCfg" in jump_block[:600]

    pretrain_block = registry[registry.index("OmniBot-Go2W-Jump-Flat-Pretrain-MLP-v0") :]
    assert "Go2WJumpFlatPretrainEnvCfg" in pretrain_block[:700]
    assert "Go2WJumpFlatPretrainMLPPPORunnerCfg" in pretrain_block[:700]

    assert "class Go2WJumpMLPPPORunnerCfg(Go2WFlatMLPPPORunnerCfg)" in runners
    assert "class Go2WJumpFlatPretrainMLPPPORunnerCfg(Go2WJumpMLPPPORunnerCfg)" in runners
    assert 'self.experiment_name = "omnibot_go2w_jump_olympus_mlp_v0"' in runners
    assert "Olympus" not in registry
    assert "Olympus" not in runners


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
    assert term.takeoff_event.item()

    sensor.data.net_forces_w.zero_()
    robot.data.root_pos_w[:, 2] = 0.47
    env.common_step_counter += 1
    term.update_state()
    assert term.all_wheels_airborne_achieved.item()

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
    spec.loader.exec_module(reference)
    trajectory = reference.JumpReferenceTrajectory.from_json(REFERENCE)
    sample = trajectory.sample(torch.tensor([-1.0, metadata["apex_time"], 2.0]))

    assert trajectory.duration == pytest.approx(0.8)
    assert sample._fields == (
        "base_position",
        "base_linear_velocity",
        "base_pitch",
        "base_pitch_rate",
        "joint_position",
        "joint_velocity",
    )
    torch.testing.assert_close(sample.base_position[0], trajectory.base_position[0])
    torch.testing.assert_close(sample.base_position[-1], trajectory.base_position[-1])
