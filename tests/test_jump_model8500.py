"""Contracts for the retained model_8500 recipe and two-stage training."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "locolab/source/locomotion/tasks"
PROFILE = TASKS / "Go2W/jump_env_cfg.py"


def load_stage_settings():
    tree = ast.parse(PROFILE.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_jump_training_stage"
    )
    namespace = {
        "RewTerm": SimpleNamespace,
        "mdp": SimpleNamespace(
            jump_heading_error_l2="heading",
            jump_heading_yaw_velocity_l2="yaw_velocity",
            jump_heading_lateral_velocity_l2="lateral_velocity",
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROFILE), "exec"), namespace)
    return namespace["apply_jump_training_stage"]


def make_cfg():
    return SimpleNamespace(
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(0.2, 0.5))),
            jump_command=SimpleNamespace(heading_hold_duration=0.0, sequence_duration=1.2),
        ),
        events=SimpleNamespace(
            reset_base=SimpleNamespace(params={"jump_probability": 0.2, "phase_time_range": (0.0, 0.8)})
        ),
        rewards=SimpleNamespace(
            track_lin_vel_xy=SimpleNamespace(params={}),
            reference_joint_position=SimpleNamespace(weight=80.0),
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(velocity_commands=SimpleNamespace(params={})),
            critic=SimpleNamespace(velocity_commands=SimpleNamespace(params={})),
        ),
    )


@pytest.mark.parametrize("stage,speed,probability", [("skill", (0.2, 0.5), 0.8), ("speed", (0.2, 1.5), 0.2)])
def test_stages_preserve_jump_recipe_and_scope_direction_constraints(stage, speed, probability):
    cfg = make_cfg()
    original_rewards = cfg.rewards
    load_stage_settings()(cfg, stage)
    assert cfg.training_stage == stage
    assert cfg.commands.base_velocity.ranges.lin_vel_x == speed
    assert cfg.events.reset_base.params["jump_probability"] == probability
    assert cfg.events.reset_base.params["phase_time_range"] == (0.0, 0.8)
    assert cfg.rewards is original_rewards
    assert cfg.rewards.reference_joint_position.weight == 80.0
    enabled = stage == "speed"
    assert cfg.commands.jump_command.heading_hold_duration == (2.0 if enabled else 0.0)
    assert cfg.observations.policy.velocity_commands.params["lock_jump_heading"] == enabled
    assert cfg.observations.critic.velocity_commands.params["lock_jump_heading"] == enabled
    assert cfg.rewards.track_lin_vel_xy.params["lock_jump_heading"] == enabled
    if enabled:
        assert cfg.rewards.jump_heading.weight == -20.0
        assert cfg.rewards.jump_yaw_velocity.weight == -5.0
        assert cfg.rewards.jump_lateral_velocity.weight == -10.0
    else:
        assert cfg.rewards.jump_heading is cfg.rewards.jump_yaw_velocity is cfg.rewards.jump_lateral_velocity is None


def test_restoring_skill_stage_removes_speed_stage_direction_constraints():
    cfg = make_cfg()
    configure = load_stage_settings()
    configure(cfg, "speed")
    configure(cfg, "skill")
    assert cfg.commands.jump_command.heading_hold_duration == 0.0
    assert not cfg.observations.policy.velocity_commands.params["lock_jump_heading"]
    assert cfg.rewards.jump_heading is cfg.rewards.jump_yaw_velocity is cfg.rewards.jump_lateral_velocity is None


def test_forward_heading_hold_covers_the_complete_new_reference_and_recovery():
    cfg = make_cfg()
    cfg.commands.jump_command.sequence_duration = 2.2
    load_stage_settings()(cfg, "speed")
    assert cfg.commands.jump_command.heading_hold_duration == 2.2


def test_unknown_stage_is_rejected_without_mutation():
    cfg = make_cfg()
    with pytest.raises(ValueError):
        load_stage_settings()(cfg, "unknown")
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.2, 0.5)
    assert cfg.events.reset_base.params["jump_probability"] == 0.2


def test_only_retained_jump_task_is_registered():
    tree = ast.parse((TASKS / "__init__.py").read_text())
    registry = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            kwargs = {arg.arg: arg.value for arg in node.keywords}
            if "id" in kwargs and isinstance(kwargs["id"], ast.Constant):
                registry[kwargs["id"].value] = ast.unparse(kwargs["kwargs"])
    assert [name for name in registry if "Jump" in name] == ["OmniBot-Go2W-Jump-MLP-v0"]
    assert "jump_env_cfg:Go2WJumpEnvCfg" in registry["OmniBot-Go2W-Jump-MLP-v0"]
    assert "jump_env_cfg:Go2WJumpPlayEnvCfg" in registry["OmniBot-Go2W-Jump-MLP-v0"]


def test_reference_and_historical_limits_are_preserved():
    reference = ROOT / "locolab/source/locomotion/mdp/reference_motions/go2w_jump_20cm_fast_full_dynamics_to.json"
    assert (
        hashlib.sha256(reference.read_bytes()).hexdigest()
        == "b6503f202e0cbc0c70ca7653629d2b00d427139267f6ad3912d94a5d88ca3f39"
    )
    text = PROFILE.read_text()
    assert 'self.scene.robot.actuators["legs"].effort_limit_sim = 23.5' in text
    assert 'self.scene.robot.actuators["legs"].velocity_limit_sim = 30.0' in text
    assert "lin_vel_x=(0.2, 0.5)" in text
    assert "lin_vel_x = (0.35, 0.35)" in text


def test_cli_enforces_stage_initialization():
    text = (ROOT / "locolab/scripts/locomotion/train.py").read_text()
    assert 'args_cli.jump_stage == "skill" and (args_cli.resume or args_cli.warm_start_rough)' in text
    assert 'args_cli.jump_stage == "speed" and not args_cli.resume' in text
    assert "apply_jump_training_stage(env_cfg, args_cli.jump_stage)" in text
