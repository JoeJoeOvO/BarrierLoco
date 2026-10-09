"""CPU contracts for the accepted smooth-TO training reference."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "locolab/source/locomotion/mdp"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REF = load_file("smooth_reference_under_test", MDP / "jump_reference.py")
EXPORT = load_file("smooth_export_under_test", ROOT / "locolab/scripts/locomotion/export_jump_reference.py")


def payload():
    return json.loads(REF.DEFAULT_JUMP_REFERENCE_PATH.read_text())


def test_current_profile_tracks_actual_full_state_and_keeps_legacy_loadable():
    trajectory = REF.load_jump_reference()
    profile = REF.GO2W_JUMP_PROFILE
    assert (profile.takeoff_time, profile.apex_time, profile.landing_time, profile.sequence_duration) == (
        0.38,
        0.62,
        0.88,
        1.8,
    )
    assert trajectory.time.numel() == 91
    assert trajectory.metadata["state_source"] == "independent_replay_actual_simulated_state"
    assert trajectory.metadata["joint_targets_used_for_reward"] is False
    assert trajectory.metadata["validation"]["success"]
    assert not trajectory.metadata["validation"]["robustness_certified"]
    torch.testing.assert_close(trajectory.base_position[0, :2], torch.zeros(2))
    assert float(trajectory.base_position[0, 2]) == pytest.approx(0.40227586)
    sample = trajectory.sample(torch.tensor([0.0, 0.51, 0.62, 1.8, 2.2]))
    assert sample.wheel_velocity.shape == (5, 4)
    assert sample.base_angular_velocity_world.shape == (5, 3)
    assert sample.base_orientation.shape == (5, 4)
    torch.testing.assert_close(sample.base_orientation.norm(dim=-1), torch.ones(5))
    assert sample.wheel_velocity[0].abs().min() > 12.0
    legacy = REF.load_jump_reference(REF.GO2W_LEGACY_JUMP_PROFILE.reference_path)
    assert legacy.duration == pytest.approx(0.8)
    assert legacy.sample(torch.tensor([0.4])).base_orientation is None


def test_quaternion_interpolation_handles_equivalent_antipodal_representations():
    original = payload()
    flipped = copy.deepcopy(original)
    for i in range(1, len(flipped["time"]), 2):
        flipped["base_orientation"][i] = [-v for v in flipped["base_orientation"][i]]
    a = REF.JumpReferenceTrajectory(original)
    b = REF.JumpReferenceTrajectory(flipped)
    query = torch.tensor([0.01, 0.39, 0.61, 1.79])
    qa, qb = a.sample(query).base_orientation, b.sample(query).base_orientation
    torch.testing.assert_close((qa * qb).sum(-1).abs(), torch.ones(4))


def test_export_is_a_rigid_frame_change_not_a_retiming_or_joint_target_substitution():
    p = payload()
    angle = 0.4
    yaw_q = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    states = []
    for i, time in enumerate(p["time"]):
        xyz = EXPORT.rotate_z(p["base_position"][i], angle)
        states.append(
            {
                "time": time,
                "base_position": [xyz[0] + 3.0, xyz[1] - 2.0, xyz[2]],
                "base_orientation_wxyz": EXPORT.quat_multiply(yaw_q, p["base_orientation"][i]),
                "base_linear_velocity": EXPORT.rotate_z(p["base_linear_velocity"][i], angle),
                "base_angular_velocity_world": EXPORT.rotate_z(p["base_angular_velocity_world"][i], angle),
                "joint_position": p["joint_position"][i],
                "joint_velocity": p["joint_velocity"][i],
                "wheel_joint_position": p["wheel_position"][i],
                "wheel_joint_velocity": p["wheel_velocity"][i],
                "wheel_contact": [not (0.38 <= time < 0.88)] * 4,
            }
        )
    replay = {
        "accepted": True,
        "input_sha256": "source",
        "policy_used": False,
        "normal_ground_reset_and_warmup": True,
        "record": {"states": states},
        "metrics": {},
    }
    metadata = {
        "independent_validation_success": True,
        "independent_validation_input_sha256": "source",
        "method": "test",
        "seed_checkpoint_sha256": "seed",
        "vx_command": 1.0,
        "mass_kg": 19.523,
        "time_step": 0.02,
        "self_collisions_enabled": True,
        "leg_effort_limit_nm": 23.5,
        "leg_velocity_limit_rad_s": 30.0,
        "independent_validation_scope": "test",
        "batch_acceptance_count": 1,
        "batch_env_count": 1,
        "leg_joint_names": p["metadata"]["leg_joint_names"],
        "wheel_joint_names": p["metadata"]["wheel_joint_names"],
    }
    converted = EXPORT.convert({"metadata": metadata}, replay, candidate_sha="candidate", replay_sha="replay")
    assert converted["time"] == p["time"]
    assert converted["joint_position"] == p["joint_position"]
    assert converted["joint_velocity"] == p["joint_velocity"]
    for key in ("base_position", "base_linear_velocity", "base_angular_velocity_world", "base_orientation"):
        torch.testing.assert_close(torch.tensor(converted[key]), torch.tensor(p[key]), atol=1e-6, rtol=1e-6)
    replay["input_sha256"] = "wrong"
    with pytest.raises(ValueError, match="does not correspond"):
        EXPORT.convert({"metadata": metadata}, replay, candidate_sha="", replay_sha="")


def test_full_state_loader_rejects_missing_pairs_nonfinite_values_and_bad_quaternions():
    p = payload()
    p.pop("wheel_velocity")
    with pytest.raises(ValueError, match="supplied together"):
        REF.JumpReferenceTrajectory(p)
    p = payload()
    p["joint_velocity"][3][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        REF.JumpReferenceTrajectory(p)
    p = payload()
    p["base_orientation"][3] = [2, 0, 0, 0]
    with pytest.raises(ValueError, match="unit length"):
        REF.JumpReferenceTrajectory(p)
