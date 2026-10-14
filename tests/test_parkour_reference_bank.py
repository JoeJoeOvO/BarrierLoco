"""Real archived teachers: IDs, phase sampling, provenance and terrain contracts."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parkour_bank_test", ROOT / "locolab/source/locomotion/mdp/parkour_reference.py"
)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


@pytest.fixture(scope="module")
def bank():
    return CORE.ParkourReferenceBank()


def test_seven_ids_truthful_approval_and_fixed_contract(bank):
    assert len(bank.names) == 7
    assert sum(e["strict_accepted"] for e in bank.entries) == 2
    assert sum(e["selection_status"] == "user_reviewed" for e in bank.entries) == 2
    assert bank.descriptors.shape == (7, 3)
    assert (bank.times[:, 0] < bank.times[:, 1]).all()
    assert (bank.times[:, 1] < bank.times[:, 2]).all()
    assert len(set(bank.times[:, 0].tolist())) > 1
    assert bank.leg_names[0] == "FR_hip_joint" and bank.wheel_names[-1] == "RL_foot_joint"


@pytest.mark.parametrize("time", [0.0, 0.2025, 0.40, 0.65, 1.0, 1.8, 2.0, 2.2])
def test_sampling_each_motion_and_finite_shapes(bank, time):
    ids = torch.arange(14) % 7
    sample = bank.sample(ids, torch.full((14,), time))
    for value in sample.values():
        assert value.shape[0] == 14 and torch.isfinite(value).all()
        torch.testing.assert_close(value[:7], value[7:])
    torch.testing.assert_close(sample["quaternion"].norm(dim=-1), torch.ones(14))
    assert set(sample["contacts"].unique().tolist()) <= {0.0, 1.0}


def test_initial_state_changes_only_xy_origin_not_height_or_pose(bank):
    sample = bank.sample(torch.arange(7), torch.zeros(7))
    torch.testing.assert_close(sample["position"][:, :2], torch.zeros(7, 2))
    for i, entry in enumerate(bank.entries):
        original = entry["replay"]["states"]
        assert sample["position"][i, 2] == torch.tensor(original["position"][0][2])
        torch.testing.assert_close(sample["joint_position"][i], torch.tensor(original["joint_position"][0]))
        torch.testing.assert_close(sample["quaternion"][i], torch.tensor(original["quaternion"][0]))


def test_cmd_describes_measured_touchdown_not_physical_gap_width(bank):
    for i, entry in enumerate(bank.entries):
        measured = entry["replay"]["metrics"]
        torch.testing.assert_close(bank.descriptors[i], torch.tensor([measured["height"], *measured["landing_xy"]]))
    assert bank.descriptors[3, 1] > 0.8  # 40 cm gap is NOT a 40 cm base displacement


def test_rolling_exit_continues_but_ramp_terminal_pose_is_held(bank):
    ids = torch.arange(7)
    start = bank.sample(ids, torch.full((7,), 1.8))
    end = bank.sample(ids, torch.full((7,), 2.2))
    assert (end["position"][:5, 0] > start["position"][:5, 0]).all()
    torch.testing.assert_close(end["position"][5:], start["position"][5:])
    torch.testing.assert_close(end["velocity"][5:], torch.zeros(2, 3))
    torch.testing.assert_close(end["joint_velocity"], torch.zeros(7, 12))
    torch.testing.assert_close(end["omega"], torch.zeros(7, 3))
    torch.testing.assert_close(end["wheel_velocity"][5:], torch.zeros(2, 4))
    assert end["contacts"].all()


def test_geometry_preserves_recorded_gaps_and_cross_slopes(bank):
    for i, entry in enumerate(bank.entries):
        mesh = CORE.terrain_mesh(entry)
        assert mesh.is_watertight
        if i < 3:
            assert mesh.bounds[1, 2] == 0
            continue
        target = entry["replay"]["target"]
        near = target["near_x"] - bank.origins[i, 0].item()
        points = mesh.triangles_center
        top = mesh.face_normals[:, 2] > 0.5
        # No positive-Z top face covers the missing strip.
        in_gap = (points[:, 0] > near + 0.001) & (points[:, 0] < near + target["width"] - 0.001)
        assert not ((points[:, 2] > -0.5) & top & in_gap).any()
        if i >= 5:
            assert (mesh.face_normals[top, 1] > 0.3).any()
            assert (mesh.face_normals[top, 1] < -0.3).any()


def test_bad_id_or_nonfinite_time_rejected(bank):
    with pytest.raises(ValueError):
        bank.sample(torch.tensor([7]), torch.zeros(1))
    with pytest.raises(ValueError):
        bank.sample(torch.tensor([0]), torch.tensor([float("nan")]))


def test_speed_layout_and_global_reset_mixture(bank):
    ids = torch.arange(100000)
    motion, rolling = CORE.environment_layout(ids, 3)
    assert rolling.float().mean() == pytest.approx(0.3)
    assert (motion[rolling] < 3).all()  # physically flat, never an unrequested pit
    assert motion[:10].tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    old, no_rolling = CORE.environment_layout(ids)
    torch.testing.assert_close(old, ids % 7)
    assert not no_rolling.any()
    torch.manual_seed(42)
    mode, _ = CORE.sample_reset_phases(bank, motion, (5 / 7, 1 / 7, 0.5 / 7, 0.25 / 7, 0.25 / 7))
    reference = (mode != 0) & ~rolling
    assert reference.float().mean() == pytest.approx(0.20, abs=0.004)


@pytest.mark.parametrize("slots", [-1, 4, 1.5])
def test_speed_layout_rejects_bad_slot_count(slots):
    with pytest.raises(ValueError):
        CORE.environment_layout(torch.arange(10), slots)


def test_missing_training_authorization_rejected(tmp_path):
    with gzip.open(CORE.BANK_PATH, "rt") as stream:
        payload = json.load(stream)
    payload["training_use_authorized"] = False
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(payload, stream)
    with pytest.raises(ValueError, match="explicitly selected"):
        CORE.ParkourReferenceBank(path)


def test_reset_curriculum_preserves_phase_zero_and_limits_recovery_weight(bank):
    torch.manual_seed(7)
    ids = torch.arange(70000) % 7
    mode, time = CORE.sample_reset_phases(bank, ids)
    frequency = torch.bincount(mode, minlength=5).float() / len(mode)
    torch.testing.assert_close(frequency, torch.tensor([0.30, 0.40, 0.15, 0.10, 0.05]), atol=0.008, rtol=0)
    assert (time[mode <= 1] == 0).all()
    assert (time[mode == 2] < bank.times[ids[mode == 2], 0]).all()
    assert (time[mode == 3] > bank.times[ids[mode == 3], 0]).all()
    assert (time[mode == 3] < bank.times[ids[mode == 3], 2]).all()
    assert (time[mode == 4] > bank.times[ids[mode == 4], 2]).all()
    assert (time <= bank.duration).all()


@pytest.mark.parametrize("weights", [(1.0, 0.0, 0.0), (0.3, 0.4, 0.15, 0.1, 0.2), (0.3, 0.4, 0.15, -0.1, 0.25)])
def test_invalid_reset_curricula_rejected(bank, weights):
    with pytest.raises(ValueError):
        CORE.sample_reset_phases(bank, torch.arange(7), weights)
