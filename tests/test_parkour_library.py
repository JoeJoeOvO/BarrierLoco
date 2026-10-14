"""Promotion requires verified replay evidence; failed motions remain diagnostic."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_parkour_leap_to import PARKOUR, TO, flight

PATH = Path(__file__).resolve().parents[1] / "locolab/scripts/locomotion/solve_parkour_library.py"
SPEC = importlib.util.spec_from_file_location("parkour_library_test", PATH)
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)


def fixture(tmp_path, monkeypatch, accepted=True, criteria_version=3):
    goal = PARKOUR.command_target(criteria_version=criteria_version)
    states = flight()
    if not accepted:
        states["position"][:, 140:150, 2] = 0.18
    for name, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[name] = torch.zeros(1, 361, width)
    states["quaternion"] = torch.tensor([1.0, 0, 0, 0]).repeat(1, 361, 1)
    metrics = PARKOUR.parkour_leap_metrics(states, goal)
    metadata = dict(
        target=asdict(goal),
        leg_effort_limit_nm=[23.5, 23.5, 35.0] * 4,
        leg_velocity_limit_rad_s=[30.0, 30.0, 20.0] * 4,
        leg_action_scale=0.25,
        wheel_action_scale=10.0,
        motion_mode="parkour_leap",
        self_collisions_enabled=True,
        leg_joint_names=TO.LEG_JOINT_NAMES,
        wheel_joint_names=TO.WHEEL_JOINT_NAMES,
        mass_kg=19.523,
        gravity=9.81,
        actuator_contract="shared",
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    for key, filename in (
        ("script_sha256", "solve_jump_target_to.py"),
        ("core_sha256", "jump_target_to.py"),
        ("forward_leap_sha256", "forward_leap_to.py"),
        ("parkour_leap_sha256", "parkour_leap_to.py"),
    ):
        (replay / filename).write_text("unit test source fixture\n")
        metadata[key] = LIB.sha(replay / filename)
    candidate = tmp_path / "candidate.json"
    LIB.write_json(candidate, metadata)
    result = dict(
        metadata,
        independent_replay=True,
        accepted=bool(metrics["accepted"][0]),
        input_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        policy_used=False,
        external_wrenches_used=False,
        production_reference_replaced=False,
        actual_physics_steps=360,
        simulation_dt=0.005,
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in metrics.items()},
    )
    LIB.write_json(replay / "replay.json", result)
    if accepted:
        LIB.write_json(replay / "reference.json", TO.export_reference(result))
    (replay / "replay.mp4").write_bytes(b"mocked video file; ffprobe is mocked in unit tests")
    monkeypatch.setattr(LIB, "video_frame_count", lambda path: 90)
    return goal, candidate, replay


def test_valid_reference_can_be_promoted(tmp_path, monkeypatch):
    goal, candidate, replay = fixture(tmp_path, monkeypatch)
    result = LIB.audit_replay(goal, candidate, replay)
    assert result["accepted"] and result["reference_sha256"]
    assert result["physics_steps"] == 360


def test_goal_first_replay_is_recomputed_and_explicitly_versioned(tmp_path, monkeypatch):
    goal, candidate, replay = fixture(tmp_path, monkeypatch, criteria_version=4)
    result = LIB.audit_replay(goal, candidate, replay)
    assert result["accepted"]
    assert result["criteria_version"] == 4
    assert result["acceptance_profile"] == "flat_goal_first_v4"
    record = json.loads((replay / "replay.json").read_text())
    record["states"]["position"][120][1] = 0.1
    LIB.write_json(replay / "replay.json", record)
    with pytest.raises(ValueError, match="acceptance differs"):
        LIB.audit_replay(goal, candidate, replay)


def test_rejected_motion_is_diagnostic_and_cannot_export_reference(tmp_path, monkeypatch):
    goal, candidate, replay = fixture(tmp_path, monkeypatch, accepted=False)
    result = LIB.audit_replay(goal, candidate, replay)
    assert not result["accepted"] and result["reference_sha256"] is None
    (replay / "reference.json").write_text("{}")
    with pytest.raises(ValueError, match="Rejected motion"):
        LIB.audit_replay(goal, candidate, replay)


@pytest.mark.parametrize("failure", ["target", "source", "input", "clock", "wrench", "scale", "lie"])
def test_bad_evidence_never_promotes(tmp_path, monkeypatch, failure):
    goal, candidate, replay = fixture(tmp_path, monkeypatch)
    path = replay / "replay.json"
    r = json.loads(path.read_text())
    if failure == "target":
        r["target"]["height"] = 0.5
    elif failure == "source":
        (replay / "parkour_leap_to.py").write_text("changed")
    elif failure == "input":
        r["input_sha256"] = "wrong"
    elif failure == "clock":
        r["actual_physics_steps"] = 361
    elif failure == "wrench":
        r["external_wrenches_used"] = True
    elif failure == "scale":
        r["leg_action_scale"] = 0.5
    else:
        r["states"]["position"][140][2] = 0.18
    LIB.write_json(path, r)
    with pytest.raises((ValueError, AssertionError)):
        LIB.audit_replay(goal, candidate, replay)


def test_frame_count_falls_back_to_existing_imageio_ffmpeg(monkeypatch):
    monkeypatch.setattr(LIB.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(count_frames_and_secs=lambda path: (90, 1.8)))
    assert LIB.video_frame_count("fixture.mp4") == 90


def test_frame_count_prefers_ffprobe_when_available(monkeypatch):
    monkeypatch.setattr(LIB.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(LIB.subprocess, "check_output", lambda args: b'{"streams":[{"nb_frames":"90"}]}')
    assert LIB.video_frame_count("fixture.mp4") == 90


def test_incomplete_video_rejected(tmp_path, monkeypatch):
    goal, candidate, replay = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(LIB, "video_frame_count", lambda path: 89)
    with pytest.raises(ValueError, match="frame count"):
        LIB.audit_replay(goal, candidate, replay)


def test_resume_chooses_best_completed_candidate_and_preserves_accepted(tmp_path):
    goal = PARKOUR.command_target()
    for stage, score, accepted in (("batch", 2.0, True), ("refine", 1.0, False)):
        LIB.write_json(tmp_path / stage / "progress.json", {"complete": True})
        LIB.write_json(
            tmp_path / stage / "one/candidate.json",
            dict(target=asdict(goal), metrics=dict(score=score, accepted=accepted)),
        )
    paths = LIB.completed_candidates(tmp_path, [("one", goal)])
    assert paths == [tmp_path / "batch/one/candidate.json"]
    LIB.write_json(tmp_path / "refine/progress.json", {"complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        LIB.completed_candidates(tmp_path, [("one", goal)])


def test_library_selection_matches_feasibility_first_solver(tmp_path):
    goal = PARKOUR.command_target()
    for stage, score, priority in (("batch", 2.0, [1, 0, 0.5, 1, 2]), ("refine", 1.0, [1, 0, 0.7, 1, 1])):
        LIB.write_json(tmp_path / stage / "progress.json", {"complete": True})
        LIB.write_json(
            tmp_path / stage / "one/candidate.json",
            dict(target=asdict(goal), metrics=dict(score=score, accepted=False), search_priority=priority),
        )
    assert LIB.completed_candidates(tmp_path, [("one", goal)]) == [tmp_path / "batch/one/candidate.json"]


def test_search_entries_preserve_per_goal_seeds_and_require_performance_baselines():
    goal = PARKOUR.command_target()
    raw = [
        dict(name="one", target=asdict(goal), seed_candidate="/one/seed.json", performance_baseline="/one/replay.json")
    ]
    assert LIB.search_entries(raw, [("one", goal)], performance=True) == raw
    del raw[0]["performance_baseline"]
    with pytest.raises(ValueError, match="performance_baseline"):
        LIB.search_entries(raw, [("one", goal)], performance=True)


def test_audit_recomputes_performance_guards_and_detects_baseline_changes(tmp_path, monkeypatch):
    goal, candidate, replay_dir = fixture(tmp_path, monkeypatch)
    replay = json.loads((replay_dir / "replay.json").read_text())
    baseline = dict(metrics=replay["metrics"].copy(), input_sha256="baseline fixture")
    cand = json.loads(candidate.read_text())
    cand.update(performance_refinement=True, performance_baseline=baseline)
    LIB.write_json(candidate, cand)
    replay.update(performance_refinement=True, performance_baseline=baseline, input_sha256=LIB.sha(candidate))
    metrics = PARKOUR.performance_refinement_metrics(
        {key: torch.as_tensor(value)[None] for key, value in replay["metrics"].items()}, goal, baseline["metrics"]
    )
    replay["metrics"] = {key: value[0].tolist() for key, value in metrics.items()}
    LIB.write_json(replay_dir / "replay.json", replay)
    LIB.write_json(replay_dir / "reference.json", TO.export_reference(replay))
    assert LIB.audit_replay(goal, candidate, replay_dir)["accepted"]
    replay["performance_baseline"]["metrics"]["height"] += 0.1
    LIB.write_json(replay_dir / "replay.json", replay)
    with pytest.raises(ValueError, match="baseline changed"):
        LIB.audit_replay(goal, candidate, replay_dir)


def test_physical_gap_audit_requires_same_actual_collider(tmp_path, monkeypatch):
    from test_gap_leap_to import GAP, certificate, crossing

    _, candidate, replay_dir = fixture(tmp_path, monkeypatch)
    states, goal = crossing()
    for name, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[name] = torch.zeros(1, 361, width)
    states["quaternion"] = torch.tensor([1.0, 0, 0, 0]).repeat(1, 361, 1)
    (replay_dir / "gap_leap_to.py").write_text("unit test source fixture\n")
    additions = dict(
        motion_mode="gap_leap",
        target=asdict(goal),
        gap_geometry=certificate(goal),
        gap_leap_sha256=LIB.sha(replay_dir / "gap_leap_to.py"),
    )
    cand = json.loads(candidate.read_text())
    cand.update(additions)
    LIB.write_json(candidate, cand)
    replay = json.loads((replay_dir / "replay.json").read_text())
    replay.update(
        additions,
        input_sha256=LIB.sha(candidate),
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in GAP.gap_leap_metrics(states, goal).items()},
    )
    LIB.write_json(replay_dir / "replay.json", replay)
    LIB.write_json(replay_dir / "reference.json", TO.export_reference(replay))
    audit = LIB.audit_replay(goal, candidate, replay_dir)
    assert audit["accepted"] and audit["physical_gap_validated"]
    replay["gap_geometry"]["collision_approximation"] = "convexHull"
    LIB.write_json(replay_dir / "replay.json", replay)
    with pytest.raises(ValueError, match="terrain changed"):
        LIB.audit_replay(goal, candidate, replay_dir)


def test_tilted_ramp_audit_checks_inclined_scene_and_normal_contact_metrics(tmp_path, monkeypatch):
    from test_tilted_ramp_to import RAMP, certificate, landing

    _, candidate, replay_dir = fixture(tmp_path, monkeypatch)
    states, goal = landing()
    for name, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[name] = torch.zeros(1, 361, width)
    additions = dict(motion_mode="tilted_ramp", target=asdict(goal), gap_geometry=certificate(goal))
    for key, filename in (("gap_leap_sha256", "gap_leap_to.py"), ("tilted_ramp_sha256", "tilted_ramp_to.py")):
        (replay_dir / filename).write_text("unit test source fixture\n")
        additions[key] = LIB.sha(replay_dir / filename)
    cand = json.loads(candidate.read_text())
    cand.update(additions)
    LIB.write_json(candidate, cand)
    replay = json.loads((replay_dir / "replay.json").read_text())
    replay.update(
        additions,
        input_sha256=LIB.sha(candidate),
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in RAMP.tilted_ramp_metrics(states, goal).items()},
    )
    LIB.write_json(replay_dir / "replay.json", replay)
    LIB.write_json(replay_dir / "reference.json", TO.export_reference(replay))
    audit = LIB.audit_replay(goal, candidate, replay_dir)
    assert audit["accepted"] and audit["tilted_ramp_validated"] and not audit["physical_gap_validated"]
    (replay_dir / "tilted_ramp_to.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        LIB.audit_replay(goal, candidate, replay_dir)
