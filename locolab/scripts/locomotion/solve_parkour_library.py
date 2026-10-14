"""Bounded multi-command parkour TO, with a fresh rendered replay for EVERY goal.

Failed candidates remain diagnostic videos. Only independently accepted,
hash-checked references enter catalog.json. No RL or production writes occur.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from gap_leap_to import GapLeapTarget, gap_leap_metrics, validate_gap_geometry
from jump_target_to import export_reference, parse_target_batch
from parkour_leap_to import candidate_priority, command_target, parkour_leap_metrics, performance_refinement_metrics
from tilted_ramp_to import TiltedRampTarget, tilted_ramp_metrics, validate_ramp_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def video_frame_count(path):
    if shutil.which("ffprobe"):
        probe = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "stream=nb_frames", "-of", "json", str(path)]
            )
        )
        return int(probe["streams"][0]["nb_frames"])
    # The existing Isaac video writer already depends on imageio-ffmpeg and
    # its bundled executable. No system package/stack installation is needed.
    import imageio_ffmpeg

    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    return frames


def completed_candidates(output, goals):
    output = Path(output)
    stages = [output / "batch"]
    if (output / "refine").exists():
        stages.append(output / "refine")
    for stage in stages:
        if not json.loads((stage / "progress.json").read_text()).get("complete"):
            raise ValueError(f"Search is incomplete: {stage}")
    chosen = []
    for name, goal in goals:
        choices = []
        for stage in stages:
            path = stage / name / "candidate.json"
            value = json.loads(path.read_text())
            if value["target"] != asdict(goal):
                raise ValueError(f"Stored command differs: {name}")
            choices.append((candidate_priority(value), path))
        chosen.append(min(choices, key=lambda item: item[0])[1])
    return chosen


def search_entries(raw, goals, performance=False):
    """Keep each command's own control seed and fixed replay baseline."""
    entries = []
    for original, (name, goal) in zip(raw, goals):
        entry = dict(name=name, target=asdict(goal))
        for key in ("seed_candidate", "performance_baseline"):
            if original.get(key):
                entry[key] = original[key]
            elif performance:
                raise ValueError(f"Performance refinement requires {key} for {name}")
        entries.append(entry)
    return entries


def audit_replay(goal, candidate_path, replay_dir):
    """Artifact evidence, not process exit code, decides whether to promote."""
    candidate_path, replay_dir = Path(candidate_path), Path(replay_dir)
    candidate = json.loads(candidate_path.read_text())
    replay = json.loads((replay_dir / "replay.json").read_text())
    if candidate["target"] != asdict(goal) or replay["target"] != asdict(goal):
        raise ValueError("Candidate/replay target differs from requested command")
    if replay["input_sha256"] != sha(candidate_path) or not replay["independent_replay"]:
        raise ValueError("Replay provenance mismatch")
    if replay["policy_used"] or replay["external_wrenches_used"] or replay["production_reference_replaced"]:
        raise ValueError("Expected actuator-only offline dynamics")
    if replay["leg_effort_limit_nm"] != [23.5, 23.5, 35.0] * 4:
        raise ValueError("Unexpected effort limits")
    if replay["leg_velocity_limit_rad_s"] != [30.0, 30.0, 20.0] * 4:
        raise ValueError("Unexpected velocity limits")
    if replay["leg_action_scale"] != 0.25 or replay["wheel_action_scale"] != 10.0:
        raise ValueError("Go2W action scales changed")
    is_ramp = isinstance(goal, TiltedRampTarget)
    is_gap = isinstance(goal, GapLeapTarget) and not is_ramp
    mode = "tilted_ramp" if is_ramp else ("gap_leap" if is_gap else "parkour_leap")
    if replay["motion_mode"] != mode or not replay["self_collisions_enabled"]:
        raise ValueError("Unexpected model mode/collision settings")
    for key, filename in (
        ("script_sha256", "solve_jump_target_to.py"),
        ("core_sha256", "jump_target_to.py"),
        ("forward_leap_sha256", "forward_leap_to.py"),
        ("parkour_leap_sha256", "parkour_leap_to.py"),
    ):
        if replay[key] != sha(replay_dir / filename) or candidate[key] != replay[key]:
            raise ValueError("Solver source changed between optimization and replay")
    if is_gap or is_ramp:
        if (
            replay["gap_leap_sha256"] != sha(replay_dir / "gap_leap_to.py")
            or candidate["gap_leap_sha256"] != replay["gap_leap_sha256"]
        ):
            raise ValueError("Physical-gap source changed between optimization and replay")
        if candidate["gap_geometry"] != replay["gap_geometry"]:
            raise ValueError("Physical terrain changed between optimization and replay")
        (validate_ramp_geometry if is_ramp else validate_gap_geometry)(replay["gap_geometry"], goal)
    if is_ramp and (
        replay["tilted_ramp_sha256"] != sha(replay_dir / "tilted_ramp_to.py")
        or candidate["tilted_ramp_sha256"] != replay["tilted_ramp_sha256"]
    ):
        raise ValueError("Tilted-ramp source changed between optimization and replay")
    if replay["actual_physics_steps"] != len(replay["states"]["position"]) - 1 or replay["actual_physics_steps"] != 360:
        raise ValueError("Recorded time does not match actual physics steps")
    states = {k: torch.tensor(v)[None] for k, v in replay["states"].items()}
    if not all(torch.isfinite(value).all() for value in states.values()):
        raise ValueError("Non-finite replay state")
    metric_function = tilted_ramp_metrics if is_ramp else (gap_leap_metrics if is_gap else parkour_leap_metrics)
    metrics = metric_function(states, goal, replay["simulation_dt"])
    if replay.get("performance_refinement"):
        if (
            not candidate.get("performance_refinement")
            or candidate["performance_baseline"] != replay["performance_baseline"]
        ):
            raise ValueError("Performance baseline changed between search and replay")
        metrics = performance_refinement_metrics(metrics, goal, replay["performance_baseline"]["metrics"])
    if metrics["accepted"].item() != replay["accepted"]:
        raise ValueError("Recomputed acceptance differs")
    for key, value in metrics.items():
        torch.testing.assert_close(value[0], torch.as_tensor(replay["metrics"][key]), atol=1e-4, rtol=1e-4)
    reference = replay_dir / "reference.json"
    if replay["accepted"]:
        if export_reference(replay) != json.loads(reference.read_text()):
            raise ValueError("Reference differs from actual independently replayed states")
    elif reference.exists():
        raise ValueError("Rejected motion must not be exported")
    video = replay_dir / "replay.mp4"
    if video_frame_count(video) != 90:
        raise ValueError("Unexpected frame count")
    return dict(
        accepted=replay["accepted"],
        metrics=replay["metrics"],
        requested=asdict(goal),
        candidate_sha256=sha(candidate_path),
        replay_sha256=sha(replay_dir / "replay.json"),
        video_sha256=sha(video),
        reference_sha256=sha(reference) if replay["accepted"] else None,
        physics_steps=360,
        actual_state_samples=361,
        video_frames=90,
        criteria_version=goal.criteria_version,
        acceptance_profile=(
            "flat_goal_first_v4" if not (is_gap or is_ramp) and goal.criteria_version == 4 else "original"
        ),
        robustness_certified=False,
        physical_gap_validated=bool(is_gap and replay["accepted"]),
        tilted_ramp_validated=bool(is_ramp and replay["accepted"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--seed_candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--candidates_per_goal", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--refine_iterations", type=int, default=20)
    parser.add_argument("--performance_refinement", action="store_true")
    parser.add_argument("--sparse_refinement", action="store_true")
    parser.add_argument(
        "--replay_only", action="store_true", help="Resume auditing/recording completed searches without rerunning TO"
    )
    args = parser.parse_args()
    raw_targets = json.loads(args.targets.read_text())
    goals = parse_target_batch(raw_targets, command_target)
    if args.output.exists() and not args.replay_only:
        parser.error("Output must be a new directory")
    if args.candidates_per_goal < 4 or args.iterations < 1 or args.refine_iterations < 0:
        parser.error("Invalid population/iteration budget")
    if args.sparse_refinement and args.candidates_per_goal < 6:
        parser.error("Sparse refinement needs at least six candidates per goal")
    frozen = args.output / "solver"
    entries = search_entries(raw_targets, goals, args.performance_refinement)
    targets_path = args.output / "targets.json"
    source_files = (
        "solve_jump_target_to.py",
        "jump_target_to.py",
        "forward_leap_to.py",
        "parkour_leap_to.py",
        "gap_leap_to.py",
        "tilted_ramp_to.py",
    )
    if args.replay_only:
        report = json.loads((args.output / "library.json").read_text())
        if report["targets"] != entries:
            parser.error("Resume requires the original targets")
        for filename in source_files:
            if sha(frozen / filename) != sha(Path(__file__).with_name(filename)):
                parser.error("Resume must use the original frozen solver/metric sources")
        report.setdefault("audit_recovery_history", []).append(report["entries"])
        report.update(entries=[], accepted_count=0, status="running", stage="replay recovery")
        shutil.copyfile(Path(__file__), args.output / "driver_replay_recovery.py")
    else:
        args.output.mkdir(parents=True)
        frozen.mkdir()
        for filename in (*source_files, Path(__file__).name):
            shutil.copyfile(Path(__file__).with_name(filename), frozen / filename)
        write_json(targets_path, entries)
        report = dict(
            status="running",
            stage="batch",
            targets=entries,
            target_count=len(goals),
            accepted_count=0,
            entries=[],
            commands=[],
            seed_candidate_sha256=sha(args.seed_candidate),
            production_reference_replaced=False,
            rl_training_started=False,
        )
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1] / "lib"), PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    base = [sys.executable, "-u", str(frozen / "solve_jump_target_to.py"), "--source_root", str(args.source_root)]
    base += [
        "--device",
        f"cuda:{args.gpu}",
        "--headless",
        f"--kit_args=--/renderer/multiGpu/enabled=false --/renderer/activeGpu={args.gpu}",
    ]

    def save():
        write_json(args.output / "library.json", report)

    def run(label, arguments):
        cmd = base + arguments
        command = dict(label=label, argv=cmd, status="running")
        report["commands"].append(command)
        save()
        with (args.output / f"{label}.log").open("w") as stream:
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        command.update(status="finished", exit_code=result.returncode)
        save()
        return result.returncode

    def search(label, target_file, iterations, noise):
        destination = args.output / label
        run(
            label,
            [
                "--output",
                str(destination),
                "--motion_mode",
                "parkour_leap",
                "--target_batch",
                str(target_file),
                "--resume",
                str(args.seed_candidate),
                "--num_envs",
                str(len(goals) * args.candidates_per_goal),
                "--replicas",
                "1",
                "--iterations",
                str(iterations),
                "--noise_scale",
                str(noise),
                "--canonical_origins",
                "--stop_on_success",
            ]
            + (["--performance_refinement"] if args.performance_refinement else [])
            + (["--sparse_refinement"] if args.sparse_refinement else []),
        )
        progress = destination / "progress.json"
        if not progress.exists() or not json.loads(progress.read_text()).get("complete"):
            raise RuntimeError(f"{label} failed; inspect {label}.log")
        return [destination / name / "candidate.json" for name, _ in goals]

    try:
        if not args.replay_only:
            candidates = search(
                "batch",
                targets_path,
                args.iterations,
                0.15 if args.sparse_refinement else (0.2 if args.performance_refinement else 0.7),
            )
            if args.refine_iterations:
                report["stage"] = "refine"
                refine_targets = args.output / "refine_targets.json"
                write_json(
                    refine_targets, [dict(entry, seed_candidate=str(path)) for entry, path in zip(entries, candidates)]
                )
                search(
                    "refine",
                    refine_targets,
                    args.refine_iterations,
                    0.07 if args.sparse_refinement else (0.12 if args.performance_refinement else 0.45),
                )
        candidates = completed_candidates(args.output, goals)
        catalog = []
        for i, ((name, goal), candidate) in enumerate(zip(goals, candidates)):
            report["stage"] = f"replay {i+1}/{len(goals)}: {name}"
            replay = args.output / f"replay_{name}"
            reused = args.replay_only and (replay / "replay.json").is_file()
            if reused:
                code = None  # no exit status inferred for recovered processes
                for command in report["commands"]:
                    if command["label"] == f"replay_{name}" and command["status"] == "running":
                        command["status"] = "artifact_recovered_after_driver_stop"
            else:
                if replay.exists():
                    raise RuntimeError(f"Incomplete replay retained at {replay}; inspect before retrying")
                code = run(f"replay_{name}", ["--output", str(replay), "--replay", str(candidate), "--video"])
            entry = dict(
                name=name,
                target=asdict(goal),
                candidate=str(candidate.relative_to(args.output)),
                replay=str(replay.relative_to(args.output)),
                process_exit_code=code,
                replay_reused=reused,
            )
            try:
                audit = audit_replay(goal, candidate, replay)
                write_json(replay / "audit.json", audit)
                entry.update(status="accepted" if audit["accepted"] else "diagnostic", metrics=audit["metrics"])
                if audit["accepted"]:
                    output_reference = args.output / "references" / f"{name}.json"
                    output_reference.parent.mkdir(exist_ok=True)
                    shutil.copyfile(replay / "reference.json", output_reference)
                    catalog.append(
                        dict(
                            name=name,
                            target=asdict(goal),
                            reference=str(output_reference.relative_to(args.output)),
                            sha256=sha(output_reference),
                        )
                    )
                    report["accepted_count"] += 1
            except Exception as error:
                entry.update(status="replay_or_audit_failed", error=str(error))
            report["entries"].append(entry)
            write_json(
                args.output / "catalog.json",
                dict(entries=catalog, accepted_count=len(catalog), production_reference_replaced=False),
            )
            save()
            print("PARKOUR_LIBRARY_TARGET " + json.dumps(entry), flush=True)
        report.update(
            status="complete" if report["accepted_count"] == len(goals) else "completed_with_rejections",
            stage="finished",
        )
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
