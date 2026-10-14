"""Physical-gap or independent tilted-ramp TO and video replay on one GPU.

Each width has its OWN real collision scene. Failed searches are retained as
diagnostics, never relabelled as successful crossings or training references.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from gap_leap_to import GapLeapTarget
from parkour_leap_to import candidate_priority
from solve_parkour_library import audit_replay, sha, write_json
from tilted_ramp_to import TiltedRampTarget

SOURCES = (
    "solve_jump_target_to.py",
    "jump_target_to.py",
    "forward_leap_to.py",
    "parkour_leap_to.py",
    "gap_leap_to.py",
    "tilted_ramp_to.py",
    "solve_parkour_library.py",
    "solve_gap_library.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--seed_candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=float, nargs="+", default=[0.4, 0.6])
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--refine_iterations", type=int, default=16)
    parser.add_argument("--near_x", type=float, default=6.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--trigger_distance", type=float, default=0.40)
    parser.add_argument("--forward_speed", type=float, default=1.0)
    parser.add_argument(
        "--tilted_ramps", action="store_true", help="Solve independent left/right cross-sloped landing ramps"
    )
    parser.add_argument("--ramp_sides", type=int, choices=(-1, 1), nargs="+", default=[1, -1])
    parser.add_argument("--ramp_tilt", type=float, default=20.0)
    parser.add_argument("--ramp_length", type=float, default=1.6)
    parser.add_argument("--ramp_width", type=float, default=0.7)
    parser.add_argument("--ramp_offset_y", type=float, default=0.4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; previous motions and source snapshots are preserved")
    if (
        len(set(args.widths)) != len(args.widths)
        or args.num_envs < 4
        or args.iterations < 1
        or args.refine_iterations < 0
    ):
        parser.error("Invalid widths or search budget")
    goals = [
        GapLeapTarget(
            width=w,
            near_x=args.near_x,
            depth=args.depth,
            trigger_distance=args.trigger_distance,
            forward_speed=args.forward_speed,
        )
        for w in args.widths
    ]
    if args.tilted_ramps:
        if len(args.widths) != 1 or len(set(args.ramp_sides)) != len(args.ramp_sides):
            parser.error("Tilted ramps require one X-gap width and unique left/right selectors")
        # Preserve the ramp-specific orientation/corridor defaults rather than
        # silently inheriting the more restrictive straight-gap task limits.
        goals = [
            TiltedRampTarget(
                width=args.widths[0],
                near_x=args.near_x,
                depth=args.depth,
                trigger_distance=args.trigger_distance,
                forward_speed=args.forward_speed,
                side=side,
                tilt_degrees=args.ramp_tilt,
                platform_length=args.ramp_length,
                platform_width=args.ramp_width,
                offset_y=args.ramp_offset_y,
            )
            for side in args.ramp_sides
        ]
    args.output.mkdir(parents=True)
    frozen = args.output / "solver"
    frozen.mkdir()
    for name in SOURCES:
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1] / "lib"), PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    base = [
        sys.executable,
        "-u",
        str(frozen / "solve_jump_target_to.py"),
        "--source_root",
        str(args.source_root),
        "--device",
        f"cuda:{args.gpu}",
        "--headless",
        f"--kit_args=--/renderer/multiGpu/enabled=false --/renderer/activeGpu={args.gpu}",
    ]
    report = dict(
        status="running",
        stage="initializing",
        targets=[asdict(g) for g in goals],
        entries=[],
        commands=[],
        accepted_count=0,
        source_root=str(args.source_root),
        initial_seed_sha256=sha(args.seed_candidate),
        production_reference_replaced=False,
        rl_training_started=False,
    )
    catalog = []

    def save():
        write_json(args.output / "report.json", report)
        write_json(
            args.output / "catalog.json",
            dict(entries=catalog, accepted_count=len(catalog), production_reference_replaced=False),
        )

    def run(label, arguments):
        report["stage"] = label
        command = dict(label=label, argv=base + arguments, status="running")
        report["commands"].append(command)
        save()
        with (args.output / f"{label}.log").open("w") as stream:
            result = subprocess.run(command["argv"], env=env, stdout=stream, stderr=subprocess.STDOUT)
        command.update(status="finished", exit_code=result.returncode)
        save()
        return result.returncode

    def search(label, goal, seed, count, noise):
        destination = args.output / label
        is_ramp = isinstance(goal, TiltedRampTarget)
        ramp_args = (
            [
                "--ramp_side",
                str(goal.side),
                "--ramp_tilt",
                str(goal.tilt_degrees),
                "--ramp_length",
                str(goal.platform_length),
                "--ramp_width",
                str(goal.platform_width),
                "--ramp_offset_y",
                str(goal.offset_y),
            ]
            if is_ramp
            else []
        )
        run(
            label,
            [
                "--output",
                str(destination),
                "--motion_mode",
                "tilted_ramp" if is_ramp else "gap_leap",
                "--gap_width",
                str(goal.width),
                "--gap_near_x",
                str(goal.near_x),
                "--gap_depth",
                str(goal.depth),
                "--gap_trigger_distance",
                str(goal.trigger_distance),
                "--forward_speed",
                str(goal.forward_speed),
                "--resume",
                str(seed),
                "--num_envs",
                str(args.num_envs),
                "--replicas",
                "1",
                "--iterations",
                str(count),
                "--noise_scale",
                str(noise),
                "--canonical_origins",
                "--stop_on_success",
            ]
            + ramp_args,
        )
        path = destination / "progress.json"
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            raise RuntimeError(f"Search did not complete; inspect {label}.log")
        return destination / "candidate.json"

    seed = args.seed_candidate
    try:
        for goal in goals:
            name = (
                f"ramp_{'left' if goal.side == 1 else 'right'}_{goal.tilt_degrees:g}deg"
                if isinstance(goal, TiltedRampTarget)
                else f"gap_{goal.width * 100:g}cm"
            ).replace(".", "p")
            candidate = search(name + "_search", goal, seed, args.iterations, 0.55)
            if args.refine_iterations and not json.loads(candidate.read_text())["metrics"]["accepted"]:
                refined = search(name + "_refine", goal, candidate, args.refine_iterations, 0.3)
                candidate = min((candidate, refined), key=lambda p: candidate_priority(json.loads(p.read_text())))
            replay_dir = args.output / (name + "_replay")
            code = run(name + "_replay", ["--output", str(replay_dir), "--replay", str(candidate), "--video"])
            entry = dict(
                name=name,
                target=asdict(goal),
                candidate=str(candidate.relative_to(args.output)),
                replay=str(replay_dir.relative_to(args.output)),
                process_exit_code=code,
            )
            try:
                audit = audit_replay(goal, candidate, replay_dir)
                write_json(replay_dir / "audit.json", audit)
                entry.update(status="accepted" if audit["accepted"] else "diagnostic", metrics=audit["metrics"])
                if audit["accepted"]:
                    reference = args.output / "references" / (name + ".json")
                    reference.parent.mkdir(exist_ok=True)
                    shutil.copyfile(replay_dir / "reference.json", reference)
                    catalog.append(
                        dict(
                            name=name,
                            target=asdict(goal),
                            reference=str(reference.relative_to(args.output)),
                            sha256=sha(reference),
                        )
                    )
                    report["accepted_count"] += 1
            except Exception as error:
                entry.update(status="replay_or_audit_failed", error=str(error))
            report["entries"].append(entry)
            save()
            print("GAP_TARGET " + json.dumps(entry), flush=True)
            seed = candidate  # control initial guess ONLY; never reuse states/certificates on another gap
        report.update(
            status="complete" if len(catalog) == len(goals) else "completed_with_rejections", stage="finished"
        )
        save()
    except BaseException as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
