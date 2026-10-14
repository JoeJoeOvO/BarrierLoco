"""Build an expanded bank from independently accepted goal-grid TO replays."""

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

from parkour_leap_to import command_target
from solve_parkour_library import audit_replay

HEIGHTS = (0.15, 0.20, 0.25)
X_TARGETS = (0.40, 0.60, 0.80)
Y_TARGETS = (-0.15, 0.0, 0.15)


def coverage_report(entries):
    by_height = {h: [] for h in HEIGHTS}
    by_xy = {}
    for entry in entries:
        target = entry["requested_goal"]
        h, x, y = (round(float(target[k]), 4) for k in ("height", "landing_x", "landing_y"))
        if (h, x, y) not in set(itertools.product(HEIGHTS, X_TARGETS, Y_TARGETS)):
            raise ValueError("Replay is outside the explicitly selected 27-goal grid")
        by_height[h].append(entry["id"])
        by_xy.setdefault((x, y), set()).add(h)
    paired = [xy for xy, levels in by_xy.items() if len(levels) >= 2]
    # A sparse set with only one height cannot teach independent height control.
    ready = all(len(v) >= 3 for v in by_height.values()) and len(paired) >= 4
    return dict(
        requested_goals=27,
        accepted_goals=len(entries),
        accepted_per_height={str(h): len(v) for h, v in by_height.items()},
        xy_cells_with_multiple_heights=[list(xy) for xy in sorted(paired)],
        training_ready=ready,
        required_coverage="at least 3 accepted goals at each height and 4 XY cells with multiple heights",
    )


def balanced_sampling(entries):
    """25% retained motions; 25% at each new height, uniform XY within each height."""
    groups = {h: [] for h in HEIGHTS}
    for i, entry in enumerate(entries[7:], 7):
        groups[round(float(entry["requested_goal"]["height"]), 4)].append(i)
    if any(not ids for ids in groups.values()):
        raise ValueError("Every height needs validated references before sampling")
    period = math.lcm(7, *(len(v) for v in groups.values()))
    schedule = []
    for i in range(period):
        schedule.append(i % 7)
        for ids in groups.values():
            schedule.append(ids[i % len(ids)])
    weights = [schedule.count(i) / len(schedule) for i in range(len(entries))]
    return schedule, weights


def build(library, base_bank, output):
    library, base_bank, output = Path(library), Path(base_bank), Path(output)
    if output.exists():
        raise ValueError("Refuse to overwrite a retained training bank")
    report = json.loads((library / "library.json").read_text())
    if report["status"] not in ("complete", "completed_with_rejections"):
        raise ValueError("TO search and all independent replays must finish first")
    targets = {
        tuple(round(float(e["target"][k]), 4) for k in ("height", "landing_x", "landing_y")) for e in report["targets"]
    }
    if targets != set(itertools.product(HEIGHTS, X_TARGETS, Y_TARGETS)):
        raise ValueError("Library must evaluate the full selected 27-goal grid")
    with gzip.open(base_bank, "rt") as stream:
        baseline = json.load(stream)
    if baseline["schema_version"] != 1 or len(baseline["entries"]) != 7:
        raise ValueError("Expected the retained seven-motion base bank")
    accepted = []
    rejected = []
    for item in report["entries"]:
        if item["status"] != "accepted":
            rejected.append(dict(name=item["name"], status=item["status"], metrics=item.get("metrics")))
            continue
        candidate = (library / item["candidate"]).resolve()
        replay_dir = (library / item["replay"]).resolve()
        if library.resolve() not in candidate.parents or library.resolve() not in replay_dir.parents:
            raise ValueError("Candidate and replay must be contained in the selected library")
        goal = command_target(**item["target"])
        audit = audit_replay(goal, candidate, replay_dir)
        if not audit["accepted"]:
            raise ValueError("Library label differs from independent replay acceptance")
        replay = json.loads((replay_dir / "replay.json").read_text())
        if replay["metrics"]["height_error"] > 0.02 or replay["metrics"]["landing_xy_error"] > 0.05:
            raise ValueError("Requested goal was not reached within its original tolerance")
        accepted.append(
            dict(
                id="grid_" + item["name"],
                strict_accepted=True,
                selection_status=f"accepted_criteria_v{goal.criteria_version}",
                acceptance_criteria_version=goal.criteria_version,
                source_sha256=audit["replay_sha256"],
                replay=replay,
                requested_goal=item["target"],
                independent_audit=audit,
            )
        )
    if len(accepted) != report["accepted_count"] or len(report["entries"]) != 27:
        raise ValueError("Incomplete or inconsistent per-goal replay accounting")
    coverage = coverage_report(accepted)
    coverage_path = output.with_suffix(".coverage.json")
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(dict(coverage, rejected=rejected), indent=2) + "\n")
    if not coverage["training_ready"]:
        raise ValueError("Insufficient independent height/XY coverage; see " + str(coverage_path))
    entries = baseline["entries"] + accepted
    schedule, weights = balanced_sampling(entries)
    payload = dict(
        schema_version=2,
        training_use_authorized=True,
        authorization="2026-10-13: expand independent height/landing TO coverage and matched single-stage sampling",
        base_bank_sha256=hashlib.sha256(base_bank.read_bytes()).hexdigest(),
        strict_acceptance_overridden=False,
        acceptance_versions=sorted({entry["acceptance_criteria_version"] for entry in accepted}),
        coverage=coverage,
        entries=entries,
        motion_schedule=schedule,
        sampling_weights=weights,
        command_source="measured replay height and first-touchdown XY; never arbitrary relabelled commands",
    )
    output.write_bytes(gzip.compress(json.dumps(payload, allow_nan=False).encode(), mtime=0))
    result = dict(
        coverage,
        motions=len(entries),
        bank=str(output),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        legacy_jump_fraction=0.25,
        new_height_fraction_each=0.25,
        ppo_updates=0,
    )
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base_bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("GOAL_GRID_BANK_READY " + json.dumps(build(args.library, args.base_bank, args.output)), flush=True)


if __name__ == "__main__":
    main()
