"""Package selected measured replays for explicitly authorized mixed-TO training."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from gap_leap_to import GapLeapTarget, gap_mesh
from tilted_ramp_to import TiltedRampTarget, tilted_ramp_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include_reviewed_candidates", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; do not overwrite an archived training bank")
    catalog = json.loads((args.collection / "catalog.json").read_text())
    entries = []
    for item in catalog["entries"]:
        path = args.collection / item["files"]["replay.json"]["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["files"]["replay.json"]["sha256"]:
            raise ValueError("Replay digest mismatch")
        if not item["strict_accepted"] and not args.include_reviewed_candidates:
            raise ValueError("Manual/diagnostic references require explicit training selection")
        replay = json.loads(path.read_text())
        entry = dict(
            id=item["id"],
            strict_accepted=item["strict_accepted"],
            selection_status=item["selection_status"],
            source_sha256=digest,
            replay=replay,
        )
        if replay["motion_mode"] == "gap_leap":
            mesh = gap_mesh(GapLeapTarget(**replay["target"]))
        elif replay["motion_mode"] == "tilted_ramp":
            mesh = tilted_ramp_mesh(TiltedRampTarget(**replay["target"]))
        else:
            mesh = None
        if mesh is not None:
            entry["terrain_mesh"] = dict(vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist())
        entries.append(entry)
    payload = dict(
        schema_version=1,
        training_use_authorized=True,
        authorization="2026-10-10: user selected all seven TOs, fixed vx=1 m/s, scratch MLP training",
        strict_acceptance_overridden=False,
        entries=entries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as stream:
        json.dump(payload, stream, allow_nan=False)
    print(
        json.dumps(
            {
                "entries": len(entries),
                "bank": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
