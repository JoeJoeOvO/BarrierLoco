"""Validate LocoLab distribution archives and installed runtime resources."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

JUMP_REFERENCE = PurePosixPath(
    "locolab/source/locomotion/mdp/reference_motions/go2w_jump_20cm_fast_full_dynamics_to.json"
)
PARKOUR_REFERENCES = {
    PurePosixPath("locolab/source/locomotion/mdp/reference_motions") / name
    for name in ("go2w_parkour_bank.json.gz", "go2w_parkour_goal_grid.json.gz")
}


def check_bank_bytes(data: bytes) -> None:
    # Reject missing LFS content as well as corrupted compressed references.
    json.loads(gzip.decompress(data))


PLANNER_MODULES = {
    PurePosixPath("locolab/planner") / name
    for name in ("__init__.py", "navigation.py", "network.py", "occupancy.py", "ppo.py", "skills.py", "jump_probe.py")
}

GO2W_USD_ROOT = PurePosixPath("locolab/source/locomotion/assets/Go2W/usd")
M20_RESOURCES = {
    PurePosixPath("locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf"),
    PurePosixPath("locolab/source/locomotion/mdp/m20/LICENSE"),
    PurePosixPath("locolab/source/locomotion/mdp/m20/LICENSE-robot_lab"),
    PurePosixPath("locolab/source/locomotion/mdp/m20/README.md"),
}
REQUIRED_USD_FILES = {
    GO2W_USD_ROOT / "go2w.usd",
    GO2W_USD_ROOT / "configuration/go2w_base.usd",
    GO2W_USD_ROOT / "configuration/go2w_description_base.usd",
    GO2W_USD_ROOT / "configuration/go2w_description_physics.usd",
    GO2W_USD_ROOT / "configuration/go2w_description_sensor.usd",
    GO2W_USD_ROOT / "configuration/go2w_physics.usd",
    GO2W_USD_ROOT / "configuration/go2w_robot.usd",
    GO2W_USD_ROOT / "configuration/go2w_sensor.usd",
}


def _require_paths(archive: Path, members: set[PurePosixPath], required: set[PurePosixPath]) -> None:
    missing = sorted(str(path) for path in required.difference(members))
    if missing:
        raise RuntimeError(f"{archive} is missing required package data: {missing}")


def check_wheel(path: Path) -> None:
    """Assert that the wheel contains every runtime USD and jump reference."""
    with zipfile.ZipFile(path) as archive:
        members = {PurePosixPath(name) for name in archive.namelist()}
        _require_paths(
            path, members, REQUIRED_USD_FILES | {JUMP_REFERENCE} | M20_RESOURCES | PARKOUR_REFERENCES | PLANNER_MODULES
        )

        for reference in PARKOUR_REFERENCES:
            check_bank_bytes(archive.read(str(reference)))
        with archive.open(str(JUMP_REFERENCE)) as stream:
            payload = json.load(stream)
        if not payload.get("time") or not payload.get("metadata"):
            raise RuntimeError(f"{path} contains an invalid jump reference")


def check_sdist(path: Path) -> None:
    """Assert that rebuilding from the sdist preserves all runtime resources."""
    sdist_usd_root = PurePosixPath("source/locomotion/assets/Go2W/usd")
    sdist_resources = {
        PurePosixPath("source/locomotion/mdp/reference_motions") / JUMP_REFERENCE.name,
        *(sdist_usd_root / item.relative_to(GO2W_USD_ROOT) for item in REQUIRED_USD_FILES),
    }
    with tarfile.open(path, mode="r:gz") as archive:
        members = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) > 1
        }
    sdist_resources.update(item.relative_to("locolab") for item in M20_RESOURCES | PARKOUR_REFERENCES | PLANNER_MODULES)
    _require_paths(path, members, sdist_resources)


def check_installed_resources() -> None:
    """Resolve resources relative to an installed wheel, as runtime code does."""
    import locolab

    if importlib.util.find_spec("locolab.source.locomotion") is None:
        raise RuntimeError("installed wheel does not expose the canonical locolab.source namespace")
    package_root = Path(locolab.__file__).resolve().parent
    reference = package_root / JUMP_REFERENCE.relative_to("locolab")
    usd_files = [package_root / item.relative_to("locolab") for item in REQUIRED_USD_FILES]

    for item in PARKOUR_REFERENCES:
        check_bank_bytes((package_root / item.relative_to("locolab")).read_bytes())

    for item in M20_RESOURCES | PLANNER_MODULES:
        resource = package_root / item.relative_to("locolab")
        if not resource.is_file() or resource.stat().st_size == 0:
            raise RuntimeError(f"installed runtime resource is missing or empty: {resource}")

    if not reference.is_file():
        raise RuntimeError(f"installed jump reference is missing: {reference}")
    missing_usd = sorted(str(path) for path in usd_files if not path.is_file())
    if missing_usd:
        raise RuntimeError(f"installed Go2W USD files are missing: {missing_usd}")

    with reference.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not payload.get("time") or not payload.get("metadata"):
        raise RuntimeError(f"installed jump reference is invalid: {reference}")
    if not all(path.stat().st_size > 0 for path in usd_files):
        raise RuntimeError("one or more installed Go2W USD files are empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.installed:
        if args.wheel or args.sdist:
            parser.error("--installed cannot be combined with archive paths")
    elif not args.wheel or not args.sdist:
        parser.error("--wheel and --sdist are required unless --installed is used")
    return args


def main() -> None:
    args = parse_args()
    if args.installed:
        check_installed_resources()
        return
    check_wheel(args.wheel)
    check_sdist(args.sdist)


if __name__ == "__main__":
    main()
