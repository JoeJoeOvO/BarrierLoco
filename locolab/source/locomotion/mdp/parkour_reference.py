"""Measured multi-motion TO bank; no policy/Isaac imports or trajectory warping."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

BANK_PATH = Path(__file__).with_name("reference_motions") / "go2w_parkour_bank.json.gz"
GOAL_GRID_BANK_PATH = Path(__file__).with_name("reference_motions") / "go2w_parkour_goal_grid.json.gz"
FIELDS = (
    "position",
    "velocity",
    "quaternion",
    "omega",
    "joint_position",
    "joint_velocity",
    "wheel_position",
    "wheel_velocity",
    "contacts",
)


def environment_layout(env_ids, locomotion_slots=0, motion_schedule=None, flat_motion_ids=None):
    """Keep seven jump/three rolling slots while allowing any validated bank size."""
    if not isinstance(locomotion_slots, int) or not 0 <= locomotion_slots <= 3:
        raise ValueError("Expected zero to three flat locomotion slots")
    schedule = tuple(range(7)) if motion_schedule is None else tuple(motion_schedule)
    flat = (0, 1, 2) if flat_motion_ids is None else tuple(flat_motion_ids)
    if not schedule or not flat or any(type(i) is not int or i < 0 for i in (*schedule, *flat)):
        raise ValueError("Motion schedules must contain nonnegative integer IDs")
    block = env_ids // (7 + locomotion_slots)
    slot = env_ids % (7 + locomotion_slots)
    locomotion = slot >= 7
    choices = torch.tensor(schedule, device=env_ids.device, dtype=torch.long)
    flat_choices = torch.tensor(flat, device=env_ids.device, dtype=torch.long)
    jump = choices[(block * 7 + slot) % len(schedule)]
    rolling = flat_choices[(block * locomotion_slots + slot - 7) % len(flat)]
    return torch.where(locomotion, rolling, jump), locomotion


def sample_reset_phases(bank, motion_ids, weights=(0.30, 0.40, 0.15, 0.10, 0.05)):
    """Ground approach / phase zero / preload / flight / recovery, by probability.

    Sampling categories, not uniform wall time, prevents the long recovery
    tail from consuming most reset examples. No simulated state is retimed.
    """
    weights = torch.as_tensor(weights, device=motion_ids.device, dtype=torch.float32)
    if weights.shape != (5,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Expected five finite nonnegative reset weights")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6):
        raise ValueError("Reset weights must sum to one")
    modes = torch.multinomial(weights, len(motion_ids), replacement=True)
    takeoff, _, landing = bank.times[motion_ids].unbind(-1)
    u = torch.rand(len(motion_ids), device=motion_ids.device)
    time = torch.zeros_like(u)
    time = torch.where(modes == 2, 0.02 + u * (takeoff - 0.06).clamp_min(0), time)
    time = torch.where(modes == 3, takeoff + 0.02 + u * (landing - takeoff - 0.04), time)
    time = torch.where(modes == 4, landing + 0.02 + u * (bank.duration - landing - 0.02), time)
    return modes, time


class ParkourReferenceBank:
    def __init__(self, path=BANK_PATH, device="cpu"):
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        version = payload["schema_version"]
        if version not in (1, 2) or not payload["training_use_authorized"]:
            raise ValueError("An explicitly selected training bank is required")
        self.entries = payload["entries"]
        self.names = [e["id"] for e in self.entries]
        if len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError("Motion names must be nonempty and unique")
        if version == 1 and len(self.names) != 7:
            raise ValueError("The baseline bank requires the seven selected motions")
        if version == 2:
            if payload.get("base_bank_sha256") != hashlib.sha256(BANK_PATH.read_bytes()).hexdigest():
                raise ValueError("Goal-grid bank must preserve the retained baseline")
            with gzip.open(BANK_PATH, "rt") as stream:
                baseline = json.load(stream)["entries"]
            if self.entries[:7] != baseline or len(self.entries) <= 7:
                raise ValueError("Goal-grid bank must retain every original entry unchanged")
            if any(
                not entry["strict_accepted"]
                or not entry["replay"]["accepted"]
                or entry["replay"]["motion_mode"] != "parkour_leap"
                for entry in self.entries[7:]
            ):
                raise ValueError("New grid motions require independently accepted flat-ground replays")
        self.motion_schedule = tuple(payload.get("motion_schedule", range(len(self.names))))
        if (
            not self.motion_schedule
            or any(type(i) is not int or not 0 <= i < len(self.names) for i in self.motion_schedule)
            or set(self.motion_schedule) != set(range(len(self.names)))
        ):
            raise ValueError("Invalid or incomplete motion allocation schedule")
        weights = payload.get("sampling_weights", [1 / len(self.names)] * len(self.names))
        if len(weights) != len(self.names) or any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError("Expected positive finite per-motion sampling weights")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Motion sampling weights must sum to one")
        for i, weight in enumerate(weights):
            if not math.isclose(self.motion_schedule.count(i) / len(self.motion_schedule), weight, abs_tol=1e-6):
                raise ValueError("Scene allocation and reset sampling weights disagree")
        self.flat_motion_ids = tuple(
            i for i, entry in enumerate(self.entries) if entry["replay"]["motion_mode"] == "parkour_leap"
        )
        if not self.flat_motion_ids:
            raise ValueError("Ordinary rolling requires at least one flat reference")
        self.device = device
        self.schema_version = version
        self.is_flat = torch.tensor(
            [i in self.flat_motion_ids for i in range(len(self.names))], device=device, dtype=torch.bool
        )
        self.sampling_weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.flat_sampling_weights = self.sampling_weights * self.is_flat
        self.flat_sampling_weights /= self.flat_sampling_weights.sum()
        self.dt, self.duration, self.sequence_duration = 0.005, 1.8, 2.2
        self.states = {}
        descriptors, times, origins = [], [], []
        for entry in self.entries:
            r = entry["replay"]
            s, m = r["states"], r["metrics"]
            if (
                not r["independent_replay"]
                or r["policy_used"]
                or r["external_wrenches_used"]
                or r["actual_physics_steps"] != 360
                or r["simulation_dt"] != self.dt
                or r["control_dt"] != 0.02
            ):
                raise ValueError("Bank must contain actual independent 5 ms physics samples")
            if r["leg_action_scale"] != 0.25 or r["wheel_action_scale"] != 10:
                raise ValueError("Unexpected action scales")
            if r["leg_effort_limit_nm"] != [23.5, 23.5, 35.0] * 4:
                raise ValueError("Unexpected torque limits")
            if r["leg_velocity_limit_rad_s"] != [30.0, 30.0, 20.0] * 4:
                raise ValueError("Unexpected joint speed limits")
            if entry["strict_accepted"] != r["accepted"]:
                raise ValueError("Do not relabel manual selection as strict acceptance")
            # Keep the fixed terrain/world heading. No yaw rotation or scaling
            # of physics states; terrain and reference receive identical XY offsets.
            origin = torch.tensor(s["position"][0], device=device)
            origins.append(origin)
            for key in FIELDS:
                value = torch.tensor(s[key], device=device, dtype=torch.float32)
                if len(value) != 361 or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid bank samples: {entry['id']} / {key}")
                if key == "position":
                    value[:, :2] -= origin[:2]
                self.states.setdefault(key, []).append(value)
            takeoff = round(m["takeoff_time"] / self.dt)
            landing = round(m["landing_time"] / self.dt)
            apex = takeoff + torch.tensor(s["position"])[takeoff : landing + 1, 2].argmax().item()
            if not 0 < takeoff < apex < landing < 360:
                raise ValueError("Reference must have a measured main flight")
            times.append([takeoff * self.dt, apex * self.dt, landing * self.dt])
            # Conditions describe the actual teacher, not unmet requested
            # targets, physical gap width or eventual rolling recovery position.
            descriptors.append([m["height"], *m["landing_xy"]])
        self.states = {k: torch.stack(v) for k, v in self.states.items()}
        if not torch.allclose(
            self.states["quaternion"].norm(dim=-1), torch.ones(len(self.names), 361, device=device), atol=1e-4, rtol=0
        ):
            raise ValueError("Invalid reference quaternions")
        self.origins = torch.stack(origins)
        self.descriptors = torch.tensor(descriptors, device=device)
        self.times = torch.tensor(times, device=device)
        self.leg_names = self.entries[0]["replay"]["leg_joint_names"]
        self.wheel_names = self.entries[0]["replay"]["wheel_joint_names"]
        if any(
            e["replay"]["leg_joint_names"] != self.leg_names or e["replay"]["wheel_joint_names"] != self.wheel_names
            for e in self.entries
        ):
            raise ValueError("Reference joint ordering differs")

    def layout(self, env_ids, locomotion_slots=0):
        return environment_layout(env_ids, locomotion_slots, self.motion_schedule, self.flat_motion_ids)

    def resample_flat_motions(self, motion_ids, locomotion):
        """Only exchange teachers on identical flat geometry; obstacle IDs stay fixed."""
        result = motion_ids.clone()
        eligible = self.is_flat[motion_ids] & ~locomotion
        count = int(eligible.sum())
        if count:
            result[eligible] = torch.multinomial(self.flat_sampling_weights, count, replacement=True)
        return result

    def sample(self, motion_ids, time):
        if motion_ids.shape != time.shape or time.ndim != 1 or not torch.isfinite(time).all():
            raise ValueError("Expected finite per-environment times and matching motion IDs")
        if ((motion_ids < 0) | (motion_ids >= len(self.names))).any():
            raise ValueError("Unknown motion ID")
        index = time.clamp(0, self.duration) / self.dt
        lo = index.floor().long().clamp(0, 360)
        hi = (lo + 1).clamp_max(360)
        blend = (index - lo).clamp(0, 1)[:, None]
        result = {}
        for key, values in self.states.items():
            a, b = values[motion_ids, lo], values[motion_ids, hi]
            if key == "quaternion":
                b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
                result[key] = torch.nn.functional.normalize(a + blend * (b - a), dim=-1)
            elif key == "contacts":
                result[key] = a  # discrete contact schedule, never fractional contact
            else:
                result[key] = a + blend * (b - a)
        # Continue the measured exit velocity on rolling surfaces; hold the
        # terminal ramp pose. This avoids a 0.4 s stationary target for a 1 m/s exit.
        extra = (time - self.duration).clamp_min(0)
        ramps = torch.tensor([e["replay"]["motion_mode"] == "tilted_ramp" for e in self.entries], device=time.device)[
            motion_ids
        ]
        result["position"] = result["position"].clone()
        result["position"][:, :2] += extra[:, None] * result["velocity"][:, :2] * (~ramps)[:, None]
        result["velocity"] = torch.where((ramps & (extra > 0))[:, None], 0, result["velocity"])
        # This explicit recovery continuation is outside the recorded 1.8 s.
        # A held pose must not simultaneously request nonzero joint/angular velocity.
        hold = extra > 0
        for key in ("joint_velocity", "omega"):
            result[key] = torch.where(hold[:, None], 0, result[key])
        result["velocity"][:, 2] = torch.where(hold, 0, result["velocity"][:, 2])
        result["wheel_velocity"] = torch.where((ramps & hold)[:, None], 0, result["wheel_velocity"])
        result["wheel_position"] += extra[:, None] * result["wheel_velocity"]
        result["contacts"] = torch.where(hold[:, None], 1, result["contacts"])
        return result


def terrain_mesh(entry):
    """Exact recorded collision mesh translated to reference XY origin."""
    import numpy as np
    import trimesh

    r = entry["replay"]
    if r["motion_mode"] == "parkour_leap":
        mesh = trimesh.creation.box(extents=(24, 8, 1))
        mesh.apply_translation((0, 0, -0.5))
        return mesh
    # Geometry is recorded at bank assembly, then checked again at runtime.
    geom = entry["terrain_mesh"]
    vertices = np.asarray(geom["vertices"], dtype="<f4")
    faces = np.asarray(geom["faces"], dtype="<i4")
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    if digest != r["gap_geometry"]["mesh_sha256"]:
        raise ValueError("Terrain mesh differs from the independently replayed scene")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    origin = r["states"]["position"][0]
    mesh.apply_translation((-origin[0], -origin[1], 0))
    return mesh
