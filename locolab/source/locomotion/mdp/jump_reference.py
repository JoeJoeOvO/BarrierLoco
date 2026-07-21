"""Runtime loading and interpolation for trajectory-optimized jump references."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch

REFERENCE_MOTION_DIR = Path(__file__).with_name("reference_motions")
DEFAULT_JUMP_REFERENCE_PATH = REFERENCE_MOTION_DIR / "go2w_jump_20cm_fast_full_dynamics_to.json"
GO2W_REFERENCE_LEG_JOINTS = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
__all__ = [
    "DEFAULT_JUMP_REFERENCE_PATH",
    "GO2W_JUMP_PROFILE",
    "GO2W_REFERENCE_LEG_JOINTS",
    "JumpProfile",
    "JumpReferenceSample",
    "JumpReferenceTrajectory",
    "load_jump_reference",
    "resolve_reference_path",
]


@dataclass(frozen=True)
class JumpProfile:
    """Single source of truth for a timed jump and its reference contract."""

    reference_path: Path
    takeoff_time: float
    apex_time: float
    landing_time: float
    sequence_duration: float
    minimum_height: float
    jump_probability: float
    leg_joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        schedule = (self.takeoff_time, self.apex_time, self.landing_time, self.sequence_duration)
        if not 0.0 < schedule[0] < schedule[1] < schedule[2] < schedule[3]:
            raise ValueError(f"Jump profile schedule is not strictly increasing: {schedule}.")
        if self.minimum_height <= 0.0:
            raise ValueError("minimum_height must be positive.")
        if not 0.0 <= self.jump_probability <= 1.0:
            raise ValueError("jump_probability must be in [0, 1].")
        if not self.leg_joint_names:
            raise ValueError("leg_joint_names must not be empty.")

    def validate_reference_metadata(self, metadata: dict) -> None:
        expected_times = {
            "target_takeoff_time": self.takeoff_time,
            "target_apex_time": self.apex_time,
            "target_landing_time": self.landing_time,
            "sequence_duration": self.sequence_duration,
        }
        for key, expected in expected_times.items():
            actual = metadata.get(key)
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Jump-reference metadata {key!r} is {actual!r}; expected {expected}.")
        reference_height = metadata.get("target_jump_height")
        if reference_height is None or float(reference_height) < self.minimum_height:
            raise ValueError(
                "Jump-reference target height must be at least the profile minimum "
                f"({reference_height!r} < {self.minimum_height})."
            )
        joint_names = tuple(metadata.get("leg_joint_names", ()))
        if joint_names != self.leg_joint_names:
            raise ValueError(f"Jump-reference leg_joint_names are {joint_names!r}; expected {self.leg_joint_names!r}.")


GO2W_JUMP_PROFILE = JumpProfile(
    reference_path=DEFAULT_JUMP_REFERENCE_PATH,
    takeoff_time=0.36,
    apex_time=0.54,
    landing_time=0.72,
    sequence_duration=0.80,
    minimum_height=0.18,
    jump_probability=0.8,
    leg_joint_names=GO2W_REFERENCE_LEG_JOINTS,
)


class JumpReferenceSample(NamedTuple):
    """Runtime state required by reset and active jump rewards."""

    base_position: torch.Tensor
    base_linear_velocity: torch.Tensor
    base_pitch: torch.Tensor
    base_pitch_rate: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


class JumpReferenceTrajectory:
    """Validated tensor representation of one offline TO trajectory."""

    def __init__(
        self,
        payload: dict,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        profile: JumpProfile | None = None,
    ):
        self.metadata = dict(payload.get("metadata", {}))
        self.time = torch.as_tensor(payload["time"], device=device, dtype=dtype)
        self.base_position = torch.as_tensor(payload["base_position"], device=device, dtype=dtype)
        self.base_linear_velocity = torch.as_tensor(payload["base_linear_velocity"], device=device, dtype=dtype)
        self.base_pitch = torch.as_tensor(payload["base_pitch"], device=device, dtype=dtype)
        self.base_pitch_rate = torch.as_tensor(payload["base_pitch_rate"], device=device, dtype=dtype)
        self.joint_position = torch.as_tensor(payload["joint_position"], device=device, dtype=dtype)
        self.joint_velocity = torch.as_tensor(payload["joint_velocity"], device=device, dtype=dtype)
        self._validate()
        if profile is not None:
            profile.validate_reference_metadata(self.metadata)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        profile: JumpProfile | None = None,
    ) -> JumpReferenceTrajectory:
        path = resolve_reference_path(path)
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(payload, device=device, dtype=dtype, profile=profile)

    @property
    def duration(self) -> float:
        return float(self.time[-1].item())

    def sample(self, reference_time: torch.Tensor) -> JumpReferenceSample:
        """Linearly interpolate the reference state at each query time."""
        if reference_time.ndim != 1:
            raise ValueError(f"reference_time must be one-dimensional, got shape {tuple(reference_time.shape)}.")
        if reference_time.device != self.time.device:
            raise ValueError(
                f"reference_time is on {reference_time.device}, but the reference is on {self.time.device}."
            )

        query = reference_time.to(dtype=self.time.dtype).clamp(self.time[0], self.time[-1])
        right = torch.searchsorted(self.time, query, right=True).clamp(min=1, max=self.time.numel() - 1)
        left = right - 1
        interval = torch.clamp_min(self.time[right] - self.time[left], torch.finfo(self.time.dtype).eps)
        blend = (query - self.time[left]) / interval

        def interpolate(values: torch.Tensor) -> torch.Tensor:
            shape = (blend.shape[0],) + (1,) * (values.ndim - 1)
            alpha = blend.reshape(shape)
            return values[left] + alpha * (values[right] - values[left])

        return JumpReferenceSample(
            base_position=interpolate(self.base_position),
            base_linear_velocity=interpolate(self.base_linear_velocity),
            base_pitch=interpolate(self.base_pitch),
            base_pitch_rate=interpolate(self.base_pitch_rate),
            joint_position=interpolate(self.joint_position),
            joint_velocity=interpolate(self.joint_velocity),
        )

    def _validate(self) -> None:
        sample_count = self.time.numel()
        if sample_count < 2 or self.time.ndim != 1:
            raise ValueError("A jump reference must contain at least two one-dimensional time samples.")
        if not torch.all(self.time[1:] > self.time[:-1]):
            raise ValueError("Jump-reference time samples must be strictly increasing.")

        expected_shapes = {
            "base_position": (sample_count, 3),
            "base_linear_velocity": (sample_count, 3),
            "base_pitch": (sample_count,),
            "base_pitch_rate": (sample_count,),
            "joint_position": (sample_count, 12),
            "joint_velocity": (sample_count, 12),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != expected:
                raise ValueError(f"Reference field {name!r} has shape {actual}; expected {expected}.")

        required_metadata = {"target_jump_height", "flight_start_time", "flight_end_time", "mass", "gravity"}
        missing = required_metadata.difference(self.metadata)
        if missing:
            raise ValueError(f"Jump-reference metadata is missing: {sorted(missing)}.")


def resolve_reference_path(path: str | Path) -> Path:
    """Resolve either an explicit path or a filename in the bundled motion directory."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    bundled = REFERENCE_MOTION_DIR / candidate
    if bundled.is_file():
        return bundled.resolve()
    raise FileNotFoundError(f"Jump reference not found at {candidate} or {bundled}.")


@lru_cache(maxsize=16)
def load_jump_reference(
    path: str | Path = DEFAULT_JUMP_REFERENCE_PATH,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    profile: JumpProfile = GO2W_JUMP_PROFILE,
) -> JumpReferenceTrajectory:
    """Load and cache a trajectory once for each training device and dtype."""
    resolved = resolve_reference_path(path)
    return JumpReferenceTrajectory.from_json(resolved, device=device, dtype=dtype, profile=profile)
