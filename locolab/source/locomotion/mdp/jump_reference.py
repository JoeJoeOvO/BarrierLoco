"""Runtime loading and interpolation for trajectory-optimized jump references."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch

REFERENCE_MOTION_DIR = Path(__file__).with_name("reference_motions")
DEFAULT_JUMP_REFERENCE_PATH = REFERENCE_MOTION_DIR / "go2w_jump_20cm_fast_full_dynamics_to.json"
__all__ = [
    "DEFAULT_JUMP_REFERENCE_PATH",
    "JumpReferenceSample",
    "JumpReferenceTrajectory",
    "load_jump_reference",
    "resolve_reference_path",
]


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

    def __init__(self, payload: dict, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.metadata = dict(payload.get("metadata", {}))
        self.time = torch.as_tensor(payload["time"], device=device, dtype=dtype)
        self.base_position = torch.as_tensor(payload["base_position"], device=device, dtype=dtype)
        self.base_linear_velocity = torch.as_tensor(payload["base_linear_velocity"], device=device, dtype=dtype)
        self.base_pitch = torch.as_tensor(payload["base_pitch"], device=device, dtype=dtype)
        self.base_pitch_rate = torch.as_tensor(payload["base_pitch_rate"], device=device, dtype=dtype)
        self.joint_position = torch.as_tensor(payload["joint_position"], device=device, dtype=dtype)
        self.joint_velocity = torch.as_tensor(payload["joint_velocity"], device=device, dtype=dtype)
        self._validate()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> JumpReferenceTrajectory:
        path = resolve_reference_path(path)
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(payload, device=device, dtype=dtype)

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
) -> JumpReferenceTrajectory:
    """Load and cache a trajectory once for each training device and dtype."""
    resolved = resolve_reference_path(path)
    return JumpReferenceTrajectory.from_json(resolved, device=device, dtype=dtype)
