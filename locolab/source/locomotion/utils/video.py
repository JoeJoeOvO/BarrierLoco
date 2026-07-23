"""Utilities for deterministic robot-following locomotion videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_VIDEO_DURATION_S = 20.0
DEFAULT_FOLLOW_EYE = (2.2, 2.2, 1.3)
DEFAULT_FOLLOW_LOOKAT = (0.0, 0.0, 0.15)


@dataclass(frozen=True)
class VideoRecordingPlan:
    """Frame and simulation-step counts for one play recording."""

    fps: int
    frame_count: int
    simulation_steps: int


def make_video_recording_plan(
    step_dt: float,
    duration_s: float = DEFAULT_VIDEO_DURATION_S,
    frame_count: int | None = None,
) -> VideoRecordingPlan:
    """Build a recording plan whose encoded duration matches the requested seconds."""
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")

    fps = max(int(round(1.0 / step_dt)), 1)
    if frame_count is None:
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        frame_count = max(int(round(duration_s / step_dt)), 1)
    elif frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")

    # RecordVideo/MoviePy emits one fewer encoded frame than the number of
    # captured simulation samples, so run one additional step.
    return VideoRecordingPlan(fps=fps, frame_count=frame_count, simulation_steps=frame_count + 1)


def configure_robot_follow_view(viewer_cfg: Any) -> None:
    """Track environment zero's robot while preserving tuned task camera offsets."""
    if viewer_cfg.origin_type not in {"asset_root", "asset_body"}:
        viewer_cfg.eye = DEFAULT_FOLLOW_EYE
        viewer_cfg.lookat = DEFAULT_FOLLOW_LOOKAT
    viewer_cfg.origin_type = "asset_root"
    viewer_cfg.asset_name = "robot"
    viewer_cfg.env_index = 0
