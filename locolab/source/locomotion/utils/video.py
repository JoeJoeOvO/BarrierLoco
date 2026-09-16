"""Utilities for deterministic robot-following locomotion videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

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


def configure_terrain_view(env_cfg: Any, terrain_name: str) -> int:
    """Select a populated curriculum column and fix the camera relative to its tile."""
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None or not generator.curriculum:
        raise ValueError("Terrain video selection requires a curriculum terrain generator.")
    terrains = generator.sub_terrains
    if terrain_name not in terrains:
        raise ValueError(f"Unknown video terrain {terrain_name!r}; choose from {list(terrains)}")
    total = sum(cfg.proportion for cfg in terrains.values())
    if total <= 0:
        raise ValueError("Terrain proportions must have a positive sum.")
    # Match Isaac Lab TerrainGenerator._generate_curriculum_terrains, including
    # its boundary offset, and TerrainImporter environment-to-column assignment.
    columns = []
    for column in range(generator.num_cols):
        cumulative = 0.0
        for name, cfg in terrains.items():
            cumulative += cfg.proportion / total
            if column / generator.num_cols + 0.001 < cumulative:
                if name == terrain_name:
                    columns.append(column)
                break
    candidates = [
        index
        for index in range(env_cfg.scene.num_envs)
        if int(index // (env_cfg.scene.num_envs / generator.num_cols)) in columns
    ]
    if not candidates:
        raise ValueError(f"No environment occupies terrain {terrain_name!r}; increase num_envs or terrain columns.")
    index = candidates[len(candidates) // 2]
    viewer = env_cfg.viewer
    viewer.origin_type = "env"
    viewer.env_index = index
    viewer.eye = (5.5, 5.5, 4.5)
    viewer.lookat = (0.0, 0.0, 0.4)
    return index


class TerrainVideoView(gym.Wrapper):
    """Keep the selected tile in view when curriculum resets move its environment."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._video_origin = None
        self._refresh_view()

    def _refresh_view(self) -> None:
        base = self.unwrapped
        index = base.cfg.viewer.env_index
        origin = tuple(base.scene.env_origins[index].tolist())
        if origin != self._video_origin:
            base.viewport_camera_controller.update_view_to_env()
            self._video_origin = origin

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._refresh_view()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._refresh_view()
        return result
