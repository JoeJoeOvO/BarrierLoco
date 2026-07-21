"""Trimesh terrain generators for Go2W high-platform training and play."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .high_platform_cfg import MeshHighPlatformSequenceTerrainCfg, MeshHighPlatformTerrainCfg


def high_platform_terrain(
    difficulty: float, cfg: MeshHighPlatformTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate flat ground followed by one centered raised platform."""
    if cfg.num_height_levels < 2:
        raise ValueError("num_height_levels must be at least 2")
    if cfg.ground_thickness <= 0.0:
        raise ValueError("ground_thickness must be positive")

    size_x, size_y = cfg.size
    platform_end_x = cfg.platform_start_x + cfg.platform_length
    if cfg.platform_length <= 0.0 or cfg.platform_start_x < 0.0 or platform_end_x > size_x:
        raise ValueError("platform must fit inside the terrain along x")
    if cfg.platform_width <= 0.0 or cfg.platform_width > size_y:
        raise ValueError("platform_width must be positive and fit inside the terrain along y")
    if not 0.0 <= cfg.spawn_x < cfg.platform_start_x:
        raise ValueError("spawn_x must be on the flat approach before the platform")

    min_height, max_height = cfg.platform_height_range
    if min_height <= 0.0 or max_height < min_height:
        raise ValueError("platform_height_range must contain positive ascending values")

    clamped_difficulty = float(np.clip(difficulty, 0.0, 1.0))
    level = min(int(clamped_difficulty * cfg.num_height_levels), cfg.num_height_levels - 1)
    platform_height = min_height + level * (max_height - min_height) / (cfg.num_height_levels - 1)

    ground = trimesh.creation.box(
        extents=(size_x, size_y, cfg.ground_thickness),
        transform=trimesh.transformations.translation_matrix((0.5 * size_x, 0.5 * size_y, -0.5 * cfg.ground_thickness)),
    )
    platform = trimesh.creation.box(
        extents=(cfg.platform_length, cfg.platform_width, platform_height),
        transform=trimesh.transformations.translation_matrix(
            (cfg.platform_start_x + 0.5 * cfg.platform_length, 0.5 * size_y, 0.5 * platform_height)
        ),
    )
    origin = np.array([cfg.spawn_x, 0.5 * size_y, 0.0])
    return [ground, platform], origin


def high_platform_sequence_terrain(
    difficulty: float, cfg: MeshHighPlatformSequenceTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the deterministic ascending platform sequence used for play."""
    del difficulty
    if cfg.ground_thickness <= 0.0:
        raise ValueError("ground_thickness must be positive")
    if not cfg.platform_heights or any(height <= 0.0 for height in cfg.platform_heights):
        raise ValueError("platform_heights must contain positive values")
    if any(next_height <= height for height, next_height in zip(cfg.platform_heights, cfg.platform_heights[1:])):
        raise ValueError("platform_heights must be strictly ascending")

    size_x, size_y = cfg.size
    platform_stride = cfg.platform_length + cfg.platform_gap
    final_end_x = cfg.platform_start_x + (len(cfg.platform_heights) - 1) * platform_stride + cfg.platform_length
    if cfg.platform_length <= 0.0 or cfg.platform_gap < 0.0 or cfg.platform_start_x < 0.0 or final_end_x > size_x:
        raise ValueError("the platform sequence must fit inside the terrain along x")
    if cfg.platform_width <= 0.0 or cfg.platform_width > size_y:
        raise ValueError("platform_width must be positive and fit inside the terrain along y")
    if not 0.0 <= cfg.spawn_x < cfg.platform_start_x:
        raise ValueError("spawn_x must be on the flat approach before the platform sequence")

    ground = trimesh.creation.box(
        extents=(size_x, size_y, cfg.ground_thickness),
        transform=trimesh.transformations.translation_matrix((0.5 * size_x, 0.5 * size_y, -0.5 * cfg.ground_thickness)),
    )
    meshes = [ground]
    for index, height in enumerate(cfg.platform_heights):
        start_x = cfg.platform_start_x + index * platform_stride
        meshes.append(
            trimesh.creation.box(
                extents=(cfg.platform_length, cfg.platform_width, height),
                transform=trimesh.transformations.translation_matrix(
                    (start_x + 0.5 * cfg.platform_length, 0.5 * size_y, 0.5 * height)
                ),
            )
        )

    origin = np.array([cfg.spawn_x, 0.5 * size_y, 0.0])
    return meshes, origin
