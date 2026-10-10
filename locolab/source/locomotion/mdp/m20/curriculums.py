# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""M20 reward curriculum adapted to environment-local state and Go2W terrain resets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..curriculums import terrain_levels_reward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_level_from_mean(mean_level: torch.Tensor) -> torch.Tensor:
    """Map mean terrain level to the upstream regularization multiplier."""
    level = torch.nan_to_num(mean_level.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(level <= 0.0, 0.0, torch.exp(torch.clamp(level, max=3.0) - 3.0))


def gait_level(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Read one scalar per environment instance, never a Python module global."""
    del env_ids
    if not hasattr(env, "_m20_gait_level"):
        env._m20_gait_level = gait_level_from_mean(env.scene.terrain.terrain_levels.float().mean())
    return env._m20_gait_level


def terrain_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Distance curriculum plus M20 reward scaling; skip travel assessment at initial reset."""
    mean_level = terrain_levels_reward(env, env_ids, command_name="base_velocity")
    env._m20_gait_level = gait_level_from_mean(mean_level)
    return mean_level
