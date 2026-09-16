# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""M20 command mixture, including heading-controlled in-place turns."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class M20VelocityCommand(UniformVelocityCommand):
    """Reserve disjoint probability intervals for zero, lateral, forward and yaw commands."""

    def __init__(self, cfg: M20VelocityCommandCfg, env):
        proportions = (cfg.rel_zero_vel_envs, cfg.rel_only_lin_y_envs, cfg.rel_only_lin_x_envs, cfg.rel_only_ang_z_envs)
        if any(not math.isfinite(p) or p < 0.0 for p in proportions) or sum(proportions) > 1.0:
            raise ValueError("M20 special-command proportions must be finite, nonnegative and sum to at most one.")
        if cfg.rel_only_ang_z_envs > 0.0 and not cfg.heading_command:
            raise ValueError("M20 in-place turns require heading_command=True.")
        super().__init__(cfg, env)
        self._full_lin_x = tuple(cfg.ranges.lin_vel_x)
        self._full_lin_y = tuple(cfg.ranges.lin_vel_y)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        # Some Isaac Lab versions pass slice(None) from CommandManager.reset().
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        super()._resample_command(env_ids)
        small = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) <= 0.2
        self.vel_command_b[env_ids[small], :2] = 0.0

        slots = torch.rand(env_ids.numel(), device=self.device)
        start = 0.0
        for kind, proportion in (
            ("zero", self.cfg.rel_zero_vel_envs),
            ("y", self.cfg.rel_only_lin_y_envs),
            ("x", self.cfg.rel_only_lin_x_envs),
            ("yaw", self.cfg.rel_only_ang_z_envs),
        ):
            ids = env_ids[(slots >= start) & (slots < start + proportion)]
            start += proportion
            self.vel_command_b[ids] = 0.0
            self.is_standing_env[ids] = kind == "zero"
            self.is_heading_env[ids] = kind == "yaw"
            if kind in ("x", "y"):
                axis, limits = (0, self._full_lin_x) if kind == "x" else (1, self._full_lin_y)
                self.vel_command_b[ids, axis] = torch.empty(ids.numel(), device=self.device).uniform_(*limits)
            elif kind == "yaw":
                self.heading_target[ids] = torch.empty(ids.numel(), device=self.device).uniform_(
                    *self.cfg.ranges.heading
                )


@configclass
class M20VelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = M20VelocityCommand
    rel_zero_vel_envs: float = 0.2
    rel_only_lin_y_envs: float = 0.02
    rel_only_lin_x_envs: float = 0.02
    rel_only_ang_z_envs: float = 0.2
