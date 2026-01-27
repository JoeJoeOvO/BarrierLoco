from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def velocity_cmd_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy",
    reward_threshold: float = 0.8,
    step_size: float = 0.1,
    max_lin_vel_x: tuple[float, float] | None = None,
    max_lin_vel_y: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Curriculum that gradually increases velocity command range based on tracking reward.

    When the tracking reward exceeds the threshold, the velocity command range is expanded
    by `step_size` in each direction, up to the specified maximum limits.

    Args:
        env: The environment instance.
        env_ids: Environment indices that have been reset.
        command_name: Name of the velocity command term.
        reward_term_name: Name of the reward term to monitor (e.g., "track_lin_vel_xy").
        reward_threshold: Fraction of max reward that triggers range expansion (0-1).
        step_size: Amount to expand range in each direction per step.
        max_lin_vel_x: Maximum (min, max) limits for x velocity. None to skip x adjustment.
        max_lin_vel_y: Maximum (min, max) limits for y velocity. None to skip y adjustment.

    Returns:
        Current maximum linear velocity x for logging.
    """
    # Get reward statistics
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward_sum = env.reward_manager._episode_sums[reward_term_name][env_ids]
    avg_reward = torch.mean(reward_sum) / env.max_episode_length_s

    # Get command configuration
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    # Check if reward exceeds threshold and expand range
    if avg_reward > reward_term.weight * reward_threshold:
        if max_lin_vel_x is not None:
            lin_vel_x_min = max(ranges.lin_vel_x[0] - step_size, max_lin_vel_x[0])
            lin_vel_x_max = min(ranges.lin_vel_x[1] + step_size, max_lin_vel_x[1])
            ranges.lin_vel_x = (lin_vel_x_min, lin_vel_x_max)

        if max_lin_vel_y is not None:
            lin_vel_y_min = max(ranges.lin_vel_y[0] - step_size, max_lin_vel_y[0])
            lin_vel_y_max = min(ranges.lin_vel_y[1] + step_size, max_lin_vel_y[1])
            ranges.lin_vel_y = (lin_vel_y_min, lin_vel_y_max)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
