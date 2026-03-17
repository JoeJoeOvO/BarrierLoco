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
    reward_threshold: float = 0.8,
) -> torch.Tensor:
    """Curriculum that expands linear velocity command range based on tracking reward.

    Args:
        env: The environment instance.
        env_ids: Environment indices that have been reset.
        reward_term_name: Name of the reward term to monitor.
        reward_threshold: Fraction of max reward that triggers range expansion (0-1).

    Returns:
        Current maximum linear velocity x for logging.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    avg_reward_per_second = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    )

    # Calculate normalized reward
    # Note: _episode_sums stores weight * func() * dt per step, so avg_reward_per_second
    # is already in units of weight * mean_func_value. The theoretical max is just weight
    # (when func() returns 1.0 every step).
    max_reward_per_second = reward_term.weight
    normalized_reward = avg_reward_per_second / max_reward_per_second

    if env.common_step_counter % env.max_episode_length == 0:
        if normalized_reward > reward_threshold:
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
    reward_threshold: float = 0.8,
) -> torch.Tensor:
    """Curriculum that expands angular velocity command range based on tracking reward.

    Args:
        env: The environment instance.
        env_ids: Environment indices that have been reset.
        reward_term_name: Name of the reward term to monitor.
        reward_threshold: Fraction of max reward that triggers range expansion (0-1).

    Returns:
        Current maximum angular velocity z for logging.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    avg_reward_per_second = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    )

    # Calculate normalized reward (see lin_vel_cmd_levels for detailed explanation)
    max_reward_per_second = reward_term.weight
    normalized_reward = avg_reward_per_second / max_reward_per_second

    if env.common_step_counter % env.max_episode_length == 0:
        if normalized_reward > reward_threshold:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


class VelocityCurriculumWrapper:
    """Wrapper class for velocity command curriculum with state tracking.

    Uses accumulated success/failure counts across episodes to determine
    when to expand velocity command ranges, providing more stable progression.

    Supports both linear and angular velocity curriculum via separate instances.
    """

    def __init__(self):
        self.succeeded_count = 0
        self.__name__ = "lin_vel_cmd_curriculum"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "base_velocity",
        reward_term_name: str = "track_lin_vel_xy",
        reward_threshold: float = 0.8,
        step_size: float = 0.1,
        max_lin_vel_x: tuple[float, float] | None = None,
        max_lin_vel_y: tuple[float, float] | None = None,
        max_ang_vel_z: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Curriculum that gradually increases velocity command range based on tracking reward.

        Args:
            env: The environment instance.
            env_ids: Environment indices that have been reset.
            command_name: Name of the velocity command term.
            reward_term_name: Name of the reward term to monitor.
            reward_threshold: Fraction of max reward that triggers range expansion (0-1).
                This is compared against the normalized reward (0-1), where 1.0 means
                perfect tracking.
            step_size: Amount to expand range in each direction per step.
            max_lin_vel_x: Maximum (min, max) limits for x velocity.
            max_lin_vel_y: Maximum (min, max) limits for y velocity.
            max_ang_vel_z: Maximum (min, max) limits for z angular velocity.

        Returns:
            Current maximum linear velocity x (or angular velocity z) for logging.
        """
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges

        if len(env_ids) == 0:
            if max_ang_vel_z is not None and max_lin_vel_x is None:
                return torch.tensor(ranges.ang_vel_z[1], device=env.device)
            return torch.tensor(ranges.lin_vel_x[1], device=env.device)

        # Get reward per second for each completed episode
        # Note: _episode_sums stores weight * func() * dt per step, so dividing by
        # max_episode_length_s gives weight * mean_func_value. The theoretical max
        # (perfect tracking) is just weight.
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward_sum_per_sec = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
        threshold_value = reward_term.weight * reward_threshold

        # Update success count based on each environment's performance
        self.succeeded_count += (reward_sum_per_sec > threshold_value).sum().item()
        self.succeeded_count -= (reward_sum_per_sec <= threshold_value).sum().item()
        self.succeeded_count = max(0, self.succeeded_count)

        # Expand range only when accumulated successes exceed num_envs
        if self.succeeded_count > env.num_envs:
            if max_lin_vel_x is not None:
                lin_vel_x_min = max(ranges.lin_vel_x[0] - step_size, max_lin_vel_x[0])
                lin_vel_x_max = min(ranges.lin_vel_x[1] + step_size, max_lin_vel_x[1])
                ranges.lin_vel_x = (lin_vel_x_min, lin_vel_x_max)

            if max_lin_vel_y is not None:
                lin_vel_y_min = max(ranges.lin_vel_y[0] - step_size, max_lin_vel_y[0])
                lin_vel_y_max = min(ranges.lin_vel_y[1] + step_size, max_lin_vel_y[1])
                ranges.lin_vel_y = (lin_vel_y_min, lin_vel_y_max)

            if max_ang_vel_z is not None:
                ang_vel_z_min = max(ranges.ang_vel_z[0] - step_size, max_ang_vel_z[0])
                ang_vel_z_max = min(ranges.ang_vel_z[1] + step_size, max_ang_vel_z[1])
                ranges.ang_vel_z = (ang_vel_z_min, ang_vel_z_max)

            # Reset counter after expansion
            self.succeeded_count = 0

        if max_ang_vel_z is not None and max_lin_vel_x is None:
            return torch.tensor(ranges.ang_vel_z[1], device=env.device)
        return torch.tensor(ranges.lin_vel_x[1], device=env.device)


lin_vel_cmd_curriculum = VelocityCurriculumWrapper()
