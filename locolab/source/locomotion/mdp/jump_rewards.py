"""Rewards used by the canonical Go2W timed-jump task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .jump_reference import JumpReferenceSample, JumpReferenceTrajectory, load_jump_reference

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command.jump_command import BinaryJumpCommand


__all__ = [
    "jump_angular_velocity_l2",
    "jump_ascent_reference_height_l2",
    "jump_ascent_reference_height_tanh_reward",
    "jump_ascent_vertical_velocity_reward",
    "jump_attitude_l2",
    "jump_inactive_base_height_l2",
    "jump_inactive_joint_deviation_l1",
    "jump_inactive_lin_vel_z_l2",
    "jump_inactive_track_ang_vel_z_exp",
    "jump_pre_takeoff_reference_joint_position_reward",
    "jump_pre_takeoff_reference_joint_velocity_reward",
    "jump_reference_joint_position_reward",
    "jump_wheel_velocity_l2",
]


def _jump_command_term(env: ManagerBasedRLEnv, command_name: str) -> BinaryJumpCommand:
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "update_state") or not hasattr(term, "reference_time"):
        raise TypeError(f"Command term {command_name!r} does not provide jump-reference phase tracking.")
    term.update_state()
    return term


def _gate_to_current_four_wheel_flight(
    term: BinaryJumpCommand,
    mask: torch.Tensor,
    required: bool,
) -> torch.Tensor:
    """Optionally require the scheduled flight phase and zero wheel contacts."""
    if not required:
        return mask
    if term.contact_state.ndim != 2 or term.contact_state.shape[-1] != 4:
        raise ValueError(f"Expected four wheel contacts, got shape {tuple(term.contact_state.shape)}.")
    current_four_wheel_flight = term.active & term.is_in_flight & ~term.contact_state.any(dim=-1)
    return mask & current_four_wheel_flight


def _reference_context(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    asset_cfg: SceneEntityCfg,
) -> tuple[BinaryJumpCommand, object, JumpReferenceTrajectory, JumpReferenceSample, torch.Tensor]:
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    state = asset.data.root_pos_w
    step = int(getattr(env, "common_step_counter", getattr(term, "_last_update_step", -1)))
    cache_key = (step, reference_path, str(state.device), state.dtype)
    cached = getattr(term, "_reference_reward_cache", None)
    if cached is None or cached[0] != cache_key:
        trajectory = load_jump_reference(reference_path, device=state.device, dtype=state.dtype)
        sample = trajectory.sample(term.reference_time)
        cached = (cache_key, trajectory, sample)
        term._reference_reward_cache = cached
    _, trajectory, sample = cached
    valid = term.active & (term.reference_time <= trajectory.duration)
    return term, asset, trajectory, sample, valid


def _masked_gaussian(normalized_error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if normalized_error.ndim == 1:
        squared_error = normalized_error.square()
    else:
        squared_error = normalized_error.square().mean(dim=tuple(range(1, normalized_error.ndim)))
    return torch.exp(-squared_error) * valid.to(squared_error.dtype)


def _positive_std(std: float, name: str = "std") -> None:
    if std <= 0.0:
        raise ValueError(f"{name} must be positive, got {std}.")


def jump_ascent_vertical_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    ascent_end_time: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    require_all_wheels_airborne: bool = False,
) -> torch.Tensor:
    """Densely reward every positive vertical velocity during the scheduled ascent.

    This intentionally returns velocity in m/s without normalization or
    saturation. The configured reward weight therefore directly represents
    the requested gain on upward velocity, while the paired trajectory-height
    cost penalizes overshoot.
    """
    _positive_std(ascent_end_time, "ascent_end_time")
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    upward_velocity = torch.clamp_min(asset.data.root_lin_vel_w[:, 2], 0.0)
    scheduled_ascent = term.active & (term.reference_time <= ascent_end_time)
    scheduled_ascent = _gate_to_current_four_wheel_flight(term, scheduled_ascent, require_all_wheels_airborne)
    return upward_velocity * scheduled_ascent.to(upward_velocity.dtype)


def _jump_ascent_reference_height_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    ascent_end_time: float,
    asset_cfg: SceneEntityCfg,
    require_all_wheels_airborne: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relative-height tracking error and its scheduled-ascent mask."""
    _positive_std(ascent_end_time, "ascent_end_time")
    term, asset, trajectory, sample, valid = _reference_context(env, command_name, reference_path, asset_cfg)
    desired_relative_height = sample.base_position[:, 2] - trajectory.base_position[0, 2]
    actual_relative_height = asset.data.root_pos_w[:, 2] - term.trigger_height
    scheduled_ascent = valid & (term.reference_time <= ascent_end_time)
    scheduled_ascent = _gate_to_current_four_wheel_flight(term, scheduled_ascent, require_all_wheels_airborne)
    return actual_relative_height - desired_relative_height, scheduled_ascent


def jump_ascent_reference_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    ascent_end_time: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    require_all_wheels_airborne: bool = False,
) -> torch.Tensor:
    """Penalize squared error from the TO ballistic-height curve through apex."""
    error, scheduled_ascent = _jump_ascent_reference_height_error(
        env,
        command_name,
        reference_path,
        ascent_end_time,
        asset_cfg,
        require_all_wheels_airborne,
    )
    return error.square() * scheduled_ascent.to(error.dtype)


def jump_ascent_reference_height_tanh_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    ascent_end_time: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    require_all_wheels_airborne: bool = False,
) -> torch.Tensor:
    """Provide a smooth dense bonus as height approaches the TO curve."""
    _positive_std(std)
    error, scheduled_ascent = _jump_ascent_reference_height_error(
        env,
        command_name,
        reference_path,
        ascent_end_time,
        asset_cfg,
        require_all_wheels_airborne,
    )
    reward = 1.0 - torch.tanh(torch.abs(error) / std)
    return reward * scheduled_ascent.to(reward.dtype)


def jump_attitude_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch tilt only while a jump command is active."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    return tilt_error * term.active.to(tilt_error.dtype)


def jump_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll, pitch, and yaw angular velocity during a jump."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    angular_speed_squared = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return angular_speed_squared * term.active.to(angular_speed_squared.dtype)


def jump_reference_joint_position_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    require_all_wheels_airborne: bool = False,
) -> torch.Tensor:
    """Track the 12-DoF leg pose from the optimized full-dynamics trajectory."""
    _positive_std(std)
    term, asset, _, sample, valid = _reference_context(env, command_name, reference_path, asset_cfg)
    valid = _gate_to_current_four_wheel_flight(term, valid, require_all_wheels_airborne)
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if actual.shape[-1] != sample.joint_position.shape[-1]:
        raise ValueError(f"Expected 12 tracked leg joints, got {actual.shape[-1]}.")
    return _masked_gaussian((actual - sample.joint_position) / std, valid)


def jump_pre_takeoff_reference_joint_position_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    takeoff_time: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track the TO leg pose during the fixed, grounded takeoff sequence."""
    _positive_std(takeoff_time, "takeoff_time")
    _positive_std(std)
    term, asset, _, sample, valid = _reference_context(env, command_name, reference_path, asset_cfg)
    pre_takeoff = valid & (term.reference_time < takeoff_time)
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if actual.shape[-1] != sample.joint_position.shape[-1]:
        raise ValueError(f"Expected 12 tracked leg joints, got {actual.shape[-1]}.")
    return _masked_gaussian((actual - sample.joint_position) / std, pre_takeoff)


def jump_pre_takeoff_reference_joint_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    takeoff_time: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track TO leg velocity before scheduled flight starts."""
    _positive_std(takeoff_time, "takeoff_time")
    _positive_std(std)
    term, asset, _, sample, valid = _reference_context(env, command_name, reference_path, asset_cfg)
    pre_takeoff = valid & (term.reference_time < takeoff_time)
    actual = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if actual.shape[-1] != sample.joint_velocity.shape[-1]:
        raise ValueError(f"Expected 12 tracked leg joints, got {actual.shape[-1]}.")
    return _masked_gaussian((actual - sample.joint_velocity) / std, pre_takeoff)


def jump_inactive_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    jump_command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track yaw only when no jump is requested."""
    _positive_std(std)
    term = _jump_command_term(env, jump_command_name)
    asset = env.scene[asset_cfg.name]
    error = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    reward = torch.exp(-torch.square(error / std))
    return reward * (~term.active).to(reward.dtype)


def jump_inactive_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stance-height error only when no jump is requested."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return error * (~term.active).to(error.dtype)


def jump_inactive_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical motion only for trigger-zero stance behavior."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    penalty = torch.square(asset.data.root_lin_vel_w[:, 2])
    return penalty * (~term.active).to(penalty.dtype)


def jump_inactive_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the default pose for command zero without suppressing TO crouching."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.abs(actual - default).sum(dim=-1)
    return penalty * (~term.active).to(penalty.dtype)


def jump_wheel_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Keep all four wheel joints still only during scheduled flight."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    wheel_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.mean(torch.square(wheel_velocity), dim=-1)
    return penalty * term.is_in_flight.to(penalty.dtype)
