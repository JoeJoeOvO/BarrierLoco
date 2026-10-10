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
    "jump_heading_error_l2",
    "jump_heading_yaw_velocity_l2",
    "jump_heading_lateral_velocity_l2",
    "jump_landing_vertical_velocity_l2",
    "jump_pre_takeoff_reference_joint_position_reward",
    "jump_pre_takeoff_reference_joint_velocity_reward",
    "jump_reference_joint_position_reward",
    "jump_reference_joint_velocity_reward",
    "jump_track_lin_vel_xy_exp",
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
    # Sampling clamps to the last TO pose. Keep that pose during the fixed
    # recovery extension, instead of creating another gap after TO ends.
    valid = term.active
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
    scheduled_ascent = term.active & term.is_in_flight & (term.reference_time <= ascent_end_time)
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
    scheduled_ascent = valid & term.is_in_flight & (term.reference_time <= ascent_end_time)
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


def jump_reference_joint_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_path: str,
    start_time: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track TO leg velocity through flight and landing, independent of contact."""
    _positive_std(std)
    term, asset, trajectory, sample, valid = _reference_context(env, command_name, reference_path, asset_cfg)
    valid = valid & (term.reference_time >= start_time)
    actual = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if actual.shape[-1] != sample.joint_velocity.shape[-1]:
        raise ValueError(f"Expected 12 tracked leg joints, got {actual.shape[-1]}.")
    target = torch.where(
        (term.reference_time >= trajectory.duration).unsqueeze(-1),
        torch.zeros_like(sample.joint_velocity),
        sample.joint_velocity,
    )
    return _masked_gaussian((actual - target) / std, valid)


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
    velocity_command_name: str | None = None,
) -> torch.Tensor:
    """Keep the default pose for command zero without suppressing TO crouching."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.abs(actual - default).sum(dim=-1)
    mask = ~term.active
    if velocity_command_name is not None:
        velocity_command = env.command_manager.get_command(velocity_command_name)
        mask &= torch.linalg.vector_norm(velocity_command[:, :3], dim=-1) < 0.1
    return penalty * mask.to(penalty.dtype)


def jump_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    jump_command_name: str,
    std: float,
    active_multiplier: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lock_jump_heading: bool = False,
) -> torch.Tensor:
    """Maintain commanded horizontal motion through a jump, in the yaw frame.

    Full body-frame velocity would mix vertical velocity into vx when pitching.
    Neither the reference's zero initial vx nor an in-air zero command is used.
    """
    _positive_std(std)
    term = _jump_command_term(env, jump_command_name)
    asset = env.scene[asset_cfg.name]
    w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    vx, vy = asset.data.root_lin_vel_w[:, :2].unbind(dim=-1)
    velocity = torch.stack((yaw.cos() * vx + yaw.sin() * vy, -yaw.sin() * vx + yaw.cos() * vy), dim=-1)
    if lock_jump_heading:
        velocity = torch.where(
            term.heading_tracking_mask[:, None], term.velocity_in_trigger_frame(asset.data.root_lin_vel_w), velocity
        )
    target = env.command_manager.get_command(command_name)[:, :2]
    reward = torch.exp(-torch.sum((velocity - target).square(), dim=-1) / std**2)
    return reward * (1.0 + (active_multiplier - 1.0) * term.active.to(reward.dtype))


def jump_heading_error_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Keep the yaw captured at trigger through takeoff, flight and recovery."""
    term = _jump_command_term(env, command_name)
    return term.heading_error().square() * term.heading_tracking_mask


def jump_heading_yaw_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Suppress world-z angular velocity during the held jump direction."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2].square() * term.heading_tracking_mask


def jump_heading_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track lateral command in the fixed trigger frame (zero for forward jumps)."""
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    lateral = term.velocity_in_trigger_frame(asset.data.root_lin_vel_w)[:, 1]
    target = env.command_manager.get_command(velocity_command_name)[:, 1]
    return (lateral - target).square() * term.heading_tracking_mask


def jump_wheel_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize each airborne wheel, including flight beyond the fixed jump clock.

    The two entity configs must use the same explicit leg order. Grounded wheels
    remain free to track forward velocity even inside the scheduled flight phase.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    wheel_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    airborne = torch.linalg.vector_norm(forces, dim=-1) <= contact_force_threshold
    if wheel_velocity.shape != airborne.shape or wheel_velocity.shape[-1] != 4:
        raise ValueError("Expected four wheel velocities and four matching contact states.")
    return (wheel_velocity.square() * airborne.to(wheel_velocity.dtype)).mean(dim=-1)


def jump_landing_vertical_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    descent_start_time: float,
    velocity_tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Damp residual vertical speed on contact after apex, including late landings.

    This is a contact-settling cost, not a measurement of pre-impact velocity.
    Ballistic descent and the grounded takeoff push are deliberately unpenalized.
    """
    term = _jump_command_term(env, command_name)
    asset = env.scene[asset_cfg.name]
    excess_speed = (asset.data.root_lin_vel_w[:, 2].abs() - velocity_tolerance).clamp_min(0.0)
    landing_contact = (term.reference_time >= descent_start_time) & term.contact_state.any(dim=-1)
    return excess_speed.square() * landing_contact.to(excess_speed.dtype)
