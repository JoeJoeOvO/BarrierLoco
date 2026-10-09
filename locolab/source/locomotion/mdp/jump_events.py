"""Timed reference-state initialization for Go2W jumping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .jump_reference import load_jump_reference

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_ranges(
    ranges: list[tuple[float, float]],
    count: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    bounds = torch.tensor(ranges, device=device, dtype=dtype)
    return math_utils.sample_uniform(bounds[:, 0], bounds[:, 1], (count, len(ranges)), device=device)


def _ensure_timed_jump_buffers(env: ManagerBasedEnv, dtype: torch.dtype) -> None:
    device = torch.device(env.device)
    specifications = {
        "_jump_episode_mask": ((env.num_envs,), torch.bool),
        "_jump_initial_phase_time": ((env.num_envs,), dtype),
        "_jump_trigger_position": ((env.num_envs, 3), dtype),
        "_jump_target_height": ((env.num_envs,), dtype),
        "_jump_reset_pending": ((env.num_envs,), torch.bool),
    }
    for name, (shape, expected_dtype) in specifications.items():
        value = getattr(env, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.device != device
            or value.dtype != expected_dtype
        ):
            setattr(env, name, torch.zeros(shape, dtype=expected_dtype, device=device))


def _resolve_leg_joint_ids(
    env: ManagerBasedEnv,
    asset: Articulation,
    asset_name: str,
    leg_joint_names: list[str],
    expected_count: int,
) -> torch.Tensor:
    cache = getattr(env, "_jump_leg_joint_ids", None)
    if not isinstance(cache, dict):
        cache = {}
        env._jump_leg_joint_ids = cache
    cache_key = (asset_name, tuple(leg_joint_names))
    if cache_key not in cache:
        joint_ids, matched_names = asset.find_joints(leg_joint_names, preserve_order=True)
        if len(joint_ids) != expected_count:
            raise ValueError(f"Expected {expected_count} Go2W reference joints, matched {matched_names}.")
        cache[cache_key] = torch.as_tensor(joint_ids, dtype=torch.long, device=asset.device)
    return cache[cache_key]


def _stratified_phase_times(
    count: int,
    lower: float,
    upper: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample the whole interval once per reset batch while retaining randomness."""
    if count == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if upper == lower:
        return torch.full((count,), lower, device=device, dtype=dtype)
    unit = (torch.arange(count, device=device, dtype=dtype) + torch.rand(count, device=device, dtype=dtype)) / count
    samples = lower + (upper - lower) * unit
    return samples[torch.randperm(count, device=device)]


def reset_timed_jump_or_flat_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    jump_probability: float,
    reference_path: str,
    leg_joint_names: list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    phase_time_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    align_reference_height: bool = False,
    preserve_reference_joint_positions: bool = False,
) -> None:
    """Sample one episode-level jump flag, then reset to jump phase or flat state.

    Non-jump environments reproduce the original flat reset.  Jump environments
    draw a stratified phase from the selected motion so every sufficiently
    large reset batch covers preparation, flight, and landing/recovery.
    """
    if not 0.0 <= jump_probability <= 1.0:
        raise ValueError(f"jump_probability must be in [0, 1], got {jump_probability}.")

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device).flatten()
    if env_ids.numel() == 0:
        return

    dtype = asset.data.default_root_state.dtype
    trajectory = load_jump_reference(reference_path, device=asset.device, dtype=dtype)
    phase_lower, phase_upper = phase_time_range
    # Config floats and float32 reference tensors may round the same endpoint
    # differently (e.g. 1.4 vs 1.399999976). Permit only numerical roundoff.
    duration_tolerance = 4 * torch.finfo(dtype).eps * max(1.0, trajectory.duration)
    if phase_lower < 0.0 or phase_upper < phase_lower or phase_upper > trajectory.duration + duration_tolerance:
        raise ValueError(f"phase_time_range must lie in [0, {trajectory.duration}], got {phase_time_range}.")
    phase_lower = min(phase_lower, trajectory.duration)
    phase_upper = min(phase_upper, trajectory.duration)

    # Begin with the exact flat-task reset semantics for every environment.
    default_root_state = asset.data.default_root_state[env_ids].clone()
    pose_samples = _sample_ranges(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        len(env_ids),
        asset.device,
        dtype,
    )
    root_position = default_root_state[:, :3] + env.scene.env_origins[env_ids] + pose_samples[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    root_orientation = math_utils.quat_mul(default_root_state[:, 3:7], orientation_delta)
    velocity_samples = _sample_ranges(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        len(env_ids),
        asset.device,
        dtype,
    )
    root_velocity = default_root_state[:, 7:13].clone() + velocity_samples
    joint_position = asset.data.default_joint_pos[env_ids].clone()
    joint_velocity = asset.data.default_joint_vel[env_ids].clone()
    joint_position *= math_utils.sample_uniform(*joint_position_range, joint_position.shape, device=asset.device)
    joint_velocity *= math_utils.sample_uniform(*joint_velocity_range, joint_velocity.shape, device=asset.device)

    jump_mask = torch.rand(len(env_ids), device=asset.device) < jump_probability
    phase_time = torch.zeros(len(env_ids), device=asset.device, dtype=dtype)
    jump_ids = jump_mask.nonzero(as_tuple=False).flatten()
    phase_time[jump_ids] = _stratified_phase_times(jump_ids.numel(), phase_lower, phase_upper, asset.device, dtype)
    trigger_position = root_position.clone()

    if jump_ids.numel() > 0:
        sample = trajectory.sample(phase_time[jump_ids])
        if align_reference_height:
            # Full-state references must place the sampled joints at their
            # physical ground-relative height, not add the default standing z.
            height = (
                env.scene.env_origins[env_ids[jump_ids], 2] + trajectory.base_position[0, 2] + pose_samples[jump_ids, 2]
            )
            trigger_position[jump_ids, 2] = height
            root_position[jump_ids, 2] = height
        zero_angle = torch.zeros(jump_ids.numel(), device=asset.device, dtype=dtype)
        yaw_quat = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, pose_samples[jump_ids, 5])
        displacement = sample.base_position - trajectory.base_position[0]
        root_position[jump_ids] += math_utils.quat_apply(yaw_quat, displacement)

        pitch_quat = math_utils.quat_from_euler_xyz(zero_angle, sample.base_pitch, zero_angle)
        if getattr(sample, "base_orientation", None) is not None:
            pitch_quat = sample.base_orientation
        root_orientation[jump_ids] = math_utils.quat_mul(root_orientation[jump_ids], pitch_quat)
        root_velocity[jump_ids, :3] = math_utils.quat_apply(yaw_quat, sample.base_linear_velocity)
        reference_angular_velocity = torch.stack(
            (
                torch.zeros_like(sample.base_pitch_rate),
                sample.base_pitch_rate,
                torch.zeros_like(sample.base_pitch_rate),
            ),
            dim=-1,
        )
        if getattr(sample, "base_angular_velocity_world", None) is not None:
            reference_angular_velocity = sample.base_angular_velocity_world
        root_velocity[jump_ids, 3:] = math_utils.quat_apply(yaw_quat, reference_angular_velocity)

        leg_joint_ids = _resolve_leg_joint_ids(
            env, asset, asset_cfg.name, leg_joint_names, sample.joint_position.shape[-1]
        )
        joint_position[jump_ids[:, None], leg_joint_ids] = sample.joint_position
        joint_velocity[jump_ids[:, None], leg_joint_ids] = sample.joint_velocity
        if getattr(sample, "wheel_velocity", None) is not None:
            wheel_names = list(trajectory.metadata["wheel_joint_names"])
            wheel_joint_ids = asset.find_joints(wheel_names, preserve_order=True)[0]
            if len(wheel_joint_ids) != 4:
                raise ValueError("Wheel reference must resolve exactly four joints")
            joint_position[jump_ids[:, None], wheel_joint_ids] = sample.wheel_position
            joint_velocity[jump_ids[:, None], wheel_joint_ids] = sample.wheel_velocity

    joint_position_limits = asset.data.soft_joint_pos_limits[env_ids].clone()
    if preserve_reference_joint_positions and jump_ids.numel() > 0:
        # Keep physically validated reference states intact inside the true
        # articulation limits. Ordinary flat resets retain their soft limits.
        joint_position_limits[jump_ids] = asset.data.joint_pos_limits[env_ids[jump_ids]]
    joint_position.clamp_(joint_position_limits[..., 0], joint_position_limits[..., 1])
    joint_velocity_limits = asset.data.soft_joint_vel_limits[env_ids]
    # Isaac Lab fills actuator soft velocity limits on the first control
    # application; before then the buffer is zero, not a zero-speed constraint.
    physical_velocity_limits = getattr(asset.data, "joint_vel_limits", None)
    if physical_velocity_limits is not None:
        joint_velocity_limits = torch.where(
            joint_velocity_limits > 0, joint_velocity_limits, physical_velocity_limits[env_ids]
        )
    joint_velocity.clamp_(-joint_velocity_limits, joint_velocity_limits)

    asset.write_root_pose_to_sim(torch.cat((root_position, root_orientation), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)

    _ensure_timed_jump_buffers(env, dtype)
    env._jump_episode_mask[env_ids] = jump_mask
    env._jump_initial_phase_time[env_ids] = phase_time
    env._jump_trigger_position[env_ids] = trigger_position
    env._jump_target_height[env_ids] = float(trajectory.metadata["target_jump_height"])
    env._jump_reset_pending[env_ids] = True


__all__ = ["reset_timed_jump_or_flat_state"]
