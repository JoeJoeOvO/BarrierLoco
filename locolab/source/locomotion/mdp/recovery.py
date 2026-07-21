"""State reset and reward helpers for Go2W recovery training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def normalized_pose_distribution(
    normal_pose_probability: float,
    special_poses: dict[str, dict[str, Any]],
) -> tuple[list[str], torch.Tensor]:
    """Validate and normalize a recovery pose distribution."""
    pose_names = ["normal", *special_poses]
    weights = [float(normal_pose_probability)]
    weights.extend(float(pose.get("probability", 0.0)) for pose in special_poses.values())
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Recovery pose probabilities must be non-negative.")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("Recovery pose distribution must have positive total probability.")
    return pose_names, torch.tensor([weight / total for weight in weights], dtype=torch.float32)


def sample_pose_indices(
    normal_pose_probability: float,
    special_poses: dict[str, dict[str, Any]],
    count: int,
    device: str | torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Sample one pose label per environment for both root and joint reset."""
    pose_names, probabilities = normalized_pose_distribution(normal_pose_probability, special_poses)
    pose_indices = torch.multinomial(probabilities.to(device=device), count, replacement=True)
    return pose_names, pose_indices


def _sample_ranges(
    ranges: dict[str, tuple[float, float]],
    keys: Sequence[str],
    count: int,
    device: str | torch.device,
) -> torch.Tensor:
    from isaaclab.utils import math as math_utils

    bounds = torch.tensor([ranges.get(key, (0.0, 0.0)) for key in keys], device=device)
    return math_utils.sample_uniform(bounds[:, 0], bounds[:, 1], (count, len(keys)), device=device)


def reset_robot_state_multi_pose(
    env,
    env_ids: torch.Tensor,
    normal_pose_probability: float,
    normal_pose_range: dict[str, tuple[float, float]],
    normal_velocity_range: dict[str, tuple[float, float]],
    normal_joint_position_range: tuple[float, float],
    normal_joint_velocity_range: tuple[float, float],
    special_poses: dict[str, dict[str, Any]],
    sample_joint_positions_from_limits: bool = False,
    asset_cfg=None,
) -> None:
    """Reset root and joints from one shared recovery-pose sample."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import math as math_utils

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    pose_names, pose_indices = sample_pose_indices(
        normal_pose_probability,
        special_poses,
        len(env_ids),
        asset.device,
    )

    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, :3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7].clone()
    velocities = root_states[:, 7:13].clone()

    joint_ids = asset_cfg.joint_ids
    joint_positions = asset.data.default_joint_pos[env_ids][:, joint_ids].clone()
    joint_velocities = asset.data.default_joint_vel[env_ids][:, joint_ids].clone()
    joint_position_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_velocity_limits = asset.data.soft_joint_vel_limits[env_ids][:, joint_ids]

    if sample_joint_positions_from_limits:
        finite_limits = (
            torch.isfinite(joint_position_limits[..., 0])
            & torch.isfinite(joint_position_limits[..., 1])
            & (joint_position_limits[..., 1] > joint_position_limits[..., 0])
        )
        lower = torch.where(finite_limits, joint_position_limits[..., 0], joint_positions)
        upper = torch.where(finite_limits, joint_position_limits[..., 1], joint_positions)
        sampled = math_utils.sample_uniform(lower, upper, joint_positions.shape, asset.device)
        joint_positions = torch.where(finite_limits, sampled, joint_positions)

    for pose_index, pose_name in enumerate(pose_names):
        selected = torch.nonzero(pose_indices == pose_index, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            continue
        pose = special_poses.get(pose_name)
        pose_range = normal_pose_range if pose is None else pose["pose_range"]
        velocity_range = normal_velocity_range if pose is None else pose["velocity_range"]
        joint_position_range = (
            normal_joint_position_range
            if pose is None
            else pose.get("joint_position_range", normal_joint_position_range)
        )
        joint_velocity_range = (
            normal_joint_velocity_range
            if pose is None
            else pose.get("joint_velocity_range", normal_joint_velocity_range)
        )

        pose_samples = _sample_ranges(
            pose_range,
            ("x", "y", "z", "roll", "pitch", "yaw"),
            len(selected),
            asset.device,
        )
        velocity_samples = _sample_ranges(
            velocity_range,
            ("x", "y", "z", "roll", "pitch", "yaw"),
            len(selected),
            asset.device,
        )
        positions[selected] += pose_samples[:, :3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_samples[:, 3],
            pose_samples[:, 4],
            pose_samples[:, 5],
        )
        orientations[selected] = math_utils.quat_mul(orientations[selected], orientation_delta)
        velocities[selected] += velocity_samples

        if not sample_joint_positions_from_limits:
            joint_positions[selected] += math_utils.sample_uniform(
                *joint_position_range,
                joint_positions[selected].shape,
                asset.device,
            )
        joint_velocities[selected] += math_utils.sample_uniform(
            *joint_velocity_range,
            joint_velocities[selected].shape,
            asset.device,
        )

    joint_positions.clamp_(joint_position_limits[..., 0], joint_position_limits[..., 1])
    joint_velocities.clamp_(-joint_velocity_limits, joint_velocity_limits)
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_positions, joint_velocities, joint_ids=joint_ids, env_ids=env_ids)


def upright_orientation_reward(env, asset_cfg=None) -> torch.Tensor:
    """Reward alignment of the base's up axis with world up."""
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)


def wheel_air_count(env, sensor_cfg, force_threshold: float) -> torch.Tensor:
    """Count wheels without recent contact."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_force = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1)
    return (max_force < force_threshold).sum(dim=1).float()


def recovering_joint_velocity_penalty(
    env,
    sensor_cfg,
    asset_cfg,
    max_joint_velocity: float,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize excessive leg speed until every wheel regains contact."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_force = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1)
    recovering = (max_force > force_threshold).sum(dim=1) < max_force.shape[1]
    joint_velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    excess = (joint_velocities.abs() - max_joint_velocity).clamp_min(0.0).sum(dim=1)
    return excess * recovering
