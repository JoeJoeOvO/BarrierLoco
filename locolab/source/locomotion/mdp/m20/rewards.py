# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""Active M20 rewards, adapted for reset-safe histories and Go2W link names.

Source: DeepRoboticsLab/rl_training@6d317dfb33060226139e38600510e1751372eb5d.
See the migration notes and licenses in this package.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.envs.mdp as lab_mdp
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .. import rewards as common_rewards
from .curriculums import gait_level


def joint_torques_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return lab_mdp.joint_torques_l2(env, asset_cfg) * gait_level(env)


def action_rate_l2(env) -> torch.Tensor:
    return lab_mdp.action_rate_l2(env) * gait_level(env)


class ActionSmoothnessPenalty(common_rewards.ActionSmoothnessPenalty):
    """Use the existing reset-aware second difference, scaled by the M20 curriculum."""

    def __call__(self, env, action_term_names: Sequence[str] | None = None) -> torch.Tensor:
        return super().__call__(env, action_term_names=action_term_names) * gait_level(env)


def contact_forces(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    excess = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) - threshold
    return excess.clamp(min=0.0).sum(dim=1) * gait_level(env)


def joint_pos_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=1)
    speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    deviation = torch.linalg.vector_norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )
    return torch.where(
        (command > command_threshold) | (speed > velocity_threshold), deviation, stand_still_scale * deviation
    )


def joint_pos_penalty_relax_side_cmd(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    side_scale: float = 0.5,
    y_cmd_threshold: float = 0.1,
    xz_norm_max: float = 0.1,
) -> torch.Tensor:
    """Reduce the default posture cost for pure lateral commands in either direction."""
    reward = joint_pos_penalty(env, command_name, asset_cfg)
    command = env.command_manager.get_command(command_name)
    side = (command[:, 1].abs() > y_cmd_threshold) & (torch.linalg.vector_norm(command[:, [0, 2]], dim=1) < xz_norm_max)
    return reward * torch.where(side, side_scale, 1.0)


def joint_pos_penalty_except_turn_side_cmd(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    terrain_height_threshold: float = 0.06,
    high_terrain_penalty_scale: float = 0.1,
) -> torch.Tensor:
    reward = joint_pos_penalty(env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold)
    command = env.command_manager.get_command(command_name)
    turn = (command[:, 2].abs() > ang_cmd_threshold) & (torch.linalg.vector_norm(command[:, :2], dim=1) < xy_norm_max)
    side = (command[:, 1].abs() > y_cmd_threshold) & (torch.linalg.vector_norm(command[:, [0, 2]], dim=1) < xz_norm_max)
    hits = env.scene[sensor_cfg.name].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits) & (hits.abs() <= 1e6)
    count = valid.sum(dim=1)
    height = torch.where(valid, hits, 0.0).sum(dim=1) / count.clamp(min=1) - env.scene.env_origins[:, 2]
    high_terrain = (height > terrain_height_threshold) & (count > 0)
    scale = torch.where(high_terrain, high_terrain_penalty_scale, 1.0)
    return reward * (~(turn | side)) * scale


def joint_pos_penalty_local_terrain(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    terrain_height_threshold: float = 0.06,
    high_terrain_penalty_scale: float = 0.1,
    footprint_half_length: float = 0.3,
    footprint_half_width: float = 0.25,
    lookahead_distance: float = 0.5,
    min_valid_rays: int = 10,
) -> torch.Tensor:
    """Relax posture on local height variation under the robot and ahead of its command.

    The 5th-to-95th percentile height span ignores isolated outliers and absolute
    tile elevation. The preview follows the planar command even when blocked;
    zero planar commands inspect only the footprint. Slopes can also activate
    this terrain-variation gate. Turn/side exemptions and static scaling remain.
    """
    reward = joint_pos_penalty(env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold)
    command = env.command_manager.get_command(command_name)
    turn = (command[:, 2].abs() > ang_cmd_threshold) & (torch.linalg.vector_norm(command[:, :2], dim=1) < xy_norm_max)
    side = (command[:, 1].abs() > y_cmd_threshold) & (torch.linalg.vector_norm(command[:, [0, 2]], dim=1) < xz_norm_max)

    asset = env.scene[asset_cfg.name]
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    valid = torch.isfinite(hits).all(dim=-1) & (hits[..., 2].abs() <= 1e6)
    relative = torch.where(valid[..., None], hits - asset.data.root_pos_w[:, None, :], 0.0)
    # The command and the preview corridor are in the yaw frame, not the tilted base frame.
    heading = asset.data.heading_w[:, None]
    cos_yaw, sin_yaw = heading.cos(), heading.sin()
    x = cos_yaw * relative[..., 0] + sin_yaw * relative[..., 1]
    y = -sin_yaw * relative[..., 0] + cos_yaw * relative[..., 1]
    epsilon = 1e-4  # Preserve grid-boundary rays after world-coordinate float32 rounding.
    footprint = (x.abs() <= footprint_half_length + epsilon) & (y.abs() <= footprint_half_width + epsilon)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    direction = command[:, :2] / speed.clamp(min=1e-6)[:, None]
    along = x * direction[:, 0, None] + y * direction[:, 1, None]
    across = -x * direction[:, 1, None] + y * direction[:, 0, None]
    preview = (
        (speed[:, None] > command_threshold)
        & (along >= -epsilon)
        & (along <= lookahead_distance + epsilon)
        & (across.abs() <= footprint_half_width + epsilon)
    )
    selected = valid & (footprint | preview)
    heights = torch.where(selected, hits[..., 2], float("nan"))
    quantiles = torch.nanquantile(heights, torch.tensor([0.05, 0.95], device=hits.device, dtype=hits.dtype), dim=1)
    local_span = quantiles[1] - quantiles[0]
    rough = (selected.sum(dim=1) >= min_valid_rays) & (local_span > terrain_height_threshold)
    scale = torch.where(rough, high_terrain_penalty_scale, 1.0)
    return reward * (~(turn | side)) * scale


def joint_mirror_unscaled(env, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """Diagonal angle penalty with the upright factor and no terrain-level dependency."""
    asset = env.scene[asset_cfg.name]
    key = (asset_cfg.name, tuple(tuple(pair) for pair in mirror_joints))
    if not hasattr(env, "_m20_mirror_ids"):
        env._m20_mirror_ids = {}
    if key not in env._m20_mirror_ids:
        env._m20_mirror_ids[key] = [(asset.find_joints(a)[0], asset.find_joints(b)[0]) for a, b in mirror_joints]
    reward = torch.zeros(env.num_envs, device=env.device)
    for left, right in env._m20_mirror_ids[key]:
        reward += (asset.data.joint_pos[:, left] - asset.data.joint_pos[:, right]).square().sum(dim=1)
    reward /= max(len(mirror_joints), 1)
    upright = (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
    return reward * upright


def joint_mirror(env, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """Retain the original scaling for the unregistered historical M20 recipe."""
    return joint_mirror_unscaled(env, asset_cfg, mirror_joints) * gait_level(env)


def feet_slide_ang_z_cmd_unscaled(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    cmd_lin_threshold: float = 0.1,
    cmd_ang_threshold: float = 0.1,
) -> torch.Tensor:
    """Preserve upstream body-relative wheel-center velocity, not physical tire slip."""
    asset = env.scene[asset_cfg.name]
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contacts = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > 1.0
    velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] - asset.data.root_lin_vel_w[:, None, :]
    quats = asset.data.root_quat_w[:, None, :].expand(-1, velocity.shape[1], -1)
    velocity_b = quat_apply_inverse(quats.reshape(-1, 4), velocity.reshape(-1, 3)).reshape_as(velocity)
    reward = (torch.linalg.vector_norm(velocity_b[..., :2], dim=-1) * contacts).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.vector_norm(command[:, :2], dim=1) < cmd_lin_threshold) & (
        command[:, 2].abs() > cmd_ang_threshold
    )
    return reward * gate


def feet_slide_ang_z_cmd(env, command_name, sensor_cfg, asset_cfg, cmd_lin_threshold=0.1, cmd_ang_threshold=0.1):
    """Preserve gait scaling for the historical M20 recipe."""
    return feet_slide_ang_z_cmd_unscaled(
        env, command_name, sensor_cfg, asset_cfg, cmd_lin_threshold, cmd_ang_threshold
    ) * gait_level(env)


class FeetAirTime(ManagerTermBase):
    """Yaw-gated landing reward with a per-environment, reset-aware clearance history."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        self._height = torch.zeros_like(env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2])
        self._peak = torch.zeros_like(self._height)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._initialized[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        threshold: float = 0.2,
        cmd_threshold: float = 0.1,
        foot_height_threshold: float = 0.0,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        height = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2, None]
        fresh = ~self._initialized
        self._height[fresh] = height[fresh]
        self._peak[fresh] = height[fresh]
        self._initialized.fill_(True)
        contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        self._peak.copy_(torch.where(contact, self._peak, torch.maximum(self._peak, height)))
        cleared = (self._peak - self._height) > foot_height_threshold
        landed = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = ((air_time - threshold) * landed * cleared).sum(dim=1)
        self._height.copy_(torch.where(contact, height, self._height))
        self._peak.copy_(torch.where(contact, height, self._peak))
        gate = env.command_manager.get_command(command_name)[:, 2].abs() > cmd_threshold
        return reward * gate * gait_level(env)


def rotation_gait_status_unscaled(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    lin_vel_threshold: float = 0.5,
    ang_vel_threshold: float = 0.05,
) -> torch.Tensor:
    """Selectors must order the two diagonal pairs as FL, RR, FR, RL."""
    contacts = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    heights = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    delta = heights[:, 2:].mean(dim=1) - heights[:, :2].mean(dim=1)
    pattern = (contacts[:, :2].all(dim=1) & (delta > target_height)) | (
        contacts[:, 2:].all(dim=1) & (-delta > target_height)
    )
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.vector_norm(command[:, :2], dim=1) < lin_vel_threshold) & (
        command[:, 2].abs() > ang_vel_threshold
    )
    return pattern.float() * gate


def rotation_gait_status(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    lin_vel_threshold: float = 0.5,
    ang_vel_threshold: float = 0.05,
) -> torch.Tensor:
    """Retain the original scaling for the unregistered historical M20 recipe."""
    return rotation_gait_status_unscaled(
        env, command_name, sensor_cfg, asset_cfg, target_height, lin_vel_threshold, ang_vel_threshold
    ) * gait_level(env)


class RotationGaitSymmetryUnscaled(ManagerTermBase):
    """Upstream five-second duty-cycle reward with valid sample counts for partial resets."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._window = max(1, round(5.0 / env.step_dt))
        self._history = torch.zeros(env.num_envs, 4, self._window, device=env.device)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._index = 0

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._history[ids] = 0.0
        self._count[ids] = 0

    def _update_duty(self, env, sensor_cfg):
        contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        self._history[:, :, self._index] = contact.float()
        self._index = (self._index + 1) % self._window
        self._count.add_(1).clamp_(max=self._window)
        # Keep the source's zero-padded five-second average and ramp, but restart the ramp per reset environment.
        duty = self._history.mean(dim=2)
        return duty

    def __call__(
        self,
        env,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        target_duty: float = 0.5,
        std: float = 0.2,
        lin_vel_threshold: float = 0.5,
        ang_vel_threshold: float = 0.05,
    ) -> torch.Tensor:
        duty = self._update_duty(env, sensor_cfg)
        error = (duty[:, :2].mean(dim=1) - target_duty).square() + (duty[:, 2:].mean(dim=1) - target_duty).square()
        reward = torch.exp(-error / std**2) * (self._count / self._window)
        command = env.command_manager.get_command(command_name)
        gate = (torch.linalg.vector_norm(command[:, :2], dim=1) < lin_vel_threshold) & (
            command[:, 2].abs() > ang_vel_threshold
        )
        return reward * gate


class RotationGaitSymmetry(RotationGaitSymmetryUnscaled):
    """Preserve gait scaling only for the unregistered historical M20 recipe."""

    def __call__(
        self,
        env,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        target_duty: float = 0.5,
        std: float = 0.2,
        lin_vel_threshold: float = 0.5,
        ang_vel_threshold: float = 0.05,
    ) -> torch.Tensor:
        return super().__call__(
            env, command_name, sensor_cfg, target_duty, std, lin_vel_threshold, ang_vel_threshold
        ) * gait_level(env)


def bad_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    return (gravity[:, 2] > 0.0) | (gravity[:, :2].abs() > 0.7).any(dim=1)


def bad_orientation_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return bad_orientation(env, asset_cfg).float()


def joint_mirror_except_side_cmd(
    env,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    command_name: str,
    y_cmd_threshold: float = 0.1,
    xz_norm_max: float = 0.1,
) -> torch.Tensor:
    """Release diagonal angle coupling for pure lateral commands, in either direction."""
    command = env.command_manager.get_command(command_name)
    side = (command[:, 1].abs() > y_cmd_threshold) & (torch.linalg.vector_norm(command[:, [0, 2]], dim=1) < xz_norm_max)
    return joint_mirror_unscaled(env, asset_cfg, mirror_joints) * (~side)


def rotation_gait_status_per_wheel(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    height_difference_std: float = 0.05,
    wheel_radius: float = 0.086,
    wheel_width: float = 0.045,
    wheel_center_offsets: tuple[float, ...] = (0.048, -0.048, -0.048, 0.048),
    max_ray_distance: float = 0.15,
    lin_vel_threshold: float = 0.5,
    ang_vel_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward matching local clearances of the two airborne diagonal wheels.

    Selectors order wheels FL, RR, FR, RL. The URDF cylinder center offset,
    radius and width give each tilted wheel's lowest point above its closest
    valid terrain scan hit. Rays beyond max_ray_distance cannot earn credit.
    Only the difference between the two swing-wheel clearances is scored;
    there is no absolute target or excessive-height cost. Support wheels must
    both contact, while swing wheels must both be airborne with positive
    clearance. Discrete scan hits approximate
    terrain height and can be inaccurate at step edges.
    """
    if height_difference_std <= 0 or max_ray_distance <= 0 or wheel_radius <= 0 or wheel_width <= 0:
        raise ValueError("Height difference scale and wheel dimensions must be positive")
    contacts = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if len(wheel_center_offsets) != 4:
        raise ValueError("Expected four wheel center offsets in FL, RR, FR, RL order")
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    # The collision-cylinder axle is link-local Y. Compute its world direction
    # from Isaac's wxyz quaternion; wheel spin around Y leaves this axis unchanged.
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    w, x, y, z = quat.unbind(dim=-1)
    axle = torch.stack((2 * (x * y - w * z), 1 - 2 * (x.square() + z.square()), 2 * (y * z + w * x)), dim=-1)
    offsets = positions.new_tensor(wheel_center_offsets)
    positions = positions + axle * offsets[None, :, None]
    axle_z = axle[..., 2].clamp(-1.0, 1.0)
    vertical_extent = wheel_radius * (1 - axle_z.square()).clamp(min=0).sqrt() + 0.5 * wheel_width * axle_z.abs()
    hits = env.scene[terrain_sensor_cfg.name].data.ray_hits_w
    valid_hits = torch.isfinite(hits).all(dim=-1) & (hits.abs() <= 1e6).all(dim=-1)
    valid_positions = torch.isfinite(positions).all(dim=-1) & (positions.abs() <= 1e6).all(dim=-1)
    safe_hits = torch.where(valid_hits[..., None], hits, 0.0)
    safe_positions = torch.where(valid_positions[..., None], positions, 0.0)
    distances = (safe_positions[:, :, None, :2] - safe_hits[:, None, :, :2]).square().sum(dim=-1)
    distances = distances.masked_fill(~valid_hits[:, None, :], float("inf"))
    distance, nearest = distances.min(dim=-1)
    ground = safe_hits[..., 2].gather(1, nearest)
    clearance = torch.where(valid_positions, safe_positions[..., 2] - ground - vertical_extent, 0.0)
    valid = valid_positions & (distance <= max_ray_distance**2)
    airborne = (~contacts) & valid & (clearance > 0.0)
    height_difference = clearance[:, [0, 2]] - clearance[:, [1, 3]]
    score = torch.exp(-(height_difference / height_difference_std).square())
    first_swings = airborne[:, :2].all(dim=1) & contacts[:, 2:].all(dim=1)
    second_swings = airborne[:, 2:].all(dim=1) & contacts[:, :2].all(dim=1)
    reward = score[:, 0] * first_swings + score[:, 1] * second_swings
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.vector_norm(command[:, :2], dim=1) < lin_vel_threshold) & (
        command[:, 2].abs() > ang_vel_threshold
    )
    return reward * gate


class RotationGaitPerWheelDuty(RotationGaitSymmetryUnscaled):
    """Per-wheel duty error plus within-diagonal swing timer synchronization."""

    def __call__(
        self,
        env,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        target_duty: float = 0.5,
        std: float = 0.2,
        swing_time_std: float = 0.1,
        lin_vel_threshold: float = 0.5,
        ang_vel_threshold: float = 0.05,
    ) -> torch.Tensor:
        if std <= 0 or swing_time_std <= 0:
            raise ValueError("Duty and swing time standard deviations must be positive")
        duty = self._update_duty(env, sensor_cfg)
        # Keep the old scale when the two wheels in each diagonal already agree.
        error = 0.5 * (duty - target_duty).square().sum(dim=1)
        air_time = env.scene.sensors[sensor_cfg.name].data.current_air_time[:, sensor_cfg.body_ids]
        differences = air_time[:, [0, 2]] - air_time[:, [1, 3]]
        timing_error = (differences / swing_time_std).square().mean(dim=1)
        reward = torch.exp(-error / std**2 - timing_error) * (self._count / self._window)
        command = env.command_manager.get_command(command_name)
        gate = (torch.linalg.vector_norm(command[:, :2], dim=1) < lin_vel_threshold) & (
            command[:, 2].abs() > ang_vel_threshold
        )
        return reward * gate
