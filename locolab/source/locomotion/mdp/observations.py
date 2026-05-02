from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def external_force_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """External wrench applied to the configured body, flattened as force+torque."""
    asset: Articulation = env.scene[asset_cfg.name]
    external_wrench = torch.zeros((env.num_envs, 6), device=env.device)
    if asset._external_force_b is not None and asset._external_torque_b is not None:
        external_wrench[:, :3] = asset._external_force_b[:, asset_cfg.body_ids[0]]
        external_wrench[:, 3:] = asset._external_torque_b[:, asset_cfg.body_ids[0]]
    return external_wrench


def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint acceleration of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def body_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The linear velocity of specified bodies in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    # body_lin_vel_w shape: (num_envs, num_bodies, 3)
    vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    return vel.reshape(vel.shape[0], -1)


def body_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    body_contact_force = body_contact_force.reshape(body_contact_force.shape[0], -1)

    return body_contact_force


def wheel_contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0) -> torch.Tensor:
    """Binary wheel contact state for each configured body."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_norm = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return (contact_force_norm > force_threshold).float()


def terrain_height_stats(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Compact terrain summary from height scan: mean, std, min, max."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    valid_mask = torch.isfinite(heights)
    valid_count = valid_mask.sum(dim=1).clamp(min=1)

    masked_heights = torch.where(valid_mask, heights, torch.zeros_like(heights))
    mean_heights = masked_heights.sum(dim=1) / valid_count

    centered = torch.where(valid_mask, heights - mean_heights.unsqueeze(1), torch.zeros_like(heights))
    std_heights = torch.sqrt((centered.square().sum(dim=1) / valid_count).clamp(min=0.0))

    min_heights = torch.where(valid_mask, heights, torch.full_like(heights, float("inf"))).min(dim=1).values
    max_heights = torch.where(valid_mask, heights, torch.full_like(heights, float("-inf"))).max(dim=1).values
    min_heights = torch.where(valid_mask.any(dim=1), min_heights, torch.zeros_like(min_heights))
    max_heights = torch.where(valid_mask.any(dim=1), max_heights, torch.zeros_like(max_heights))

    return torch.stack((mean_heights, std_heights, min_heights, max_heights), dim=1)
