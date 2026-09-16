"""Go2W-specific safe resets and explicitly mass-coupled inertia randomization."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as lab_mdp
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from .geometry import collision_lower_bound, load_collision_geometry

GO2W_URDF = Path(__file__).resolve().parents[2] / "assets/Go2W/urdf/go2w.urdf"
SAFE_PLATFORM_TERRAINS = {
    "MeshPyramidStairsTerrainCfg",
    "MeshInvertedPyramidStairsTerrainCfg",
    "MeshRandomGridTerrainCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfInvertedPyramidSlopedTerrainCfg",
}


def _reset_geometry(env, asset, asset_name: str):
    cache = getattr(env, "_m20_reset_geometry", None)
    if cache is None:
        cache = env._m20_reset_geometry = {}
    if asset_name not in cache:
        reset_params = env.cfg.events.reset_robot_joints.params
        if tuple(reset_params["position_range"]) != (1.0, 1.0):
            raise ValueError("Geometry-aware Go2W reset requires the fixed default-joint reset.")
        defaults = asset.data.default_joint_pos
        if not torch.equal(defaults, defaults[0].expand_as(defaults)):
            raise ValueError("Geometry-aware Go2W reset requires shared default joint positions.")
        limits = asset.data.soft_joint_pos_limits[0]
        positions = defaults[0].clamp(limits[:, 0], limits[:, 1]).detach().cpu().tolist()
        model = load_collision_geometry(GO2W_URDF, dict(zip(asset.joint_names, positions)))
        cache[asset_name] = (model.bounding_radius, model.tensors(asset.device))
    return cache[asset_name]


def reset_root_state_with_clearance(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    minimum_clearance: float = 0.03,
    safe_spawn_half_extent: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Lift only as needed, while keeping the full robot inside the terrain's safe center.

    Current M20 terrains have >=2 m central platforms, or a height-field origin at
    the maximum height of the central 2x2 m region. The inner 1.6x1.6 m region
    leaves room for height-field vertex shifts. No stale height-sensor data is used.
    """
    if not math.isfinite(minimum_clearance) or minimum_clearance < 0.0:
        raise ValueError("minimum_clearance must be finite and nonnegative.")
    if not 0.0 < safe_spawn_half_extent <= 0.8:
        raise ValueError("safe_spawn_half_extent must be in (0, 0.8].")
    asset = env.scene[asset_cfg.name]
    radius, geometry = _reset_geometry(env, asset, asset_cfg.name)
    for axis in ("x", "y"):
        if max(abs(value) for value in pose_range.get(axis, (0.0, 0.0))) + radius > safe_spawn_half_extent:
            raise ValueError("Reset translation plus robot collision envelope leaves the safe terrain center.")
    terrain_cfg = env.scene.terrain.cfg
    if terrain_cfg.terrain_type == "generator":
        for terrain in terrain_cfg.terrain_generator.sub_terrains.values():
            if type(terrain).__name__ == "HfRandomUniformTerrainCfg":
                continue
            if type(terrain).__name__ not in SAFE_PLATFORM_TERRAINS or getattr(terrain, "platform_width", 0.0) < 2.0:
                raise ValueError("Geometry-aware reset requires a >=2 m central platform or random height field.")
    elif terrain_cfg.terrain_type != "plane":
        raise ValueError("Geometry-aware reset supports only the configured generator or a plane.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    if len(env_ids) == 0:
        return
    lab_mdp.reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
    quaternion = asset.data.root_quat_w[env_ids]
    vertical = torch.zeros(len(env_ids), 3, device=asset.device)
    vertical[:, 2] = 1.0
    lowest = collision_lower_bound(quat_apply_inverse(quaternion, vertical), geometry)
    pose = torch.cat((asset.data.root_pos_w[env_ids], quaternion), dim=1)
    target_z = env.scene.env_origins[env_ids, 2] + minimum_clearance - lowest
    pose[:, 2] = torch.maximum(pose[:, 2], target_z)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)


def randomize_mass_consistent_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float] = (0.85, 1.15),
):
    """I = I_default * (mass / default_mass) * per-body scalar model error.

    Apply one positive scalar to the complete tensor, preserving principal axes
    and physical inertia inequalities. Recompute the reference on every call so
    reset/subset use cannot compound earlier noise or undo mass randomization.
    """
    low, high = inertia_distribution_params
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
        raise ValueError("Inertia multipliers must be finite, positive and ordered.")
    asset = env.scene[asset_cfg.name]
    inertias = asset.root_physx_view.get_inertias().clone()
    device = inertias.device
    ids = torch.arange(inertias.shape[0], device=device) if env_ids is None else env_ids.to(device=device)
    bodies = torch.arange(asset.num_bodies, device=device)[asset_cfg.body_ids]
    if ids.numel() == 0 or bodies.numel() == 0:
        return
    selection = (ids[:, None], bodies)
    current_mass = asset.root_physx_view.get_masses()[selection]
    default_mass = asset.data.default_mass.to(device)[selection]
    ratio = torch.where(default_mass > 0.0, current_mass / default_mass.clamp(min=1e-12), 1.0)
    reference = asset.data.default_inertia.to(device)[selection] * ratio[..., None]
    noise = sample_uniform(low, high, (ids.numel(), bodies.numel(), 1), device=device)
    inertias[selection] = reference * noise
    asset.root_physx_view.set_inertias(inertias, ids)
