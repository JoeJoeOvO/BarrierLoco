"""Go2W high-platform curriculum and deterministic play course."""

from __future__ import annotations

import copy
import math

import isaaclab.terrains as terrain_gen
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.tasks.velocity_env_cfg import LocomotionEventCfg
from locolab.source.locomotion.terrains.high_platform_cfg import (
    MeshHighPlatformSequenceTerrainCfg,
    MeshHighPlatformTerrainCfg,
)

from .rough_env_cfg import Go2WRoughCommandsCfg, Go2WRoughEnvCfg, Go2WRoughRewardsCfg, Go2WRoughSceneCfg


def joint_pair_symmetry_l2(
    env,
    asset_cfg: SceneEntityCfg,
    joint_pairs: list[list[str]],
    opposite_sign: bool = True,
) -> torch.Tensor:
    """Penalize position asymmetry for explicitly paired joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    cache_key = (asset_cfg.name, tuple(tuple(pair) for pair in joint_pairs), opposite_sign)
    cache = getattr(env, "_high_platform_joint_pair_cache", None)
    if cache is None:
        cache = {}
        env._high_platform_joint_pair_cache = cache
    if cache_key not in cache:
        cache[cache_key] = [(asset.find_joints(left)[0], asset.find_joints(right)[0]) for left, right in joint_pairs]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for left_ids, right_ids in cache[cache_key]:
        left_position = asset.data.joint_pos[:, left_ids]
        right_position = asset.data.joint_pos[:, right_ids]
        error = left_position + right_position if opposite_sign else left_position - right_position
        penalty += torch.sum(torch.square(error), dim=-1)
    if joint_pairs:
        penalty /= len(joint_pairs)
    return penalty


GO2W_HIGH_PLATFORM_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=36,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "high_platform": MeshHighPlatformTerrainCfg(
            proportion=1.0,
            platform_height_range=(0.10, 0.80),
            num_height_levels=36,
            platform_start_x=2.5,
            platform_length=1.5,
            platform_width=2.0,
            spawn_x=1.0,
            ground_thickness=1.0,
        ),
    },
)

GO2W_HIGH_PLATFORM_PLAY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    size=(24.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "high_platform_sequence": MeshHighPlatformSequenceTerrainCfg(
            proportion=1.0,
            platform_heights=(0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
            platform_start_x=2.5,
            platform_length=1.5,
            platform_gap=1.5,
            platform_width=2.0,
            spawn_x=1.0,
            ground_thickness=1.0,
        ),
    },
)


@configclass
class Go2WHighPlatformSceneCfg(Go2WRoughSceneCfg):
    """Rough-terrain sensors with the dedicated platform curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = copy.deepcopy(GO2W_HIGH_PLATFORM_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 15


@configclass
class Go2WHighPlatformCommandsCfg(Go2WRoughCommandsCfg):
    """Positive-x command with a fixed world heading."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=2.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.5, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-math.pi, math.pi),
            heading=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.5, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-math.pi, math.pi),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class Go2WHighPlatformEventsCfg(LocomotionEventCfg):
    """Training randomization used by the validated platform run."""

    wheel_resistance = EventTerm(
        func=mdp.wheel_resistance_xy,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "resample_probability": 0.01,
            "linear_res_coef_range": (0.0, 2.0),
            "quadratic_res_coef_range": (0.0, 0.2),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )


@configclass
class Go2WHighPlatformRewardsCfg(Go2WRoughRewardsCfg):
    """Frozen reward profile from the validated HighPlatform training run."""

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=5.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    track_lin_vel_xy_enhance = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )
    track_ang_vel_z_enhance = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", ".*_hip", "base"]),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ratio_threshold": 5.0,
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    hip_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"])},
    )
    hip_lr_symmetry = RewTerm(
        func=joint_pair_symmetry_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_pairs": [
                ["FL_hip_joint", "FR_hip_joint"],
                ["RL_hip_joint", "RR_hip_joint"],
            ],
            "opposite_sign": True,
        },
    )
    joint_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )


@configclass
class Go2WHighPlatformEnvCfg(Go2WRoughEnvCfg):
    """Training environment for the 10--80 cm platform curriculum."""

    scene: Go2WHighPlatformSceneCfg = Go2WHighPlatformSceneCfg(num_envs=8192, env_spacing=2.5)
    commands: Go2WHighPlatformCommandsCfg = Go2WHighPlatformCommandsCfg()
    rewards: Go2WHighPlatformRewardsCfg = Go2WHighPlatformRewardsCfg()
    events: Go2WHighPlatformEventsCfg = Go2WHighPlatformEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0.0, 0.0),
        }


@configclass
class Go2WHighPlatformPlayEnvCfg(Go2WHighPlatformEnvCfg):
    """Deterministic six-platform evaluation course."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = copy.deepcopy(GO2W_HIGH_PLATFORM_PLAY_TERRAIN_CFG)
        self.scene.terrain.max_init_terrain_level = 0
        self.episode_length_s = 30.0

        command_ranges = self.commands.base_velocity.ranges
        command_ranges.lin_vel_x = (1.0, 1.0)
        command_ranges.lin_vel_y = (0.0, 0.0)
        command_ranges.ang_vel_z = (-math.pi, math.pi)
        command_ranges.heading = (0.0, 0.0)
        command_limits = self.commands.base_velocity.limit_ranges
        command_limits.lin_vel_x = (1.0, 1.0)
        command_limits.lin_vel_y = (0.0, 0.0)
        command_limits.ang_vel_z = (-math.pi, math.pi)
        command_limits.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False

        self.curriculum.terrain_levels = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.viewer.eye = (-3.0, 3.5, 2.2)
        self.viewer.lookat = (1.0, 0.0, 0.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
        self.events.wheel_resistance = None
