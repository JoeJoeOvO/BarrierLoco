"""Go2W rough terrain environment configuration."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.mdp.m20 import rewards as m20_rewards
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCommandsCfg,
    LocomotionCurriculumCfg,
    LocomotionRewardsCfg,
)

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WSceneCfg

# ==============================================================================
# Go2W Rough Scene Configuration
# ==============================================================================

GO2W_HIMLOCO_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.05, 0.25), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class Go2WRoughSceneCfg(Go2WSceneCfg):
    """Scene configuration for Go2W on rough terrain."""

    # Use a tighter footprint for base-height reward on rough terrain so the
    # target tracks the support region more closely near steps and boxes.
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground/terrain"],
    )
    # Retained for diagnostics and the historical M20 recipe. Active posture
    # relaxation uses height_scanner's footprint and directional preview instead.
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground/terrain"],
    )

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = GO2W_HIMLOCO_ROUGH_TERRAIN_CFG
        self.terrain.max_init_terrain_level = 5
        self.terrain.physics_material.static_friction = 0.8
        self.terrain.physics_material.dynamic_friction = 0.8
        # Keep a longer contact history for reactive obstacle-lift shaping.
        self.contact_forces.history_length = 6
        self.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]


# ==============================================================================
# Go2W Rough Commands Configuration
# ==============================================================================


@configclass
class Go2WRoughCommandsCfg(LocomotionCommandsCfg):
    """Fixed, capped commands for Go2W rough terrain."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.9,
        heading_command=True,
        heading_control_stiffness=2.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ==============================================================================
# Go2W Rough Curriculum Configuration
# ==============================================================================


@configclass
class Go2WRoughCurriculumCfg(LocomotionCurriculumCfg):
    """Keep distance-based terrain progression with fixed reward weights."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_reward)
    lin_vel_cmd = None
    ang_vel_cmd = None


# ==============================================================================
# Go2W Rough Rewards Configuration
# ==============================================================================


@configclass
class Go2WRoughRewardsCfg(LocomotionRewardsCfg):
    """Reward terms for Go2W training on rough terrain.

    Independently defined from LocomotionRewardsCfg (not inherited from Go2WRewardsCfg).
    """

    stay_alive = None
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    penalize_ang_vel_z = RewTerm(
        func=mdp.penalize_ang_vel_z,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    velocity_slack = RewTerm(
        func=mdp.velocity_slack,
        weight=0.25,
        params={"command_name": "base_velocity", "min_ratio": 0.3, "max_ratio": 1.5, "command_threshold": 0.0},
    )
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
    base_height = None
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # -- contact regularization
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", "base"]),
        },
    )
    # M20's 100 N threshold scaled by nominal mass: 19.523 / 34.4914.
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 56.6,
        },
    )
    penalize_lin_vel_xy = RewTerm(
        func=mdp.penalize_lin_vel_xy,
        weight=-0.05,
        params={"command_name": "base_velocity"},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_se2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            # Apply only to near-zero translation and yaw commands.
            "command_threshold": 0.06,
            "angular_command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )

    # -- posture regularization
    hip_joint_pos_penalty = RewTerm(
        func=m20_rewards.joint_pos_penalty_relax_side_cmd,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "side_scale": 0.5,
            "y_cmd_threshold": 0.1,
            "xz_norm_max": 0.1,
        },
    )
    thigh_joint_pos_penalty = RewTerm(
        func=m20_rewards.joint_pos_penalty_local_terrain,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "ang_cmd_threshold": 0.1,
            "y_cmd_threshold": 0.1,
            "xy_norm_max": 0.1,
            "xz_norm_max": 0.1,
            "terrain_height_threshold": 0.06,
            "high_terrain_penalty_scale": 0.1,
            "footprint_half_length": 0.3,
            "footprint_half_width": 0.25,
            "lookahead_distance": 0.5,
            "min_valid_rays": 10,
        },
    )
    calf_joint_pos_penalty = RewTerm(
        func=m20_rewards.joint_pos_penalty_local_terrain,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "ang_cmd_threshold": 0.1,
            "y_cmd_threshold": 0.1,
            "xy_norm_max": 0.1,
            "xz_norm_max": 0.1,
            "terrain_height_threshold": 0.06,
            "high_terrain_penalty_scale": 0.1,
            "footprint_half_length": 0.3,
            "footprint_half_width": 0.25,
            "lookahead_distance": 0.5,
            "min_valid_rays": 10,
        },
    )
    joint_mirror = RewTerm(
        func=m20_rewards.joint_mirror_except_side_cmd,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "y_cmd_threshold": 0.1,
            "xz_norm_max": 0.1,
            "mirror_joints": [
                ["FL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"],
                ["FR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"],
            ],
        },
    )
    rotation_gait_status = RewTerm(
        func=m20_rewards.rotation_gait_status_per_wheel,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "RR_foot", "FR_foot", "RL_foot"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "RR_foot", "FR_foot", "RL_foot"], preserve_order=True
            ),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_difference_std": 0.05,
            "wheel_radius": 0.086,
            "wheel_width": 0.045,
            "wheel_center_offsets": (0.048, -0.048, -0.048, 0.048),
            "max_ray_distance": 0.15,
            "lin_vel_threshold": 0.5,
            "ang_vel_threshold": 0.05,
        },
    )
    feet_air_time_ang_z_M20 = None
    feet_slide_ang_z_cmd = RewTerm(
        func=m20_rewards.feet_slide_ang_z_cmd_unscaled,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "cmd_lin_threshold": 0.1,
            "cmd_ang_threshold": 0.1,
        },
    )
    rotation_gait_symmetry = RewTerm(
        func=m20_rewards.RotationGaitPerWheelDuty,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "RR_foot", "FR_foot", "RL_foot"], preserve_order=True
            ),
            "target_duty": 0.5,
            "std": 0.2,
            "swing_time_std": 0.1,
            "lin_vel_threshold": 0.5,
            "ang_vel_threshold": 0.05,
        },
    )
    wheel_air_time_excess = RewTerm(
        func=mdp.wheel_air_time_excess,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_air_time": 0.3,
            "lin_vel_threshold": 0.5,
            "ang_vel_threshold": 0.05,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,
        weight=-0.002,
        params={"action_term_names": ["joint_pos", "joint_vel"]},
    )


# ==============================================================================
# Go2W Rough Environment Configuration
# ==============================================================================


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=8192, env_spacing=2.5)
    commands: Go2WRoughCommandsCfg = Go2WRoughCommandsCfg()
    curriculum: Go2WRoughCurriculumCfg = Go2WRoughCurriculumCfg()
    rewards: Go2WRoughRewardsCfg = Go2WRoughRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt
        self.terminations.bad_orientation = None


@configclass
class Go2WRoughPlayEnvCfg(Go2WRoughEnvCfg):
    """Go2W rough terrain play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Fewer terrain patches for easier visual inspection.
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
