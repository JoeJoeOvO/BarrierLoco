"""Go2W rough terrain environment configuration."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
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
    """Terrain-only curriculum; velocity ranges remain fixed."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_reward,
        params={"command_name": "base_velocity"},
    )
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

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=5.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
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
    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-5.0,
        params={
            "target_height": 0.42,
            "tolerance": 0.02,
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
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
    penalize_lin_vel_xy = RewTerm(
        func=mdp.penalize_lin_vel_xy,
        weight=-0.05,
        params={"command_name": "base_velocity"},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )

    # -- posture regularization
    hip_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


# ==============================================================================
# Go2W Rough Environment Configuration
# ==============================================================================


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WRoughCommandsCfg = Go2WRoughCommandsCfg()
    curriculum: Go2WRoughCurriculumCfg = Go2WRoughCurriculumCfg()
    rewards: Go2WRoughRewardsCfg = Go2WRoughRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
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
