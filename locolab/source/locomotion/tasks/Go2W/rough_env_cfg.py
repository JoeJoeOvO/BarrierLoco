"""Go2W rough terrain environment configuration."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    ROUGH_TERRAIN_CFG,
    LocomotionCommandsCfg,
    LocomotionCurriculumCfg,
)

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WRewardsCfg, Go2WSceneCfg

# ==============================================================================
# Go2W Rough Scene Configuration
# ==============================================================================


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
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        self.terrain.max_init_terrain_level = 5


# ==============================================================================
# Go2W Rough Environment Configuration
# ==============================================================================
@configclass
class Go2WRoughCommandsCfg(LocomotionCommandsCfg):
    """Command configuration for Go2W rough terrain with curriculum-compatible ranges."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.8,
        heading_command=True,
        heading_control_stiffness=2.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-math.pi / 4, math.pi / 4),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class Go2WRoughCurriculumCfg(LocomotionCurriculumCfg):
    """Curriculum terms for Go2W training on rough terrain.

    Per-axis velocity curriculum (参考 LocoLeggedWheel):
    - x/ang 分轴独立维护课程进度，互不干扰
    - 双向调整：成功率低于 lower_threshold 时降级，避免训练卡在过难任务
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # x 轴线速度课程：初始 ±0.5，逐步扩展到 ±1.5，失败时回退到初始范围
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "reward_threshold": 0.8,
            "lower_threshold": 0.6,
            "step_size": 0.1,
            "min_lin_vel_x": (-0.5, 0.5),
            "max_lin_vel_x": (-1.0, 1.0),
        },
    )

    # 角速度课程：初始 ±0.5，逐步扩展到 ±1.0，失败时回退到初始范围
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_curriculum,
        params={
            "reward_term_name": "track_ang_vel_z",
            "reward_threshold": 0.8,
            "lower_threshold": 0.6,
            "step_size": 0.1,
            "min_ang_vel_z": (-0.5, 0.5),
            "max_ang_vel_z": (-math.pi / 4, math.pi / 4),
        },
    )


@configclass
class Go2WRoughRewardsCfg(Go2WRewardsCfg):
    """Reward terms for Go2W training on rough terrain."""

    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-1.0,
        params={
            "target_height": 0.42,
            "tolerance": 0.02,
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WRoughCommandsCfg = Go2WRoughCommandsCfg()
    curriculum: Go2WRoughCurriculumCfg = Go2WRoughCurriculumCfg()
    rewards: Go2WRoughRewardsCfg = Go2WRoughRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class Go2WRoughPlayEnvCfg(Go2WRoughEnvCfg):
    """Go2W rough terrain play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
