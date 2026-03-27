"""Go2W rough terrain environment configuration."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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

    def __post_init__(self):
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        self.terrain.max_init_terrain_level = 2


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
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class Go2WRoughCurriculumCfg(LocomotionCurriculumCfg):
    """Curriculum terms for Go2W training on rough terrain."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy",
            "reward_threshold": 0.8,
            "step_size": 0.1,
            "max_lin_vel_x": (-1.5, 1.5),
        },
    )


@configclass
class Go2WRoughRewardsCfg(Go2WRewardsCfg):
    """Reward terms for Go2W training on rough terrain.

    Overrides joint_pos_deviation from Go2WRewardsCfg:
    - hip: keep stand_still_scale=2.0, moving_scale=0.5 (姿态稳定，但允许适当偏转)
    - thigh/calf: moving_scale=0.2 (抬腿自由度更大，rough 地形穿越需要)
    Adds feet_stumble on top.
    """

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )

    # Hip: 保持较强约束维持横向姿态稳定，移动时适度放松
    joint_pos_deviation_hip = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"]),
            "stand_still_scale": 2.0,
            "moving_scale": 0.5,
            "velocity_threshold": 0.5,
            "ang_vel_threshold": 0.4,
        },
    )

    # Thigh/Calf: 降低 moving 惩罚，允许抬腿和跨障碍
    joint_pos_deviation_leg = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"]),
            "stand_still_scale": 2.0,
            "moving_scale": 0.2,
            "velocity_threshold": 0.5,
            "ang_vel_threshold": 0.4,
        },
    )

    # 关闭继承自 flat 的 joint_pos_deviation（统一用上面两项替代）
    joint_pos_deviation = None  # type: ignore[assignment]


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WRoughCommandsCfg = Go2WRoughCommandsCfg()
    curriculum: Go2WRoughCurriculumCfg = Go2WRoughCurriculumCfg()
    rewards: Go2WRoughRewardsCfg = Go2WRoughRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0


@configclass
class Go2WRoughPlayEnvCfg(Go2WRoughEnvCfg):
    """Go2W rough terrain play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # 缩小地形用于 play 测试，增加地形种类
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5
