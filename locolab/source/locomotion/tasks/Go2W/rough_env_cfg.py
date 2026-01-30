"""Go2W rough terrain environment configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.tasks.velocity_env_cfg import ROUGH_TERRAIN_CFG, LocomotionCurriculumCfg

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


# ==============================================================================
# Go2W Rough Environment Configuration
# ==============================================================================
@configclass
class Go2WRoughCurriculumCfg(LocomotionCurriculumCfg):
    """Curriculum terms for Go2W training on rough terrain."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class Go2WRoughRewardsCfg(Go2WRewardsCfg):
    """Reward terms for Go2W training on rough terrain."""

    # 降低关节偏离惩罚，允许更大幅度抬腿
    joint_pos_deviation = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.2,  # 从 -1.0 降到 -0.2
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"]),
            "stand_still_scale": 2.0,
            "velocity_threshold": 3,
            "ang_vel_threshold": 0.2,
        },
    )
    joint_pos_deviation_hip = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"]),
            "stand_still_scale": 2.0,
            "velocity_threshold": 3,
            "ang_vel_threshold": 0.2,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=4096, env_spacing=2.5)
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
