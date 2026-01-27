"""Go2W flat terrain environment configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCurriculumCfg,
    LocomotionEnvCfg,
    LocomotionRewardsCfg,
    LocomotionSceneCfg,
)

# ==============================================================================
# Go2W Scene Configuration
# ==============================================================================


@configclass
class Go2WSceneCfg(LocomotionSceneCfg):
    """Scene configuration for Go2W on flat terrain."""

    robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        self.terrain.terrain_type = "plane"
        self.terrain.terrain_generator = None


# ==============================================================================
# Go2W Actions Configuration (Wheel-Legged Robot)
# ==============================================================================


@configclass
class Go2WActionsCfg:
    """Action specifications for Go2W wheel-legged robot."""

    # Joints - Legs (Position Control)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"],
        scale=0.25,
        use_default_offset=True,
    )
    # Joints - Wheels (Velocity Control)
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_foot_.*"],
        scale=10.0,
        use_default_offset=True,
    )


# ==============================================================================
# Go2W Rewards Configuration
# ==============================================================================


@configclass
class Go2WRewardsCfg(LocomotionRewardsCfg):
    """Reward terms for Go2W wheel-legged robot."""

    # -- base stability (additional)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,  # 从-2.0降低，减少高度惩罚
        params={
            "target_height": 0.45,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # -- joint regularization
    joint_pos_deviation = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.5,  # 从-1.0降低
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.1,
            "ang_vel_threshold": 0.1,  # 转向时不惩罚关节偏离
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-8,  # 从-2.5e-7降低5倍，主要问题项
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,  # 从-2e-5降低
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,  # 从-2e-4降低
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_power_l1 = RewTerm(
        func=mdp.dof_power_l1,
        weight=-1e-5,  # 从-2e-5降低
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,  # 从-2.0降低4倍
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )

    # -- action regularization (legs)
    action_rate_legs = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    action_smoothness_legs = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )

    # -- action regularization (wheels) - 轮子需要快速响应，大幅降低惩罚
    action_rate_wheels = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-0.001,  # 从-0.01降低10倍
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )
    action_smoothness_wheels = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.001,  # 从-0.01降低10倍
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )

    # -- contact regularization
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf", "base"]),
        },
    )


# ==============================================================================
# Go2W Curriculum Configuration
# ==============================================================================


@configclass
class Go2WCurriculumCfg(LocomotionCurriculumCfg):
    """Curriculum terms for Go2W training."""

    velocity_curriculum = CurrTerm(
        func=mdp.velocity_cmd_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy",
            "reward_threshold": 0.9,
            "step_size": 0.1,
            "max_lin_vel_x": (-2.5, 2.5),
            "max_lin_vel_y": (-0.3, 0.3),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.terrain_levels = None


# ==============================================================================
# Go2W Flat Environment Configuration
# ==============================================================================


@configclass
class Go2WFlatEnvCfg(LocomotionEnvCfg):
    """Go2W velocity tracking environment on flat terrain."""

    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2WActionsCfg = Go2WActionsCfg()
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()
    curriculum: Go2WCurriculumCfg = Go2WCurriculumCfg()


@configclass
class Go2WFlatPlayEnvCfg(Go2WFlatEnvCfg):
    """Go2W play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 1
