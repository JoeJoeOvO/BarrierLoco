"""Go2W flat terrain environment configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCurriculumCfg,
    LocomotionEnvCfg,
    LocomotionObservationsCfg,
    LocomotionRewardsCfg,
    LocomotionSceneCfg,
    LocomotionTerminationsCfg,
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

    # Joints - Legs (Position Control) - 小scale限制腿部运动范围
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"],
        scale=0.25,  # 从0.25降到0.1，限制腿部动作幅度
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

    # ==========================================================================
    # BASE STABILITY & REGULARIZATION
    # ==========================================================================

    # -- base stability (additional)
    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-5.0,
        params={
            "target_height": 0.42,
            "tolerance": 0.02,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    hip_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"]),
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_acc_l2_wheel = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_power_l1 = RewTerm(
        func=mdp.dof_power_l1,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )

    # -- action regularization (legs)
    action_rate_legs = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    action_smoothness_legs = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )

    # -- action regularization (wheels)
    action_rate_wheels = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-5e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )
    action_smoothness_wheels = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-5e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )

    # -- contact regularization
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf", "base"]),
        },
    )


# ==============================================================================
# Go2W Observations Configuration
# ==============================================================================


@configclass
class Go2WObservationsCfg(LocomotionObservationsCfg):
    """Observation specifications for Go2W - only leg joints in policy obs."""

    @configclass
    class PolicyCfg(LocomotionObservationsCfg.PolicyCfg):
        """Observations for policy group - only leg joint positions."""

        def __post_init__(self):
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])
            }
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*", ".*_foot_.*"]
                )
            }
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(LocomotionObservationsCfg.CriticCfg):
        """Observations for critic group - only leg joint positions."""

        def __post_init__(self):
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])
            }
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*", ".*_foot_.*"]
                )
            }
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ==============================================================================
# Go2W Curriculum Configuration
# ==============================================================================


@configclass
class Go2WCurriculumCfg(LocomotionCurriculumCfg):
    """Curriculum terms for Go2W training."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain_levels = None


# ==============================================================================
# Go2W Terminations Configuration
# ==============================================================================


@configclass
class Go2WFlatTerminationsCfg(LocomotionTerminationsCfg):
    """Termination terms for Go2W wheel-legged robot."""

    base_contact = DoneTerm(
        func=mdp.DelayedIllegalContact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0,
            "delay_steps": 5,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.DelayedBadOrientation,
        params={
            "limit_angle": 1.5,
            "delay_steps": 5,
        },
    )


# ==============================================================================
# Go2W Flat Environment Configuration
# ==============================================================================


@configclass
class Go2WFlatEnvCfg(LocomotionEnvCfg):
    """Go2W velocity tracking environment on flat terrain."""

    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2WActionsCfg = Go2WActionsCfg()
    observations: Go2WObservationsCfg = Go2WObservationsCfg()
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()
    terminations: Go2WFlatTerminationsCfg = Go2WFlatTerminationsCfg()
    curriculum: Go2WCurriculumCfg = Go2WCurriculumCfg()


@configclass
class Go2WFlatPlayEnvCfg(Go2WFlatEnvCfg):
    """Go2W play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
