"""Go2W flat terrain environment configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG, GO2W_SDK_JOINT_NAMES
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCurriculumCfg,
    LocomotionEnvCfg,
    LocomotionObservationsCfg,
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
        # Export/deploy needs the SDK joint order explicitly.
        self.robot.joint_sdk_names = GO2W_SDK_JOINT_NAMES


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
    """Reward terms for Go2W wheel-legged robot.

    Reference: ETH RSL "Learning Robust Autonomous Navigation and Locomotion
    for Wheeled-Legged Robots" (Science Robotics 2024), Section S3.
    """

    # ==========================================================================
    # BASE STABILITY (ETH-style)
    # ==========================================================================

    # -- base height with tolerance band (ETH: r_h = max(0, |h-0.42|-0.05))
    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-5.0,
        params={
            "target_height": 0.42,
            "tolerance": 0.02,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # ==========================================================================
    # JOINT REGULARIZATION (ETH: r_s for legs only, r_tau for all)
    # ==========================================================================

    # -- joint motion: velocity^2 + 0.01*acceleration^2 (ETH: r_s, legs only)
    # ETH r_s = -c_k * Σ(q̇² + 0.01*q̈²), ratio acc/vel = 0.01
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-7,  # conservative step from original -2.5e-7; ETH ratio is aspirational target
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*", ".*_foot_.*"])
        },
    )

    # -- torque penalty (ETH: r_tau = -Σ||τ_i||^2, all joints)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    dof_power_l1 = RewTerm(
        func=mdp.dof_power_l1,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # -- joint position limits (ETH: r_jc, knee constraint soft penalty only)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_.*"])},
    )

    # ==========================================================================
    # ACTION REGULARIZATION (ETH: 1st + 2nd order target smoothness)
    # ==========================================================================

    # -- action rate (1st order finite diff, legs)
    action_rate_legs = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )
    # -- action smoothness (2nd order finite diff, legs, ETH: target smoothness)
    action_smoothness_legs = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])},
    )

    # -- action rate (1st order, wheels, lighter penalty)
    action_rate_wheels = RewTerm(
        func=mdp.action_rate_l2_mask,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )
    # -- action smoothness (2nd order, wheels)
    action_smoothness_wheels = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"])},
    )

    # ==========================================================================
    # CONTACT REGULARIZATION (ETH: r_bc = bad contacts except wheels)
    # ==========================================================================

    # -- undesired contacts (ETH: penalize contacts except wheels)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf", "base"]),
        },
    )

    # -- joint position default deviation (wheeled-legged specific: prevents legs floating)
    # NOTE: ETH ANYmal doesn't need this because its leg structure naturally resists floating.
    # Go2W's wheels allow the policy to exploit leg-floating as a local optimum without this term.
    joint_pos_deviation = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
            "stand_still_scale": 2.0,
            "velocity_threshold": 3,
            "ang_vel_threshold": 0.2,
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
# Go2W Flat Environment Configuration
# ==============================================================================


@configclass
class Go2WFlatEnvCfg(LocomotionEnvCfg):
    """Go2W velocity tracking environment on flat terrain."""

    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2WActionsCfg = Go2WActionsCfg()
    observations: Go2WObservationsCfg = Go2WObservationsCfg()
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()
    curriculum: Go2WCurriculumCfg = Go2WCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Increase zero-command training ratio for better stand-still behavior
        self.commands.base_velocity.rel_standing_envs = 0.25


@configclass
class Go2WFlatPlayEnvCfg(Go2WFlatEnvCfg):
    """Go2W play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Play/deploy should use clean observations.
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        # Disable training-time randomization for evaluation/deploy parity.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
