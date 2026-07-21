"""Go2W flat terrain environment configuration."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG, GO2W_SDK_JOINT_NAMES
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCommandsCfg,
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
# Go2W Commands Configuration
# ==============================================================================


@configclass
class Go2WFlatCommandsCfg(LocomotionCommandsCfg):
    """Flat commands with an easy start and bounded curriculum limits."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.8,
        heading_command=True,
        heading_control_stiffness=2.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ==============================================================================
# Go2W Rewards Configuration
# ==============================================================================


@configclass
class Go2WRewardsCfg(LocomotionRewardsCfg):
    """Reward terms for Go2W wheel-legged robot on flat terrain."""

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=5.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
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
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"])},
    )
    thigh_calf_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"])},
    )
    penalize_lin_vel_xy = RewTerm(
        func=mdp.penalize_lin_vel_xy,
        weight=-0.05,
        params={"command_name": "base_velocity"},
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", "base"]),
        },
    )
    thigh_calf_contact = RewTerm(
        func=mdp.no_leg_ground_contact,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_.*"])},
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

        feet_contact_force = ObsTerm(
            func=mdp.body_contact_force,
            scale=0.01,
            clip=(-100, 100),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        wheel_contact_state = ObsTerm(
            func=mdp.wheel_contact_state,
            clip=(0.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "force_threshold": 1.0},
        )

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
    """Flat terrain and velocity-command curricula for Go2W."""

    lin_vel_cmd = CurrTerm(
        func=mdp.lin_vel_cmd_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy",
            "reward_threshold": 0.75,
            "lower_threshold": 0.35,
            "step_size": 0.1,
            "min_lin_vel_x": (-0.5, 0.5),
            "min_lin_vel_y": (-0.2, 0.2),
            "max_lin_vel_x": (-2.0, 2.0),
            "max_lin_vel_y": (-0.5, 0.5),
        },
    )
    ang_vel_cmd = CurrTerm(
        func=mdp.ang_vel_cmd_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_ang_vel_z",
            "reward_threshold": 0.75,
            "lower_threshold": 0.35,
            "step_size": 0.1,
            "min_ang_vel_z": (-0.5, 0.5),
            "max_ang_vel_z": (-2.0, 2.0),
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
    commands: Go2WFlatCommandsCfg = Go2WFlatCommandsCfg()
    observations: Go2WObservationsCfg = Go2WObservationsCfg()
    rewards: Go2WRewardsCfg = Go2WRewardsCfg()
    curriculum: Go2WCurriculumCfg = Go2WCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.gpu_max_rigid_patch_count = 2**19


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
