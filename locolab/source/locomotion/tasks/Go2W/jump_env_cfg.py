"""Go2W rolling locomotion with repeatable timed jumps."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionCommandsCfg,
    LocomotionEventCfg,
    LocomotionTerminationsCfg,
)
from locolab.source.locomotion.terrains.grid_plane import MovingJumpPlaneTerrainImporter

from .flat_env_cfg import Go2WCurriculumCfg, Go2WFlatEnvCfg, Go2WObservationsCfg, Go2WRewardsCfg, Go2WSceneCfg

JUMP_PROFILE = mdp.GO2W_JUMP_PROFILE
JUMP_REFERENCE_PATH = str(JUMP_PROFILE.reference_path)
GO2W_REFERENCE_LEG_JOINTS = list(JUMP_PROFILE.leg_joint_names)


def apply_jump_training_stage(cfg, stage: str):
    """Preserve the skill recipe and hold forward heading in the speed stage."""
    if stage not in ("skill", "speed"):
        raise ValueError(f"Unknown jump training stage: {stage}")
    cfg.training_stage = stage
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5) if stage == "skill" else (0.2, 1.5)
    # Dense reference-state practice first; mostly ordinary ground starts once
    # the same policy is resumed for varied-speed rolling and jumping.
    cfg.events.reset_base.params["jump_probability"] = 0.8 if stage == "skill" else 0.2
    held_heading = stage == "speed"
    cfg.commands.jump_command.heading_hold_duration = (
        max(2.0, cfg.commands.jump_command.sequence_duration) if held_heading else 0.0
    )
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.velocity_commands.params["lock_jump_heading"] = held_heading
    cfg.rewards.track_lin_vel_xy.params["lock_jump_heading"] = held_heading
    cfg.rewards.jump_heading = (
        RewTerm(func=mdp.jump_heading_error_l2, weight=-20.0, params={"command_name": "jump_command"})
        if held_heading
        else None
    )
    cfg.rewards.jump_yaw_velocity = (
        RewTerm(func=mdp.jump_heading_yaw_velocity_l2, weight=-5.0, params={"command_name": "jump_command"})
        if held_heading
        else None
    )
    cfg.rewards.jump_lateral_velocity = (
        RewTerm(
            func=mdp.jump_heading_lateral_velocity_l2,
            weight=-10.0,
            params={"command_name": "jump_command", "velocity_command_name": "base_velocity"},
        )
        if held_heading
        else None
    )


@configclass
class Go2WJumpSceneCfg(Go2WSceneCfg):
    """Flat scene used to learn a vertical jump and landing."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.class_type = MovingJumpPlaneTerrainImporter


@configclass
class Go2WJumpCommandsCfg(LocomotionCommandsCfg):
    """Hold vx through each episode and sample repeated 0/1 jumps during motion."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.2, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=None,
    )
    jump_command = mdp.BinaryJumpCommandCfg(
        asset_name="robot",
        contact_sensor_name="contact_forces",
        foot_body_names=".*_foot",
        resampling_time_range=(2.0, 4.0),
        jump_probability=JUMP_PROFILE.jump_probability,
        initial_trigger_delay_range=(1.0, 2.0),
        require_grounded_trigger=True,
        minimum_forward_speed=0.1,
        contact_force_threshold=1.0,
        minimum_jump_height=JUMP_PROFILE.minimum_height,
        landing_velocity_threshold=0.2,
        landing_recovery_time=0.10,
        landing_attitude_limit=0.5,
        takeoff_time=JUMP_PROFILE.takeoff_time,
        apex_time=JUMP_PROFILE.apex_time,
        landing_time=JUMP_PROFILE.landing_time,
        # Preserve the complete validated TO and hold its final pose for 0.4 s.
        # Completion remains fixed-time, never contact-latched.
        sequence_duration=JUMP_PROFILE.sequence_duration + 0.4,
        debug_vis=False,
    )


@configclass
class Go2WJumpObservationsCfg(Go2WObservationsCfg):
    """Keep the flat observation layout and append one jump-command scalar."""

    @configclass
    class PolicyCfg(Go2WObservationsCfg.PolicyCfg):
        velocity_commands = ObsTerm(
            func=mdp.velocity_and_jump_commands,
            clip=(-100, 100),
            params={
                "velocity_command_name": "base_velocity",
                "jump_command_name": "jump_command",
            },
        )

    @configclass
    class CriticCfg(Go2WObservationsCfg.CriticCfg):
        velocity_commands = ObsTerm(
            func=mdp.velocity_and_jump_commands,
            clip=(-100, 100),
            params={
                "velocity_command_name": "base_velocity",
                "jump_command_name": "jump_command",
            },
        )
        # Privileged fixed-schedule phase; the actor still receives only the
        # public binary jump interface.
        jump_phase = ObsTerm(
            func=mdp.jump_phase,
            clip=(0.0, 1.0),
            params={"command_name": "jump_command"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WJumpEventsCfg(LocomotionEventCfg):
    """Mostly start on the ground; retain 20% reference resets as auxiliary practice."""

    reset_base = EventTerm(
        func=mdp.reset_timed_jump_or_flat_state,
        mode="reset",
        params={
            "jump_probability": 0.2,
            "reference_path": JUMP_REFERENCE_PATH,
            "leg_joint_names": GO2W_REFERENCE_LEG_JOINTS,
            "align_reference_height": True,
            "preserve_reference_joint_positions": True,
            # These are the original flat-task reset ranges.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "joint_position_range": (0.8, 1.2),
            "joint_velocity_range": (-1.0, 1.0),
            "phase_time_range": (0.0, JUMP_PROFILE.sequence_duration),
        },
    )
    # The combined event writes root and joint state exactly once.
    reset_robot_joints = None


@configclass
class Go2WJumpRewardsCfg(Go2WRewardsCfg):
    """Continuous TO pose guidance with contact-aware wheel and landing costs."""

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=50.0)
    jump_heading = None
    jump_yaw_velocity = None
    jump_lateral_velocity = None
    track_lin_vel_xy = RewTerm(
        func=mdp.jump_track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "jump_command_name": "jump_command",
            "std": 0.5,
            "active_multiplier": 2.0,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.jump_inactive_track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "jump_command_name": "jump_command",
            "std": 0.5,
        },
    )
    base_height = RewTerm(
        func=mdp.jump_inactive_base_height_l2,
        weight=-5.0,
        params={"command_name": "jump_command", "target_height": 0.42},
    )
    lin_vel_z = RewTerm(
        func=mdp.jump_inactive_lin_vel_z_l2,
        weight=-1.0,
        params={"command_name": "jump_command"},
    )
    stand_still = RewTerm(
        func=mdp.jump_inactive_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "jump_command",
            "velocity_command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    hip_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"]),
        },
    )
    thigh_calf_default = RewTerm(
        func=mdp.jump_inactive_joint_deviation_l1,
        weight=-0.25,
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    penalize_lin_vel_xy = None
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS)},
    )

    # Pose guidance must not disappear on delayed takeoff, partial contact, or
    # landing. Ascent shaping follows the fixed flight clock even before all
    # wheels leave the ground, so failed takeoffs still receive height feedback.
    reference_joint_position = RewTerm(
        func=mdp.jump_reference_joint_position_reward,
        weight=80.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "std": math.radians(15.0),
            "require_all_wheels_airborne": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS, preserve_order=True),
        },
    )
    # Covered by the continuous term above; do not double the pre-takeoff reward.
    pre_takeoff_reference_joint_position = None
    pre_takeoff_reference_joint_velocity = RewTerm(
        func=mdp.jump_pre_takeoff_reference_joint_velocity_reward,
        weight=20.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "takeoff_time": JUMP_PROFILE.takeoff_time,
            "std": 3.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS, preserve_order=True),
        },
    )
    reference_joint_velocity = RewTerm(
        func=mdp.jump_reference_joint_velocity_reward,
        weight=20.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "start_time": JUMP_PROFILE.takeoff_time,
            "std": 3.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS, preserve_order=True),
        },
    )

    takeoff_upward_velocity = RewTerm(
        func=mdp.jump_ascent_vertical_velocity_reward,
        weight=100.0,
        params={
            "command_name": "jump_command",
            "ascent_end_time": JUMP_PROFILE.apex_time,
            "require_all_wheels_airborne": False,
        },
    )
    ascent_height_error = RewTerm(
        func=mdp.jump_ascent_reference_height_l2,
        weight=-200.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "ascent_end_time": JUMP_PROFILE.apex_time,
            "require_all_wheels_airborne": False,
        },
    )
    ascent_height_tanh = RewTerm(
        func=mdp.jump_ascent_reference_height_tanh_reward,
        weight=20.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "ascent_end_time": JUMP_PROFILE.apex_time,
            "std": 0.05,
            "require_all_wheels_airborne": False,
        },
    )
    jump_attitude = RewTerm(
        func=mdp.jump_attitude_l2,
        weight=-5.0,
        params={"command_name": "jump_command"},
    )
    jump_angular_velocity = RewTerm(
        func=mdp.jump_angular_velocity_l2,
        weight=-2.0,
        params={"command_name": "jump_command"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    jump_wheel_velocity = RewTerm(
        func=mdp.jump_wheel_velocity_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True
            ),
            "contact_force_threshold": 1.0,
        },
    )
    landing_vertical_velocity = RewTerm(
        func=mdp.jump_landing_vertical_velocity_l2,
        weight=-2.0,
        params={
            "command_name": "jump_command",
            "descent_start_time": JUMP_PROFILE.apex_time,
            "velocity_tolerance": 0.2,
        },
    )


@configclass
class Go2WJumpTerminationsCfg(LocomotionTerminationsCfg):
    """Debounce large attitude excursions before terminating a jump episode."""

    bad_orientation = DoneTerm(
        func=mdp.DelayedBadOrientation,
        params={"limit_angle": 1.0, "delay_steps": 20},
    )


@configclass
class Go2WJumpCurriculumCfg(Go2WCurriculumCfg):
    """Keep the forward-command range fixed throughout jump training."""

    lin_vel_cmd = None
    ang_vel_cmd = None


@configclass
class Go2WJumpEnvCfg(Go2WFlatEnvCfg):
    """Train rolling -> jump -> recover -> rolling, several times per episode."""

    scene: Go2WJumpSceneCfg = Go2WJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WJumpCommandsCfg = Go2WJumpCommandsCfg()
    observations: Go2WJumpObservationsCfg = Go2WJumpObservationsCfg()
    events: Go2WJumpEventsCfg = Go2WJumpEventsCfg()
    rewards: Go2WJumpRewardsCfg = Go2WJumpRewardsCfg()
    terminations: Go2WJumpTerminationsCfg = Go2WJumpTerminationsCfg()
    curriculum: Go2WJumpCurriculumCfg = Go2WJumpCurriculumCfg()
    training_stage: str = "baseline"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        # Historical model_8500 simulation limits only. Shared robot assets and
        # deployment keep their corrected per-joint limits; this is not a
        # hardware-specification change.
        self.scene.robot.actuators["legs"].effort_limit_sim = 23.5
        self.scene.robot.actuators["legs"].velocity_limit_sim = 30.0
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class Go2WJumpPlayEnvCfg(Go2WJumpEnvCfg):
    """Ground-start evaluation: roll first, then trigger jumps at fixed intervals."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.jump_command.jump_probability = 1.0
        self.commands.jump_command.initial_trigger_delay_range = (2.0, 2.0)
        self.commands.jump_command.resampling_time_range = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.events.reset_base.params["jump_probability"] = 0.0
        self.events.reset_base.params["phase_time_range"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
