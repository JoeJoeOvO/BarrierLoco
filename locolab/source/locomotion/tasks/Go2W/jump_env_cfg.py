"""Canonical Go2W vertical-jump task with dense ascent shaping."""

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
from locolab.source.locomotion.terrains.grid_plane import GridPlaneTerrainImporter

from .flat_env_cfg import Go2WCurriculumCfg, Go2WFlatEnvCfg, Go2WObservationsCfg, Go2WRewardsCfg, Go2WSceneCfg

JUMP_PROFILE = mdp.GO2W_JUMP_PROFILE
JUMP_REFERENCE_PATH = str(JUMP_PROFILE.reference_path)
GO2W_REFERENCE_LEG_JOINTS = list(JUMP_PROFILE.leg_joint_names)


@configclass
class Go2WJumpSceneCfg(Go2WSceneCfg):
    """Flat scene used to learn a vertical jump and landing."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.class_type = GridPlaneTerrainImporter


@configclass
class Go2WJumpCommandsCfg(LocomotionCommandsCfg):
    """Zero planar velocity plus one episode-level 0/1 jump command."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=None,
    )
    jump_command = mdp.BinaryJumpCommandCfg(
        asset_name="robot",
        contact_sensor_name="contact_forces",
        foot_body_names=".*_foot",
        # The reset event samples one trigger per episode.  Keep the command
        # horizon beyond the episode so it cannot resample halfway through.
        resampling_time_range=(1000.0, 1000.0),
        jump_probability=JUMP_PROFILE.jump_probability,
        contact_force_threshold=1.0,
        minimum_jump_height=JUMP_PROFILE.minimum_height,
        landing_velocity_threshold=0.2,
        landing_recovery_time=0.06,
        takeoff_time=JUMP_PROFILE.takeoff_time,
        apex_time=JUMP_PROFILE.apex_time,
        landing_time=JUMP_PROFILE.landing_time,
        sequence_duration=JUMP_PROFILE.sequence_duration,
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
    """Sample one episode flag and initialize jump episodes across all phases."""

    reset_base = EventTerm(
        func=mdp.reset_timed_jump_or_flat_state,
        mode="reset",
        params={
            "jump_probability": JUMP_PROFILE.jump_probability,
            "reference_path": JUMP_REFERENCE_PATH,
            "leg_joint_names": GO2W_REFERENCE_LEG_JOINTS,
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
    """Final dense-ascent reward profile for the timed 20 cm jump."""

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=50.0)
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    hip_default = RewTerm(
        func=mdp.jump_inactive_joint_deviation_l1,
        weight=-0.5,
        params={
            "command_name": "jump_command",
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

    reference_joint_position = RewTerm(
        func=mdp.jump_reference_joint_position_reward,
        weight=80.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "std": math.radians(15.0),
            "require_all_wheels_airborne": True,
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS, preserve_order=True),
        },
    )
    pre_takeoff_reference_joint_position = RewTerm(
        func=mdp.jump_pre_takeoff_reference_joint_position_reward,
        weight=80.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "takeoff_time": JUMP_PROFILE.takeoff_time,
            "std": math.radians(15.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_REFERENCE_LEG_JOINTS, preserve_order=True),
        },
    )
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

    takeoff_upward_velocity = RewTerm(
        func=mdp.jump_ascent_vertical_velocity_reward,
        weight=100.0,
        params={
            "command_name": "jump_command",
            "ascent_end_time": JUMP_PROFILE.apex_time,
            "require_all_wheels_airborne": True,
        },
    )
    ascent_height_error = RewTerm(
        func=mdp.jump_ascent_reference_height_l2,
        weight=-200.0,
        params={
            "command_name": "jump_command",
            "reference_path": JUMP_REFERENCE_PATH,
            "ascent_end_time": JUMP_PROFILE.apex_time,
            "require_all_wheels_airborne": True,
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
            "require_all_wheels_airborne": True,
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
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_.*"]),
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
    """Disable velocity curricula for the stationary jump task."""

    lin_vel_cmd = None
    ang_vel_cmd = None


@configclass
class Go2WJumpEnvCfg(Go2WFlatEnvCfg):
    """Train Go2W to jump once for command one and remain stable for zero."""

    scene: Go2WJumpSceneCfg = Go2WJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WJumpCommandsCfg = Go2WJumpCommandsCfg()
    observations: Go2WJumpObservationsCfg = Go2WJumpObservationsCfg()
    events: Go2WJumpEventsCfg = Go2WJumpEventsCfg()
    rewards: Go2WJumpRewardsCfg = Go2WJumpRewardsCfg()
    terminations: Go2WJumpTerminationsCfg = Go2WJumpTerminationsCfg()
    curriculum: Go2WJumpCurriculumCfg = Go2WJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0


@configclass
class Go2WJumpPlayEnvCfg(Go2WJumpEnvCfg):
    """Clean deterministic evaluation configuration for a one-shot jump."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.jump_command.jump_probability = 1.0
        self.events.reset_base.params["jump_probability"] = 1.0
        self.events.reset_base.params["phase_time_range"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
