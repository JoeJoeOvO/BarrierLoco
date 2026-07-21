"""DreamWaQ environment for Go2W stand-up recovery."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.tasks.velocity_env_cfg import LocomotionEventCfg, LocomotionTerminationsCfg

from .dreamwaq_env_cfg import Go2WRoughDreamWaQEnvCfg
from .rough_env_cfg import Go2WRoughCommandsCfg, Go2WRoughRewardsCfg

LEG_JOINTS = [".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]
ALL_JOINTS = [*LEG_JOINTS, ".*_foot_.*"]
WHEELS = ".*_foot"

NORMAL_POSE_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.40, 0.52),
    "roll": (-0.15, 0.15),
    "pitch": (-0.15, 0.15),
    "yaw": (-math.pi, math.pi),
}
NORMAL_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.2, 0.2),
}
FALLEN_VELOCITY_RANGE = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.8, 0.8),
    "pitch": (-0.8, 0.8),
    "yaw": (-0.8, 0.8),
}

RECOVERY_POSES = {
    "supine": {
        "probability": 0.20,
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.16, 0.30),
            "roll": (-0.35, 0.35),
            "pitch": (-1.9, -1.2),
            "yaw": (-math.pi, math.pi),
        },
        "velocity_range": FALLEN_VELOCITY_RANGE,
        "joint_velocity_range": (-0.2, 0.2),
    },
    "upside_down": {
        "probability": 0.10,
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.16, 0.30),
            "roll": (math.pi - 0.35, math.pi + 0.35),
            "pitch": (-0.35, 0.35),
            "yaw": (-math.pi, math.pi),
        },
        "velocity_range": FALLEN_VELOCITY_RANGE,
        "joint_velocity_range": (-0.2, 0.2),
    },
    "prone": {
        "probability": 0.20,
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.16, 0.30),
            "roll": (-0.35, 0.35),
            "pitch": (1.2, 1.9),
            "yaw": (-math.pi, math.pi),
        },
        "velocity_range": FALLEN_VELOCITY_RANGE,
        "joint_velocity_range": (-0.2, 0.2),
    },
    "side_left": {
        "probability": 0.20,
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.16, 0.30),
            "roll": (1.2, 1.9),
            "pitch": (-0.35, 0.35),
            "yaw": (-math.pi, math.pi),
        },
        "velocity_range": FALLEN_VELOCITY_RANGE,
        "joint_velocity_range": (-0.2, 0.2),
    },
    "side_right": {
        "probability": 0.20,
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.16, 0.30),
            "roll": (-1.9, -1.2),
            "pitch": (-0.35, 0.35),
            "yaw": (-math.pi, math.pi),
        },
        "velocity_range": FALLEN_VELOCITY_RANGE,
        "joint_velocity_range": (-0.2, 0.2),
    },
}


@configclass
class Go2WRecoverCommandsCfg(Go2WRoughCommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        ranges = self.base_velocity.ranges
        limits = self.base_velocity.limit_ranges
        ranges.lin_vel_x = limits.lin_vel_x = (-0.8, 0.8)
        ranges.lin_vel_y = limits.lin_vel_y = (-0.2, 0.2)
        ranges.ang_vel_z = limits.ang_vel_z = (-0.5, 0.5)


@configclass
class Go2WRecoverRewardsCfg(Go2WRoughRewardsCfg):
    """Recovery-specific terms; unchanged rough terms remain inherited."""

    track_lin_vel_xy_enhance = None
    track_ang_vel_z_enhance = None
    penalize_lin_vel_xy = None
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", ".*_hip", "base"]),
        },
    )
    hip_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"])},
    )
    joint_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_.*", ".*_calf_.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    upright_orientation = RewTerm(func=mdp.upright_orientation_reward, weight=3.0)
    wheel_air = RewTerm(
        func=mdp.wheel_air_count,
        weight=-2.0,
        params={
            "force_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEELS),
        },
    )
    recovering_joint_velocity = RewTerm(
        func=mdp.recovering_joint_velocity_penalty,
        weight=-1.0,
        params={
            "max_joint_velocity": 2.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEELS),
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-250.0,
        params={
            "target_height": 0.42,
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class Go2WRecoverEventsCfg(LocomotionEventCfg):
    reset_base = None
    reset_robot_joints = None
    reset_robot_state = EventTerm(
        func=mdp.reset_robot_state_multi_pose,
        mode="reset",
        params={
            "normal_pose_probability": 0.10,
            "normal_pose_range": NORMAL_POSE_RANGE,
            "normal_velocity_range": NORMAL_VELOCITY_RANGE,
            "normal_joint_position_range": (-0.05, 0.05),
            "normal_joint_velocity_range": (0.0, 0.0),
            "special_poses": RECOVERY_POSES,
            "sample_joint_positions_from_limits": True,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS),
        },
    )


@configclass
class Go2WRecoverTerminationsCfg(LocomotionTerminationsCfg):
    base_contact = DoneTerm(
        func=mdp.illegal_contact_delayed,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
            "delay_steps": 200,
        },
    )
    bad_orientation = None


@configclass
class Go2WRecoverDreamWaQEnvCfg(Go2WRoughDreamWaQEnvCfg):
    commands: Go2WRecoverCommandsCfg = Go2WRecoverCommandsCfg()
    rewards: Go2WRecoverRewardsCfg = Go2WRecoverRewardsCfg()
    events: Go2WRecoverEventsCfg = Go2WRecoverEventsCfg()
    terminations: Go2WRecoverTerminationsCfg = Go2WRecoverTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        # Recovery pose z-ranges are offsets from this baseline and therefore
        # encode the intended absolute reset heights above each env origin.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
        self.scene.base_height_scanner.mesh_prim_paths = ["/World/ground"]


@configclass
class Go2WRecoverDreamWaQPlayEnvCfg(Go2WRecoverDreamWaQEnvCfg):
    """Small recovery evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.observations.policy_history.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
