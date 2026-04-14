"""WF_TRON1 rough terrain environment configuration."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.limx import LIMX_WF_TRON1_CFG, WF_TRON1_JOINT_NAMES
from locolab.source.locomotion.tasks.velocity_env_cfg import (
    LocomotionEnvCfg,
    LocomotionObservationsCfg,
    LocomotionSceneCfg,
)

LEG_JOINT_NAMES = [
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
]
WHEEL_JOINT_NAMES = ["wheel_L_Joint", "wheel_R_Joint"]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES


@configclass
class WFTRON1SceneCfg(LocomotionSceneCfg):
    """Scene configuration for WF_TRON1 on rough terrain."""

    robot = LIMX_WF_TRON1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )

    def __post_init__(self):
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.terrain.max_init_terrain_level = 5
        self.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        )


@configclass
class WFTRON1ActionsCfg:
    """Action specifications for WF_TRON1."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=20.0,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class WFTRON1CommandsCfg:
    """Velocity command configuration for WF_TRON1."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-2.5, 2.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class WFTRON1RewardsCfg:
    """Reward configuration migrated from limx_robot_lab WF_TRON1."""

    stay_alive = RewTerm(func=mdp.stay_alive_startup, weight=5.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=7.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_lin_vel_xy_exp_enhance = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.02},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp_enhance = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.1},
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_yaw_l2_clip, weight=-0.1, params={"clip": 20.0})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_w_l2_clip, weight=-2.0, params={"clip": 1.0})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.7664,
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES)},
    )
    dof_power_l1 = RewTerm(
        func=mdp.dof_power_l1,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES)},
    )
    joint_pos_limits_torque_regularization = RewTerm(
        func=mdp.joint_pos_limits_torque_regularization,
        weight=-150.0,
        params={
            "pos_limit_threshold": 0.1,
            "torque_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_pos_limits_vel_regularization = RewTerm(
        func=mdp.joint_pos_limits_vel_regularization,
        weight=-150.0,
        params={
            "pos_limit_threshold": 0.1,
            "vel_threshold": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )

    dof_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_mask, weight=-0.05, params={"mask": [0, 1, 2, 3, 4, 5]})
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.03,
        params={"mask": [0, 1, 2, 3, 4, 5]},
    )
    action_rate_l2_wheel = RewTerm(func=mdp.action_rate_l2_mask, weight=-0.2, params={"mask": [6, 7]})
    action_smoothness_wheel = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.1,
        params={"mask": [6, 7]},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
        },
    )


@configclass
class WFTRON1ObservationsCfg(LocomotionObservationsCfg):
    """Observation configuration for standard PPO WF_TRON1 training."""

    @configclass
    class PolicyCfg(LocomotionObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.base_ang_vel.scale = 0.1
            self.projected_gravity.scale = 1.0
            self.joint_pos.scale = 1.0
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*", preserve_order=True)
            }
            self.joint_vel.scale = 0.05
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)
            }

    @configclass
    class CriticCfg(LocomotionObservationsCfg.CriticCfg):
        external_force_torque = ObsTerm(
            func=mdp.external_force_torque,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_Link")},
        )
        wheel_contact_force = ObsTerm(
            func=mdp.body_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel_[LR]_Link")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scanner = None
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        terrain_height_stats = None

        def __post_init__(self):
            super().__post_init__()
            self.base_ang_vel.scale = 0.1
            self.projected_gravity.scale = 1.0
            self.joint_pos.scale = 1.0
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*", preserve_order=True)
            }
            self.joint_vel.scale = 0.05
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)
            }

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WFTRON1EventsCfg:
    """Event configuration for WF_TRON1 rough training."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.7),
            "num_buckets": 48,
            "make_consistent": True,
        },
    )
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_Link"]),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!.*base_Link).*"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_Link"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_Link"]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.5, 0.5)},
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_armature,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    align_actuators_delay = EventTerm(func=mdp.align_actuators_delay, mode="reset")
    wheel_resistance = EventTerm(
        func=mdp.wheel_resistance_xy,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "resample_probability": 0.01,
            "linear_res_coef_range": (0.0, 5.0),
            "quadratic_res_coef_range": (0.0, 0.5),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*wheel_[LR]_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_[LR]_Link"),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "force_range": {"x": (-500.0, 500.0), "y": (-500.0, 500.0), "z": (0.0, 0.0)},
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 0.0)},
            "probability": 0.002,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class WFTRON1TerminationsCfg:
    """Termination configuration for WF_TRON1."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "base_Link"]),
            "threshold": 1.0,
        },
    )
    illegal_contact_knee = DoneTerm(
        func=mdp.illegal_contact_delayed,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["knee_.*"]),
            "threshold": 1.0,
            "delay_steps": 10,
        },
    )


@configclass
class WFTRON1CurriculumCfg:
    """Curriculum configuration for WF_TRON1."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class WFTRON1RoughEnvCfg(LocomotionEnvCfg):
    """WF_TRON1 rough terrain environment."""

    scene: WFTRON1SceneCfg = WFTRON1SceneCfg(num_envs=8192, env_spacing=2.5)
    actions: WFTRON1ActionsCfg = WFTRON1ActionsCfg()
    observations: WFTRON1ObservationsCfg = WFTRON1ObservationsCfg()
    commands: WFTRON1CommandsCfg = WFTRON1CommandsCfg()
    rewards: WFTRON1RewardsCfg = WFTRON1RewardsCfg()
    events: WFTRON1EventsCfg = WFTRON1EventsCfg()
    terminations: WFTRON1TerminationsCfg = WFTRON1TerminationsCfg()
    curriculum: WFTRON1CurriculumCfg = WFTRON1CurriculumCfg()
    joint_names = WF_TRON1_JOINT_NAMES

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt


@configclass
class WFTRON1RoughPlayEnvCfg(WFTRON1RoughEnvCfg):
    """WF_TRON1 rough terrain play environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base = None
        self.events.reset_robot_joints = None
        self.events.randomize_joint_armature = None
        self.events.randomize_joint_default_pos = None
        self.events.align_actuators_delay = None
        self.events.wheel_resistance = None
        self.events.randomize_push_robot = None
