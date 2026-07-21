"""Shared configuration for Go2W front-wheel handstand and rear-wheel upright tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import ArticulationCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG
from locolab.source.locomotion.tasks.velocity_env_cfg import LocomotionCommandsCfg

from .flat_env_cfg import Go2WCurriculumCfg, Go2WFlatEnvCfg, Go2WRewardsCfg, Go2WSceneCfg


def track_lin_vel_xy_yaw_frame_ungated_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track ground-frame planar velocity without assuming a flat body pose."""
    asset: RigidObject = env.scene[asset_cfg.name]
    velocity_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)
    error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_ungated_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track world yaw rate without the flat-task uprightness gate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def penalize_lin_vel_xy_yaw_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar command error in the yaw-aligned ground frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    velocity_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)
    return torch.sum(torch.abs(env.command_manager.get_command(command_name)[:, :2] - velocity_yaw[:, :2]), dim=1)


def lin_vel_z_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize world-frame vertical velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def ang_vel_xy_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize world-frame roll and pitch angular velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, :2]), dim=1)


@configclass
class Go2WTwoWheelBalanceCommandsCfg(LocomotionCommandsCfg):
    """Ground-frame longitudinal commands shared by both balance orientations."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
        limit_ranges=None,
    )


@configclass
class Go2WTwoWheelBalanceRewardsCfg(Go2WRewardsCfg):
    """Pose-independent reward terms shared by handstand and upright."""

    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_yaw_frame_ungated_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_z_world_ungated_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z = RewTerm(func=lin_vel_z_world_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=ang_vel_xy_world_l2, weight=-0.05)
    penalize_lin_vel_xy = RewTerm(
        func=penalize_lin_vel_xy_yaw_frame,
        weight=-0.05,
        params={"command_name": "base_velocity"},
    )
    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-5.0,
        params={
            "target_height": 0.63,
            "tolerance": 0.03,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class Go2WTwoWheelBalanceCurriculumCfg(Go2WCurriculumCfg):
    """Keep the fixed balance-task command range."""

    lin_vel_cmd = None
    ang_vel_cmd = None


@configclass
class Go2WTwoWheelBalanceEnvCfg(Go2WFlatEnvCfg):
    """Common environment behavior for the two mirrored balance poses."""

    commands: Go2WTwoWheelBalanceCommandsCfg = Go2WTwoWheelBalanceCommandsCfg()
    rewards: Go2WTwoWheelBalanceRewardsCfg = Go2WTwoWheelBalanceRewardsCfg()
    curriculum: Go2WTwoWheelBalanceCurriculumCfg = Go2WTwoWheelBalanceCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-3.14, 3.14),
        }
        self.terminations.bad_orientation = None


GO2W_HANDSTAND_CFG = GO2W_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        rot=(0.70710678, 0.0, 0.70710678, 0.0),
        joint_pos={
            "F[LR]_hip_joint": 0.0,
            "F[LR]_thigh_joint": -0.686,
            "F[LR]_calf_joint": -1.16,
            "R[LR]_hip_joint": 0.0,
            "R[LR]_thigh_joint": 1.7,
            "R[LR]_calf_joint": -1.853,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
)


@configclass
class Go2WHandstandSceneCfg(Go2WSceneCfg):
    robot = GO2W_HANDSTAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Go2WHandstandRewardsCfg(Go2WTwoWheelBalanceRewardsCfg):
    orientation = RewTerm(
        func=mdp.orientation_l2,
        weight=5.0,
        params={"desired_gravity": [1.0, 0.0, 0.0]},
    )
    front_wheel_contact = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
        },
    )
    rear_wheel_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
        },
    )
    airborne_wheel_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["R[LR]_foot_joint"])},
    )


@configclass
class Go2WHandstandEnvCfg(Go2WTwoWheelBalanceEnvCfg):
    scene: Go2WHandstandSceneCfg = Go2WHandstandSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: Go2WHandstandRewardsCfg = Go2WHandstandRewardsCfg()


@configclass
class Go2WHandstandPlayEnvCfg(Go2WHandstandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_clean_play(self)


GO2W_UPRIGHT_CFG = GO2W_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        rot=(0.70710678, 0.0, -0.70710678, 0.0),
        joint_pos={
            "F[LR]_hip_joint": 0.0,
            "F[LR]_thigh_joint": 0.234,
            "F[LR]_calf_joint": -1.853,
            "R[LR]_hip_joint": 0.0,
            "R[LR]_thigh_joint": 2.456,
            "R[LR]_calf_joint": -1.16,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
)


@configclass
class Go2WUprightSceneCfg(Go2WSceneCfg):
    robot = GO2W_UPRIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Go2WUprightRewardsCfg(Go2WTwoWheelBalanceRewardsCfg):
    orientation = RewTerm(
        func=mdp.orientation_l2,
        weight=5.0,
        params={"desired_gravity": [-1.0, 0.0, 0.0]},
    )
    rear_wheel_contact = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
        },
    )
    front_wheel_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
        },
    )
    airborne_wheel_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_foot_joint"])},
    )


@configclass
class Go2WUprightEnvCfg(Go2WTwoWheelBalanceEnvCfg):
    scene: Go2WUprightSceneCfg = Go2WUprightSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: Go2WUprightRewardsCfg = Go2WUprightRewardsCfg()


@configclass
class Go2WUprightPlayEnvCfg(Go2WUprightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_clean_play(self)
        self.viewer.eye = (2.2, 2.2, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.15)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0


def _configure_clean_play(env: Go2WTwoWheelBalanceEnvCfg) -> None:
    env.scene.num_envs = 32
    env.observations.policy.enable_corruption = False
    env.observations.critic.enable_corruption = False
    env.events.physics_material = None
    env.events.add_base_mass = None
    env.events.randomize_joint_friction = None
    env.events.randomize_actuator_gains = None
    env.events.base_external_force_torque = None
