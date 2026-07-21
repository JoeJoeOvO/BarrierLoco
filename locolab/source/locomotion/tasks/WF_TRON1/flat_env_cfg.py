"""WF_TRON1 flat terrain environment configuration."""

from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import WFTRON1RoughEnvCfg


@configclass
class WFTRON1FlatEnvCfg(WFTRON1RoughEnvCfg):
    """WF_TRON1 flat terrain environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.base_height_l2 = None
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7


@configclass
class WFTRON1FlatPlayEnvCfg(WFTRON1FlatEnvCfg):
    """WF_TRON1 flat terrain play environment."""

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
