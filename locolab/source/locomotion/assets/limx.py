"""Configuration for Limx robots used in locomotion tasks."""

from __future__ import annotations

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from locolab.source.locomotion.actuators import DelayedDCMotorCfg

_ASSETS_DIR = pathlib.Path(__file__).parent.resolve()


LIMX_WF_TRON1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_ASSETS_DIR / "Limx" / "WF_TRON1A" / "urdf" / "robot.urdf"),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9664),
        joint_pos={".*_Joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=80.0,
            saturation_effort=160.0,
            velocity_limit=15.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.0749,
            friction=0.0,
            viscous_friction=0.0,
            min_delay=2,
            max_delay=5,
            min_sensor_delay=1,
            max_sensor_delay=2,
        ),
        "wheels": DelayedDCMotorCfg(
            joint_names_expr=["wheel_L_Joint", "wheel_R_Joint"],
            effort_limit=40.0,
            saturation_effort=80.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.8,
            armature=0.0105,
            friction=0.0,
            viscous_friction=0.0,
            min_delay=2,
            max_delay=5,
            min_sensor_delay=1,
            max_sensor_delay=2,
        ),
    },
)


WF_TRON1_JOINT_NAMES = [
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "wheel_L_Joint",
    "wheel_R_Joint",
]
