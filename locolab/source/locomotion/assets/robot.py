"""Configuration for OmniBot robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# USD model directory - relative to this file
_ASSETS_DIR = pathlib.Path(__file__).parent.resolve()

# Unitree Go2W Configuration (Wheel-Legged Robot)
GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSETS_DIR / "Go2W" / "usd" / "go2w.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "hips": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=2.0,
            friction=0.01,
        ),
        "thighs": IdealPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=2.0,
            friction=0.01,
        ),
        "calves": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=2.0,
            friction=0.01,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.01,
        ),
    },
)

# SDK joint order for deployment (not part of ArticulationCfg)
# fmt: off
GO2W_SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
]
# fmt: on
