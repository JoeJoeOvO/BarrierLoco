# Copyright (c) 2025 Deep Robotics
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
"""Dormant M20 Rough recipe reference; intentionally not registered as a training task.

Source: DeepRoboticsLab/rl_training@6d317dfb33060226139e38600510e1751372eb5d.
Active training uses the original Go2W Rough task while individual changes are evaluated.
See locolab/source/locomotion/mdp/m20/README.md for adaptations and validation.
"""

import copy
import math

import isaaclab.envs.mdp as lab_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import stand_still_joint_deviation_l1
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.terminations import terrain_out_of_bounds

from locolab.source.locomotion import mdp
from locolab.source.locomotion.mdp.m20 import curriculums
from locolab.source.locomotion.mdp.m20 import events as m20_events
from locolab.source.locomotion.mdp.m20 import rewards
from locolab.source.locomotion.mdp.m20.commands import M20VelocityCommandCfg

from .dreamwaq_env_cfg import Go2WDreamWaQObservationsCfg
from .flat_env_cfg import Go2WFlatEnvCfg, Go2WSceneCfg

LEG_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
WHEEL_JOINTS = [".*_foot_joint"]
DIAGONAL_FEET = ["FL_foot", "RR_foot", "FR_foot", "RL_foot"]

# Spell out the source's Isaac Lab 2.3.2 terrain defaults to avoid version-dependent changes.
M20_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.025, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.16),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class Go2WM20SceneCfg(Go2WSceneCfg):
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground/terrain"],
    )

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = copy.deepcopy(M20_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 5
        self.terrain.physics_material.static_friction = 1.0
        self.terrain.physics_material.dynamic_friction = 1.0
        self.terrain.physics_material.restitution = 1.0
        self.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]
        self.contact_forces.history_length = 3
        # Inherit per-joint official-model effort/velocity limits from GO2W_CFG.


@configclass
class Go2WM20CommandsCfg:
    base_velocity = M20VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=M20VelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
        rel_zero_vel_envs=0.2,
        rel_only_lin_y_envs=0.02,
        rel_only_lin_x_envs=0.02,
        rel_only_ang_z_envs=0.2,
    )


@configclass
class Go2WM20EventsCfg:
    physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.35, 1.5),
            "dynamic_friction_range": (0.35, 1.5),
            "restitution_range": (0.0, 0.7),
            "num_buckets": 1024,
        },
    )
    add_base_mass = EventTerm(
        func=lab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    scale_body_mass = EventTerm(
        func=lab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base$).*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_inertia = EventTerm(
        func=m20_events.randomize_mass_consistent_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.85, 1.15),
        },
    )
    randomize_com = EventTerm(
        func=lab_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )
    base_external_force_torque = EventTerm(
        func=lab_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    reset_robot_joints = EventTerm(
        func=lab_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=lab_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=m20_events.reset_root_state_with_clearance,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "minimum_clearance": 0.03,
            "safe_spawn_half_extent": 0.8,
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (0.0, 0.0),
            },
        },
    )
    push_robot = EventTerm(
        func=lab_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class Go2WM20RewardsCfg:
    """Only the 22 active upstream terms; no inheritance of baseline Go2W shaping."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    lin_vel_z_l2 = RewTerm(func=lab_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=lab_mdp.ang_vel_xy_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=lab_mdp.flat_orientation_l2, weight=-50.0)
    joint_torques_l2 = RewTerm(
        func=rewards.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    joint_acc_l2 = RewTerm(
        func=lab_mdp.joint_acc_l2,
        weight=-2e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    joint_acc_wheel_l2 = RewTerm(
        func=lab_mdp.joint_acc_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
    )
    # Preserve upstream's planar-only gate: this also applies during pure yaw commands.
    stand_still = RewTerm(
        func=stand_still_joint_deviation_l1,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    hipx_joint_pos_penalty = RewTerm(
        func=rewards.joint_pos_penalty,
        weight=-3.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    hipy_joint_pos_penalty = RewTerm(
        func=rewards.joint_pos_penalty_except_turn_side_cmd,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
        },
    )
    knee_joint_pos_penalty = RewTerm(
        func=rewards.joint_pos_penalty_except_turn_side_cmd,
        weight=-0.75,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
        },
    )
    joint_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"],
                ["FR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"],
            ],
        },
    )
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=-0.01)
    action_smooth_l2 = RewTerm(
        func=rewards.ActionSmoothnessPenalty,
        weight=-0.025,
        params={"action_term_names": ["joint_pos", "joint_vel"]},
    )
    undesired_contacts = RewTerm(
        func=lab_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot$).*"), "threshold": 1.0},
    )
    # Scale M20's 100 N threshold by nominal model mass: 19.523 / 34.4914.
    contact_forces = RewTerm(
        func=rewards.contact_forces,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 56.6},
    )
    feet_air_time_ang_z_M20 = RewTerm(
        func=rewards.FeetAirTime,
        weight=50.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=DIAGONAL_FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=DIAGONAL_FEET, preserve_order=True),
            "threshold": 0.2,
            "cmd_threshold": 0.1,
            "foot_height_threshold": 0.0,
        },
    )
    feet_slide_ang_z_cmd = RewTerm(
        func=rewards.feet_slide_ang_z_cmd,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=DIAGONAL_FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=DIAGONAL_FEET, preserve_order=True),
        },
    )
    rotation_gait_status = RewTerm(
        func=rewards.rotation_gait_status,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=DIAGONAL_FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=DIAGONAL_FEET, preserve_order=True),
        },
    )
    rotation_gait_symmetry = RewTerm(
        func=rewards.RotationGaitSymmetry,
        weight=15.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=DIAGONAL_FEET, preserve_order=True),
            "target_duty": 0.5,
            "std": 0.2,
            "lin_vel_threshold": 0.5,
            "ang_vel_threshold": 0.05,
        },
    )
    bad_orientation_penalty = RewTerm(func=rewards.bad_orientation_penalty, weight=-1000.0)


@configclass
class Go2WM20TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(func=terrain_out_of_bounds, params={"distance_buffer": 3.0}, time_out=True)
    bad_orientation = DoneTerm(func=rewards.bad_orientation)


@configclass
class Go2WM20CurriculumCfg:
    terrain_levels = CurrTerm(func=curriculums.terrain_levels)
    gait_level = CurrTerm(func=curriculums.gait_level)


@configclass
class Go2WM20RoughEnvCfg(Go2WFlatEnvCfg):
    scene: Go2WM20SceneCfg = Go2WM20SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WM20CommandsCfg = Go2WM20CommandsCfg()
    events: Go2WM20EventsCfg = Go2WM20EventsCfg()
    rewards: Go2WM20RewardsCfg = Go2WM20RewardsCfg()
    terminations: Go2WM20TerminationsCfg = Go2WM20TerminationsCfg()
    curriculum: Go2WM20CurriculumCfg = Go2WM20CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt


@configclass
class Go2WM20RoughPlayEnvCfg(Go2WM20RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # Keep all six terrain types represented by retaining the 20 columns.
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.max_init_terrain_level = 4
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_body_mass = None
        self.events.randomize_inertia = None
        self.events.randomize_com = None
        self.events.base_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"yaw": (-math.pi, math.pi)}
        self.events.reset_base.params["velocity_range"] = {}


@configclass
class Go2WM20RoughDreamWaQEnvCfg(Go2WM20RoughEnvCfg):
    """Use the existing DreamWaQ history and velocity targets with the same six-part M20 recipe."""

    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()


@configclass
class Go2WM20RoughDreamWaQPlayEnvCfg(Go2WM20RoughPlayEnvCfg):
    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_history.enable_corruption = False
