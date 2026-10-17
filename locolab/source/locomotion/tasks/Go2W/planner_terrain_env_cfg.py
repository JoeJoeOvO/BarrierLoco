"""Planner scene scaffold: terrain, goal, observations and reset/termination only.

The 16D actions are native joint actions for diagnostics, NOT a Planner policy.
Rewards are intentionally empty until the skill adapter and objective are designed.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.mdp import planner_terrain as terrain_mdp
from locolab.source.locomotion.terrains.planner_course_cfg import planner_terrain_cfg

from .flat_env_cfg import Go2WFlatEnvCfg


@configclass
class EmptyCfg:
    pass


@configclass
class PlannerTerrainObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        gravity = ObsTerm(func=mdp.projected_gravity)
        joint_position = ObsTerm(func=mdp.joint_pos_rel)
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel)
        previous_action = ObsTerm(func=mdp.last_action)
        goal_xy = ObsTerm(func=terrain_mdp.goal_vector)
        height_map = ObsTerm(func=terrain_mdp.height_map)
        height_validity = ObsTerm(func=terrain_mdp.height_validity)
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PlannerTerrainTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal = DoneTerm(func=terrain_mdp.reached_goal)
    boundary = DoneTerm(func=terrain_mdp.left_course)
    pit = DoneTerm(func=terrain_mdp.fell_into_gap)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class Go2WPlannerTerrainEnvCfg(Go2WFlatEnvCfg):
    commands: EmptyCfg = EmptyCfg()
    rewards: EmptyCfg = EmptyCfg()
    curriculum: EmptyCfg = EmptyCfg()
    observations: PlannerTerrainObservationsCfg = PlannerTerrainObservationsCfg()
    terminations: PlannerTerrainTerminationsCfg = PlannerTerrainTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.seed = 42
        self.scene.num_envs = 256
        self.episode_length_s = 30.0
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = planner_terrain_cfg()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.62, 0.68), roughness=0.85
        )
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=900.0)
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.2, size=(6.0, 4.0))
        self.scene.height_scanner.offset.pos = (1.5, 0.0, 20.0)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # Shared GoalGrid dynamics are a provisional scene contract. Existing
        # skill assets stay untouched; skill-to-skill dynamics must be resolved
        # when wiring their frozen policies into a single physical robot.
        self.events.reset_base.params["pose_range"] = {"x": (-0.10, 0.10), "y": (-0.15, 0.15), "yaw": (-0.10, 0.10)}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None


@configclass
class Go2WPlannerTerrainPlayEnvCfg(Go2WPlannerTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 60
        self.scene.terrain.max_init_terrain_level = 5
