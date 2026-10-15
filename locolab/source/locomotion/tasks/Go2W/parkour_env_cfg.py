"""Shared GoalGrid environment components; not a separately registered task."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.mdp import parkour_jump as pm
from locolab.source.locomotion.mdp.parkour_reference import GOAL_GRID_BANK_PATH
from locolab.source.locomotion.tasks.velocity_env_cfg import LocomotionEventCfg
from locolab.source.locomotion.terrains.parkour_bank import ParkourBankTerrainCfg

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WObservationsCfg, Go2WRewardsCfg, Go2WSceneCfg
from .jump_env_cfg import Go2WJumpCommandsCfg, Go2WJumpCurriculumCfg, Go2WJumpTerminationsCfg


def reward(kind, weight):
    return RewTerm(func=pm.parkour_reward, weight=weight, params={"kind": kind})


@configclass
class Go2WParkourSceneCfg(Go2WSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terrain = ParkourBankTerrainCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=self.terrain.physics_material,
            visual_material=None,
        )
        self.sky_light.spawn = sim_utils.DomeLightCfg(intensity=900.0)


@configclass
class Go2WParkourCommandsCfg(Go2WJumpCommandsCfg):
    base_velocity = None
    jump_command = None
    parkour = pm.ParkourJumpCommandCfg(
        forward_speed_range=(0.2, 1.5),
        locomotion_slots=3,
        approach_timeout=6.0,
        rolling_exit_duration=1.0,
        landing_command_mode="fixed",
        speed_endpoint_probability=0.30,
        approach_velocity_std=0.25,
        approach_speed_error_floor=0.5,
        reference_xy_feedback_gain=2.0,
        reference_xy_feedback_limit=0.75,
    )

    def __post_init__(self):
        pass


@configclass
class Go2WParkourObservationsCfg(Go2WObservationsCfg):
    @configclass
    class PolicyCfg(Go2WObservationsCfg.PolicyCfg):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "parkour"})

    @configclass
    class CriticCfg(Go2WObservationsCfg.CriticCfg):
        # Dynamic remaining displacement is privileged; actor goals stay fixed.
        velocity_commands = ObsTerm(func=pm.parkour_critic_commands)
        jump_phase = ObsTerm(func=pm.parkour_phase, clip=(0.0, 1.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WParkourEventsCfg(LocomotionEventCfg):
    reset_base = EventTerm(
        func=pm.reset_parkour_state,
        mode="reset",
        params={
            # 30% rolling lanes + 50% ground jump approaches + 20% RSI.
            "reset_weights": (5 / 7, 1 / 7, 0.5 / 7, 0.25 / 7, 0.25 / 7),
            "locomotion_slots": 3,
            "ground_reference_pose": True,
        },
    )
    reset_robot_joints = None


@configclass
class Go2WParkourRewardsCfg(Go2WRewardsCfg):
    stay_alive = reward("alive", 5.0)
    # RewardManager multiplies by dt=.02: these are +30/-30 per full attempt,
    # and +5/-5 for RSI flight/recovery practice, not 1500 per terminal event.
    task_success_bonus = reward("success", 1500.0)
    termination_penalty = RewTerm(func=pm.parkour_failure_cost, weight=-1500.0)
    track_lin_vel_xy = reward("approach_velocity", 20.0)
    approach_speed_error = reward("approach_speed_error", -10.0)
    track_ang_vel_z = reward("approach_yaw", 1.0)
    base_height = reward("inactive_height", -30.0)
    lin_vel_z = reward("inactive_vz", -1.0)
    hip_default = reward("inactive_hip", -0.5)
    # These flat-pose costs would oppose reference tucking and sloped landing.
    orientation = reward("inactive_orientation", -2.0)
    ang_vel_xy = reward("inactive_angular_velocity", -0.2)
    thigh_calf_default = reward("inactive_thigh_calf", -0.25)
    stand_still = None
    penalize_lin_vel_xy = None
    reference_joint_position = reward("joint_position", 80.0)
    reference_joint_velocity = reward("joint_velocity", 20.0)
    reference_base_position = reward("base_position", 20.0)
    reference_base_velocity = reward("base_velocity", 15.0)
    reference_attitude = reward("attitude", -5.0)
    reference_angular_velocity = reward("angular_velocity", -2.0)
    takeoff_upward_velocity = reward("upward_velocity", 100.0)
    reference_height_error = reward("height_error", -200.0)
    ascent_height_tanh = reward("height_tanh", 20.0)
    reference_contacts = reward("contacts", 5.0)
    all_wheels_airborne = reward("airborne", 10.0)
    landing_vertical_velocity = reward("landing_vz", -2.0)
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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    approach_joint_position = reward("approach_joint_position", 10.0)
    landing_lead_error = reward("landing_lead_error", -60.0)
    recovery_position_error = reward("recovery_position_error", -80.0)
    # dt=.02 gives -100 * landing_error^2 once: -1 at 10 cm, -4 at 20 cm.
    landing_position_error = reward("landing_position_error", -5000.0)


@configclass
class Go2WParkourTerminationsCfg(Go2WJumpTerminationsCfg):
    # One approach + one 2.2 s motion per episode, avoiding another jump after
    # passing the fixed obstacle. Generic timeout still covers failure to approach.
    task_succeeded = DoneTerm(func=pm.parkour_succeeded)
    task_failed = DoneTerm(func=pm.parkour_deadline_failed)
    missed_target_or_pit = DoneTerm(func=pm.parkour_failed)


@configclass
class Go2WParkourBaseEnvCfg(Go2WFlatEnvCfg):
    """Shared single-stage recipe: rolling and measured TO jumps.

    Actor commands are [vx_yaw, vy_yaw, yaw_rate, jump, height, target_dx, target_dy].
    Height and landing XY are fixed TO-start-frame descriptors, not live error.
    The critic alone observes remaining displacement and the fixed-time phase.
    Approach/exit speeds vary; the measured TO timing and geometry never warp.
    """

    command_semantics_version: int = 3
    critic_command_semantics_version: int = 2
    training_stage: str = "goal_grid_single_stage"
    reward_recipe_version: int = 2
    reference_bank_path: str = str(GOAL_GRID_BANK_PATH)
    scene: Go2WParkourSceneCfg = Go2WParkourSceneCfg(num_envs=4096, env_spacing=32.0)
    commands: Go2WParkourCommandsCfg = Go2WParkourCommandsCfg()
    observations: Go2WParkourObservationsCfg = Go2WParkourObservationsCfg()
    events: Go2WParkourEventsCfg = Go2WParkourEventsCfg()
    rewards: Go2WParkourRewardsCfg = Go2WParkourRewardsCfg()
    terminations: Go2WParkourTerminationsCfg = Go2WParkourTerminationsCfg()
    curriculum: Go2WJumpCurriculumCfg = Go2WJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.terrain.reference_bank_path = self.reference_bank_path
        self.scene.terrain.locomotion_slots = self.commands.parkour.locomotion_slots
        props = self.scene.robot.spawn.articulation_props
        props.enabled_self_collisions = True
        props.solver_position_iteration_count = 8
        props.solver_velocity_iteration_count = 4
        # Inherit shared Go2W limits (23.5/23.5/35 Nm, 30/30/20 rad/s),
        # not the historical Jump task's uniform leg caps. Scales stay .25/10.
