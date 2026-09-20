"""Rolling/trot/bound through WTW continuous commands, on Go2W flat terrain.

Implements the paper's Appendix D phase convention, not the contradictory gait
names in upstream play.py. This is a Go2W/Isaac Lab port of the gait mechanism,
not a reproduction of the original Go1 RMA training stack.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.mdp import wtw_rewards
from locolab.source.locomotion.mdp.command.wtw_command_cfg import WTWCommandCfg

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WObservationsCfg, Go2WRewardsCfg


@configclass
class Go2WGaitCommandsCfg:
    base_velocity = WTWCommandCfg()


@configclass
class Go2WGaitObservationsCfg(Go2WObservationsCfg):
    @configclass
    class PolicyCfg(Go2WObservationsCfg.PolicyCfg):
        velocity_commands = ObsTerm(func=wtw_rewards.wtw_commands)
        gait_clock = ObsTerm(func=wtw_rewards.wtw_clocks)

    @configclass
    class CriticCfg(Go2WObservationsCfg.CriticCfg):
        velocity_commands = ObsTerm(func=wtw_rewards.wtw_commands)
        gait_clock = ObsTerm(func=wtw_rewards.wtw_clocks)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WGaitRewardsCfg(Go2WRewardsCfg):
    # WTW released train.py: dof_pos disabled, height cost 10, clearance cost 30,
    # phase force/velocity costs 4, vertical/roll-pitch rate costs .02/.001.
    stand_still = RewTerm(
        func=wtw_rewards.wtw_stand_still,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"]),
        },
    )
    # Hip reference remains zero for the current planar height IK.
    hip_default = RewTerm(
        func=wtw_rewards.wtw_joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*"])},
    )
    thigh_calf_default = None
    penalize_lin_vel_xy = None
    stay_alive = None
    # Strengthen height tracking 5x relative to the released WTW height cost.
    base_height = RewTerm(func=wtw_rewards.wtw_height, weight=-50.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.02)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.001)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    tracking_contacts_shaped_force = RewTerm(func=wtw_rewards.wtw_swing_force, weight=-4.0)
    tracking_contacts_shaped_vel = RewTerm(func=wtw_rewards.wtw_stance_velocity, weight=-4.0)
    feet_clearance_cmd_linear = RewTerm(func=wtw_rewards.wtw_clearance, weight=-30.0)
    raibert_heuristic = RewTerm(func=wtw_rewards.wtw_raibert, weight=-10.0)
    swing_wheel_velocity = RewTerm(func=wtw_rewards.wtw_swing_wheel_velocity, weight=-0.002)


@configclass
class Go2WGaitEnvCfg(Go2WFlatEnvCfg):
    commands: Go2WGaitCommandsCfg = Go2WGaitCommandsCfg()
    observations: Go2WGaitObservationsCfg = Go2WGaitObservationsCfg()
    rewards: Go2WGaitRewardsCfg = Go2WGaitRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 20 s episodes contain an in-episode command change at 10 s, as WTW.
        self.episode_length_s = 20.0
        # Flat's global velocity curriculum expects a UniformVelocityCommand;
        # do not apply it to the 10-D parameter distribution.
        self.curriculum.lin_vel_cmd = None
        self.curriculum.ang_vel_cmd = None


@configclass
class Go2WGaitPlayEnvCfg(Go2WGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.standing_probability = 0.0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
