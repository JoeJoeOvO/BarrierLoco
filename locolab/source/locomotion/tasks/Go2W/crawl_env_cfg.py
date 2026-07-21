"""Go2W low-body forward-crawling task derived from flat training."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp
from locolab.source.locomotion.assets.robot import GO2W_CFG
from locolab.source.locomotion.tasks.velocity_env_cfg import LocomotionCommandsCfg

from .flat_env_cfg import Go2WCurriculumCfg, Go2WFlatEnvCfg, Go2WRewardsCfg, Go2WSceneCfg

GO2W_CRAWL_CFG = GO2W_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 1.5,
            ".*_calf_joint": -2.6,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
)


@configclass
class Go2WCrawlSceneCfg(Go2WSceneCfg):
    """Flat scene initialized in the trained low crawl pose."""

    robot = GO2W_CRAWL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Go2WCrawlCommandsCfg(LocomotionCommandsCfg):
    """Strictly forward commands with the trained curriculum limits."""

    base_velocity = mdp.UniformVelocityCommandYawFrameCfg(
        asset_name="robot",
        class_type=mdp.UniformVelocityCommandYawFrame,
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.2, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(
            lin_vel_x=(0.1, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class Go2WCrawlRewardsCfg(Go2WRewardsCfg):
    """Flat rewards with the trained low body-height target."""

    base_height = RewTerm(
        func=mdp.base_height_tolerance,
        weight=-5.0,
        params={
            "target_height": 0.22,
            "tolerance": 0.03,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class Go2WCrawlCurriculumCfg(Go2WCurriculumCfg):
    """Expand only the positive longitudinal command range."""

    lin_vel_cmd = CurrTerm(
        func=mdp.lin_vel_cmd_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy",
            "reward_threshold": 0.75,
            "lower_threshold": 0.35,
            "step_size": 0.1,
            "min_lin_vel_x": (0.2, 0.5),
            "min_lin_vel_y": (0.0, 0.0),
            "max_lin_vel_x": (0.1, 1.0),
            "max_lin_vel_y": (0.0, 0.0),
        },
    )
    ang_vel_cmd = None


@configclass
class Go2WCrawlEnvCfg(Go2WFlatEnvCfg):
    """Go2W low-body forward-crawling training environment."""

    scene: Go2WCrawlSceneCfg = Go2WCrawlSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WCrawlCommandsCfg = Go2WCrawlCommandsCfg()
    rewards: Go2WCrawlRewardsCfg = Go2WCrawlRewardsCfg()
    curriculum: Go2WCrawlCurriculumCfg = Go2WCrawlCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class Go2WCrawlPlayEnvCfg(Go2WCrawlEnvCfg):
    """Small, clean evaluation configuration for the crawl policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.viewer.eye = (2.2, 2.2, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.15)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
