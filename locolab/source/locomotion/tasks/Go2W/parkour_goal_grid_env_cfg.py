"""Height/landing grid training with a matched, independently replayed TO per command."""

from isaaclab.utils import configclass

from locolab.source.locomotion.mdp.parkour_reference import GOAL_GRID_BANK_PATH

from .parkour_env_cfg import Go2WParkourBaseEnvCfg


@configclass
class Go2WParkourGoalGridEnvCfg(Go2WParkourBaseEnvCfg):
    training_stage: str = "goal_grid_single_stage"
    reference_bank_path: str = str(GOAL_GRID_BANK_PATH)
    required_reference_bank_version: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["resample_flat_motion"] = True


@configclass
class Go2WParkourGoalGridPlayEnvCfg(Go2WParkourGoalGridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 40
        self.events.reset_base.params["reference_probability"] = 0.0
        self.events.reset_base.params["resample_flat_motion"] = False
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
