"""Extended gait commands; inherit the original robot, physics, rewards and observations."""

from isaaclab.utils import configclass

from .gait_env_cfg import Go2WGaitEnvCfg, Go2WGaitPlayEnvCfg


@configclass
class Go2WGaitCurriculumEnvCfg(Go2WGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.lin_vel_x = (-5.0, 5.0)
        self.commands.base_velocity.footswing_height = (0.0, 0.15)
        self.commands.base_velocity.curriculum_enabled = True


@configclass
class Go2WGaitCurriculumPlayEnvCfg(Go2WGaitPlayEnvCfg):
    """Evaluate the full trained range without restarting the training curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.lin_vel_x = (-5.0, 5.0)
        self.commands.base_velocity.footswing_height = (0.0, 0.15)
