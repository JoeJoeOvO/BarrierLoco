from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .velocity_command import UniformVelocityCommandYawFrame


@configclass
class UniformVelocityCommandYawFrameCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandYawFrame
