"""Configuration for the binary jump command."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .jump_command import BinaryJumpCommand


@configclass
class BinaryJumpCommandCfg(CommandTermCfg):
    """Configuration for a latched 0/1 jump trigger."""

    class_type: type = BinaryJumpCommand
    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    foot_body_names: str | list[str] = ".*_foot"
    jump_probability: float = 0.8
    contact_force_threshold: float = 1.0
    minimum_jump_height: float = 0.05
    landing_velocity_threshold: float = 0.2
    landing_recovery_time: float = 0.06
    # Fixed one-shot schedule matching the bundled full-rigid-body reference.
    takeoff_time: float = 0.36
    apex_time: float = 0.54
    landing_time: float = 0.72
    sequence_duration: float = 0.80
