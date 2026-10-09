"""Configuration for the binary jump command."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from ..jump_reference import GO2W_JUMP_PROFILE
from .jump_command import BinaryJumpCommand


@configclass
class BinaryJumpCommandCfg(CommandTermCfg):
    """Configuration for a latched 0/1 jump trigger."""

    class_type: type = BinaryJumpCommand
    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    foot_body_names: str | list[str] = ".*_foot"
    jump_probability: float = GO2W_JUMP_PROFILE.jump_probability
    contact_force_threshold: float = 1.0
    minimum_jump_height: float = GO2W_JUMP_PROFILE.minimum_height
    landing_velocity_threshold: float = 0.2
    landing_recovery_time: float = 0.06
    # Zero delay preserves the legacy reset-trigger behavior. Moving-jump tasks
    # wait for rolling motion before the first automatic request.
    initial_trigger_delay_range: tuple[float, float] = (0.0, 0.0)
    require_grounded_trigger: bool = False
    minimum_forward_speed: float = 0.0
    landing_attitude_limit: float = 1.0
    # Stage two holds the trigger heading through a fixed recovery interval.
    # Zero preserves the original skill-stage command/observation behavior.
    heading_hold_duration: float = 0.0
    # Fixed one-shot schedule matching the bundled full-rigid-body reference.
    takeoff_time: float = GO2W_JUMP_PROFILE.takeoff_time
    apex_time: float = GO2W_JUMP_PROFILE.apex_time
    landing_time: float = GO2W_JUMP_PROFILE.landing_time
    sequence_duration: float = GO2W_JUMP_PROFILE.sequence_duration
