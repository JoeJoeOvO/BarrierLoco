"""WTW command configuration: paper phase convention, Go2W-safe sampling ranges."""

import math

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from ..wtw_curriculum import DEFAULT_SCHEDULE
from .wtw_command import WTWCommand


@configclass
class WTWCommandCfg(CommandTermCfg):
    class_type: type = WTWCommand
    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = False
    sampling_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    standing_probability: float = 0.10
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-math.pi, math.pi)
    heading_command: bool = True
    heading: tuple[float, float] = (-math.pi, math.pi)
    rel_heading_envs: float = 0.8
    heading_control_stiffness: float = 2.0
    body_height: tuple[float, float] = (0.22, 0.46)
    # Only default-pose rewards use IK; actions, observations and resets stay Flat.
    height_conditioned_reward_reference: bool = True
    frequency: tuple[float, float] = (1.5, 4.0)
    # Center-plus-signed-offset sampling, applied only on command resampling.
    phase_jitter_deg: float = 5.0
    max_pair_phase_error_deg: float = 10.0
    gait_duration: float = 0.5
    footswing_height: tuple[float, float] = (0.0, 0.10)
    curriculum_enabled: bool = False
    curriculum_steps_per_iteration: int = 24
    curriculum_schedule: tuple = DEFAULT_SCHEDULE
    kappa: float = 0.07
    # From Go2W URDF cylinder collision geometry in FR, FL, RR, RL order.
    wheel_radius: float = 0.086
    wheel_half_width: float = 0.0225
    wheel_center_offsets: tuple[float, float, float, float] = (-0.048, 0.048, -0.048, 0.048)
