"""Configuration objects for delayed actuator models."""

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.utils import configclass

from .actuator_pd import DelayedDCMotor, DelayedImplicitActuator


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for an implicit actuator with command delay."""

    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 0


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for a DC motor actuator with command and sensor delay."""

    class_type: type = DelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0
    min_sensor_delay: int = 0
    max_sensor_delay: int = 0
