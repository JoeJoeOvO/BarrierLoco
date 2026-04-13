"""DSR-CENet Go2W environment configurations.

Simpler than DreamWaQ env config: DSR uses recurrent hidden states instead of
observation history, so no ``policy_history`` group is needed. The DSR-specific
extras are a velocity-supervision group and a clean response-target group.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WFlatPlayEnvCfg, Go2WObservationsCfg
from .rough_env_cfg import Go2WRoughEnvCfg, Go2WRoughPlayEnvCfg


@configclass
class Go2WDSRObservationsCfg(Go2WObservationsCfg):
    """Observation groups for DSR-CENet.

    Only adds ``dsr_velocity`` (base_lin_vel) for velocity supervision.
    No history buffer needed — DSR uses GRU hidden states. We also expose a
    clean response-target group for multi-step future-response supervision.
    """

    @configclass
    class DSRVelocityCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ResponseTargetCfg(ObsGroup):
        """Clean k-step response target: lin vel, ang vel, projected gravity."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    dsr_velocity: DSRVelocityCfg = DSRVelocityCfg()
    dsr_response_target: ResponseTargetCfg = ResponseTargetCfg()


# ==============================================================================
# Flat terrain
# ==============================================================================


@configclass
class Go2WFlatDSREnvCfg(Go2WFlatEnvCfg):
    observations: Go2WDSRObservationsCfg = Go2WDSRObservationsCfg()


@configclass
class Go2WFlatDSRPlayEnvCfg(Go2WFlatPlayEnvCfg):
    observations: Go2WDSRObservationsCfg = Go2WDSRObservationsCfg()


# ==============================================================================
# Rough terrain
# ==============================================================================


@configclass
class Go2WRoughDSREnvCfg(Go2WRoughEnvCfg):
    observations: Go2WDSRObservationsCfg = Go2WDSRObservationsCfg()


@configclass
class Go2WRoughDSRPlayEnvCfg(Go2WRoughPlayEnvCfg):
    observations: Go2WDSRObservationsCfg = Go2WDSRObservationsCfg()
