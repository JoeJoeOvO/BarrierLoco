"""DreamWaQ-specific Go2W environment configurations."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WFlatPlayEnvCfg, Go2WObservationsCfg
from .rough_env_cfg import Go2WRoughEnvCfg, Go2WRoughPlayEnvCfg


@configclass
class Go2WDreamWaQObservationsCfg(Go2WObservationsCfg):
    """Observation groups needed by the DreamWaQ architecture."""

    @configclass
    class PolicyHistoryCfg(Go2WObservationsCfg.PolicyCfg):
        history_length = 5

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DreamWaQVelocityCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy_history: PolicyHistoryCfg = PolicyHistoryCfg()
    dreamwaq_velocity: DreamWaQVelocityCfg = DreamWaQVelocityCfg()


@configclass
class Go2WFlatDreamWaQEnvCfg(Go2WFlatEnvCfg):
    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()


@configclass
class Go2WFlatDreamWaQPlayEnvCfg(Go2WFlatPlayEnvCfg):
    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_history.enable_corruption = False


@configclass
class Go2WRoughDreamWaQEnvCfg(Go2WRoughEnvCfg):
    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()


@configclass
class Go2WRoughDreamWaQPlayEnvCfg(Go2WRoughPlayEnvCfg):
    observations: Go2WDreamWaQObservationsCfg = Go2WDreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_history.enable_corruption = False
