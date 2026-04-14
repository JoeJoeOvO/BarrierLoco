"""DreamWaQ-specific WF_TRON1 environment configurations."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp

from .flat_env_cfg import WFTRON1FlatEnvCfg, WFTRON1FlatPlayEnvCfg
from .rough_env_cfg import WFTRON1ObservationsCfg, WFTRON1RoughEnvCfg, WFTRON1RoughPlayEnvCfg


@configclass
class WFTRON1DreamWaQObservationsCfg(WFTRON1ObservationsCfg):
    """Observation groups needed by the DreamWaQ architecture."""

    @configclass
    class PolicyHistoryCfg(WFTRON1ObservationsCfg.PolicyCfg):
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
class WFTRON1FlatDreamWaQEnvCfg(WFTRON1FlatEnvCfg):
    observations: WFTRON1DreamWaQObservationsCfg = WFTRON1DreamWaQObservationsCfg()


@configclass
class WFTRON1FlatDreamWaQPlayEnvCfg(WFTRON1FlatPlayEnvCfg):
    observations: WFTRON1DreamWaQObservationsCfg = WFTRON1DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_history.enable_corruption = False


@configclass
class WFTRON1RoughDreamWaQEnvCfg(WFTRON1RoughEnvCfg):
    observations: WFTRON1DreamWaQObservationsCfg = WFTRON1DreamWaQObservationsCfg()


@configclass
class WFTRON1RoughDreamWaQPlayEnvCfg(WFTRON1RoughPlayEnvCfg):
    observations: WFTRON1DreamWaQObservationsCfg = WFTRON1DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy_history.enable_corruption = False
