"""DreamWaQ-specific Go2W environment configurations."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locolab.source.locomotion import mdp

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WFlatPlayEnvCfg, Go2WObservationsCfg
from .rough_env_cfg import Go2WRoughEnvCfg, Go2WRoughPlayEnvCfg


@configclass
class Go2WDreamWaQObservationsCfg(Go2WObservationsCfg):
    """Observation groups needed by the DreamWaQ / RVCL architecture.

    Extra groups beyond baseline DreamWaQ (policy / critic / policy_history / dreamwaq_velocity):
    - ``proprio_critic``: critic observations without terrain exteroception (used by V_prop and ΔV heads).
    - ``event_height_target``: supervision target for the encoder's terrain-height prediction head.
    - ``event_contact_target``: supervision target for the encoder's foot-contact prediction head.
    """

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

    @configclass
    class ProprioCriticCfg(Go2WObservationsCfg.CriticCfg):
        """Critic observations without terrain exteroception (for V_prop / ΔV proprio branch)."""

        def __post_init__(self):
            super().__post_init__()
            # Remove terrain-related exteroception so V_prop represents
            # "value achievable from proprio alone" and ΔV captures the terrain-conditional residual.
            self.height_scanner = None
            self.terrain_height_stats = None

    @configclass
    class EventHeightTargetCfg(ObsGroup):
        """Terrain-height summary target for encoder's event-aware auxiliary supervision."""

        terrain_height_stats = ObsTerm(
            func=mdp.terrain_height_stats,
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EventContactTargetCfg(ObsGroup):
        """Foot-contact binary target for encoder's event-aware auxiliary supervision."""

        wheel_contact_state = ObsTerm(
            func=mdp.wheel_contact_state,
            clip=(0.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "force_threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy_history: PolicyHistoryCfg = PolicyHistoryCfg()
    dreamwaq_velocity: DreamWaQVelocityCfg = DreamWaQVelocityCfg()
    proprio_critic: ProprioCriticCfg = ProprioCriticCfg()
    event_height_target: EventHeightTargetCfg = EventHeightTargetCfg()
    event_contact_target: EventContactTargetCfg = EventContactTargetCfg()


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
