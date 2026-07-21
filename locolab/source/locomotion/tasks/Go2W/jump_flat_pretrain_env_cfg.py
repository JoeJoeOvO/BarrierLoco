"""Flat-locomotion pretraining with a Jump-compatible observation layout."""

from isaaclab.utils import configclass

from locolab.source.locomotion import mdp

from .flat_env_cfg import Go2WFlatCommandsCfg, Go2WFlatEnvCfg
from .jump_env_cfg import Go2WJumpObservationsCfg, Go2WJumpSceneCfg


@configclass
class Go2WJumpFlatPretrainCommandsCfg(Go2WFlatCommandsCfg):
    """Keep Flat velocity commands and append an always-zero jump trigger."""

    jump_command = mdp.BinaryJumpCommandCfg(
        asset_name="robot",
        contact_sensor_name="contact_forces",
        foot_body_names=".*_foot",
        resampling_time_range=(1000.0, 1000.0),
        jump_probability=0.0,
        debug_vis=False,
    )


@configclass
class Go2WJumpFlatPretrainEnvCfg(Go2WFlatEnvCfg):
    """Train Flat walking while preserving the Jump actor/critic input sizes."""

    scene: Go2WJumpSceneCfg = Go2WJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: Go2WJumpFlatPretrainCommandsCfg = Go2WJumpFlatPretrainCommandsCfg()
    observations: Go2WJumpObservationsCfg = Go2WJumpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Avoid fetching the remote arrow marker asset in headless training.
        self.commands.base_velocity.debug_vis = False


@configclass
class Go2WJumpFlatPretrainPlayEnvCfg(Go2WJumpFlatPretrainEnvCfg):
    """Deterministic evaluation configuration for the compatible Flat policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_joint_friction = None
        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
