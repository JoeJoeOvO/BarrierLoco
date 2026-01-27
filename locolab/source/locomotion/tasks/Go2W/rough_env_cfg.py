"""Go2W rough terrain environment configuration."""

from isaaclab.utils import configclass

from locolab.source.locomotion.tasks.velocity_env_cfg import ROUGH_TERRAIN_CFG

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WSceneCfg

# ==============================================================================
# Go2W Rough Scene Configuration
# ==============================================================================


@configclass
class Go2WRoughSceneCfg(Go2WSceneCfg):
    """Scene configuration for Go2W on rough terrain."""

    def __post_init__(self):
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = ROUGH_TERRAIN_CFG


# ==============================================================================
# Go2W Rough Environment Configuration
# ==============================================================================


@configclass
class Go2WRoughEnvCfg(Go2WFlatEnvCfg):
    """Go2W velocity tracking environment on rough terrain."""

    scene: Go2WRoughSceneCfg = Go2WRoughSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2WRoughPlayEnvCfg(Go2WRoughEnvCfg):
    """Go2W rough terrain play environment configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 1
