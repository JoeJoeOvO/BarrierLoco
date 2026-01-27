"""Go2W robot environment configurations."""

from .flat_env_cfg import Go2WFlatEnvCfg, Go2WFlatPlayEnvCfg
from .rough_env_cfg import Go2WRoughEnvCfg, Go2WRoughPlayEnvCfg

__all__ = [
    "Go2WFlatEnvCfg",
    "Go2WFlatPlayEnvCfg",
    "Go2WRoughEnvCfg",
    "Go2WRoughPlayEnvCfg",
]
