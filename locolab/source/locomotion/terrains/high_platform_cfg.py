"""Configuration objects for the Go2W high-platform terrains."""

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .high_platform import high_platform_sequence_terrain, high_platform_terrain


@configclass
class MeshHighPlatformTerrainCfg(SubTerrainBaseCfg):
    """Flat approach followed by a centered raised platform."""

    function = high_platform_terrain

    platform_height_range: tuple[float, float] = (0.10, 0.80)
    num_height_levels: int = 36
    platform_start_x: float = 2.5
    platform_length: float = 1.5
    platform_width: float = 2.0
    spawn_x: float = 1.0
    ground_thickness: float = 1.0


@configclass
class MeshHighPlatformSequenceTerrainCfg(SubTerrainBaseCfg):
    """Flat approach followed by separate ascending platforms."""

    function = high_platform_sequence_terrain

    platform_heights: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    platform_start_x: float = 2.5
    platform_length: float = 1.5
    platform_gap: float = 1.5
    platform_width: float = 2.0
    spawn_x: float = 1.0
    ground_thickness: float = 1.0
