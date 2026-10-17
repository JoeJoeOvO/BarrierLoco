"""Equal family allocation; rows represent difficulty, not a learned curriculum."""

from copy import deepcopy

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from .planner_course import FAMILIES, planner_course_terrain
from .planner_gaps import diverse_gap_terrain


@configclass
class PlannerCourseCfg(SubTerrainBaseCfg):
    function = planner_course_terrain
    family: str = "flat"
    vary_geometry: bool = False


TERRAIN_FAMILIES = FAMILIES + tuple(ROUGH_TERRAINS_CFG.sub_terrains)


def planner_terrain_cfg():
    # Equal columns allow independent per-family evaluation. Copy to avoid mutating
    # Isaac Lab's global preset (including its random-grid square-tile requirement).
    terrains = {name: PlannerCourseCfg(proportion=0.1, family=name) for name in FAMILIES}
    for name, original in ROUGH_TERRAINS_CFG.sub_terrains.items():
        terrains[name] = deepcopy(original)
        terrains[name].proportion = 0.1
    return TerrainGeneratorCfg(
        seed=42,
        curriculum=True,
        size=(16.0, 16.0),
        border_width=5.0,
        num_rows=6,
        num_cols=len(TERRAIN_FAMILIES),
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        color_scheme="none",
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        sub_terrains=terrains,
    )


def configure_continuous_terrain(cfg):
    """Versioned mixed-navigation geometry, without changing skill tasks."""
    for name in FAMILIES:
        cfg.scene.terrain.terrain_generator.sub_terrains[name].vary_geometry = True
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        cfg.scene.terrain.terrain_generator.sub_terrains[name].step_height_range = (0.04, 0.25)


@configclass
class DiverseGapCfg(SubTerrainBaseCfg):
    function = diverse_gap_terrain
    family: str = "single_gap"


def configure_diverse_gap_terrain(cfg):
    from .planner_gaps import GAP_FAMILIES

    generator = cfg.scene.terrain.terrain_generator
    generator.num_cols = len(GAP_FAMILIES)
    generator.sub_terrains = {name: DiverseGapCfg(proportion=0.2, family=name) for name in GAP_FAMILIES}
