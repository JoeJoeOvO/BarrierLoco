"""Synchronous flat terrain importer for large environment grids."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporterCfg


class GridPlaneTerrainImporter(TerrainImporter):
    """Create a local mesh plane while retaining Isaac Lab's grid origins.

    The stock plane references a remote USD asset.  A local two-triangle mesh
    avoids that asynchronous load for large rendered clone grids.
    """

    def __init__(self, cfg: TerrainImporterCfg):
        cfg.validate()
        if cfg.num_envs is None or cfg.num_envs < 1:
            raise ValueError("Grid plane requires at least one environment.")
        if cfg.env_spacing is None or cfg.env_spacing <= 0.0:
            raise ValueError("Grid plane requires a positive environment spacing.")

        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore[union-attr]
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        num_rows = math.ceil(cfg.num_envs / int(math.sqrt(cfg.num_envs)))
        num_cols = math.ceil(cfg.num_envs / num_rows)
        size = (num_rows * cfg.env_spacing, num_cols * cfg.env_spacing)
        self.import_mesh("terrain", make_plane(size, height=0.0))
        self.configure_env_origins()
        self.set_debug_vis(cfg.debug_vis)
