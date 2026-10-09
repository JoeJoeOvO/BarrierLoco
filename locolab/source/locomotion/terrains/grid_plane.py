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

    border_margin: float = 0.0

    def __init__(self, cfg: TerrainImporterCfg):
        if sim_utils.SimulationContext.instance() is None:
            raise RuntimeError("GridPlaneTerrainImporter requires an active SimulationContext.")
        super().__init__(cfg)

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Import the plane as a local mesh while preserving the parent lifecycle."""
        del size
        if self.cfg.num_envs is None or self.cfg.num_envs < 1:
            raise ValueError("Grid plane requires at least one environment.")
        if self.cfg.env_spacing is None or self.cfg.env_spacing <= 0.0:
            raise ValueError("Grid plane requires a positive environment spacing.")

        num_rows = math.ceil(self.cfg.num_envs / int(math.sqrt(self.cfg.num_envs)))
        num_cols = math.ceil(self.cfg.num_envs / num_rows)
        mesh_size = (
            num_rows * self.cfg.env_spacing + 2 * self.border_margin,
            num_cols * self.cfg.env_spacing + 2 * self.border_margin,
        )
        self.import_mesh(name, make_plane(mesh_size, height=0.0))


class MovingJumpPlaneTerrainImporter(GridPlaneTerrainImporter):
    """Room for 10 s at a 1.5 m/s command, plus reset noise and speed overshoot."""

    border_margin: float = 20.0
