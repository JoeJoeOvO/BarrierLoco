"""One fixed, real collision scene per selected Go2W motion, evenly allocated."""

import numpy as np
import torch
import trimesh
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from ..mdp.parkour_reference import BANK_PATH, ParkourReferenceBank, terrain_mesh


class ParkourBankTerrainImporter(TerrainImporter):
    def import_ground_plane(self, name, size=None):
        import omni.usd
        from pxr import UsdPhysics

        if self.cfg.env_spacing < 30:
            raise ValueError("Recorded banks need at least 30 m between environment origins")
        bank = ParkourReferenceBank(self.cfg.reference_bank_path)
        origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing).cpu().numpy()
        templates = [terrain_mesh(e) for e in bank.entries]
        motions, _ = bank.layout(torch.arange(self.cfg.num_envs), self.cfg.locomotion_slots)
        vertices, faces, offset = [], [], 0
        for i, origin in enumerate(origins):
            mesh = templates[int(motions[i])]
            vertices.append(mesh.vertices + origin)
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
        mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)
        self.import_mesh(name, mesh)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.cfg.prim_path + f"/{name}/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("none")


@configclass
class ParkourBankTerrainCfg(TerrainImporterCfg):
    class_type: type = ParkourBankTerrainImporter
    reference_bank_path: str = str(BANK_PATH)
    locomotion_slots: int = 0
