from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "locolab/source/locomotion/mdp/events.py"
GRID_PLANE = ROOT / "locolab/source/locomotion/terrains/grid_plane.py"


class _SceneEntityCfg:
    def __init__(self, name: str, body_ids=None):
        self.name = name
        self.body_ids = body_ids


class _WheelResistanceBase:
    pass


def _load_wheel_resistance_term():
    """Keep testing the shared event even though HighPlatform no longer enables it."""
    events_tree = ast.parse(EVENTS.read_text())
    term_node = next(
        node for node in events_tree.body if isinstance(node, ast.ClassDef) and node.name == "wheel_resistance_xy"
    )

    module = ast.Module(body=[term_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "math_utils": SimpleNamespace(quat_apply_inverse=lambda _quat, vector: vector),
        "ManagerTermBase": _WheelResistanceBase,
        "EventTermCfg": object,
        "ManagerBasedEnv": object,
        "SceneEntityCfg": _SceneEntityCfg,
        "RigidObject": object,
        "Articulation": object,
        "ContactSensor": object,
    }
    exec(compile(module, str(EVENTS), "exec"), namespace)
    return namespace[term_node.name]


def _load_grid_plane_module(simulation_context):
    class SimulationContext:
        @classmethod
        def instance(cls):
            return simulation_context

    class TerrainImporter:
        def __init__(self, cfg):
            self.parent_initialized = True
            cfg.validate()
            self.cfg = cfg
            self.device = SimulationContext.instance().device
            self.terrain_prim_paths = []
            self.terrain_origins = None
            self.env_origins = None
            self._terrain_flat_patches = {}
            if cfg.terrain_type != "plane":
                raise ValueError("test adapter only supports plane terrain")
            self.import_ground_plane("terrain")
            self.configure_env_origins()
            self.set_debug_vis(cfg.debug_vis)

        def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
            self.remote_plane = (name, size)

        def import_mesh(self, name, mesh):
            self.imported_mesh = (name, mesh)
            self.terrain_prim_paths.append(name)

        def configure_env_origins(self):
            self.origins_configured = True

        def set_debug_vis(self, debug_vis):
            self.debug_vis = debug_vis

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    sim = ModuleType("isaaclab.sim")
    sim.SimulationContext = SimulationContext
    terrains = ModuleType("isaaclab.terrains")
    terrains.__path__ = []
    terrains.TerrainImporter = TerrainImporter
    terrains_trimesh = ModuleType("isaaclab.terrains.trimesh")
    terrains_trimesh.__path__ = []
    terrain_utils = ModuleType("isaaclab.terrains.trimesh.utils")
    terrain_utils.make_plane = lambda size, height: SimpleNamespace(size=size, height=height)

    modules = {
        "isaaclab": isaaclab,
        "isaaclab.sim": sim,
        "isaaclab.terrains": terrains,
        "isaaclab.terrains.trimesh": terrains_trimesh,
        "isaaclab.terrains.trimesh.utils": terrain_utils,
    }
    spec = importlib.util.spec_from_file_location("_go2w_grid_plane", GRID_PLANE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def test_wheel_resistance_honors_partial_env_ids() -> None:
    term_type = _load_wheel_resistance_term()
    term = term_type.__new__(term_type)
    term.device = "cpu"
    term.linear_resistance = torch.zeros(4)
    term.quadratic_resistance = torch.zeros(4)

    setter_calls = []

    class Asset:
        def __init__(self):
            self.data = SimpleNamespace(
                body_lin_vel_w=torch.ones(4, 2, 3),
                body_quat_w=torch.zeros(4, 2, 4),
            )
            self.has_external_wrench = False
            self._external_force_b = torch.zeros(4, 2, 3)

        def set_external_force_and_torque(self, forces, torques, *, env_ids, body_ids):
            setter_calls.append((forces.clone(), torques.clone(), env_ids.clone(), body_ids))

    asset = Asset()
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.ones(4, 1, 2, 3)))

    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            return {"robot": asset}[name]

    env = SimpleNamespace(num_envs=4, device="cpu", scene=Scene())

    target_ids = torch.tensor([1, 3])
    term(
        env,
        target_ids,
        resample_probability=1.0,
        linear_res_coef_range=(1.0, 1.0),
        quadratic_res_coef_range=(0.5, 0.5),
        sensor_cfg=_SceneEntityCfg("contact_forces", [0, 1]),
        asset_cfg=_SceneEntityCfg("robot", [0, 1]),
    )

    torch.testing.assert_close(term.linear_resistance, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(term.quadratic_resistance, torch.tensor([0.0, 0.5, 0.0, 0.5]))
    assert len(setter_calls) == 1
    forces, torques, applied_env_ids, body_ids = setter_calls[0]
    torch.testing.assert_close(applied_env_ids, target_ids)
    assert forces.shape == (2, 2, 3)
    torch.testing.assert_close(torques, torch.zeros_like(torques))
    assert body_ids == [0, 1]
    torch.testing.assert_close(asset._external_force_b, torch.zeros_like(asset._external_force_b))


def test_wheel_resistance_does_not_disable_an_existing_external_wrench() -> None:
    term_type = _load_wheel_resistance_term()
    term = term_type.__new__(term_type)
    term.device = "cpu"
    term.linear_resistance = torch.zeros(1)
    term.quadratic_resistance = torch.zeros(1)

    class Asset:
        def __init__(self):
            self.has_external_wrench = True
            self.data = SimpleNamespace(
                body_lin_vel_w=torch.zeros(1, 1, 3),
                body_quat_w=torch.zeros(1, 1, 4),
            )

        def set_external_force_and_torque(self, forces, torques, *, env_ids, body_ids):
            del env_ids, body_ids
            # Match Isaac Lab: the public setter derives one articulation-wide
            # enable bit only from the wrench supplied by this call.
            self.has_external_wrench = bool(forces.any() or torques.any())

    asset = Asset()
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.ones(1, 1, 1, 3)))

    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            return {"robot": asset}[name]

    env = SimpleNamespace(num_envs=1, device="cpu", scene=Scene())
    term(
        env,
        torch.tensor([0]),
        resample_probability=0.0,
        sensor_cfg=_SceneEntityCfg("contact_forces", [0]),
        asset_cfg=_SceneEntityCfg("robot", [0]),
    )

    assert asset.has_external_wrench


def test_grid_plane_uses_parent_lifecycle_with_local_mesh_override() -> None:
    module = _load_grid_plane_module(SimpleNamespace(device="cpu"))

    class Cfg:
        num_envs = 6
        env_spacing = 2.0
        terrain_type = "plane"
        debug_vis = False

        def validate(self):
            self.was_validated = True

    importer = module.GridPlaneTerrainImporter(Cfg())

    assert importer.parent_initialized
    assert importer.origins_configured
    assert importer.terrain_prim_paths == ["terrain"]
    name, mesh = importer.imported_mesh
    assert name == "terrain"
    assert mesh.size == (6.0, 4.0)
    assert mesh.height == 0.0
    assert not hasattr(importer, "remote_plane")


def test_grid_plane_requires_an_active_simulation_context() -> None:
    module = _load_grid_plane_module(None)

    class Cfg:
        num_envs = 1
        env_spacing = 1.0
        terrain_type = "plane"
        debug_vis = False

        def validate(self):
            pass

    with pytest.raises(RuntimeError, match="active SimulationContext"):
        module.GridPlaneTerrainImporter(Cfg())
