from __future__ import annotations

import ast
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "locolab/source/locomotion/mdp/m20"
URDF = ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _defaults():
    robot = ET.parse(URDF).getroot()
    q = {}
    limits = []
    for joint in robot.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        q[name] = {"hip": 0.0, "thigh": 0.8, "calf": -1.5, "foot": 0.0}[name.split("_")[1]]
        limit = joint.find("limit")
        limits.append((float(limit.get("lower", "-1000")), float(limit.get("upper", "1000"))))
    return q, limits


def _quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


@pytest.fixture
def implementation(monkeypatch):
    package = ModuleType("_m20_reset")
    package.__path__ = [str(PACKAGE)]
    monkeypatch.setitem(sys.modules, "_m20_reset", package)
    geometry = _load(monkeypatch, "_m20_reset.geometry", PACKAGE / "geometry.py")
    lab = ModuleType("isaaclab")
    envs = ModuleType("isaaclab.envs")
    mdp = ModuleType("isaaclab.envs.mdp")
    managers = ModuleType("isaaclab.managers")
    utils = ModuleType("isaaclab.utils")
    math_utils = ModuleType("isaaclab.utils.math")
    managers.SceneEntityCfg = lambda name: SimpleNamespace(name=name, body_ids=slice(None))

    def inverse(quaternion, vector):
        q = quaternion[:, 1:]
        return (
            vector
            - 2 * quaternion[:, :1] * torch.cross(q, vector, dim=-1)
            + 2 * torch.cross(q, torch.cross(q, vector, dim=-1), dim=-1)
        )

    def reset(env, ids, pose_range, velocity_range, asset_cfg):
        asset = env.scene[asset_cfg.name]
        sample = {key: sum(values) / 2 for key, values in pose_range.items()}
        pose = torch.zeros(len(ids), 7)
        pose[:, :3] = env.scene.env_origins[ids] + torch.tensor(
            [sample.get("x", 0.0), sample.get("y", 0.0), 0.45 + sample.get("z", 0.0)]
        )
        pose[:, 3:] = _quaternion(*(sample.get(key, 0.0) for key in ("roll", "pitch", "yaw")))
        asset.write_root_pose_to_sim(pose, ids)
        env.reset_velocity_range = velocity_range

    mdp.reset_root_state_uniform = reset
    math_utils.quat_apply_inverse = inverse
    math_utils.sample_uniform = lambda low, high, size, device: torch.empty(size, device=device).uniform_(low, high)
    lab.envs = envs
    envs.mdp = mdp
    lab.managers = managers
    lab.utils = utils
    utils.math = math_utils
    for name, module in {
        "isaaclab": lab,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    events = _load(monkeypatch, "_m20_reset.events", PACKAGE / "events.py")
    return SimpleNamespace(geometry=geometry, events=events)


def _environment(n=4):
    q, limits = _defaults()

    class Asset:
        device = "cpu"
        joint_names = list(q)

        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.tensor(list(q.values())).repeat(n, 1),
                soft_joint_pos_limits=torch.tensor(limits).repeat(n, 1, 1),
                root_pos_w=torch.full((n, 3), 7.0),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            )

        def write_root_pose_to_sim(self, pose, env_ids):
            self.data.root_pos_w[env_ids] = pose[:, :3]
            self.data.root_quat_w[env_ids] = pose[:, 3:]

    class Scene(dict):
        pass

    scene = Scene(robot=Asset())
    scene.env_origins = torch.zeros(n, 3)
    scene.env_origins[:, 2] = torch.arange(n).float()
    scene.terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_type="plane"))
    return SimpleNamespace(
        num_envs=n,
        scene=scene,
        cfg=SimpleNamespace(
            events=SimpleNamespace(reset_robot_joints=SimpleNamespace(params={"position_range": (1, 1)}))
        ),
    )


def test_all_reset_angles_have_clearance_with_full_robot_geometry(implementation):
    geometry = implementation.geometry
    model = geometry.load_collision_geometry(URDF, _defaults()[0])
    assert len(model.centers) == 27
    assert model.bounding_radius + 0.25 < 0.8
    r, p = np.meshgrid(np.linspace(-0.3, 0.3, 61), np.linspace(-0.3, 0.3, 61))
    vertical = np.stack((-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)), axis=-1).reshape(-1, 3)
    low = geometry.collision_lower_bound(torch.tensor(vertical, dtype=torch.float32), model.tensors("cpu"))
    assert (0.45 + low).min().item() == pytest.approx(-0.045178, abs=1e-5)
    z = torch.maximum(torch.full_like(low, 0.45), 0.03 - low)
    assert torch.all(z + low >= 0.03 - 1e-6)
    assert z.max().item() == pytest.approx(0.525178, abs=1e-5)


def test_reset_preserves_other_envs_velocity_request_and_upright_height(implementation):
    env = _environment()
    velocity = {"x": (-0.2, 0.2)}
    implementation.events.reset_root_state_with_clearance(
        env,
        torch.tensor([1, 3]),
        {"roll": (0.3, 0.3), "pitch": (-0.3, -0.3)},
        velocity,
    )
    position = env.scene["robot"].data.root_pos_w
    torch.testing.assert_close(position[[0, 2]], torch.full((2, 3), 7.0))
    torch.testing.assert_close(position[[1, 3], 2], torch.tensor([1.525178, 3.525178]), atol=1e-5, rtol=0)
    assert env.reset_velocity_range is velocity
    implementation.events.reset_root_state_with_clearance(env, torch.tensor([0]), {}, {})
    assert position[0, 2].item() == pytest.approx(0.45)
    implementation.events.reset_root_state_with_clearance(env, torch.tensor([], dtype=torch.long), {}, {})


def test_geometry_uses_configured_joint_defaults(implementation):
    geometry = implementation.geometry
    q, _ = _defaults()
    original = geometry.load_collision_geometry(URDF, q)
    q = {key: (-1.0 if "_calf_" in key else value) for key, value in q.items()}
    changed = geometry.load_collision_geometry(URDF, q)
    assert not np.allclose(original.centers, changed.centers)


def test_unsafe_spawn_ranges_terrains_and_joint_resets_are_rejected(implementation):
    events = implementation.events
    env = _environment()
    with pytest.raises(ValueError, match="safe terrain center"):
        events.reset_root_state_with_clearance(env, torch.tensor([0]), {"x": (-1, 1)}, {})
    env.scene.terrain.cfg = SimpleNamespace(
        terrain_type="generator",
        terrain_generator=SimpleNamespace(sub_terrains={"unknown": SimpleNamespace(platform_width=3)}),
    )
    with pytest.raises(ValueError, match="central platform"):
        events.reset_root_state_with_clearance(env, torch.tensor([0]), {}, {})
    env = _environment()
    env.cfg.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
    with pytest.raises(ValueError, match="fixed default-joint"):
        events.reset_root_state_with_clearance(env, torch.tensor([0]), {}, {})


@pytest.mark.parametrize("n,b", [(1, 1), (4, 3)])
def test_mass_coupled_inertia_preserves_mass_ratio_and_never_compounds(implementation, n, b):
    tensor = torch.tensor([[2.0, 0.1, 0.2], [0.1, 3.0, 0.1], [0.2, 0.1, 4.0]]).reshape(9)
    default = tensor.repeat(n, b, 1)

    class View:
        def __init__(self):
            self.inertia = default * 99.0

        def get_inertias(self):
            return self.inertia

        def get_masses(self):
            return torch.full((n, b), 3.0)

        def set_inertias(self, values, env_ids):
            self.inertia = values

    view = View()
    asset = SimpleNamespace(
        num_bodies=b,
        root_physx_view=view,
        data=SimpleNamespace(default_inertia=default, default_mass=torch.full((n, b), 2.0)),
    )
    env = SimpleNamespace(scene={"robot": asset})
    ids = torch.tensor([n - 1])
    cfg = SimpleNamespace(name="robot", body_ids=[0])
    events = implementation.events
    events.randomize_mass_consistent_inertia(env, ids, cfg, (1.0, 1.0))
    torch.testing.assert_close(view.inertia[-1, 0], default[-1, 0] * 1.5)
    if n > 1:
        torch.testing.assert_close(view.inertia[0], default[0] * 99.0)
        torch.testing.assert_close(view.inertia[-1, 1:], default[-1, 1:] * 99.0)
    once = view.inertia.clone()
    events.randomize_mass_consistent_inertia(env, ids, cfg, (1.0, 1.0))
    torch.testing.assert_close(view.inertia, once)
    events.randomize_mass_consistent_inertia(
        env, None, SimpleNamespace(name="robot", body_ids=slice(None)), (0.85, 1.15)
    )
    ratios = view.inertia / default / 1.5
    assert torch.all((ratios >= 0.85 - 1e-6) & (ratios <= 1.15 + 1e-6))
    torch.testing.assert_close(ratios, ratios[..., :1].expand_as(ratios))
    assert torch.linalg.eigvalsh(view.inertia.reshape(n, b, 3, 3)).min() > 0


def test_m20_inherits_corrected_common_limits_without_changing_gains():
    cfg = ROOT / "locolab/source/locomotion/tasks/Go2W/m20_env_cfg.py"
    tree = ast.parse(cfg.read_text())
    scene = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WM20SceneCfg")
    assignments = {
        ast.unparse(target): node.value
        for node in ast.walk(scene)
        if isinstance(node, ast.Assign)
        for target in node.targets
    }
    assert not any(
        key.endswith(("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim")) for key in assignments
    )
    shared = ast.parse((ROOT / "locolab/source/locomotion/assets/robot.py").read_text())
    actuators = next(
        node.value for node in ast.walk(shared) if isinstance(node, ast.keyword) and node.arg == "actuators"
    )
    legs = next(value for key, value in zip(actuators.keys, actuators.values) if key.value == "legs")
    settings = {item.arg: item.value for item in legs.keywords}
    limits = ast.literal_eval(settings["velocity_limit_sim"])
    effort = ast.literal_eval(settings["effort_limit_sim"])
    assert limits == {".*_hip_joint": 30.0, ".*_thigh_joint": 30.0, ".*_calf_joint": 20.0}
    assert effort[".*_calf_joint"] == 35.0
    assert ast.literal_eval(settings["stiffness"]) == 40.0
    assert ast.literal_eval(settings["damping"]) == 1.0
    model = ET.parse(URDF).getroot()
    for joint in model.findall("joint"):
        if "_calf_" in joint.get("name"):
            assert limits[".*_calf_joint"] <= float(joint.find("limit").get("velocity"))
            assert effort[".*_calf_joint"] <= float(joint.find("limit").get("effort"))
