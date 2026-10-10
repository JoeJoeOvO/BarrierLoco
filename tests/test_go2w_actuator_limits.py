"""Keep the train/TO/deploy model tied to the per-joint Go2-W URDF limits."""

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf"


def urdf_limits():
    return {
        joint.get("name"): (float(joint.find("limit").get("effort")), float(joint.find("limit").get("velocity")))
        for joint in ET.parse(URDF).getroot().findall("joint")
        if joint.get("type") != "fixed" and joint.find("limit") is not None
    }


def selected_limits():
    return {name: ((35.0, 20.0) if "_calf_joint" in name else (23.5, 30.0)) for name in urdf_limits()}


def configured_actuators():
    tree = ast.parse((ROOT / "locolab/source/locomotion/assets/robot.py").read_text())
    robot = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GO2W_CFG" for target in node.targets)
    )
    actuators = next(keyword.value for keyword in robot.keywords if keyword.arg == "actuators")
    return eval(
        compile(ast.Expression(actuators), "robot.py", "eval"),
        {"ImplicitActuatorCfg": lambda **kwargs: SimpleNamespace(**kwargs)},
    )


def per_joint(value, name):
    if isinstance(value, dict):
        matches = [bound for pattern, bound in value.items() if re.fullmatch(pattern, name)]
        assert len(matches) == 1
        return matches[0]
    return value


def test_every_joint_has_one_actuator_with_official_model_limits():
    actuators = configured_actuators()
    assert set(actuators) == {"legs", "wheels"}  # no actuator-group/API renaming
    for name, (effort, velocity) in selected_limits().items():
        groups = [
            cfg for cfg in actuators.values() if any(re.fullmatch(pattern, name) for pattern in cfg.joint_names_expr)
        ]
        assert len(groups) == 1
        assert per_joint(groups[0].effort_limit_sim, name) == effort
        assert per_joint(groups[0].velocity_limit_sim, name) == velocity
        assert effort <= urdf_limits()[name][0] and velocity <= urdf_limits()[name][1]
    assert (actuators["legs"].stiffness, actuators["legs"].damping) == (40.0, 1.0)
    assert (actuators["wheels"].stiffness, actuators["wheels"].damping) == (0.0, 0.5)


def test_deployment_torque_order_matches_urdf_joint_names():
    cfg = yaml.safe_load((ROOT / "deploy/policy/go2w/base.yaml").read_text())["go2w"]
    assert len(cfg["joint_names"]) == len(cfg["torque_limits"]) == 16
    expected = [selected_limits()[name][0] for name in cfg["joint_names"]]
    np.testing.assert_allclose(cfg["torque_limits"], expected)
    assert cfg["joint_mapping"] == list(range(16))


def test_m20_does_not_override_corrected_common_velocity_limits():
    text = (ROOT / "locolab/source/locomotion/tasks/Go2W/m20_env_cfg.py").read_text()
    assert 'actuators["legs"].velocity_limit_sim' not in text


def test_mujoco_compiled_limits_and_clipped_actuator_forces():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "deploy/src/robot_description/go2w_description/mjcf/go2w.xml"))
    data = mujoco.MjData(model)
    assert model.nu == 16
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint).replace("_wheel_joint", "_foot_joint")
        effort = selected_limits()[name][0]
        assert model.actuator_ctrllimited[actuator]
        np.testing.assert_allclose(model.actuator_ctrlrange[actuator], [-effort, effort])
    data.ctrl[:] = 1000.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = -1000.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, model.actuator_ctrlrange[:, 0])
    data.ctrl[:] = 0.0
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
