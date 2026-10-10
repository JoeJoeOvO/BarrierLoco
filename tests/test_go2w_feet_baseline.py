from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAT_CFG = ROOT / "locolab/source/locomotion/tasks/Go2W/flat_env_cfg.py"
ROUGH_CFG = ROOT / "locolab/source/locomotion/tasks/Go2W/rough_env_cfg.py"
OBSERVATIONS = ROOT / "locolab/source/locomotion/mdp/observations.py"


def _class(tree: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(class_node: ast.ClassDef, name: str) -> ast.Assign:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


def test_every_go2w_critic_inherits_body_frame_feet_velocity() -> None:
    flat_tree = ast.parse(FLAT_CFG.read_text())
    observations_cfg = _class(flat_tree, "Go2WObservationsCfg")
    critic_cfg = _class(observations_cfg, "CriticCfg")
    feet_lin_vel = ast.unparse(_assignment(critic_cfg, "feet_lin_vel").value)

    assert "func=mdp.body_lin_vel_b" in feet_lin_vel
    assert "SceneEntityCfg('robot', body_names='.*_foot')" in feet_lin_vel

    observations_tree = ast.parse(OBSERVATIONS.read_text())
    body_lin_vel_b = next(
        node for node in observations_tree.body if isinstance(node, ast.FunctionDef) and node.name == "body_lin_vel_b"
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "quat_apply_inverse" for node in ast.walk(body_lin_vel_b)
    )


def test_rough_reward_replaces_feet_stumble_with_wheel_contact_force() -> None:
    rough_tree = ast.parse(ROUGH_CFG.read_text())
    rewards_cfg = _class(rough_tree, "Go2WRoughRewardsCfg")
    contact_force = ast.unparse(_assignment(rewards_cfg, "contact_forces").value)

    assert "feet_stumble" not in ast.unparse(rewards_cfg)
    assert "func=mdp.contact_forces" in contact_force
    assert "weight=-0.00015" in contact_force
    assert "SceneEntityCfg('contact_forces', body_names='.*_foot')" in contact_force
    assert "'threshold': 56.6" in contact_force
