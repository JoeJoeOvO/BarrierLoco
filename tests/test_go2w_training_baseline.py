from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCOMOTION = ROOT / "locolab/source/locomotion"
GO2W = LOCOMOTION / "tasks/Go2W"


def test_go2w_training_uses_the_validated_usd_dynamics_baseline() -> None:
    robot = (LOCOMOTION / "assets/robot.py").read_text()
    flat = (GO2W / "flat_env_cfg.py").read_text()

    assert "spawn=sim_utils.UsdFileCfg(" in robot
    assert '"Go2W" / "usd" / "go2w.usd"' in robot
    assert "UrdfFileCfg" not in robot
    assert "stiffness=40.0" in robot
    assert "damping=1.0" in robot
    assert "self.sim.physx.gpu_max_rigid_patch_count = 2**19" in flat


def test_flat_and_rough_share_the_tuned_go2w_command_reward_baseline() -> None:
    flat = (GO2W / "flat_env_cfg.py").read_text()
    rough = (GO2W / "rough_env_cfg.py").read_text()

    assert "class Go2WFlatCommandsCfg" in flat
    assert "lin_vel_x=(-0.5, 0.5)" in flat
    assert "limit_ranges=mdp.UniformVelocityCommandYawFrameCfg.Ranges(" in flat
    assert "lin_vel_x=(-2.0, 2.0)" in flat
    for reward in (
        "stay_alive",
        "termination_penalty",
        "stand_still",
        "hip_default",
        "thigh_calf_default",
        "penalize_lin_vel_xy",
        "action_smoothness",
        "thigh_calf_contact",
    ):
        assert f"{reward} = RewTerm" in flat

    assert 'mesh_prim_paths=["/World/ground/terrain"]' in rough
    assert 'self.height_scanner.mesh_prim_paths = ["/World/ground/terrain"]' in rough
    assert "rel_standing_envs=0.25" in rough
    assert "lin_vel_x=(-1.5, 1.5)" in rough
    assert "lin_vel_y=(-0.4, 0.4)" in rough
    assert "class Go2WRoughCurriculumCfg" in rough
    assert "lin_vel_cmd = None" in rough
    assert "ang_vel_cmd = None" in rough
    assert "run_still" not in rough
