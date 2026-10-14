"""Check real Isaac Lab configs against the three bundled checkpoint contracts."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import locolab.source.locomotion.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from locolab.source.locomotion.assets.robot import GO2W_CFG
    from locolab.source.locomotion.tasks.Go2W.rough_env_cfg import Go2WRoughEnvCfg

    contracts = {
        "OmniBot-Go2W-Gait-Flat-MLP-v0": False,
        "OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0": False,
        "OmniBot-Go2W-HighPlatform-MLP-v0": True,
        "OmniBot-Go2W-Jump-MLP-v0": True,
    }
    configs = []
    for task, self_collisions in contracts.items():
        for entry in ("env_cfg_entry_point", "play_env_cfg_entry_point"):
            cfg = load_cfg_from_registry(task, entry)
            legs = cfg.scene.robot.actuators["legs"]
            wheels = cfg.scene.robot.actuators["wheels"]
            assert (legs.effort_limit_sim, legs.velocity_limit_sim) == (23.5, 30.0), task
            assert (legs.stiffness, legs.damping) == (40.0, 1.0), task
            assert (wheels.stiffness, wheels.damping) == (0.0, 0.5), task
            assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is self_collisions, task
            assert math.isclose(cfg.sim.dt * cfg.decimation, 0.02), task
            assert cfg.actions.joint_pos.scale == 0.25, task
            assert cfg.actions.joint_vel.scale == 10.0, task
            configs.append(cfg)
            print(f"SKILL_DYNAMICS_OK task={task} config={entry}")

    # Config creation and task-local mutation must not modify other tasks or the shared asset.
    configs[0].scene.robot.actuators["legs"].effort_limit_sim = 1.0
    for cfg in configs[1:]:
        assert cfg.scene.robot.actuators["legs"].effort_limit_sim == 23.5
    for robot in (GO2W_CFG, Go2WRoughEnvCfg().scene.robot):
        assert robot.actuators["legs"].effort_limit_sim[".*_calf_joint"] == 35.0
        assert robot.actuators["legs"].velocity_limit_sim[".*_calf_joint"] == 20.0
    print("UNIFIED_SKILL_DYNAMICS_OK")
finally:
    app.close()
