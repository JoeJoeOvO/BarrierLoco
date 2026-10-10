"""Real-physics reset/observation contract for the smooth-TO task; no PPO updates."""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source_root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
known, _ = parser.parse_known_args()
sys.path[:0] = [str(known.source_root), str(known.source_root / "locolab")]
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import locolab.source.locomotion.tasks  # noqa: F401 -- register Gym tasks
    from locolab.source.locomotion import mdp
    from locolab.source.locomotion.tasks.Go2W.jump_env_cfg import Go2WJumpEnvCfg, apply_jump_training_stage

    cfg = Go2WJumpEnvCfg()
    cfg.scene.num_envs = 64
    cfg.sim.device = args.device
    cfg.seed = 42
    apply_jump_training_stage(cfg, "skill")
    assert cfg.events.reset_base.params["jump_probability"] == 0.8
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.2, 0.5)
    assert cfg.commands.jump_command.heading_hold_duration == 0.0
    apply_jump_training_stage(cfg, "speed")
    assert cfg.events.reset_base.params["jump_probability"] == 0.2
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.2, 1.5)
    assert cfg.commands.jump_command.heading_hold_duration == 2.2
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    env = gym.make("OmniBot-Go2W-Jump-MLP-v0", cfg=cfg)
    native = env.unwrapped
    obs, _ = env.reset()
    robot = native.scene["robot"]
    trajectory = mdp.load_jump_reference(device=native.device)
    assert trajectory.duration == float(torch.tensor(1.8))
    assert trajectory.metadata["state_source"] == "independent_replay_actual_simulated_state"
    leg_ids = robot.find_joints(list(trajectory.metadata["leg_joint_names"]), preserve_order=True)[0]
    wheel_ids = robot.find_joints(list(trajectory.metadata["wheel_joint_names"]), preserve_order=True)[0]
    assert torch.allclose(robot.data.joint_effort_limits, torch.full_like(robot.data.joint_effort_limits, 23.5))
    assert torch.allclose(robot.data.joint_vel_limits, torch.full_like(robot.data.joint_vel_limits, 30.0))
    ids = torch.arange(native.num_envs, device=native.device)
    params = dict(cfg.events.reset_base.params)
    params.update(
        jump_probability=1.0,
        pose_range={},
        velocity_range={},
        joint_position_range=(1.0, 1.0),
        joint_velocity_range=(0.0, 0.0),
    )
    phases = (0.0, 0.22, 0.38, 0.46, 0.62, 0.88, 1.4, 1.8)
    for time in phases:
        query = torch.full((native.num_envs,), time, device=native.device)
        sample = trajectory.sample(query)
        mdp.reset_timed_jump_or_flat_state(native, ids, **dict(params, phase_time_range=(time, time)))
        torch.testing.assert_close(
            robot.data.root_pos_w - native.scene.env_origins, sample.base_position, atol=5e-6, rtol=0
        )
        torch.testing.assert_close(
            (robot.data.root_quat_w * sample.base_orientation).sum(-1).abs(), torch.ones_like(query)
        )
        torch.testing.assert_close(robot.data.root_lin_vel_w, sample.base_linear_velocity)
        torch.testing.assert_close(robot.data.root_ang_vel_w, sample.base_angular_velocity_world)
        torch.testing.assert_close(robot.data.joint_pos[:, leg_ids], sample.joint_position)
        torch.testing.assert_close(robot.data.joint_vel[:, leg_ids], sample.joint_velocity)
        torch.testing.assert_close(robot.data.joint_pos[:, wheel_ids], sample.wheel_position)
        torch.testing.assert_close(robot.data.joint_vel[:, wheel_ids], sample.wheel_velocity)
        native.command_manager.reset(ids)
        term = native.command_manager.get_term("jump_command")
        torch.testing.assert_close(term.reference_time, query)
        assert term.active.all()
        obs = native.observation_manager.compute()
        assert obs["policy"].shape == (64, 54) and obs["critic"].shape == (64, 309)
        assert all(torch.isfinite(value).all() for value in obs.values())
    env.reset()
    for _ in range(128):
        with torch.no_grad():
            obs, reward, _, _, _ = env.step(torch.zeros(64, 16, device=native.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
    report = {
        "success": True,
        "phases": phases,
        "physics_steps_after_resets": 128,
        "actor_inputs": 54,
        "critic_inputs": 309,
        "control_duration_s": 2.2,
        "self_collisions": True,
        "ppo_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SMOOTH_REFERENCE_NATIVE_OK " + json.dumps(report), flush=True)
except BaseException as error:
    import traceback

    traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".error.json").write_text(json.dumps({"error": str(error)}) + "\n")
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
