"""Validate Rough -> binary Jump weights and rewards with 2 environments, no PPO updates."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "locolab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.modules import ActorCritic
    from rsl_rl.utils.rough_mlp_warm_start import warm_start_rough_mlp

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg import Go2WJumpMLPPPORunnerCfg

    task = "OmniBot-Go2W-Jump-MLP-v0"
    cfg = parse_env_cfg(task, device=args.device or "cpu", num_envs=2, use_fabric=False)
    cfg.seed = 42
    assert cfg.events.reset_base.params["jump_probability"] == 0.2
    assert cfg.commands.jump_command.resampling_time_range == (2.0, 4.0)
    assert abs(cfg.commands.jump_command.sequence_duration - 1.2) < 1e-6
    # Exercise genuine moving triggers without an auxiliary airborne reset.
    cfg.events.reset_base.params["jump_probability"] = 0.0
    env = gym.make(task, cfg=cfg)
    try:
        obs, _ = env.reset()
        velocity = env.unwrapped.command_manager.get_command("base_velocity")
        assert ((velocity[:, 0] >= 0.2) & (velocity[:, 0] <= 1.5)).all(), velocity
        assert not velocity[:, 1:3].any(), velocity
        torch.testing.assert_close(obs["policy"][:, 6:9], velocity[:, :3])
        assert obs["policy"].shape == (2, 54)
        assert obs["critic"].shape == (2, 309)
        jump = env.unwrapped.command_manager.get_term("jump_command")
        assert not jump.active.any()
        rewards = env.unwrapped.reward_manager
        pose = rewards.get_term_cfg("reference_joint_position")
        assert pose.params["require_all_wheels_airborne"] is False
        assert "pre_takeoff_reference_joint_position" not in rewards.active_terms
        assert len(rewards.get_term_cfg("dof_pos_limits").params["asset_cfg"].joint_ids) == 12
        assert rewards.get_term_cfg("hip_default").func.__name__ == "joint_deviation_l1"
        assert len(rewards.get_term_cfg("hip_default").params["asset_cfg"].joint_ids) == 4
        assert len(rewards.get_term_cfg("thigh_calf_default").params["asset_cfg"].joint_ids) == 8
        assert "thigh_default" not in rewards.active_terms and "calf_default" not in rewards.active_terms
        assert "joint_mirror" not in rewards.active_terms
        assert "contact_forces" not in rewards.active_terms
        wheels = rewards.get_term_cfg("jump_wheel_velocity").params
        robot = env.unwrapped.scene["robot"]
        sensor = env.unwrapped.scene.sensors["contact_forces"]
        joint_names = [robot.joint_names[i].removesuffix("_joint") for i in wheels["asset_cfg"].joint_ids]
        body_names = [sensor.body_names[i] for i in wheels["sensor_cfg"].body_ids]
        assert joint_names == body_names == ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        assert cfg.actions.joint_pos.scale == 0.25 and cfg.actions.joint_vel.scale == 10.0
        runner_cfg = Go2WJumpMLPPPORunnerCfg()
        kwargs = runner_cfg.policy.to_dict()
        assert kwargs.pop("class_name") == "ActorCritic"
        policy = ActorCritic(obs, runner_cfg.obs_groups, 16, **kwargs).to(env.unwrapped.device)
        assert policy.noise_std_type == "scalar"
        before = {k: v.clone() for k, v in policy.critic.state_dict().items()}
        report = warm_start_rough_mlp(policy, args.checkpoint, command_name="jump_trigger")
        for key, value in policy.critic.state_dict().items():
            torch.testing.assert_close(value, before[key])
        source = torch.load(args.checkpoint, map_location=env.unwrapped.device, weights_only=True)["model_state_dict"]
        with torch.no_grad():
            for flag in (0.0, 1.0):
                inputs = obs["policy"].clone()
                inputs[:, 9] = flag
                expected = torch.cat((inputs[:, :9], inputs[:, 10:]), dim=-1)
                for index in (0, 2, 4, 6):
                    expected = torch.nn.functional.linear(
                        expected, source[f"actor.{index}.weight"], source[f"actor.{index}.bias"]
                    )
                    if index != 6:
                        expected = torch.nn.functional.elu(expected)
                torch.testing.assert_close(policy.actor(inputs), expected)
            torch.testing.assert_close(policy.std, source["log_std"].exp())
        attempt_total = 0.0
        previous_attempts = jump.metrics["ground_attempts"].clone()
        for _ in range(400):
            with torch.inference_mode():
                action = policy.act_inference(obs)
            obs, reward, *_ = env.step(action)
            velocity = env.unwrapped.command_manager.get_command("base_velocity")
            assert ((velocity[:, 0] >= 0.2) & (velocity[:, 0] <= 1.5)).all(), velocity
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
            attempts = jump.metrics["ground_attempts"]
            attempt_total += (attempts - previous_attempts).clamp_min(0).sum().item()
            previous_attempts = attempts.clone()
        assert attempt_total >= 2, f"Expected repeated ground-start requests, got {attempt_total}"
        print(
            f"JUMP_ROUGH_SMOKE_OK actor=54 critic=309 steps=400 updates=0 ground_attempts={attempt_total} "
            f"source_sha={report['source_sha256']}",
            flush=True,
        )
    finally:
        env.close()
finally:
    app.close()
