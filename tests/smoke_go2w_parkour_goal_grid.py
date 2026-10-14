"""Native reset/command/reference alignment and first-landing event audit for the grid task."""

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

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.source.locomotion.mdp.parkour_jump import reset_parkour_state
    from locolab.source.locomotion.tasks.Go2W.parkour_goal_grid_env_cfg import Go2WParkourGoalGridPlayEnvCfg

    cfg = Go2WParkourGoalGridPlayEnvCfg()
    cfg.scene.num_envs = 100
    cfg.sim.device = args.device
    cfg.events.reset_base.params["resample_flat_motion"] = True
    cfg.seed = 42
    env = gym.make("OmniBot-Go2W-Parkour-GoalGrid-MLP-v0", cfg=cfg)
    native = env.unwrapped
    obs, _ = env.reset()
    term = native.command_manager.get_term("parkour")
    bank = term.bank
    robot = native.scene["robot"]
    assert bank.schema_version == 2 and len(bank.names) > 7
    assert obs["policy"].shape == (100, 57) and obs["critic"].shape == (100, 312)
    ids = torch.arange(100, device=native.device)
    original, _ = bank.layout(ids, 3)
    obstacle = ~bank.is_flat[original]
    seen = set()
    changed = False
    for _ in range(12):
        before = term.motion_ids.clone()
        obs, _ = env.reset()
        changed |= bool((before != term.motion_ids).any())
        torch.testing.assert_close(term.motion_ids[obstacle], original[obstacle])
        assert bank.is_flat[term.motion_ids[~obstacle]].all()
        seen.update(term.motion_ids.cpu().tolist())
        goal = bank.descriptors[term.motion_ids] * (~term.locomotion_only)[:, None]
        torch.testing.assert_close(obs["policy"][:, 10:13], goal, rtol=0, atol=0)
        assert not term.ground_attempt.any()
    assert changed and any(i >= 7 for i in seen)
    # Reset each selected motion into its own actual reference time, then verify
    # body and joints against that same sampled trajectory before stepping.
    reset_parkour_state(
        native,
        ids,
        reference_probability=1.0,
        phase_range=(0.4, 0.4),
        locomotion_slots=3,
        ground_reference_pose=True,
        resample_flat_motion=True,
    )
    native.command_manager.reset(ids)
    ref = term.sample()
    chosen = ~term.locomotion_only
    torch.testing.assert_close(
        robot.data.root_pos_w[chosen] - native.scene.env_origins[chosen], ref["position"][chosen], rtol=0, atol=5e-5
    )
    limits = robot.data.joint_pos_limits[:, term.leg_ids]
    expected = ref["joint_position"].clamp(limits[..., 0], limits[..., 1])
    torch.testing.assert_close(robot.data.joint_pos[chosen][:, term.leg_ids], expected[chosen], rtol=0, atol=1e-6)
    assert not term.ground_attempt.any()
    counters = torch.zeros(100, device=native.device)
    events = [0]
    compute = native.reward_manager.compute
    index = native.reward_manager.active_terms.index("landing_position_error")

    def audited_reward(dt):
        result = compute(dt)
        actual = native.reward_manager._step_reward[:, index] * dt
        expected = -100 * torch.where(term.touchdown_event, term.touchdown_error.square(), 0)
        expected *= torch.where(term.reset_mode <= 2, 1.0, 1.0 / 6.0)
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()
        counters.add_(term.touchdown_event.float())
        assert (counters <= 1).all()
        events[0] += int(term.touchdown_event.sum())
        counters[native.termination_manager.dones] = 0
        return result

    native.reward_manager.compute = audited_reward
    for _ in range(320):
        with torch.no_grad():
            obs, reward, _, _, _ = env.step(torch.zeros(100, 16, device=native.device))
        goal = bank.descriptors[term.motion_ids] * (~(term.locomotion_only | term.rolling_exit))[:, None]
        torch.testing.assert_close(obs["policy"][:, 10:13], goal, rtol=0, atol=0)
        torch.testing.assert_close(obs["critic"][:, 9:16], term.critic_command)
        torch.testing.assert_close(term.motion_ids[obstacle], original[obstacle])
        assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
    assert events[0] > 0
    report = dict(
        success=True,
        envs=100,
        steps=320,
        bank_motions=len(bank.names),
        seen_motions=sorted(seen),
        flat_reference_resampled=True,
        obstacle_references_fixed=True,
        reset_pose_command_and_reference_aligned=True,
        actor_inputs=57,
        critic_inputs=312,
        first_landing_events=events[0],
        first_landing_not_repeated=True,
        ppo_updates=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GOAL_GRID_NATIVE_OK " + json.dumps(report), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
