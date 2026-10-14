"""Real contact-event audit for the single-stage tuned environment, without a policy checkpoint."""

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
    from locolab.source.locomotion.tasks.Go2W.parkour_env_cfg import Go2WParkourFixedTargetTunedPlayEnvCfg

    cfg = Go2WParkourFixedTargetTunedPlayEnvCfg()
    assert cfg.commands.parkour.speed_endpoint_probability == 0.30
    assert cfg.commands.parkour.landing_command_mode == "fixed"
    assert cfg.command_semantics_version == 3 and cfg.critic_command_semantics_version == 2
    cfg.scene.num_envs = 30
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make("OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0", cfg=cfg)
    native = env.unwrapped
    obs, _ = env.reset()
    term = native.command_manager.get_term("parkour")
    assert obs["policy"].shape == (30, 57) and obs["critic"].shape == (30, 312)
    assert ((term.forward_speed >= 0.2) & (term.forward_speed <= 1.5)).all()
    # Exercise real contact from all seven reference flight states; this is a
    # reset contract audit, not a five-stage controller or a trained policy replay.
    ids = torch.arange(30, device=native.device)
    reset_parkour_state(
        native, ids, reference_probability=1.0, phase_range=(0.4, 0.4), locomotion_slots=3, ground_reference_pose=True
    )
    native.command_manager.reset(ids)
    obs = native.observation_manager.compute()
    assert not term.ground_attempt.any()
    counters = torch.zeros(30, device=native.device)
    audited_events = [0]
    compute = native.reward_manager.compute
    index = native.reward_manager.active_terms.index("landing_position_error")

    def audit_reward(dt):
        result = compute(dt)
        actual = native.reward_manager._step_reward[:, index] * dt
        expected = -100 * torch.where(term.touchdown_event, term.touchdown_error.square(), 0)
        expected *= torch.where(term.reset_mode <= 2, 1.0, 1.0 / 6.0)
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()
        counters.add_(term.touchdown_event.float())
        assert (counters <= 1).all(), "First landing penalty repeated inside one episode"
        audited_events[0] += int(term.touchdown_event.sum())
        counters[native.termination_manager.dones] = 0
        return result

    native.reward_manager.compute = audit_reward
    for _ in range(320):
        with torch.no_grad():
            obs, reward, _, _, _ = env.step(torch.zeros(30, 16, device=native.device))
        target = term.bank.descriptors[term.motion_ids] * (~(term.locomotion_only | term.rolling_exit))[:, None]
        torch.testing.assert_close(obs["policy"][:, 10:13], target, rtol=0, atol=0)
        torch.testing.assert_close(obs["critic"][:, 9:16], term.critic_command)
        assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values())
    assert audited_events[0] > 0
    report = dict(
        success=True,
        envs=30,
        physics_control_steps=320,
        actor_inputs=57,
        critic_inputs=312,
        actor_targets_fixed=True,
        audited_first_landing_events=audited_events[0],
        first_landing_cost_not_repeated=True,
        no_checkpoint_loaded=True,
        ppo_updates=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("TUNED_NATIVE_OK " + json.dumps(report), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
