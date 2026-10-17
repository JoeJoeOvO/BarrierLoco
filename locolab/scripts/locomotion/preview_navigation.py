"""Smoke-test OCC navigation interfaces with frozen skills, without PPO updates."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--report", type=Path)
parser.add_argument(
    "--random-planner", action="store_true", help="Use untrained network outputs; not a navigation demo."
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.planner.navigation import NavigationRollout
    from locolab.planner.network import NavigationActorCritic
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg

    cfg = Go2WPlannerTerrainEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.terminations.goal = None  # Navigation arrival changes the goal, not the episode.
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg)
    rollout = NavigationRollout(env, ROOT)
    obs = rollout.reset()
    model = NavigationActorCritic().to(env.unwrapped.device).eval()
    assert obs["occ"].shape == (args.num_envs, 2, 16, 24, 32)
    assert obs["proprio"].shape == (args.num_envs, 58)
    assert (
        rollout.goal_valid.all()
    ), f"Initial goal sampling failed: {(~rollout.goal_valid).nonzero().flatten().tolist()}"
    selected = set()
    for step in range(args.steps):
        with torch.no_grad():
            output = model.sample(obs, allowed=rollout.skills.allowed(), deterministic=True)
            likelihood, _ = model.evaluate(obs, output["skill"], output["raw_command"], rollout.skills.allowed())
            torch.testing.assert_close(likelihood, output["log_prob"])
        if not args.random_planner:
            # Explicit interface coverage, not a hand-coded navigation baseline.
            output["skill"] = torch.arange(args.num_envs, device=env.unwrapped.device) % 3
            output["command"].zero_()
        selected.update(output["skill"].tolist())
        obs, reward, done, info = rollout.step(output["skill"], output["command"])
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.isfinite(reward).all()
    # Test the arrival state machine independently of learned navigation ability:
    # put an injected goal at the current robot, and use a supported stop command.
    rollout.skills.reset(torch.arange(args.num_envs, device=env.unwrapped.device))
    rollout.goal_w[:] = env.unwrapped.scene["robot"].data.root_pos_w
    rollout.goal_w[:, 2] -= 0.3
    rollout.goal_valid[:] = True
    rollout.arrival_steps[:] = 1
    zero_skill = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
    zero_cmd = torch.zeros(args.num_envs, 10, device=env.unwrapped.device)
    zero_cmd[:, 8] = 1  # rolling/stand, zero velocity.
    _, _, _, info = rollout.step(zero_skill, zero_cmd)
    assert info["arrived"].any(), "Injected near-goal should refresh without a goal termination"
    report = dict(
        success=True,
        steps=args.steps,
        envs=args.num_envs,
        occ_shape=list(obs["occ"].shape),
        selected_skills=sorted(selected),
        injected_goal_refresh_count=int(info["arrived"].sum()),
        network_parameters=sum(p.numel() for p in model.parameters()),
        known_fraction=float(obs["occ"][:, 1].mean()),
        occupied_fraction=float(obs["occ"][:, 0].mean()),
        ppo_updates=0,
        navigation_success_validated=False,
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("NAVIGATION_INTERFACE_OK " + json.dumps(report), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
