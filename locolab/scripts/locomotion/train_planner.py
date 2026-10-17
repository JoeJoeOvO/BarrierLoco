"""From-scratch OCC navigation training with frozen skill policies and hybrid PPO."""

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=160)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--max-iterations", type=int, default=20000)
parser.add_argument("--diverse-gaps", action="store_true")
parser.add_argument("--continuous-mixed", action="store_true")
parser.add_argument("--jump-always-active", action="store_true")
parser.add_argument("--gap-only", action="store_true")
parser.add_argument("--progress-reward", type=float, default=1.0)
parser.add_argument("--arrival-reward", type=float, default=5.0)
parser.add_argument("--step-cost", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save-interval", type=int, default=100)
parser.add_argument("--minibatch-size", type=int, default=256)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--resume", type=Path)
parser.add_argument("--episode-seconds", type=float, default=30.0)
parser.add_argument(
    "--verify",
    action="store_true",
    help="Exercise goal refresh, timeout, failure and checkpoint reload before training.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.horizon < 2 or args.max_iterations < 1:
    parser.error("Need >=2 envs, >=2 horizon and >=1 iteration")
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "status.json").exists() and not args.resume:
    parser.error("Output already contains a run; choose a new directory or explicitly resume")
app = AppLauncher(args).app
success = False
stop_requested = False
status = {"state": "initializing", "pid": os.getpid(), "max_iterations": args.max_iterations}


def write_status(**changes):
    status.update(changes)
    temporary = args.output / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(args.output / "status.json")


try:
    import gymnasium as gym
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.planner.navigation import NavigationRollout
    from locolab.planner.network import NavigationActorCritic
    from locolab.planner.ppo import HybridPPO
    from locolab.planner.skills import POLICIES
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg
    from locolab.source.locomotion.terrains.planner_course_cfg import TERRAIN_FAMILIES

    def request_stop(signum, frame):
        global stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    write_status()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Go2WPlannerTerrainEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.max_init_terrain_level = 5  # Train on all six difficulty bands.
    if args.gap_only:
        from locolab.source.locomotion.terrains.planner_course_cfg import PlannerCourseCfg

        cfg.scene.terrain.terrain_generator.sub_terrains = {"gap": PlannerCourseCfg(proportion=1.0, family="gap")}
        # Start 1.5–2.5 m before the gap; allow the planner to approach and trigger.
        cfg.events.reset_base.params["pose_range"]["x"] = (2.0, 3.0)
    if args.continuous_mixed:
        from locolab.source.locomotion.terrains.planner_course_cfg import configure_continuous_terrain

        if args.gap_only:
            raise ValueError("continuous-mixed and gap-only are mutually exclusive")
        configure_continuous_terrain(cfg)
    terrain_families = ["gap"] if args.gap_only else list(TERRAIN_FAMILIES)
    if args.diverse_gaps:
        from locolab.source.locomotion.terrains.planner_course_cfg import configure_diverse_gap_terrain
        from locolab.source.locomotion.terrains.planner_gaps import GAP_FAMILIES

        if args.gap_only or not args.continuous_mixed:
            raise ValueError("diverse-gaps requires continuous-mixed, not the old gap-only route")
        configure_diverse_gap_terrain(cfg)
        terrain_families = list(GAP_FAMILIES)
    cfg.sim.device = args.device
    cfg.episode_length_s = args.episode_seconds
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg).unwrapped
    navigation = NavigationRollout(
        env,
        ROOT,
        manual_reset=True,
        gap_only=args.gap_only,
        reward_scales=(args.progress_reward, args.arrival_reward, 5.0, args.step_cost),
    )
    navigation.all_gap_terrain = args.diverse_gaps
    navigation.skills.jump_always_active = args.jump_always_active
    if args.continuous_mixed:
        navigation.directional_goals = True
        navigation.arrival_radius = 1.2
        navigation.arrival_speed = 1.0
        navigation.arrival_hold_steps = 1
    model = NavigationActorCritic().to(env.device)
    model.initialize_navigation()
    if args.gap_only or args.continuous_mixed:
        # Equal initial skill prior, without rewarding or forcing Jump selection.
        with torch.no_grad():
            model.skill_head.bias.zero_()
    optimizer = HybridPPO(model, minibatch_size=args.minibatch_size)
    observation = navigation.reset()
    start_iteration = 0
    if args.resume:
        restored = torch.load(args.resume, map_location=env.device, weights_only=False)
        model.load_state_dict(restored["model"], strict=True)
        optimizer.optimizer.load_state_dict(restored["optimizer"])
        optimizer.learning_rate = optimizer.optimizer.param_groups[0]["lr"]
        start_iteration = restored["iteration"] + 1
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        terrain_families=terrain_families,
        difficulty_rows=6,
        decision_dt=0.1,
        checkpoint_schema=1,
        proprio_dim=58,
        command_dims=[10, 3, 4],
        goal_threshold_m=navigation.arrival_radius,
        goal_speed_threshold_m_s=navigation.arrival_speed if args.continuous_mixed else None,
        goal_hold_steps=navigation.arrival_hold_steps,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=4,
        reward={
            "progress": args.progress_reward,
            "arrival": args.arrival_reward,
            "failure": -5.0,
            "per_step": -args.step_cost,
        },
        frozen_policy_sha256={
            path: hashlib.sha256((ROOT / "deploy/policy/go2w" / path).read_bytes()).hexdigest() for path in POLICIES
        },
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "locolab/planner").glob("*.py")
        },
        resume_resets_simulation=bool(args.resume),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from isaaclab.utils.io import dump_yaml

    dump_yaml(str(args.output / "env.yaml"), cfg)
    snapshot_files = list((ROOT / "locolab/planner").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "locolab/source/locomotion/tasks/Go2W/planner_terrain_env_cfg.py",
        ROOT / "locolab/source/locomotion/mdp/planner_terrain.py",
        ROOT / "locolab/source/locomotion/terrains/planner_course.py",
        ROOT / "locolab/source/locomotion/terrains/planner_course_cfg.py",
        ROOT / "locolab/source/locomotion/terrains/planner_gaps.py",
        ROOT / "locolab/source/locomotion/assets/robot.py",
    ]
    for source in snapshot_files:
        destination = args.output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    writer = SummaryWriter(str(args.output / "tensorboard"))

    def checkpoint(iteration):
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        temporary = args.output / "checkpoint.tmp"
        torch.save(payload, temporary)
        destination = args.output / f"model_{iteration:05d}.pt"
        temporary.replace(destination)
        if args.verify:
            restored = torch.load(destination, map_location=env.device, weights_only=False)
            clone = NavigationActorCritic().to(env.device)
            clone.load_state_dict(restored["model"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(model(observation)[0], clone(observation)[0])
            del clone
        return str(destination)

    if args.verify:
        with torch.no_grad():
            if args.continuous_mixed:
                root = env.scene["robot"].data.root_pos_w
                distance = (navigation.goal_w[:, :2] - root[:, :2]).norm(dim=-1)
                assert navigation.goal_valid.all()
                assert ((distance >= 8.0) & (distance <= 16.01)).all()
                if args.diverse_gaps:
                    probes = env.scene.env_origins[:, None].repeat(1, 320, 1)
                    probes[:, :, 0] += torch.linspace(-1.4, 14.4, 320, device=env.device)
                    probes[:, :, 2] = 10.0
                    down = torch.zeros_like(probes)
                    down[:, :, 2] = -1
                    ground = navigation.raycast(probes, down)[..., 2]
                    assert torch.isfinite(ground).all()
                    assert (ground.amin(dim=1) < -0.5).all()
                    torch.testing.assert_close(navigation.goal_w[:, 2], torch.zeros_like(distance), atol=1e-4, rtol=0)
                    print("DIVERSE_GAPS_OK actual_mesh_pits_and_supported_goals", flush=True)
                # Observe the actual actor input in the old inactive-recovery
                # interval, not merely a configuration flag.
                original = navigation.skills.models[2]
                captured = []

                def capture_jump(obs):
                    captured.append(obs.clone())
                    return original(obs)

                navigation.skills.models[2] = capture_jump
                navigation.skills.skill[:] = 2
                navigation.skills.timer[:] = 0.2
                navigation.skills.command[:, :4] = navigation.goal_w.new_tensor([0.8, 0.2, 0.6, 0.0])
                try:
                    navigation.skills.act()
                    assert (captured[0][:, 9] == 1).all()
                    assert (captured[0][:, 10:12] > 0).all()
                finally:
                    navigation.skills.models[2] = original
                navigation.reset()
                print("CONTINUOUS_NAV_OK goals_8_16m jump_actor_flag_1_near_end", flush=True)
            if args.gap_only:
                generator = env.scene.terrain.cfg.terrain_generator
                root = env.scene["robot"].data.root_pos_w
                half_x = generator.num_rows * generator.size[0] / 2
                row = ((root[:, 0] + half_x) / generator.size[0]).floor()
                edge = row * generator.size[0] - half_x + 6.0
                assert navigation.goal_valid.all()
                assert ((root[:, 0] - edge) * (navigation.goal_w[:, 0] - edge) < 0).all()
                probes = navigation.goal_w.clone()
                probes[:, 0] = edge + 0.1
                probes[:, 2] = 10.0
                down = torch.zeros_like(probes)
                down[:, 2] = -1
                pit_hits = navigation.raycast(probes[:, None], down[:, None])
                torch.testing.assert_close(pit_hits[:, 0, 2], torch.full_like(edge, -1.0), atol=1e-4, rtol=0)
                print("GAP_GEOMETRY_OK opposite_bank_goals real_pit_depth_1m", flush=True)
            navigation.goal_w[:] = env.scene["robot"].data.root_pos_w
            navigation.goal_w[:, 2] -= 0.3
            navigation.goal_valid[:] = True
            navigation.arrival_steps[:] = 1
            command = torch.zeros(args.num_envs, 10, device=env.device)
            command[:, 8] = 1.0
            skill = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
            _, _, done, info = navigation.step(skill, command)
            assert info["arrived"].any() and not done[info["arrived"]].any()
            env.episode_length_buf[:] = env.max_episode_length - 5
            _, _, done, info = navigation.step(skill, command)
            assert info["timeout"].any()
            assert info["terminal_observation"] is not None
            assert (env.episode_length_buf[done] == 0).all()
            # Explicitly place one robot outside the complete generated map.
            state = env.scene["robot"].data.root_state_w[:1].clone()
            generator = env.scene.terrain.cfg.terrain_generator
            state[:, 1] = generator.num_cols * generator.size[1] / 2 + 1.0
            env.scene["robot"].write_root_state_to_sim(state, env_ids=torch.tensor([0], device=env.device))
            env.sim.forward()
            _, _, done, info = navigation.step(skill, command)
            assert info["failed"][0] and done[0] and not info["timeout"][0]
        observation = navigation.reset()
        print("PLANNER_PREFLIGHT_OK goal_refresh timeout failure", flush=True)

    # Binary OCC rollouts are stored as uint8, not float32, to control GPU memory.
    shape = (args.horizon, args.num_envs)
    rollout = {
        name: torch.empty(
            (*shape, *value.shape[1:]), device=env.device, dtype=torch.uint8 if name == "occ" else torch.float32
        )
        for name, value in observation.items()
    }
    for name in ("reward", "value", "bootstrap", "log_prob"):
        rollout[name] = torch.empty(shape, device=env.device)
    rollout["done"] = torch.empty(shape, device=env.device, dtype=torch.bool)
    rollout["skill"] = torch.empty(shape, device=env.device, dtype=torch.long)
    rollout["allowed"] = torch.empty((*shape, 3), device=env.device, dtype=torch.bool)
    rollout["raw_command"] = torch.empty((*shape, 10), device=env.device)
    frozen_before = [next(policy.parameters()).detach().clone() for policy in navigation.skills.models]
    started = time.monotonic()
    ema = None
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    for iteration in range(start_iteration, args.max_iterations):
        tick = time.monotonic()
        arrivals = torch.zeros(args.num_envs, device=env.device)
        failures = torch.zeros_like(arrivals)
        timeouts = torch.zeros_like(arrivals)
        invalid = torch.zeros_like(arrivals)
        for t in range(args.horizon):
            with torch.no_grad():
                allowed = navigation.skills.allowed()
                action = model.sample(observation, allowed)
                for name, value in observation.items():
                    rollout[name][t].copy_(value)
                for name in ("skill", "raw_command", "log_prob", "value"):
                    rollout[name][t].copy_(action[name])
                rollout["allowed"][t].copy_(allowed)
                next_observation, reward, done, info = navigation.step(action["skill"], action["command"])
                assert all(torch.isfinite(value).all() for value in next_observation.values())
                assert torch.isfinite(reward).all()
                bootstrap = model(next_observation)[2]
                if info["timeout"].any():
                    terminal_value = model(info["terminal_observation"])[2]
                    bootstrap = torch.where(info["timeout"], terminal_value, bootstrap)
                bootstrap = torch.where(info["failed"], 0.0, bootstrap)
                rollout["reward"][t].copy_(reward)
                rollout["bootstrap"][t].copy_(bootstrap)
                rollout["done"][t].copy_(done)
                arrivals += info["arrived"]
                failures += info["failed"]
                timeouts += info["timeout"]
                invalid += ~info["goal_valid"]
                observation = next_observation
        metrics = optimizer.update(rollout)
        duration = time.monotonic() - tick
        ema = duration if ema is None else 0.9 * ema + 0.1 * duration
        metrics.update(
            iteration=iteration,
            completed_iterations=iteration + 1,
            iteration_seconds=duration,
            elapsed_seconds=time.monotonic() - started,
            eta_seconds=ema * (args.max_iterations - iteration - 1),
            reward=float(rollout["reward"].mean()),
            arrivals=int(arrivals.sum()),
            failures=int(failures.sum()),
            timeouts=int(timeouts.sum()),
            invalid_goal_fraction=float(invalid.mean() / args.horizon),
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        )
        for index, family in enumerate(terrain_families):
            mask = (
                torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                if args.gap_only
                else env.scene.terrain.terrain_types == index
            )
            if mask.any():
                for key, value in (("arrivals", arrivals), ("failures", failures), ("timeouts", timeouts)):
                    prefix = "spawn_terrain" if args.continuous_mixed else "terrain"
                    metrics[f"{prefix}/{family}/{key}_per_env"] = float(value[mask].mean())
        metrics["arrival_fraction_of_outcomes"] = float(
            arrivals.sum() / (arrivals.sum() + failures.sum() + timeouts.sum()).clamp(min=1)
        )
        metrics["mean_goal_distance_m"] = float(
            (navigation.goal_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2]).norm(dim=-1).mean()
        )
        metrics["mean_speed_m_s"] = float(env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1).mean())
        metrics["busy_fraction"] = float((rollout["proprio"][..., -1] > 0.5).float().mean())
        decisions = rollout["proprio"][..., -1] < 0.5
        for index, name in enumerate(("multiskill", "highplatform", "goalgrid")):
            metrics[f"skill/{name}_decision_fraction"] = float(
                ((rollout["skill"] == index) & decisions).sum() / decisions.sum().clamp(min=1)
            )
            metrics[f"skill/{name}_fraction"] = float((rollout["skill"] == index).float().mean())
        for name, value in metrics.items():
            writer.add_scalar(name, value, iteration)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if iteration % args.save_interval == 0 or iteration == args.max_iterations - 1:
            status["checkpoint"] = checkpoint(iteration)
            writer.flush()
        write_status(state="running", **metrics)
        print(
            "PLANNER_ITER "
            + json.dumps({k: v for k, v in metrics.items() if not k.startswith(("terrain/", "spawn_terrain/"))}),
            flush=True,
        )
        if stop_requested:
            break
    assert not torch.equal(
        initial, torch.cat([p.detach().flatten() for p in model.parameters()])
    ), "No parameter update"
    for policy, before in zip(navigation.skills.models, frozen_before):
        torch.testing.assert_close(next(policy.parameters()), before, atol=0.0, rtol=0.0)
        assert all(parameter.grad is None for parameter in policy.parameters())
    write_status(state="stopped" if stop_requested else "complete", checkpoint=checkpoint(iteration))
    print(("PLANNER_STOPPED " if stop_requested else "PLANNER_TRAINING_OK ") + json.dumps(status), flush=True)
    success = True
except BaseException as error:
    traceback.print_exc()
    write_status(state="failed", error=repr(error))
finally:
    if "writer" in globals():
        writer.close()
    if "env" in globals():
        env.close()
    app.close()
# Check status.json/PLANNER_TRAINING_OK as Isaac can exit the process in close().
if not success:
    sys.exit(1)
