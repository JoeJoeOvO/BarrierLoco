"""Frozen final policy, four speeds, seven jumps and ordinary rolling, three trials each."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source_root", type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument("--checkpoint", type=Path, required=True)
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
    import yaml

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.source.locomotion.tasks.Go2W.parkour_env_cfg import Go2WParkourFixedTargetTunedPlayEnvCfg

    if args.output.exists():
        raise ValueError("Preserve the existing evaluation")
    cfg = Go2WParkourFixedTargetTunedPlayEnvCfg()
    cfg.scene.num_envs = 120
    cfg.sim.device = args.device
    cfg.seed = 231
    agent = yaml.safe_load((args.checkpoint.parent / "params/agent.yaml").read_text())
    params = args.checkpoint.parent / "params"
    env_yaml = params / "env.yaml"
    if env_yaml.exists():
        recorded = yaml.load(env_yaml.read_text(), Loader=yaml.BaseLoader)
    else:
        with gzip.open(params / "env.yaml.gz", "rt") as stream:
            recorded = yaml.load(stream, Loader=yaml.BaseLoader)
    assert not agent["policy"]["actor_obs_normalization"]
    assert (
        recorded["training_stage"] == "fixed_target_single_stage_tuned"
        and int(recorded["command_semantics_version"]) == 3
    )
    assert int(recorded["critic_command_semantics_version"]) == 2
    assert int(recorded["reward_recipe_version"]) == 2
    assert cfg.events.base_external_force_torque is None
    env = gym.make("OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0", cfg=cfg)
    native = env.unwrapped
    obs, _ = env.reset()
    term = native.command_manager.get_term("parkour")
    robot = native.scene["robot"]
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    assert saved["iter"] == 4999
    assert all(torch.isfinite(v).all() for v in saved["model_state_dict"].values())
    actor = (
        torch.nn.Sequential(
            torch.nn.Linear(57, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 16),
        )
        .to(args.device)
        .eval()
    )
    actor.load_state_dict(
        {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
    )
    speeds = [0.2, 0.5, 1.0, 1.5]
    term.forward_speed[:] = torch.tensor(speeds, device=args.device).repeat_interleave(30)
    assert not term.active.any()
    obs = native.observation_manager.compute()

    def check_commands(observation):
        target = term.bank.descriptors[term.motion_ids] * (~(term.locomotion_only | term.rolling_exit))[:, None]
        torch.testing.assert_close(term.command[:, 4:], target, rtol=0, atol=0)
        torch.testing.assert_close(observation["policy"][:, 6:13], term.command)
        torch.testing.assert_close(observation["critic"][:, 9:16], term.critic_command)

    check_commands(obs)
    last = {}
    compute = native.termination_manager.compute

    def capture():
        done = compute()
        last.update(
            position=(robot.data.root_pos_w - native.scene.env_origins).clone(),
            velocity=robot.data.root_lin_vel_w.clone(),
            quaternion=robot.data.root_quat_w.clone(),
            omega=robot.data.root_ang_vel_w.clone(),
            contacts=term.contact_state.clone(),
            attempted=term.ground_attempt.clone(),
            success=term.ground_success.clone(),
            active=term.active.clone(),
            reference_time=term.reference_time.clone(),
            exit_time=term.exit_time.clone(),
            contact_foot_error=term.touchdown_error.clone(),
            reported_peak=term.flight_peak_height.clone(),
            actor_command=term.command.clone(),
            reasons={
                name: native.termination_manager.get_term(name).clone()
                for name in native.termination_manager.active_terms
            },
        )
        return done

    native.termination_manager.compute = capture
    traces = [[] for _ in range(120)]
    ended = [False] * 120
    results = []

    def summarize(i, reasons):
        trace = traces[i]
        speed = speeds[i // 30]
        motion_id = i % 10 if i % 10 < 7 else i % 10 - 7
        rolling = i % 10 >= 7
        start = next((j for j, row in enumerate(trace) if row["attempted"]), None)
        flight_peak, longest, air, positive = 0.0, 0.0, 0.0, False
        landing_error, support = None, False
        if start is not None:
            launch_z = trace[start]["position"][2]
            for row in trace[start:]:
                rise = row["position"][2] - launch_z
                if not any(row["contacts"]):
                    if support and row["velocity"][2] > 0.35 and rise > 0.025:
                        positive = True
                    air += native.step_dt
                    if positive and landing_error is None:
                        longest = max(longest, air)
                        flight_peak = max(flight_peak, rise)
                else:
                    support = True
                    if positive and air >= 0.08 and landing_error is None:
                        goal = term.bank.descriptors[motion_id, 1:].cpu().tolist()
                        landing_error = math.dist(row["position"][:2], goal)
                    air = 0.0
        ordinary = [row for j, row in enumerate(trace) if (j + 1) * native.step_dt >= 1.0 and rolling]
        approach = trace[:start] if start is not None else trace
        exit_segment = [row for row in trace if row["exit_time"] >= 0.10]
        yaw = []
        for row in trace:
            w, x, y, z = row["quaternion"]
            yaw.append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

        def mae(rows):
            return sum(abs(row["velocity"][0] - speed) for row in rows) / len(rows) if rows else None

        return dict(
            env=i,
            trial=(i // 10) % 3,
            speed=speed,
            rolling=rolling,
            motion="rolling" if rolling else term.bank.names[motion_id],
            rolling_lane=None if not rolling else motion_id,
            attempted=start is not None,
            positive_launch=positive,
            peak_body_rise_m=flight_peak,
            longest_all_air_s=longest,
            landing_error_m=landing_error,
            reported_peak_m=trace[-1]["reported_peak"],
            success=(
                (reasons == ["time_out"]) if rolling else bool(trace[-1]["success"] and "task_succeeded" in reasons)
            ),
            terminal_reasons=reasons,
            duration_s=len(trace) * native.step_dt,
            trigger_time_s=None if start is None else (start + 1) * native.step_dt,
            trigger_vx=None if start is None else trace[start]["velocity"][0],
            trigger_height_m=None if start is None else trace[start]["position"][2],
            approach_min_height_m=min(row["position"][2] for row in approach) if approach else None,
            rolling_vx_mae_m_s=mae(ordinary),
            rolling_exit_vx_mae_m_s=mae(exit_segment),
            rolling_exit_last_vx=None if not exit_segment else exit_segment[-1]["velocity"][0],
            max_abs_yaw_deg=max(map(abs, yaw)) * 180 / math.pi,
            rolling_height_mean_m=None if not ordinary else sum(row["position"][2] for row in ordinary) / len(ordinary),
            rolling_height_min_m=None if not ordinary else min(row["position"][2] for row in ordinary),
            trace=trace,
        )

    for step in range(520):
        with torch.no_grad():
            action = actor(obs["policy"])
            obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(action).all() and torch.isfinite(reward).all()
        check_commands(obs)
        sample = {k: v.cpu().tolist() for k, v in last.items() if k != "reasons"}
        done = (terminated | truncated).cpu().tolist()
        reasons = {k: v.cpu().tolist() for k, v in last["reasons"].items()}
        for i in range(120):
            if ended[i]:
                continue
            row = {k: v[i] for k, v in sample.items()}
            if not math.isfinite(row["contact_foot_error"]):
                row["contact_foot_error"] = None
            traces[i].append(row)
            if done[i] or step == 519:
                ended[i] = True
                result = summarize(i, [name for name, values in reasons.items() if values[i]])
                results.append(result)
                print("EPISODE " + json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
        if all(ended):
            break
    assert len(results) == 120
    report = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_iteration=saved["iter"],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_root=str(args.source_root),
        speeds=speeds,
        trials_per_jump_speed=3,
        rolling_trials_per_speed=9,
        ground_only=True,
        deterministic=True,
        observation_noise=False,
        domain_randomization=False,
        external_wrenches=False,
        actor_command_version=3,
        critic_command_version=2,
        actor_targets_fixed_verified=True,
        reference_reset_probability=0.0,
        ppo_updates=0,
        results=sorted(results, key=lambda x: x["env"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("FINAL_FIXED_EVAL_COMPLETE", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
