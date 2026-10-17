"""Paired flat-ground dynamics checks through the actual Planner skill adapter.

This isolates robot configuration changes, not terrain generalization or skill
switching. HighPlatform uses evaluate_highplatform.py instead.
"""

import argparse
import hashlib
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--skill", choices=("multiskill", "goalgrid"), required=True)
parser.add_argument("--dynamics", choices=("native", "unified"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--disable-self-collision", action="store_true", help="Diagnostic ablation only; does not edit tasks."
)
parser.add_argument("--jump-start", choices=("standing", "rolling"), default="standing")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.planner.occupancy import yaw_rotate
    from locolab.planner.skills import POLICIES, FrozenSkills
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.skill == "multiskill":
        velocities = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (-3.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (1.0, 0.3, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
        ]
        cases = [
            {"gait": gait, "velocity": velocity}
            for gait, velocity in itertools.product(("rolling", "trot", "bound"), velocities)
        ]
        task = "OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0"
        duration = 8.0
    else:
        cases = [
            {"height": h, "x": x, "y": y}
            for h, x, y in itertools.product((0.15, 0.2, 0.25), (0.4, 0.6, 0.8), (-0.15, 0.0, 0.15))
        ]
        task = "OmniBot-Go2W-Parkour-GoalGrid-MLP-v0"
        duration = 2.2
    repeats = 4
    count = len(cases) * repeats
    cfg = Go2WPlannerTerrainEnvCfg()
    if args.dynamics == "native":
        cfg.scene.robot = deepcopy(load_cfg_from_registry(task, "env_cfg_entry_point").scene.robot)
    if args.disable_self_collision:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.num_envs = count
    cfg.scene.env_spacing = 8.0
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.sim.device = args.device
    cfg.episode_length_s = 100.0
    cfg.terminations.goal = cfg.terminations.boundary = cfg.terminations.pit = None
    # Fixed jitter distributions and seed across paired runs.
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.02, 0.02)}
    # Diagnostic base observations contain goal_vector; a plane has no terrain_types.
    cfg.observations.policy.goal_xy = None
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg).unwrapped
    adapter = FrozenSkills(env, ROOT)
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    feet, _ = sensor.find_bodies(".*_foot")
    ids = torch.arange(count, device=env.device)
    case_ids = ids // repeats
    policy_path = ROOT / "deploy/policy/go2w" / POLICIES[0 if args.skill == "multiskill" else 2]
    report = {
        "skill": args.skill,
        "dynamics": args.dynamics,
        "task": task,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "duration_s": duration,
        "jump_start": args.jump_start,
        "seeds": [42, 43],
        "repeats_per_case_per_seed": repeats,
        "scope": "Planner adapter; paired dynamics on flat terrain; no navigation or transition claim",
        "effort_limits": robot.data.joint_effort_limits[0].tolist(),
        "velocity_limits": robot.data.joint_vel_limits[0].tolist(),
        "solver_iterations": [
            cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
        ],
        "self_collision": cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
        "cases": cases,
        "batches": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for seed in (42, 43):
        env.reset(seed=seed)
        adapter.reset(ids)
        active = torch.ones(count, dtype=torch.bool, device=env.device)
        # Settle with the same skill's idle command for one second.
        if args.skill == "goalgrid":
            adapter.skill[:] = 2
            adapter.command[:] = 0
        for _ in range(50):
            if args.skill == "goalgrid" and args.jump_start == "rolling":
                # Diagnostic ablation: show descriptors and approach at 1 m/s
                # before asserting the jump flag, matching the upstream contract.
                data = robot.data
                velocity = torch.zeros(count, 3, device=env.device)
                velocity[:, 0] = 1.0
                velocity = yaw_rotate(velocity, data.root_quat_w, inverse=True)
                descriptors = torch.tensor([[c["height"], c["x"], c["y"]] for c in cases], device=env.device)[case_ids]
                obs = torch.cat(
                    (
                        data.root_ang_vel_b * 0.2,
                        data.projected_gravity_b,
                        velocity,
                        torch.zeros(count, 1, device=env.device),
                        descriptors,
                        data.joint_pos[:, adapter.legs] - data.default_joint_pos[:, adapter.legs],
                        (data.joint_vel[:, adapter.joints] - data.default_joint_vel[:, adapter.joints]) * 0.05,
                        adapter.previous,
                    ),
                    -1,
                )
                with torch.no_grad():
                    action = adapter.models[2](obs.clamp(-100, 100)).clamp(-100, 100)
                adapter.previous.copy_(action)
            else:
                action = adapter.act()
            _, _, terminated, truncated, _ = env.step(action)
            active &= ~(terminated | truncated)
        warmup_failed = ~active.clone()
        start = robot.data.root_pos_w.clone()
        if args.skill == "multiskill":
            for index, case in enumerate(cases):
                mask = case_ids == index
                command = adapter.command[mask].clone()
                command[:, :3] = command.new_tensor(case["velocity"])
                command[:, 3] = 0.34
                command[:, 4] = 2.5
                command[:, 5:8] = command.new_tensor(
                    {"rolling": (0.0, 0.0, 0.0), "trot": (0.5, 0.0, 0.0), "bound": (0.0, 0.0, 0.5)}[case["gait"]]
                )
                command[:, 8] = 1.0 if case["gait"] == "rolling" else 0.5
                command[:, 9] = 0.0 if case["gait"] == "rolling" else 0.10
                adapter.command[mask] = command
            target = adapter.command[:, :3].clone()
        else:
            adapter.timer[:] = duration
            adapter.jump_heading[:] = robot.data.root_quat_w
            adapter.command[:] = 0
            adapter.command[:, 0] = 1.0
            target = torch.tensor([[c["height"], c["x"], c["y"]] for c in cases], device=env.device)[case_ids]
            adapter.command[:, 1:4] = target
        squared_error = torch.zeros(count, 3, device=env.device)
        samples = torch.zeros(count, device=env.device)
        max_tilt = torch.zeros(count, device=env.device)
        peak = torch.zeros_like(samples)
        airborne_count = torch.zeros_like(samples)
        took_off = torch.zeros(count, dtype=torch.bool, device=env.device)
        landed = torch.zeros_like(took_off)
        landing = torch.zeros(count, 2, device=env.device)
        for step in range(round(duration / env.step_dt)):
            with torch.inference_mode():
                actions = adapter.act()
            assert torch.isfinite(actions).all()
            _, _, terminated, truncated, _ = env.step(actions)
            # Ignore all post-autoreset state; each environment is one attempt.
            active &= ~(terminated | truncated)
            tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            max_tilt = torch.where(active, torch.maximum(max_tilt, tilt), max_tilt)
            if args.skill == "multiskill":
                if step >= 50:  # discard one-second command transient
                    velocity = yaw_rotate(robot.data.root_lin_vel_w, robot.data.root_quat_w, inverse=True)
                    measured = torch.cat((velocity[:, :2], robot.data.root_ang_vel_w[:, 2:3]), -1)
                    squared_error += (measured - target).square() * active[:, None]
                    samples += active
            else:
                relative = yaw_rotate(robot.data.root_pos_w - start, adapter.jump_heading, inverse=True)
                peak = torch.where(active, torch.maximum(peak, relative[:, 2]), peak)
                contacts = (sensor.data.net_forces_w[:, feet].norm(dim=-1) > 5.0).sum(-1)
                airborne_count = torch.where((contacts == 0) & active, airborne_count + 1, 0.0)
                took_off |= airborne_count >= 3  # >=60 ms with no wheel contact
                first_landing = active & took_off & ~landed & (contacts >= 2)
                landing[first_landing] = relative[first_landing, :2]
                landed |= first_landing
        batch = {
            "seed": seed,
            "survived": active.tolist(),
            "warmup_failed": warmup_failed.tolist(),
            "max_tilt_deg": (max_tilt * 180 / torch.pi).tolist(),
        }
        if args.skill == "multiskill":
            batch.update(
                rmse=(squared_error / samples.clamp(min=1)[:, None]).sqrt().tolist(), valid_samples=samples.tolist()
            )
        else:
            error = (landing - target[:, 1:]).norm(dim=-1)
            batch.update(
                took_off=took_off.tolist(),
                landed=landed.tolist(),
                peak_rise_m=peak.tolist(),
                height_error_m=(peak - target[:, 0]).abs().tolist(),
                landing_error_m=[float(error[i]) if landed[i] else None for i in range(count)],
                success=(active & landed & ((peak - target[:, 0]).abs() < 0.10) & (error < 0.20)).tolist(),
            )
        report["batches"].append(batch)
        save()
        print("DYNAMICS_BATCH", args.skill, args.dynamics, seed, "survived", int(active.sum()), "/", count, flush=True)
    report["complete"] = True
    save()
    print("SKILL_DYNAMICS_EVALUATION_OK", str(args.output), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in globals():
        env.close()
    app.close()
