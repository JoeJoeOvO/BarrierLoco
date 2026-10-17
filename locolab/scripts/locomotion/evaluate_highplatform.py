"""Evaluate an immutable native Go2W MLP checkpoint on four fixed-height platforms."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument("--source-manifest", type=Path, help="Optional SHA-256 manifest of the training snapshot.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", action="store_true")
parser.add_argument("--dynamics", choices=("native", "unified"), default="native")
known, _ = parser.parse_known_args()
checkpoint_path = known.checkpoint
if checkpoint_path.is_dir():
    choices = [p for p in checkpoint_path.glob("model_*.pt") if time.time() - p.stat().st_mtime > 10.0]
    checkpoint_path = max(choices, key=lambda p: int(p.stem.split("_")[-1]))
checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
reference = json.loads(known.source_manifest.read_text()) if known.source_manifest else {}
for name, expected in reference.items():
    assert hashlib.sha256((known.source_root / name).read_bytes()).hexdigest() == expected, name
print("EVALUATION_CHECKPOINT", str(checkpoint_path), checkpoint_sha, flush=True)
sys.path.insert(0, str(known.source_root))
sys.path.insert(0, str(known.source_root / "locolab"))
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.checkpoint = checkpoint_path
args.enable_cameras = args.video
app = AppLauncher(args).app

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.terrains as terrain_gen
import numpy as np
import omni.usd
import torch
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from pxr import UsdGeom
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

import locolab.source.locomotion.tasks  # noqa: F401
from locolab.source.locomotion.terrains.high_platform_cfg import MeshHighPlatformTerrainCfg
from locolab.source.locomotion.utils.platform_evaluation import PlatformProgress

HEIGHTS = [0.5, 0.6, 0.7, 0.8]
PER_HEIGHT = 16
COUNT = len(HEIGHTS) * PER_HEIGHT
DEADLINE = 10.0
TASK = "OmniBot-Go2W-HighPlatform-MLP-v0"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
env = None


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


try:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise RuntimeError("Results already exist; refusing to overwrite an evaluation")
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    if args.dynamics == "unified":
        from copy import deepcopy

        from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg

        cfg.scene.robot = deepcopy(Go2WPlannerTerrainEnvCfg().scene.robot)
    cfg.seed = 42
    cfg.scene.num_envs = COUNT
    cfg.sim.device = args.device
    cfg.episode_length_s = DEADLINE
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    assert cfg.rewards.feet_stumble.params["ratio_threshold"] == 5.0
    for name, value in vars(cfg.curriculum).items():
        if hasattr(value, "func"):
            setattr(cfg.curriculum, name, None)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        seed=42,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=len(HEIGHTS),
        curriculum=True,  # Deterministic column assignment only; manager curriculum is disabled.
        use_cache=False,
        sub_terrains={
            f"platform_{round(100 * height)}": MeshHighPlatformTerrainCfg(
                proportion=1.0 / len(HEIGHTS),
                platform_height_range=(height, height),
                num_height_levels=36,
                platform_start_x=2.5,
                platform_length=1.5,
                platform_width=2.0,
                spawn_x=1.0,
                ground_thickness=1.0,
            )
            for height in HEIGHTS
        },
    )
    for name, value in vars(cfg.events).items():
        if hasattr(value, "mode") and name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "yaw": (-0.08, 0.08)},
        "velocity_range": {},
    }
    cfg.events.reset_robot_joints.params = {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
    for group in vars(cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (0.6, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)
    command.heading_command = True
    command.rel_heading_envs = 1.0
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1000.0, 1000.0)
    command.debug_vis = False
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (-0.5, -4.0, 2.6)
    cfg.viewer.lookat = (2.0, 0.0, 0.5)
    cfg.viewer.resolution = (1280, 720)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
    observations, _ = env.reset(seed=42)
    assert observations["policy"].shape == (COUNT, 53)
    assert observations["critic"].shape == (COUNT, 307)
    policy_kwargs = agent_cfg.policy.to_dict()
    assert policy_kwargs.pop("class_name") == "ActorCritic"
    policy = (
        ActorCritic(TensorDict(observations, [COUNT]), agent_cfg.obs_groups, 16, **policy_kwargs).to(env.device).eval()
    )
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
    assert all(torch.isfinite(value).all().item() for value in saved["model_state_dict"].values())
    assert policy.load_state_dict(saved["model_state_dict"], strict=True)
    iteration = saved["iter"]
    del saved
    robot = env.scene["robot"]
    wheel_ids, wheel_names = robot.find_bodies(".*_foot")
    sensor = env.scene["contact_forces"]
    sensor_ids, sensor_names = sensor.find_bodies(".*_foot")
    assert len(wheel_ids) == 4 and wheel_names == sensor_names
    kinds = env.scene.terrain.terrain_types.clone()
    assert torch.bincount(kinds, minlength=len(HEIGHTS)).tolist() == [PER_HEIGHT] * len(HEIGHTS)
    heights = torch.tensor(HEIGHTS, device=env.device)[kinds]
    stage = omni.usd.get_context().get_stage()
    for index in range(COUNT):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot/base")
        assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    report = {
        "dynamics": args.dynamics,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_iteration": iteration,
        "source_files_verified": len(reference),
        "task": TASK,
        "num_envs": COUNT,
        "platform_heights_m": HEIGHTS,
        "platform_length_m": 1.5,
        "platform_width_m": 2.0,
        "platform_front_x_from_spawn_m": 1.5,
        "platform_back_x_from_spawn_m": 3.0,
        "deadline_s": DEADLINE,
        "speeds_m_s": [0.6, 1.0],
        "seeds": [42, 43],
        "attempts_per_height_per_speed": 32,
        "inference": "Native PyTorch deterministic mean action; no optimization or exploration noise",
        "observation_noise": False,
        "domain_randomization": False,
        "self_collisions": True,
        "reset_pose_jitter": {"x_m": [-0.1, 0.1], "y_m": [-0.2, 0.2], "yaw_rad": [-0.08, 0.08]},
        "heading_command": True,
        "heading_target_rad": 0.0,
        "heading_gain": command.heading_control_stiffness,
        "yaw_rate_limit": list(command.ranges.ang_vel_z),
        "mount_rule": (
            "All four wheel centers inside top footprint and within 6.5 cm of top+0.086 m; "
            "at least two supporting contacts >5 N; gravity_z<-0.7; continuous 0.1 s"
        ),
        "pass_rule": (
            "After mount, all wheels beyond platform back+0.05 m, body beyond back+0.35 m, "
            "all wheel centers within 8 cm of ground+0.086 m, supported and upright for 0.2 s"
        ),
        "failure_rule": (
            "First termination, timeout, or body |y|>0.75 m ends attempt; autoresets never create another attempt"
        ),
        "batches": [],
        "video_replays": [],
    }
    save_json(
        args.output / "runtime_contract.json",
        {
            "step_dt": env.step_dt,
            "effort_limits": robot.data.joint_effort_limits[0].tolist(),
            "velocity_limits": robot.data.joint_vel_limits[0].tolist(),
            "joint_names": robot.data.joint_names,
            "wheel_names": wheel_names,
            "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
            "stiffness": robot.data.default_joint_stiffness[0].tolist(),
            "damping": robot.data.default_joint_damping[0].tolist(),
            "actions": {
                name: {"joint_names": term._joint_names, "scale": term.cfg.scale}
                for name, term in env.action_manager._terms.items()
            },
            "observations": {
                name: {
                    "terms": env.observation_manager.active_terms[name],
                    "dims": env.observation_manager.group_obs_term_dim[name],
                }
                for name in ("policy", "critic")
            },
        },
    )

    def run_batch(speed, seed, focus=None):
        env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = (speed, speed)
        obs, _ = env.reset(seed=seed)
        assert (env.scene.terrain.terrain_levels == 0).all().item()
        torch.testing.assert_close(
            env.command_manager.get_command("base_velocity")[:, 0], torch.full((COUNT,), speed, device=env.device)
        )
        origins = env.scene.env_origins.clone()
        progress = PlatformProgress(heights, env.step_dt)
        initial_root = robot.data.root_pos_w - origins
        progress.max_forward = initial_root[:, 0].clone()
        initial_state = robot.data.root_state_w.clone().cpu().numpy()
        frames = []
        snapshots = []
        if focus is not None:
            for index in range(COUNT):
                prim = UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot"))
                prim.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            visible = UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{focus}/Robot"))
            visible.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            assert visible.ComputeVisibility() != UsdGeom.Tokens.invisible
            env.viewport_camera_controller.set_view_env_index(focus)
        for step in range(round(DEADLINE / env.step_dt)):
            with torch.inference_mode():
                actions = policy.act_inference(TensorDict(obs, [COUNT]))
            assert torch.isfinite(actions).all().item()
            obs, _, terminated, truncated, _ = env.step(actions)
            root = robot.data.root_pos_w - origins
            wheels = robot.data.body_pos_w[:, wheel_ids] - origins[:, None, :]
            forces = sensor.data.net_forces_w[:, sensor_ids, 2]
            gravity = robot.data.projected_gravity_b[:, 2]
            progress.update(root, wheels, forces, gravity, terminated, truncated, (step + 1) * env.step_dt)
            if step % 5 == 0:
                snapshots.append(
                    torch.cat(
                        (
                            root,
                            wheels.flatten(1),
                            forces,
                            gravity[:, None],
                            progress.active[:, None],
                            progress.mounted[:, None],
                            progress.passed[:, None],
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
            if focus is not None and step % 2 == 0 and not (terminated[focus] | truncated[focus]).item():
                frame = np.ascontiguousarray(env.render()[:, :, :3]).copy()
                label = (
                    f"{heights[focus].item()*100:.0f} cm | vx={speed:.1f} m/s | "
                    f"seed={seed} | t={(step+1)*env.step_dt:.2f}s"
                )
                cv2.putText(frame, label, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                state = "MOUNTED" if progress.mounted[focus].item() else "APPROACH / CLIMB"
                cv2.putText(frame, state, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 255), 2)
                frames.append(frame)
            if focus is not None and not progress.active[focus].item():
                break
            if not progress.active.any().item():
                break
        progress.finish()
        detail = {
            name: getattr(progress, name).tolist()
            for name in (
                "mounted",
                "passed",
                "terminated",
                "timed_out",
                "left_corridor",
                "mount_time",
                "pass_time",
                "max_forward",
            )
        }
        cases = []
        for height_id, height in enumerate(HEIGHTS):
            mask = kinds == height_id
            case = {"height_m": height, "command_m_s": speed, "seed": seed, "attempts": int(mask.sum().item())}
            for name in ("mounted", "passed", "terminated", "timed_out", "left_corridor"):
                case[name] = int(getattr(progress, name)[mask].sum().item())
            cases.append(case)
        if focus is None:
            np.savez_compressed(
                args.output / f"trajectory_v{speed}_seed{seed}.npz",
                samples=np.asarray(snapshots),
                initial_root_state=initial_state,
                dt=env.step_dt * 5,
                height_id=kinds.cpu().numpy(),
            )
        else:
            video_name = f"platform_{round(heights[focus].item()*100)}cm_v{speed}_seed{seed}.mp4"
            outcome = (
                "PASS: crossed and landed"
                if progress.passed[focus].item()
                else ("MOUNTED, CROSSING FAILED" if progress.mounted[focus].item() else "FAIL: did not mount")
            )
            if frames:
                final = frames[-1].copy()
                cv2.putText(final, outcome, (24, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 255), 2)
                with imageio.get_writer(
                    str(args.output / video_name),
                    fps=25,
                    codec="libx264",
                    quality=7,
                    macro_block_size=16,
                    ffmpeg_params=["-threads", "2"],
                ) as video_writer:
                    for frame in frames:
                        video_writer.append_data(frame)
                    for _ in range(25):
                        video_writer.append_data(final)
            return {
                "height_m": heights[focus].item(),
                "speed": speed,
                "seed": seed,
                "env_index": focus,
                "video": video_name,
                **{name: value[focus] for name, value in detail.items()},
            }
        return {"speed": speed, "seed": seed, "cases": cases, "per_env": detail}

    for speed in (0.6, 1.0):
        for seed in (42, 43):
            batch = run_batch(speed, seed)
            report["batches"].append(batch)
            save_json(args.output / "results.json", report)
            print("PLATFORM_BATCH_RESULT", json.dumps(batch["cases"]), flush=True)
    if args.video:
        reference_batch = next(batch for batch in report["batches"] if batch["speed"] == 1.0 and batch["seed"] == 42)
        for height_id in range(len(HEIGHTS)):
            focus = height_id * PER_HEIGHT
            replay = run_batch(1.0, 42, focus=focus)
            replay["agrees_with_quantitative"] = all(
                replay[name] == reference_batch["per_env"][name][focus] for name in ("mounted", "passed")
            )
            report["video_replays"].append(replay)
            save_json(args.output / "results.json", report)
            print("PLATFORM_VIDEO_RESULT", json.dumps(replay), flush=True)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    print("FIXED_PLATFORM_EVALUATION_OK", str(args.output), flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
