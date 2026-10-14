"""Record a continuous, labelled sweep of the extended Go2W command set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from cmd_sweep import build_cases
from swing_height import SwingHeightTracker, format_heights_cm


def main():
    root = Path(__file__).resolve().parents[2]
    repo = root
    sys.path[:0] = [str(repo / "locolab/scripts/locomotion"), str(repo / "locolab"), str(repo)]

    import cli_args
    from isaaclab.app import AppLauncher

    default_model = repo / "deploy/policy/go2w/multiskill_scratch80k/model_79999.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_model)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0, help="Seconds per command")
    parser.add_argument("--case-limit", type=int, default=0, help="0 records all extended commands")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0.02:
        parser.error("--seconds must be finite and at least one simulation step (0.02s)")
    cases = build_cases(extended=True)
    if args.case_limit:
        if not 1 <= args.case_limit <= len(cases):
            parser.error("--case-limit must be in [1, 76]")
        cases = cases[: args.case_limit]
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import imageio.v2 as imageio
        import isaaclab_tasks  # noqa: F401
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry
        from PIL import Image, ImageDraw, ImageFont
        from rsl_rl.runners import OnPolicyRunner

        import locolab.source.locomotion.tasks  # noqa: F401
        from locolab.source.locomotion.mdp.wtw_math import validate_commands
        from locolab.source.locomotion.mdp.wtw_rewards import _wheel_state
        from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg
        from locolab.source.locomotion.utils.video import configure_robot_follow_view

        validate_commands(torch.tensor([case["command"] for case in cases]))
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        video_path = output / "go2w_continuous_cmd_sweep.mp4"
        telemetry_path = output / "telemetry.csv"
        task = "OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")
        cfg.seed = 42
        cfg.commands.base_velocity.heading_command = False
        cfg.terminations.time_out = None
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        configure_robot_follow_view(cfg.viewer)
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = (1.8, 2.0, 1.0)
        cfg.viewer.lookat = (0.0, 0.0, 0.0)

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        raw = gym.make(task, cfg=cfg, render_mode="rgb_array")
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, cli_args.runner_cfg_to_dict(agent_cfg), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        term = env.unwrapped.command_manager.get_term("base_velocity")
        tracker = SwingHeightTracker()
        dt = env.unwrapped.step_dt
        fps = round(1 / dt)
        frames_per_case = round(args.seconds / dt)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        title_font = ImageFont.truetype(font_path, 28)
        body_font = ImageFont.truetype(font_path, 20)
        fields = [
            "frame",
            "case",
            "gait",
            "group",
            "case_time_s",
            "vx_cmd",
            "actual_vx_mps",
            "vy_cmd",
            "actual_vy_mps",
            "yaw_cmd",
            "actual_yaw_rad_s",
            "body_height_cmd_m",
            "actual_body_height_m",
            "frequency_hz",
            "swing_cmd_m",
            "wheel_height_fr_m",
            "wheel_height_fl_m",
            "wheel_height_rr_m",
            "wheel_height_rl_m",
            "swing_peak_fr_m",
            "swing_peak_fl_m",
            "swing_peak_rr_m",
            "swing_peak_rl_m",
            "failure_reset",
            "total_failure_resets",
        ]
        results = []
        total_resets = 0
        global_frame = 0
        env.reset()
        started_at = time.time()
        with (
            imageio.get_writer(
                str(video_path),
                fps=fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            ) as writer,
            telemetry_path.open("w", newline="") as csv_file,
            torch.inference_mode(),
        ):
            telemetry = csv.DictWriter(csv_file, fieldnames=fields)
            telemetry.writeheader()
            for case_index, case in enumerate(cases, start=1):
                # Intentionally do not reset environment, phase, action history, pose or velocity here.
                command = torch.tensor(case["command"], device=env.unwrapped.device)
                tracker.reset()
                case_resets = 0
                actual_vx = []
                actual_height = []
                print(
                    "CONTINUOUS_CMD_START " + json.dumps({"case": case_index, "total": len(cases), **case}),
                    flush=True,
                )
                for case_frame in range(frames_per_case):
                    if not app.is_running():
                        raise RuntimeError("Simulator closed before recording completed.")
                    term.set_command(command)
                    obs = env.get_observations()
                    _, _, dones, _ = env.step(policy(obs))
                    did_reset = bool(dones[0].item())
                    case_resets += int(did_reset)
                    total_resets += int(did_reset)
                    if did_reset:
                        tracker.reset()

                    data = term.robot.data
                    height = float(data.root_pos_w[0, 2] - env.unwrapped.scene.env_origins[0, 2])
                    velocity = data.root_lin_vel_b[0].tolist()
                    yaw_rate = float(data.root_ang_vel_b[0, 2])
                    bottom, _ = _wheel_state(term)
                    wheel_heights = (bottom[0, :, 2] - env.unwrapped.scene.env_origins[0, 2]).tolist()
                    if did_reset:
                        swing_peaks = [None] * 4
                    else:
                        swinging = ((term.leg_phase[0] >= 0.5) & (term.command[0, 8] < 1)).tolist()
                        swing_peaks = tracker.update(wheel_heights, swinging)
                    actual_vx.append(velocity[0])
                    actual_height.append(height)
                    telemetry.writerow(
                        dict(
                            zip(
                                fields,
                                [
                                    global_frame,
                                    case_index,
                                    case["gait"],
                                    case["group"],
                                    (case_frame + 1) * dt,
                                    command[0].item(),
                                    velocity[0],
                                    command[1].item(),
                                    velocity[1],
                                    command[2].item(),
                                    yaw_rate,
                                    command[3].item(),
                                    height,
                                    command[4].item(),
                                    command[9].item(),
                                    *wheel_heights,
                                    *swing_peaks,
                                    int(did_reset),
                                    total_resets,
                                ],
                            )
                        )
                    )

                    frame = env.unwrapped.render()
                    if frame is None:
                        raise RuntimeError("RGB rendering returned no frame.")
                    picture = Image.fromarray(frame)
                    draw = ImageDraw.Draw(picture, "RGBA")
                    draw.rectangle((0, 0, 1280, 151), fill=(8, 14, 22, 225))
                    draw.text(
                        (20, 10),
                        f"{case_index}/{len(cases)}  {case['gait'].upper()} | {case['group']} | continuous CMD switch",
                        font=title_font,
                        fill=(255, 255, 255, 255),
                    )
                    draw.text(
                        (20, 52),
                        f"CMD vx {command[0]:+.2f} vy {command[1]:+.2f} yaw {command[2]:+.2f} | "
                        f"height {100*command[3]:.0f} cm | freq {command[4]:.1f} Hz | swing {100*command[9]:.1f} cm",
                        font=body_font,
                        fill=(205, 225, 245, 255),
                    )
                    draw.text(
                        (20, 87),
                        f"ACTUAL vx {velocity[0]:+.2f} vy {velocity[1]:+.2f} yaw {yaw_rate:+.2f} | "
                        f"body {100*height:.1f} cm | wheel FR/FL/RR/RL: {format_heights_cm(wheel_heights)} cm",
                        font=body_font,
                        fill=(255, 255, 255, 255),
                    )
                    draw.text(
                        (20, 120),
                        f"Last complete swing peaks FR/FL/RR/RL: {format_heights_cm(swing_peaks)} cm | "
                        f"resets {case_resets}/{total_resets} | {(case_frame + 1) * dt:.1f}/{args.seconds:.1f} s",
                        font=body_font,
                        fill=(255, 145, 115, 255) if did_reset else (205, 225, 245, 255),
                    )
                    writer.append_data(np.asarray(picture))
                    global_frame += 1
                results.append(
                    {
                        "case": case_index,
                        **case,
                        "mean_actual_vx_mps": sum(actual_vx) / len(actual_vx),
                        "mean_actual_body_height_m": sum(actual_height) / len(actual_height),
                        "failure_resets": case_resets,
                    }
                )
                csv_file.flush()
                print(f"CONTINUOUS_CMD_DONE {case_index}/{len(cases)} resets={case_resets}", flush=True)
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "task": task,
            "continuous_between_commands": True,
            "timeout_reset_disabled": True,
            "failure_resets_retained": True,
            "case_count": len(cases),
            "seconds_per_case": args.seconds,
            "fps": fps,
            "frame_count": global_frame,
            "duration_s": global_frame / fps,
            "failure_resets": total_resets,
            "wall_seconds": time.time() - started_at,
            "video": str(video_path),
            "telemetry": str(telemetry_path),
            "cases": results,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("CONTINUOUS_CMD_SWEEP_OK " + json.dumps(summary), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
