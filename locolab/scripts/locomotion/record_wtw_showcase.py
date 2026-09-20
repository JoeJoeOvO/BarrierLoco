"""Record a deterministic, labelled Go2W WTW gait/height showcase (no training)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys


def build_segments(phase_presets, duration_s=8.0):
    """Use nominal gait phases and one common velocity for fair comparisons."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Segment duration must be positive and finite.")
    segments = []
    for gait in ("trot", "bound", "rolling"):
        for height in (0.34, 0.22, 0.46):
            rolling = gait == "rolling"
            segments.append(
                {
                    "gait": gait,
                    "height_m": height,
                    "duration_s": duration_s,
                    "command": [
                        0.6,
                        0.0,
                        0.0,
                        height,
                        2.5,
                        *phase_presets[gait],
                        1.0 if rolling else 0.5,
                        0.0 if rolling else 0.075,
                    ],
                }
            )
    return segments


def main():
    project_root = pathlib.Path(__file__).resolve().parents[3]
    locolab_root = project_root / "locolab"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(locolab_root))

    import cli_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="OmniBot-Go2W-Gait-Flat-MLP-v0")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--segment-limit", type=int, default=0, help="0 records all nine segments.")
    parser.add_argument("--seed", type=int, default=42)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint must name the exact trained checkpoint.")
    if args.segment_limit < 0 or args.segment_limit > 9:
        parser.error("--segment-limit must be in [0, 9].")
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
        from PIL import Image, ImageDraw, ImageFont
        from rsl_rl.runners import OnPolicyRunner

        import locolab.source.locomotion.tasks  # noqa: F401
        from locolab.source.locomotion.mdp.wtw_math import PHASE_PRESETS, validate_commands
        from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg
        from locolab.source.locomotion.utils.video import configure_robot_follow_view

        segments = build_segments(PHASE_PRESETS, args.segment_seconds)
        if args.segment_limit:
            segments = segments[: args.segment_limit]
        validate_commands(torch.tensor([segment["command"] for segment in segments]))
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        video_path = output / "go2w_gaits_heights.mp4"
        if video_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing recording: {video_path}")

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")
        cfg.seed = args.seed
        cfg.commands.base_velocity.heading_command = False
        # Each segment starts from the same neutral state, without hiding a settling period.
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        configure_robot_follow_view(cfg.viewer)
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = (1.2, 1.8, 0.8)
        cfg.viewer.lookat = (0.0, 0.0, -0.1)
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        raw = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, cli_args.runner_cfg_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        term = env.unwrapped.command_manager.get_term("base_velocity")
        robot = term.robot.data
        dt = env.unwrapped.step_dt
        fps = round(1 / dt)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        title_font = ImageFont.truetype(font_path, 30)
        body_font = ImageFont.truetype(font_path, 22)
        fields = [
            "segment",
            "gait",
            "video_time_s",
            "segment_time_s",
            "target_height_m",
            "actual_height_m",
            "target_vx_mps",
            "actual_vx_mps",
            "actual_vy_mps",
            "actual_yaw_rate",
            "contact_FR",
            "contact_FL",
            "contact_RR",
            "contact_RL",
            "reset",
            "reset_count",
        ]
        results = []
        global_frame = 0
        with (
            imageio.get_writer(
                str(video_path),
                fps=fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            ) as video,
            (output / "telemetry.csv").open("w", newline="") as csv_file,
            torch.inference_mode(),
        ):
            telemetry = csv.DictWriter(csv_file, fieldnames=fields)
            telemetry.writeheader()
            for index, segment in enumerate(segments):
                env.reset()
                command = torch.tensor(segment["command"], device=env.unwrapped.device)
                frames = round(segment["duration_s"] / dt)
                if frames < 1:
                    raise ValueError("Segment duration must include at least one control step.")
                resets = 0
                reset_flash = 0
                rows = []
                start_s = global_frame / fps
                print(f"SHOWCASE_SEGMENT_START {index + 1}/{len(segments)} {segment}", flush=True)
                for step in range(frames):
                    if not app.is_running():
                        raise RuntimeError("Simulator closed before the complete showcase was recorded.")
                    # Reassert after any automatic reset, before computing actor observations.
                    term.set_command(command)
                    obs = env.get_observations()
                    _, _, dones, _ = env.step(policy(obs))
                    reset = bool(dones[0].item())
                    resets += int(reset)
                    reset_flash = fps if reset else max(reset_flash - 1, 0)
                    height = (robot.root_pos_w[0, 2] - env.unwrapped.scene.env_origins[0, 2]).item()
                    velocity = robot.root_lin_vel_b[0].tolist()
                    yaw_rate = robot.root_ang_vel_b[0, 2].item()
                    contact = (term.sensor.data.net_forces_w[0, term.sensor_body_ids].norm(dim=-1) > 1).int().tolist()
                    row = dict(
                        zip(
                            fields,
                            [
                                index + 1,
                                segment["gait"],
                                global_frame / fps,
                                (step + 1) * dt,
                                segment["height_m"],
                                height,
                                command[0].item(),
                                velocity[0],
                                velocity[1],
                                yaw_rate,
                                *contact,
                                int(reset),
                                resets,
                            ],
                        )
                    )
                    telemetry.writerow(row)
                    rows.append(row)
                    frame = env.unwrapped.render()
                    if frame is None:
                        raise RuntimeError("RGB rendering returned no frame.")
                    picture = Image.fromarray(frame)
                    draw = ImageDraw.Draw(picture, "RGBA")
                    draw.rectangle((0, 0, 1280, 108), fill=(10, 16, 24, 220))
                    draw.text(
                        (22, 12),
                        f"{index + 1}/{len(segments)}  {segment['gait'].upper()}  |  "
                        f"Target height: {100 * segment['height_m']:.0f} cm",
                        font=title_font,
                        fill=(255, 255, 255, 255),
                    )
                    draw.text(
                        (22, 61),
                        f"vx +0.60 m/s | yaw 0 | f 2.5 Hz | duty {segment['command'][8]:.2f} | "
                        f"swing {100 * segment['command'][9]:.1f} cm",
                        font=body_font,
                        fill=(200, 220, 240, 255),
                    )
                    draw.rectangle((0, 646, 1280, 720), fill=(10, 16, 24, 220))
                    draw.text(
                        (22, 653),
                        f"Actual height {100 * height:5.1f} cm | vx {velocity[0]:+.2f} m/s | "
                        f"contacts FR FL RR RL: {' '.join(map(str, contact))}",
                        font=body_font,
                        fill=(255, 255, 255, 255),
                    )
                    status = "  AUTO RESET" if reset_flash else ""
                    draw.text(
                        (22, 684),
                        f"{(step + 1) * dt:.1f}/{segment['duration_s']:.1f} s | resets: {resets}{status} | "
                        "independent neutral start per segment",
                        font=body_font,
                        fill=(255, 120, 100, 255) if reset_flash else (200, 220, 240, 255),
                    )
                    video.append_data(np.asarray(picture))
                    if step == min(frames - 1, 6 * fps):
                        picture.save(output / f"segment_{index + 1:02d}_preview.png")
                    global_frame += 1
                # Summaries exclude the first two seconds, but the video contains every recorded frame.
                steady = [row for row in rows if row["segment_time_s"] > 2 and not row["reset"]] or rows
                result = {
                    **segment,
                    "start_s": start_s,
                    "end_s": global_frame / fps,
                    "resets": resets,
                    "mean_actual_height_m_after_2s": sum(row["actual_height_m"] for row in steady) / len(steady),
                    "mean_actual_vx_mps_after_2s": sum(row["actual_vx_mps"] for row in steady) / len(steady),
                }
                results.append(result)
                csv_file.flush()
                print(f"SHOWCASE_SEGMENT_DONE {json.dumps(result)}", flush=True)
        summary = {
            "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
            "task": args.task,
            "height_conditioned_reward_reference": cfg.commands.base_velocity.height_conditioned_reward_reference,
            "seed": args.seed,
            "fps": fps,
            "frame_count": global_frame,
            "duration_s": global_frame / fps,
            "video": str(video_path),
            "segments": results,
            "protocol": (
                "Play config; no observation noise or domain randomization; neutral reset each segment; "
                "standard phase centers; normal terminations retained; no hidden warm-up or failed-attempt filtering."
            ),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SHOWCASE_OK {video_path}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
