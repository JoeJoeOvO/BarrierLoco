"""Live, labelled command sweep for the pretrained Go2W gait policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from swing_height import SwingHeightTracker, format_heights_cm


def build_cases(extended=False):
    presets = {"rolling": (0.0, 0.0, 0.0), "trot": (0.5, 0.0, 0.0), "bound": (0.0, 0.0, 0.5)}
    cases = []

    def add(gait, group, **changes):
        values = dict(vx=0.6, vy=0.0, yaw=0.0, height=0.34, frequency=2.5, swing=0.075)
        values.update(changes)
        if gait == "rolling":
            values["swing"] = 0.0
        command = [
            values["vx"],
            values["vy"],
            values["yaw"],
            values["height"],
            values["frequency"],
            *presets[gait],
            1.0 if gait == "rolling" else 0.5,
            values["swing"],
        ]
        cases.append(dict(gait=gait, group=group, command=command))

    if extended:
        for gait in presets:
            for vx in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, 0.0):
                add(gait, "Extended speed", vx=vx)
    for gait in presets:
        for height in (0.34, 0.22, 0.46):
            add(gait, "Body height", height=height)
    for gait in ("trot", "bound"):
        for frequency in (1.5, 2.5, 4.0):
            add(gait, "Frequency", frequency=frequency)
    for gait in ("trot", "bound"):
        for swing in (0.0, 0.05, 0.10, 0.12, 0.15) if extended else (0.0, 0.05, 0.10):
            add(gait, "Swing height", swing=swing)
    for gait in presets:
        for vx in () if extended else (-0.5, 0.0, 1.0):
            add(gait, "Forward / reverse", vx=vx)
        for vy in (-0.3, 0.3):
            add(gait, "Sideways", vx=0.0, vy=vy)
        for yaw in (-1.0, 1.0):
            add(gait, "Turning", vx=0.3, yaw=yaw)
    return cases


def main():
    root = Path(__file__).resolve().parents[2]
    repo = root
    sys.path[:0] = [str(repo / "locolab/scripts/locomotion"), str(repo / "locolab"), str(repo)]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=0, help="0 repeats until the window is closed")
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument(
        "--extended", action="store_true", help="Sweep the curriculum range: vx +/-5 m/s, swing up to .15 m"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/cmd_sweep")
    parser.add_argument(
        "--reset-between-cases", action="store_true", help="Use independent starts instead of continuous transitions"
    )
    parser.add_argument("--checkpoint", default=str(repo / "deploy/policy/go2w/multiskill_scratch80k/model_79999.pt"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    cases = build_cases(args.extended)
    if args.case_limit:
        cases = cases[: args.case_limit]
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.start_case <= len(cases):
        parser.error("seconds must be positive and start-case must refer to an existing case")
    if args.case_limit < 0 or args.cycles < 0:
        parser.error("case-limit and cycles must be nonnegative")
    app = AppLauncher(args).app
    env = None
    panel = None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "session.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "extended": args.extended,
                "seconds": args.seconds,
                "continuous": not args.reset_between_cases,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        import cli_args
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import locolab.source.locomotion.tasks  # noqa: F401
        from locolab.source.locomotion.mdp.wtw_math import validate_commands
        from locolab.source.locomotion.mdp.wtw_rewards import _wheel_state
        from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg
        from locolab.source.locomotion.utils.video import configure_robot_follow_view

        validate_commands(torch.tensor([case["command"] for case in cases]))
        task = "OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0" if args.extended else "OmniBot-Go2W-Gait-Flat-MLP-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")
        cfg.seed = 42
        cfg.commands.base_velocity.heading_command = False
        if not args.reset_between_cases:
            # A training episode timeout must not interrupt a continuous demo.
            cfg.terminations.time_out = None
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        configure_robot_follow_view(cfg.viewer)
        cfg.viewer.eye = (1.8, 2.0, 1.0)
        cfg.viewer.lookat = (0.0, 0.0, 0.0)
        from isaaclab_tasks.utils import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, cli_args.runner_cfg_to_dict(agent_cfg), log_dir=None, device=args.device)
        runner.load(args.checkpoint, load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        term = env.unwrapped.command_manager.get_term("base_velocity")
        swing_tracker = SwingHeightTracker()
        state = dict(index=args.start_case - 1, change=True, hold=False, quit=False)
        labels = {}

        def advance(offset):
            state["index"] = (state["index"] + offset) % len(cases)
            state["change"] = True

        def toggle_hold():
            state["hold"] = not state["hold"]

        if not args.headless:
            import omni.ui as ui

            window = ui.Window("Go2W | LIVE COMMAND | " + Path(args.checkpoint).name, width=500, height=850)
            with (
                window.frame,
                ui.ScrollingFrame(),
                ui.VStack(spacing=5, height=0, style={"font_size": 15}),
            ):
                for key in (
                    "title",
                    "progress",
                    "speed",
                    "height",
                    "phase",
                    "raw",
                    "actual",
                    "wheel_height",
                    "swing_peak",
                    "resets",
                ):
                    labels[key] = ui.Label("Loading...", word_wrap=True, height=32)
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Previous", clicked_fn=lambda: advance(-1))
                    ui.Button("Next", clicked_fn=lambda: advance(1))
                    ui.Button("Hold / Auto", clicked_fn=toggle_hold)
                    ui.Button("Stop", clicked_fn=lambda: state.update(quit=True))
                mode_text = (
                    "Independent starts."
                    if args.reset_between_cases
                    else "CONTINUOUS: switch CMD only; no timeout reset."
                )
                ui.Label(
                    mode_text + "\nWheel order: FR / FL / RR / RL. Peaks: last complete commanded swing; -- = pending.",
                    word_wrap=True,
                )

                # Keep the V1 command/height inspector and the user's docked velocity curves together.
                chart_frame = ui.Frame(height=420)
                from locolab.source.locomotion.utils.velocity_panel import VelocityPanel

                panel = VelocityPanel(window=window, frame=chart_frame)
        dt = env.unwrapped.step_dt
        elapsed = 0.0
        resets = 0
        total_resets = 0
        cycles = 0
        updates = 0
        status_path = output / "current_cmd.json"
        env.reset()
        with (output / "events.jsonl").open("a") as events, torch.inference_mode():
            while app.is_running() and not state["quit"]:
                started = time.monotonic()
                if state["change"]:
                    swing_tracker.reset()
                    if args.reset_between_cases and updates:
                        env.reset()
                    elapsed = 0.0
                    resets = 0
                    state["change"] = False
                    case = cases[state["index"]]
                    command = torch.tensor(case["command"], device=env.unwrapped.device)
                    event = dict(
                        event="case_start",
                        case=state["index"] + 1,
                        total=len(cases),
                        **case,
                        continuous=not args.reset_between_cases,
                        episode_steps=int(env.unwrapped.episode_length_buf[0]),
                        position=term.robot.data.root_pos_w[0].tolist(),
                        phase=float(term.phase[0]),
                        total_resets=total_resets,
                    )
                    events.write(json.dumps(event) + "\n")
                    events.flush()
                    print("CMD_SWEEP " + json.dumps(event), flush=True)
                # Manual commands must be reapplied after automatic resets, before observations.
                term.set_command(command)
                obs = env.get_observations()
                _, _, dones, _ = env.step(policy(obs))
                did_reset = bool(dones[0].item())
                resets += int(did_reset)
                total_resets += int(did_reset)
                bottom, _ = _wheel_state(term)
                wheel_heights = (bottom[0, :, 2] - env.unwrapped.scene.env_origins[0, 2]).tolist()
                if did_reset:
                    swing_tracker.reset()
                    swing_peaks = list(swing_tracker.last_peak)
                else:
                    swinging = ((term.leg_phase[0] >= 0.5) & (term.command[0, 8] < 1)).tolist()
                    swing_peaks = swing_tracker.update(wheel_heights, swinging)
                if not state["hold"]:
                    elapsed += dt
                updates += 1
                if panel is not None:
                    data = term.robot.data
                    actual = [
                        float(data.root_lin_vel_b[0, 0]),
                        float(data.root_lin_vel_b[0, 1]),
                        float(data.root_ang_vel_b[0, 2]),
                    ]
                    # Reset measurements are post-teleport: clear the curves instead of drawing them.
                    panel.update(
                        updates * dt,
                        f"{state['index'] + 1}/{len(cases)} {case['gait']} | {case['group']}",
                        case["command"][:3],
                        [float("nan")] * 3 if did_reset else actual,
                        total_resets,
                        clear=did_reset or (args.reset_between_cases and elapsed <= dt),
                    )
                if updates % 5 == 0 or elapsed <= dt or did_reset:
                    values = term.command[0].tolist() if not did_reset else command.tolist()
                    data = term.robot.data
                    actual_height = float(data.root_pos_w[0, 2] - env.unwrapped.scene.env_origins[0, 2])
                    velocity = data.root_lin_vel_b[0].tolist()
                    actual_yaw = float(data.root_ang_vel_b[0, 2])
                    status = dict(
                        case=state["index"] + 1,
                        total=len(cases),
                        **case,
                        elapsed_s=elapsed,
                        hold=state["hold"],
                        resets=resets,
                        continuous=not args.reset_between_cases,
                        total_resets=total_resets,
                        episode_steps=int(env.unwrapped.episode_length_buf[0]),
                        actual_height_m=actual_height,
                        actual_velocity_mps=velocity,
                        wheel_order=["FR", "FL", "RR", "RL"],
                        actual_wheel_bottom_height_m=wheel_heights,
                        last_complete_swing_peak_m=swing_peaks,
                        actual_yaw_rad_s=actual_yaw,
                        wall_time=time.time(),
                    )
                    temporary = status_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(status, indent=2) + "\n")
                    temporary.replace(status_path)
                    texts = dict(
                        title=f"{state['index'] + 1}/{len(cases)}   {case['gait'].upper()}  |  {case['group']}",
                        progress=(
                            f"{'HOLD' if state['hold'] else 'AUTO'}  {elapsed:.1f} / {args.seconds:.1f} s"
                            f"  |  cycle {cycles + 1}"
                        ),
                        speed=f"CMD vx {values[0]:+.2f}   vy {values[1]:+.2f} m/s   yaw {values[2]:+.2f} rad/s",
                        height=(
                            f"Height {100*values[3]:.0f} cm   Freq {values[4]:.1f} Hz"
                            f"   Swing {100*values[9]:.1f} cm"
                        ),
                        phase=f"Phase [{values[5]:.2f}, {values[6]:.2f}, {values[7]:.2f}]   Duty {values[8]:.2f}",
                        raw="cmd = [" + ", ".join(f"{value:g}" for value in case["command"]) + "]",
                        actual=(
                            f"ACTUAL height {100*actual_height:.1f} cm   vx {velocity[0]:+.2f}"
                            f"   vy {velocity[1]:+.2f}   yaw {actual_yaw:+.2f}"
                        ),
                        wheel_height="ACTUAL wheel height cm: " + format_heights_cm(wheel_heights),
                        swing_peak="ACTUAL last swing peak cm: " + format_heights_cm(swing_peaks),
                        resets=f"Failure resets: case {resets} / total {total_resets}"
                        + ("  RESET" if did_reset else ""),
                    )
                    for key, label in labels.items():
                        label.text = texts[key]
                if elapsed >= args.seconds and not state["hold"] and not state["change"]:
                    if state["index"] == len(cases) - 1:
                        cycles += 1
                        if args.cycles and cycles >= args.cycles:
                            break
                    advance(1)
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
        print("CMD_SWEEP_STOPPED", flush=True)
    finally:
        if panel is not None:
            panel.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
