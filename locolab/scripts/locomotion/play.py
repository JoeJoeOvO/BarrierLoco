# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import pathlib
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Add the project root to sys.path to allow importing locolab
# Structure: project_root/locolab/scripts/locomotion/train.py
# We need to add project_root to sys.path
project_root = pathlib.Path(__file__).parents[3]
locolab_root = pathlib.Path(__file__).parents[2]
logs_root = locolab_root / "logs"
sys.path.append(str(project_root))
# Prefer local rsl_rl package in this repo over any globally installed one.
sys.path.insert(0, str(locolab_root))

from locolab.source.locomotion.utils.video import (  # noqa: E402
    DEFAULT_VIDEO_DURATION_S,
    TerrainVideoView,
    configure_robot_follow_view,
    configure_terrain_view,
    make_video_recording_plan,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_terrain", type=str, default=None, help="Fix video camera on this curriculum terrain type.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Encoded video frame target. Overrides --video_duration_s when provided.",
)
parser.add_argument(
    "--video_duration_s",
    type=float,
    default=DEFAULT_VIDEO_DURATION_S,
    help="Duration of the recorded video in seconds.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--highplatform-pretrain", action="store_true", help="Replay Rough with its HighPlatform pretraining leg limits."
)
parser.add_argument(
    "--offline-visuals", action="store_true", help="Use local terrain material and texture-free lighting."
)
parser.add_argument("--jump_stage", choices=("skill", "speed"), help="Use the matching Go2W Jump stage semantics.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--follow-camera", action="store_true", help="Track robot zero in the interactive viewport.")
parser.add_argument("--velocity-sweep", action="store_true", help="Cycle Go2W gait / omnidirectional velocity cases.")
parser.add_argument("--parkour-generalization", action="store_true", help="Probe continuous H/XY goals on flat ground.")
parser.add_argument("--parkour-sweep", action="store_true", help="Inspect all GoalGrid motions at four trained speeds.")
parser.add_argument("--parkour-sweep-seconds", type=float, default=12.0, help="Seconds per GoalGrid inspection case.")
parser.add_argument("--live-plot", action="store_true", help="Dock live velocity curves and values inside Isaac Sim.")
parser.add_argument("--sweep-seconds", type=float, default=6.0, help="Simulation seconds per velocity case.")
parser.add_argument(
    "--sweep-max-vx",
    type=float,
    default=5.0,
    help="Filter sweep cases by absolute vx; use 1 for the original-range control model.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.highplatform_pretrain and args_cli.task != "OmniBot-Go2W-Rough-MLP-v0":
    parser.error("--highplatform-pretrain requires OmniBot-Go2W-Rough-MLP-v0")
if args_cli.velocity_sweep and args_cli.task != "OmniBot-Go2W-Gait-Flat-MLP-v0":
    parser.error("--velocity-sweep requires the Go2W Gait task")
if args_cli.sweep_seconds <= 0 or not math.isfinite(args_cli.sweep_seconds):
    parser.error("--sweep-seconds must be positive and finite")
if not math.isfinite(args_cli.sweep_max_vx) or args_cli.sweep_max_vx <= 0:
    parser.error("--sweep-max-vx must be positive and finite")
if (
    args_cli.parkour_sweep or args_cli.parkour_generalization
) and args_cli.task != "OmniBot-Go2W-Parkour-GoalGrid-MLP-v0":
    parser.error("--parkour-sweep requires GoalGrid")
if args_cli.parkour_sweep and args_cli.parkour_generalization:
    parser.error("Choose reference sweep or generalization, not both")
if not math.isfinite(args_cli.parkour_sweep_seconds) or args_cli.parkour_sweep_seconds < 10:
    parser.error("--parkour-sweep-seconds must be finite and at least 10 s (one episode)")
if args_cli.live_plot and args_cli.headless:
    parser.error("--live-plot requires a desktop window")
if args_cli.jump_stage and args_cli.task != "OmniBot-Go2W-Jump-MLP-v0":
    parser.error("jump_stage requires OmniBot-Go2W-Jump-MLP-v0")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import locolab.source.locomotion.tasks  # noqa: F401
from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg

try:
    # Recent Isaac Lab versions keep RL checkpoint helpers in isaaclab_rl.
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError as exc:
    if exc.name not in {"isaaclab_rl.utils", "isaaclab_rl.utils.pretrained_checkpoint"}:
        raise
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.highplatform_pretrain:
        # Match train.py --highplatform-pretrain, not the shared asset's newer calf limits.
        env_cfg.scene.robot.actuators["legs"].effort_limit_sim = 23.5
        env_cfg.scene.robot.actuators["legs"].velocity_limit_sim = 30.0
    if args_cli.offline_visuals:
        import isaaclab.sim as sim_utils

        # Rendering only: preserve all terrain collision and friction settings.
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.62, 0.66, 0.70), roughness=0.85
        )
        env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=300.0)
    if args_cli.velocity_sweep:
        # A single robot makes curves unambiguous. Manual wz must not be overwritten
        # by the random heading controller used during normal training.
        env_cfg.scene.num_envs = 1
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.lin_vel_x = (-args_cli.sweep_max_vx, args_cli.sweep_max_vx)
    if args_cli.parkour_sweep:
        # Retain each reference's physical tile; the camera changes robots instead
        # of assigning a gap/ramp reference to an incompatible flat environment.
        env_cfg.scene.num_envs = 100
        env_cfg.events.reset_base.params["resample_flat_motion"] = False
    if args_cli.parkour_generalization:
        # Off-grid actor tests have no matching teacher trajectory. Keep physical
        # failure checks, but do not grade/reset against the old reference goal.
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 10.0
        env_cfg.events.reset_base.params["resample_flat_motion"] = False
        env_cfg.terminations.task_succeeded = None
        env_cfg.terminations.task_failed = None
        env_cfg.rewards.termination_penalty = None
        env_cfg.rewards.task_success_bonus = None
    if args_cli.follow_camera:
        configure_robot_follow_view(env_cfg.viewer)
    if args_cli.jump_stage:
        from locolab.source.locomotion.tasks.Go2W.jump_env_cfg import apply_jump_training_stage

        play_speed = env_cfg.commands.base_velocity.ranges.lin_vel_x
        apply_jump_training_stage(env_cfg, args_cli.jump_stage)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = play_speed
        env_cfg.events.reset_base.params["jump_probability"] = 0.0
    if args_cli.video:
        if args_cli.video_terrain:
            index = configure_terrain_view(env_cfg, args_cli.video_terrain)
            print(f"[INFO] Video terrain: {args_cli.video_terrain}, env_index={index}, fixed tile view")
        else:
            configure_robot_follow_view(env_cfg.viewer)

    # specify directory for logging experiments
    log_root_path = str(logs_root / "rsl_rl" / cli_args.get_output_experiment_name(agent_cfg))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video and args_cli.video_terrain:
        env = TerrainVideoView(env)
    video_plan = None
    if args_cli.video:
        video_plan = make_video_recording_plan(
            step_dt=env.unwrapped.step_dt,
            duration_s=args_cli.video_duration_s,
            frame_count=args_cli.video_length,
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_plan.frame_count,
            "fps": video_plan.fps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, cli_args.runner_cfg_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, cli_args.runner_cfg_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    telemetry = None
    velocity_panel = None
    if args_cli.live_plot:
        import csv
        from datetime import datetime

        from locolab.source.locomotion.utils.velocity_panel import VelocityPanel

        telemetry_path = pathlib.Path(log_dir) / f"velocity_live_{datetime.now():%Y%m%d_%H%M%S}.csv"
        telemetry = telemetry_path.open("w", buffering=1)
        writer = csv.writer(telemetry)
        writer.writerow(["time", "case", "cmd0", "cmd1", "cmd2", "actual0", "actual1", "actual2", "resets"])
        telemetry.flush()
        quantities = (
            (("rise H", "m"), ("landing X", "m"), ("landing Y", "m")) if args_cli.parkour_generalization else None
        )
        velocity_panel = VelocityPanel(quantities=quantities)
        print(f"[INFO] Live velocity telemetry: {telemetry_path}", flush=True)
    if args_cli.velocity_sweep:
        from locolab.source.locomotion.mdp.wtw_math import PHASE_PRESETS
        from locolab.source.locomotion.utils.gait_sweep import gait_command, velocity_cases

        # Keep narrow-range checkpoints inside their trained vx range during inspection.
        cases = [case for case in velocity_cases() if abs(case[2][0]) <= args_cli.sweep_max_vx]
    command_name = "parkour" if "Parkour-" in args_cli.task else "base_velocity"
    inspect_index = 0
    if args_cli.parkour_sweep:
        from locolab.source.locomotion.utils.parkour_sweep import inspection_cases

        native = env.unwrapped
        parkour_term = native.command_manager.get_term("parkour")
        sweep_cases = inspection_cases(parkour_term.motion_ids.tolist(), parkour_term.locomotion_only.tolist())
        if {case[0] for case in sweep_cases} != set(range(len(parkour_term.bank.names))):
            raise ValueError("Inspection layout does not cover the complete reference bank")
    inspector = None
    if args_cli.parkour_generalization:
        from locolab.source.locomotion.utils.parkour_generalization import GoalGridInspector

        inspector = GoalGridInspector(env.unwrapped)
    previous_case = -1
    reset_count = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            label = "random commands"
            if args_cli.task == "OmniBot-Go2W-HighPlatform-MLP-v0":
                label = f"HighPlatform | {pathlib.Path(resume_path).name} | 30-80 cm"
            if command_name == "parkour":
                label = f"Parkour | {pathlib.Path(resume_path).name}"
            case_changed = False
            if args_cli.velocity_sweep:
                case_number = int(timestep * dt / args_cli.sweep_seconds)
                gait, name, velocity = cases[case_number % len(cases)]
                label = f"{case_number % len(cases) + 1}/{len(cases)} {gait}: {name}"
                if case_number != previous_case:
                    env.reset()
                    case_changed = True
                    previous_case = case_number
                    print(f"[SWEEP] {label}", flush=True)
                # Reset can resample commands; reapply before every policy inference.
                env.unwrapped.command_manager.get_term("base_velocity").set_command(
                    gait_command(gait, velocity, PHASE_PRESETS)
                )
                obs = env.get_observations()
            if args_cli.parkour_sweep:
                case_number = int(timestep * dt / args_cli.parkour_sweep_seconds)
                motion_id, speed, inspect_index = sweep_cases[case_number % len(sweep_cases)]
                if case_number != previous_case:
                    env.reset()
                    controller = native.viewport_camera_controller
                    if controller is not None:
                        controller.set_view_env_index(inspect_index)
                    case_changed = True
                    previous_case = case_number
                    print(
                        f"[PARKOUR_SWEEP] case={case_number % len(sweep_cases) + 1}/{len(sweep_cases)} "
                        f"motion={parkour_term.bank.names[motion_id]} speed={speed} env={inspect_index}",
                        flush=True,
                    )
                # Auto-reset resamples speeds: reapply before every inference and
                # recompute observations so the policy receives the displayed cmd.
                parkour_term.forward_speed[:] = speed
                obs = env.get_observations()
                label = (
                    f"{case_number % len(sweep_cases) + 1}/{len(sweep_cases)} "
                    f"{parkour_term.bank.names[motion_id]} | {speed:.1f} m/s"
                )
                if velocity_panel is not None:
                    full_cmd = parkour_term.command[inspect_index]
                    robot = native.scene["robot"]
                    local = robot.data.root_pos_w[inspect_index] - native.scene.env_origins[inspect_index]
                    rise = local[2] - parkour_term.bank.states["position"][motion_id, 0, 2]
                    stage = (
                        "exit"
                        if parkour_term.rolling_exit[inspect_index]
                        else "jump" if parkour_term.active[inspect_index] else "approach"
                    )
                    error = parkour_term.touchdown_error[inspect_index].item()
                    landing_text = f"{error:.3f} m" if math.isfinite(error) else "pending"
                    velocity_panel.details.text = (
                        f"Stage: {stage} | jump={full_cmd[3].item():.0f}\n"
                        f"Target H / X / Y: {full_cmd[4].item():.3f} / {full_cmd[5].item():.3f} / "
                        f"{full_cmd[6].item():.3f} m\n"
                        f"Actual rise / X / Y: {rise.item():.3f} / {local[0].item():.3f} / {local[1].item():.3f} m\n"
                        f"First landing XY error: {landing_text}"
                    )
            if inspector is not None:
                inspector.term.forward_speed[:] = inspector.speed
                obs = inspector.actor_observation(env.get_observations())
                plot_cmd, plot_actual, details = inspector.measure()
                label = f"Attempt {inspector.index + 1} | {inspector.group} | off-grid goals"
                if velocity_panel is not None:
                    velocity_panel.details.text = details
            cmd = env.unwrapped.command_manager.get_command(command_name)[inspect_index, :3].clone()
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if telemetry is not None:
                reset = bool(dones[inspect_index].item())
                reset_count += int(reset)
                robot = env.unwrapped.scene["robot"]
                actual = torch.cat(
                    (robot.data.root_lin_vel_b[inspect_index, :2], robot.data.root_ang_vel_b[inspect_index, 2:3])
                )
                # Log the command used for this action, with post-step body-frame velocity.
                # A terminal step has already auto-reset in the vector wrapper, so omit
                # that measurement from CSV instead of presenting it as tracking success.
                if inspector is not None:
                    # Measurements are pre-step, before vector-env auto-reset can
                    # erase the landing. Targets are peak/endpoint, not trajectories.
                    cmd = cmd.new_tensor(plot_cmd)
                    actual = actual.new_tensor(plot_actual)
                values = [float("nan")] * 3 if reset or case_changed else actual.tolist()
                writer.writerow([timestep * dt, label, *cmd.tolist(), *values, reset_count])
                velocity_panel.update(
                    timestep * dt,
                    label,
                    cmd.tolist(),
                    actual.tolist(),
                    reset_count,
                    clear=reset or case_changed,
                )
            if inspector is not None and bool(dones[0]):
                inspector.finish()
                inspector.next_attempt()
        timestep += 1
        if video_plan is not None:
            # Exit the play loop after recording one video
            if timestep >= video_plan.simulation_steps:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if telemetry is not None:
        telemetry.close()
    if velocity_panel is not None:
        velocity_panel.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
