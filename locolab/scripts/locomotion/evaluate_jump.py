"""Bounded jump checkpoint inference and video capture; no PPO updates."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source_root", default=str(Path(__file__).resolve().parents[3]))
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--case",
    nargs="+",
    choices=(
        "to_phase0_vx035",
        "settled_standing_vx0",
        "settled_standing_vx035",
        "rolling_vx035",
        "rolling_vx05",
        "rolling_vx10",
        "rolling_vx15",
    ),
)
parser.add_argument("--video_dir")
parser.add_argument("--heading_lock", action="store_true")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--follow_xy", action="store_true", help="Follow horizontal motion while holding camera height fixed."
)
parser.add_argument(
    "--stochastic", action="store_true", help="Sample actions after jump trigger; warmup stays deterministic."
)
parser.add_argument("--repeats", type=int, default=1)
root_args, _ = parser.parse_known_args()
source_root = Path(root_args.source_root)
sys.path.insert(0, str(source_root))
sys.path.insert(0, str(source_root / "locolab"))

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.repeats < 1 or (args.video_dir and args.repeats != 1):
    parser.error("repeats must be positive; repeated trials cannot share video filenames")
if args.video_dir:
    args.enable_cameras = True
app = AppLauncher(args).app
failure = None

try:
    import gymnasium as gym
    import torch
    from rsl_rl.modules import ActorCritic

    import locolab.source.locomotion.tasks  # noqa: F401 -- register Gym tasks
    from locolab.source.locomotion.assets.robot import GO2W_CFG
    from locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg import Go2WJumpMLPPPORunnerCfg
    from locolab.source.locomotion.tasks.Go2W.jump_env_cfg import apply_jump_training_stage
    from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg

    cfg = parse_env_cfg(
        "OmniBot-Go2W-Jump-MLP-v0",
        device=args.device,
        num_envs=1,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args.heading_lock:
        apply_jump_training_stage(cfg, "speed")
        assert cfg.commands.jump_command.heading_hold_duration == 2.2
        assert cfg.observations.policy.velocity_commands.params["lock_jump_heading"]
        assert cfg.observations.critic.velocity_commands.params["lock_jump_heading"]
    assert type(cfg).__name__ == "Go2WJumpPlayEnvCfg"
    assert cfg.scene.robot.actuators["legs"].effort_limit_sim == 23.5
    assert cfg.scene.robot.actuators["legs"].velocity_limit_sim == 30.0
    assert GO2W_CFG.actuators["legs"].effort_limit_sim[".*_calf_joint"] == 35.0
    assert GO2W_CFG.actuators["legs"].velocity_limit_sim[".*_calf_joint"] == 20.0
    assert cfg.rewards.hip_default.weight == -0.5
    assert cfg.rewards.thigh_calf_default.weight == -0.25
    assert getattr(cfg.rewards, "thigh_default", None) is getattr(cfg.rewards, "calf_default", None) is None
    assert abs(cfg.commands.jump_command.sequence_duration - 2.2) < 1e-6
    assert cfg.episode_length_s == 10.0
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    assert cfg.commands.jump_command.apex_time == 0.62
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 20.0
    cfg.scene.terrain.num_envs = 1
    cfg.scene.terrain.env_spacing = 20.0
    if args.video_dir:
        cfg.viewer.origin_type = "world" if args.follow_xy else "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (2.5, 2.5, 1.7)
        cfg.viewer.lookat = (0.3, 0.0, 0.5)
        cfg.viewer.resolution = (1280, 720)
        if args.case and "rolling_vx035" in args.case:
            cfg.viewer.eye = (3.0, 3.0, 1.8)
            cfg.viewer.lookat = (0.8, 0.0, 0.5)
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.jump_command.jump_probability = 0.0
    cfg.commands.jump_command.resampling_time_range = (1000.0, 1000.0)
    cfg.events.reset_base.params.update(
        jump_probability=0.0,
        phase_time_range=(0.0, 0.0),
        pose_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
        joint_position_range=(1.0, 1.0),
        joint_velocity_range=(0.0, 0.0),
    )
    env = gym.make("OmniBot-Go2W-Jump-MLP-v0", cfg=cfg, render_mode="rgb_array" if args.video_dir else None)
    video_writer = None
    try:
        obs, _ = env.reset()
        native = env.unwrapped
        robot = native.scene["robot"]
        sensor = native.scene.sensors["contact_forces"]
        from isaaclab.utils.math import quat_apply

        foot_names = [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]
        feet, _ = sensor.find_bodies(foot_names, preserve_order=True)
        foot_bodies = robot.find_bodies(foot_names, preserve_order=True)[0]
        wheel_ids = robot.find_joints([name + "_joint" for name in foot_names], preserve_order=True)[0]
        wheel_offsets = torch.tensor([[0.0, side * 0.048, 0.0] for side in (-1, 1, -1, 1)], device=native.device)
        wheel_axes = torch.tensor([[0.0, 1.0, 0.0]] * 4, device=native.device)
        masses = robot.root_physx_view.get_masses().to(native.device)

        def wheel_clearance():
            q = robot.data.body_link_quat_w[0, foot_bodies]
            center = robot.data.body_link_pos_w[0, foot_bodies] + quat_apply(q, wheel_offsets)
            axis_z = quat_apply(q, wheel_axes)[:, 2].clamp(-1, 1)
            extent = 0.0225 * axis_z.abs() + 0.086 * (1 - axis_z.square()).clamp_min(0).sqrt()
            return center[:, 2] - extent - native.scene.env_origins[0, 2]

        def com_height():
            return float((robot.data.body_com_pos_w[0, :, 2] * masses[0]).sum() / masses[0].sum())

        jump = native.command_manager.get_term("jump_command")
        velocity = native.command_manager.get_term("base_velocity")
        runner_cfg = Go2WJumpMLPPPORunnerCfg()
        kwargs = runner_cfg.policy.to_dict()
        kwargs.pop("class_name")
        policy = ActorCritic(obs, runner_cfg.obs_groups, 16, **kwargs).to(native.device)
        checkpoint = torch.load(args.checkpoint, map_location=native.device, weights_only=True)
        assert obs["policy"].shape == (1, 54) and obs["critic"].shape == (1, 309)
        assert cfg.actions.joint_pos.scale == 0.25 and cfg.actions.joint_vel.scale == 10.0
        leg_ids, leg_joint_names = robot.find_joints("^(?!.*_foot_joint).*")
        assert len(leg_ids) == 12
        assert torch.allclose(
            robot.data.joint_effort_limits[:, leg_ids], torch.full((1, 12), 23.5, device=native.device)
        )
        assert torch.allclose(robot.data.joint_vel_limits[:, leg_ids], torch.full((1, 12), 30.0, device=native.device))
        print(
            "SMOOTH_TO_POLICY_NATIVE_CONTRACT_OK actor=54 critic=309 actions=16 legacy_limits_task_local=True",
            flush=True,
        )
        policy.load_state_dict(checkpoint["model_state_dict"])
        print(
            "EVAL_CHECKPOINT " + json.dumps({"path": args.checkpoint, "iteration": checkpoint.get("iter")}), flush=True
        )
        policy.eval()
        dt = native.step_dt

        def contacts():
            return torch.linalg.vector_norm(sensor.data.net_forces_w[0, feet], dim=-1) > 1.0

        def yaw():
            w, x, y, z = robot.data.root_quat_w[0].tolist()
            return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        def capture_frame():
            if args.follow_xy:
                x, y = robot.data.root_pos_w[0, :2].tolist()
                native.viewport_camera_controller.update_view_location(
                    eye=(x + 2.6, y + 2.6, 1.6), lookat=(x + 0.25, y, 0.5)
                )
            return env.render()

        def advance(vx, stochastic=False):
            velocity.command[:, :3] = torch.tensor([vx, 0.0, 0.0], device=native.device)
            observation = native.observation_manager.compute()
            with torch.inference_mode():
                action = policy.act(observation) if stochastic else policy.act_inference(observation)
            _, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(action).all() and torch.isfinite(reward).all()
            if video_writer is not None:
                video_writer.append_data(capture_frame())
            return bool(terminated.item() or truncated.item())

        results = []
        cases = [
            ("to_phase0_vx035", True, 0.35),
            ("settled_standing_vx0", False, 0.0),
            ("settled_standing_vx035", False, 0.35),
        ]
        if args.case and "rolling_vx035" in args.case:
            cases.append(("rolling_vx035", False, 0.35))
        for name, speed in (("rolling_vx05", 0.5), ("rolling_vx10", 1.0), ("rolling_vx15", 1.5)):
            if args.case and name in args.case:
                cases.append((name, False, speed))
        if args.case:
            cases = [case for case in cases if case[0] in args.case]
        for trial_index, (name, to_reset, target_vx) in enumerate(cases * args.repeats):
            reset_term = native.event_manager.get_term_cfg("reset_base")
            reset_term.params["jump_probability"] = float(to_reset)
            env.reset()
            if args.video_dir:
                import imageio.v2 as imageio

                video_dir = Path(args.video_dir)
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"{name}.mp4"
                if video_path.exists():
                    raise FileExistsError(video_path)
                video_writer = imageio.get_writer(str(video_path), fps=round(1.0 / dt), codec="libx264", quality=8)
                video_writer.append_data(capture_frame())
            warmup_failed = False
            if not to_reset:
                moving_start = name.startswith("rolling_")
                warmup_vx = target_vx if moving_start else 0.0
                settled_steps = 0
                for warmup_step in range(round((4.0 if moving_start else 3.0) / dt)):
                    if advance(warmup_vx):
                        warmup_failed = True
                        break
                    stable = (
                        bool(contacts().all())
                        and abs(float(robot.data.root_lin_vel_w[0, 2])) <= 0.1
                        and float(robot.data.projected_gravity_b[0, 2]) < -0.87758
                    )
                    if moving_start:
                        stable = bool(jump._can_trigger(torch.tensor([0], device=native.device)).item())
                    settled_steps = settled_steps + 1 if stable else 0
                    ready = stable if moving_start else settled_steps * dt >= 0.2 - 1e-6
                    if warmup_step * dt >= (2.0 if moving_start else 0.5) and ready:
                        break
                else:
                    warmup_failed = True
            if warmup_failed:
                result = {"case": name, "warmup_failed": True}
                print("GROUND_START_RESULT " + json.dumps(result), flush=True)
                results.append(result)
                if video_writer is not None:
                    video_writer.close()
                    video_writer = None
                continue
            start_z = float(robot.data.root_pos_w[0, 2])
            start_com = com_height()
            peak_com = start_com
            peak_min_wheel_clearance = 0.0
            start_vz = float(robot.data.root_lin_vel_w[0, 2])
            start_vx = float(robot.data.root_lin_vel_b[0, 0])
            start_xy = robot.data.root_pos_w[0, :2].clone()
            start_yaw = yaw()
            start_contacts = contacts().tolist()
            if not to_reset:
                jump.set_trigger(1)
            assert jump.active.item() and abs(jump.reference_time.item()) < 1e-6
            max_z, max_vz = start_z, start_vz
            ground_steps = 3 if all(start_contacts) else 0
            ground_seen = all(start_contacts)
            air_steps = 0
            max_air_steps = 0
            landed_steps = 0
            landed_stably = False
            terminated_early = False
            first_takeoff_s = first_touchdown_s = first_stable_landing_s = None
            apex_s = 0.0
            trajectory = []
            for post_step in range(round(3.0 / dt)):
                if advance(target_vx, stochastic=args.stochastic):
                    terminated_early = True
                    break
                z = float(robot.data.root_pos_w[0, 2])
                peak_com = max(peak_com, com_height())
                vz = float(robot.data.root_lin_vel_w[0, 2])
                post_time = (post_step + 1) * dt
                if z > max_z:
                    max_z = z
                    apex_s = post_time
                max_vz = max(max_vz, vz)
                contact = contacts()
                clearances = wheel_clearance()
                if not bool(contact.any()):
                    peak_min_wheel_clearance = max(peak_min_wheel_clearance, float(clearances.min()))
                displacement = robot.data.root_pos_w[0, :2] - start_xy
                dx, dy = displacement.tolist()
                yaw_difference = yaw() - start_yaw
                yaw_error = math.atan2(math.sin(yaw_difference), math.cos(yaw_difference))
                trajectory.append(
                    {
                        "time": post_time,
                        "base_z": z,
                        "base_vz": vz,
                        "base_vx_b": float(robot.data.root_lin_vel_b[0, 0]),
                        "root_quat_wxyz": robot.data.root_quat_w[0].tolist(),
                        "joint_position": robot.data.joint_pos[0, leg_ids].tolist(),
                        "joint_velocity": robot.data.joint_vel[0, leg_ids].tolist(),
                        "wheel_velocity": robot.data.joint_vel[0, wheel_ids].tolist(),
                        "wheel_clearance": clearances.tolist(),
                        "reference_time": float(jump.reference_time[0]),
                        "forward_displacement": math.cos(start_yaw) * dx + math.sin(start_yaw) * dy,
                        "lateral_displacement": -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy,
                        "yaw_error_rad": yaw_error,
                        "contact": contact.tolist(),
                        "jump_active": bool(jump.active.item()),
                    }
                )
                ground_steps = ground_steps + 1 if bool(contact.all()) else 0
                ground_seen |= ground_steps >= 3
                air_steps = air_steps + 1 if ground_seen and not bool(contact.any()) else 0
                max_air_steps = max(max_air_steps, air_steps)
                if first_takeoff_s is None and air_steps >= 3:
                    first_takeoff_s = post_time - 2 * dt
                if first_touchdown_s is None and first_takeoff_s is not None and bool(contact.any()):
                    first_touchdown_s = post_time
                upright = float(robot.data.projected_gravity_b[0, 2]) < -0.87758
                stable = max_air_steps >= 3 and bool(contact.all()) and abs(vz) <= 0.2 and upright
                landed_steps = landed_steps + 1 if stable else 0
                landed_stably |= landed_steps * dt >= 0.1 - 1e-6
                if first_stable_landing_s is None and landed_stably:
                    first_stable_landing_s = post_time
            segments = []
            segment = []
            for point in trajectory:
                if not any(point["contact"]):
                    segment.append(point)
                elif segment:
                    segments.append(segment)
                    segment = []
            if segment:
                segments.append(segment)
            main_flight = max(segments, key=len, default=[])
            held_interval = [
                point for point in trajectory if point["time"] <= cfg.commands.jump_command.heading_hold_duration + 1e-6
            ]
            result = {
                "case": name,
                "trial": trial_index,
                "stochastic": args.stochastic,
                "initial_z": start_z,
                "initial_vz": start_vz,
                "initial_vx_b": start_vx,
                "vx_command": target_vx,
                "jump_duration_s": cfg.commands.jump_command.sequence_duration,
                "xy_displacement_m": (robot.data.root_pos_w[0, :2] - start_xy).tolist(),
                "initial_contacts": start_contacts,
                "peak_rise_m": max_z - start_z,
                "whole_com_rise_m": peak_com - start_com,
                "peak_min_all_four_wheel_clearance_m": peak_min_wheel_clearance,
                "max_upward_vz_mps": max_vz,
                "ground_observed": ground_seen,
                "max_airborne_after_ground_s": max_air_steps * dt,
                "landed_stably": landed_stably,
                "terminated_early": terminated_early,
                "first_takeoff_s": first_takeoff_s,
                "apex_s": apex_s,
                "first_touchdown_s": first_touchdown_s,
                "first_stable_landing_s": first_stable_landing_s,
                "original_jump_success_metric": float(jump.metrics["jump_success"][0]),
                "takeoff_at_least_18cm": ground_seen and max_air_steps >= 3 and max_z - start_z >= 0.18,
                "heading_lock": args.heading_lock,
                "trigger_yaw_deg": math.degrees(start_yaw),
                "peak_yaw_error_main_flight_deg": max(
                    (abs(math.degrees(point["yaw_error_rad"])) for point in main_flight), default=None
                ),
                "peak_yaw_error_hold_deg": max(
                    (abs(math.degrees(point["yaw_error_rad"])) for point in held_interval), default=0.0
                ),
                "end_yaw_error_3s_deg": math.degrees(trajectory[-1]["yaw_error_rad"]) if trajectory else None,
                "end_lateral_displacement_3s_m": trajectory[-1]["lateral_displacement"] if trajectory else None,
                "max_lateral_displacement_hold_m": max(
                    (abs(point["lateral_displacement"]) for point in held_interval), default=0.0
                ),
                "main_flight_forward_distance_m": (
                    (main_flight[-1]["forward_displacement"] - main_flight[0]["forward_displacement"])
                    if main_flight
                    else 0.0
                ),
                "main_flight_lateral_distance_m": (
                    (main_flight[-1]["lateral_displacement"] - main_flight[0]["lateral_displacement"])
                    if main_flight
                    else 0.0
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / f"{name}_trial{trial_index}.states.json").write_text(json.dumps(trajectory) + "\n")
            print("GROUND_START_RESULT " + json.dumps(result), flush=True)
            results.append(result)
            if video_writer is not None:
                video_writer.close()
                video_writer = None
                print("GROUND_START_VIDEO " + str(video_path), flush=True)
        print("GROUND_START_EVAL_COMPLETE " + json.dumps(results), flush=True)
        print("JUMP_CHECKPOINT_EVALUATION_COMPLETE updates=0", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    "full_schedule_s": cfg.commands.jump_command.sequence_duration,
                    "self_collisions": True,
                    "normal_ground_reset": all(not case[1] for case in cases),
                    "leg_joint_names": leg_joint_names,
                    "wheel_joint_names": [name + "_joint" for name in foot_names],
                    "iteration": checkpoint.get("iter"),
                    "heading_lock": args.heading_lock,
                    "deterministic": not args.stochastic,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()
except BaseException as error:
    import traceback

    traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".error.json").write_text(json.dumps({"error": str(error)}) + "\n")
    failure = error
finally:
    try:
        app.close()
    finally:
        if failure is not None:
            raise SystemExit(1) from failure
