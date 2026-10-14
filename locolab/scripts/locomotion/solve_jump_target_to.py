"""Batched direct-shooting Go2W jump TO with real contact dynamics.

Optimize an open-loop PD target spline using CEM on one Isaac Lab GPU. No
policy, root wrench, airborne reset, kinematic playback, or PPO is used.
Use --replay in a fresh process for independent single-environment validation.
The production jump configuration/reference is never modified by this tool.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from forward_leap_to import ForwardLeapTarget, forward_leap_metrics, load_forward_target
from gap_leap_to import (
    GapLeapTarget,
    gap_leap_metrics,
    gap_mesh,
    gap_proposals,
    geometry_fingerprint,
    validate_gap_geometry,
)
from jump_target_to import (
    KNOT_TIMES,
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    JumpTarget,
    control_tape,
    export_reference,
    initial_distribution,
    lateral_seed,
    launch_parameter_mask,
    parameter_bounds,
    parse_target_batch,
    step_dynamics,
    trajectory_metrics,
)
from parkour_leap_to import (
    LANDMARKS,
    BatchedParkourTargets,
    ParkourLeapTarget,
    command_target,
    knot_schedule,
    launch_proposals,
    parameter_distribution,
    parkour_control_tape,
    parkour_leap_metrics,
    performance_proposals,
    performance_refinement_metrics,
    priority_order,
    search_priority,
    sparse_control_proposals,
)
from tilted_ramp_to import (
    TiltedRampTarget,
    ramp_proposals,
    tilted_ramp_mesh,
    tilted_ramp_metrics,
    validate_ramp_geometry,
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jump_height", type=float, default=0.20)
    parser.add_argument("--landing_x", type=float, default=0.10)
    parser.add_argument("--landing_y", type=float, default=0.15)
    parser.add_argument("--height_tolerance", type=float, default=0.01)
    parser.add_argument("--landing_tolerance", type=float, default=0.03)
    parser.add_argument(
        "--motion_mode",
        choices=("point_jump", "forward_leap", "parkour_leap", "gap_leap", "tilted_ramp"),
        default="point_jump",
    )
    parser.add_argument("--gap_width", type=float, default=0.4)
    parser.add_argument("--gap_near_x", type=float, default=6.0)
    parser.add_argument("--gap_depth", type=float, default=1.0)
    parser.add_argument("--gap_trigger_distance", type=float, default=0.40)
    parser.add_argument("--ramp_side", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--ramp_tilt", type=float, default=20.0)
    parser.add_argument("--ramp_length", type=float, default=1.6)
    parser.add_argument("--ramp_width", type=float, default=0.7)
    parser.add_argument("--ramp_offset_y", type=float, default=0.4)
    parser.add_argument("--forward_speed", type=float, default=1.0)
    parser.add_argument("--rear_delay", type=float, default=0.04, help="Forward-leap rear control delay in seconds")
    parser.add_argument("--launch_time_scale", type=float, default=1.0, help="Scale forward-leap motion knot times")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--warmup_seconds", type=float, default=3.0)
    parser.add_argument("--balanced_warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guarded_warmup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--position_iterations", type=int, default=8)
    parser.add_argument("--velocity_iterations", type=int, default=4)
    parser.add_argument("--canonical_origins", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_scale", type=float, help="CEM search spread, not robot action scale")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target_batch", type=Path, help="JSON array of named target dictionaries")
    parser.add_argument(
        "--performance_refinement", action="store_true", help="Preserve replayed style; improve height/air distance"
    )
    parser.add_argument(
        "--refine_launch", action="store_true", help="Freeze lateral/recovery modes; refine 15 launch modes"
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--stop_on_success", action="store_true")
    parser.add_argument("--sparse_refinement", action="store_true", help="Search local actuator/time coordinates")
    parser.add_argument("--actuator_contract", choices=("shared", "legacy_training"), default="shared")
    known, _ = parser.parse_known_args()
    sys.path[:0] = [str(known.source_root), str(known.source_root / "locolab")]
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.sparse_refinement and (
        args.performance_refinement or (args.motion_mode != "parkour_leap" and not args.replay)
    ):
        parser.error("Sparse refinement is supported for plain parkour goal batches")
    if args.noise_scale is not None and (not math.isfinite(args.noise_scale) or args.noise_scale <= 0):
        parser.error("noise_scale must be positive and finite")
    if not math.isfinite(args.rear_delay) or not 0 <= args.rear_delay <= 0.15:
        parser.error("rear_delay must be between zero and 0.15 seconds")
    if not math.isfinite(args.launch_time_scale) or not 0.5 <= args.launch_time_scale <= 1.25:
        parser.error("launch_time_scale must be between 0.5 and 1.25")
    target_entries = json.loads(args.target_batch.read_text()) if args.target_batch else None
    if args.motion_mode in ("gap_leap", "tilted_ramp") and args.target_batch:
        parser.error("Physical gap widths use separate scenes; target_batch is not supported")
    goals = (
        parse_target_batch(target_entries, command_target if args.motion_mode == "parkour_leap" else JumpTarget)
        if args.target_batch
        else None
    )
    if goals and args.motion_mode == "forward_leap":
        parser.error("Legacy forward-leap batches are unsupported; use parkour_leap")
    if goals and args.motion_mode == "parkour_leap":
        try:
            BatchedParkourTargets([goal for _, goal in goals])  # validate shared settings before opening Isaac Sim
        except ValueError as error:
            parser.error(str(error))
        args.forward_speed = goals[0][1].forward_speed
    performance_rows = {}
    if args.performance_refinement:
        if not goals or args.motion_mode != "parkour_leap":
            parser.error("Performance refinement requires a parkour target batch with per-goal replay baselines")
        for entry, (name, goal) in zip(target_entries, goals):
            if not entry.get("performance_baseline") or not entry.get("seed_candidate"):
                parser.error("Each performance goal needs seed_candidate and performance_baseline")
            path = Path(entry["performance_baseline"])
            replay = json.loads(path.read_text())
            if (
                replay.get("target") != asdict(goal)
                or not replay.get("independent_replay")
                or replay.get("actual_physics_steps") != 360
                or replay.get("motion_mode") != "parkour_leap"
                or replay.get("leg_action_scale") != 0.25
                or replay.get("wheel_action_scale") != 10.0
                or replay.get("actuator_contract") != "shared"
            ):
                parser.error(f"Performance baseline does not match command/dynamics: {name}")
            performance_rows[name] = dict(
                metrics=replay["metrics"], input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
    if args.motion_mode == "parkour_leap" and args.refine_launch:
        parser.error("Parkour v2 optimizes both launch and landing; legacy launch-only masks are unsupported")
    if goals and args.replay:
        parser.error("Batch optimization and single-candidate replay are separate operations")
    goal_count = len(goals) if goals else 1
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory; existing artifacts are never overwritten")
    if not args.replay and (args.num_envs < 4 or args.iterations < 1):
        parser.error("Optimization needs at least four environments and one iteration")
    if not args.replay and (args.replicas < 1 or args.num_envs % args.replicas or args.num_envs // args.replicas < 4):
        parser.error("num_envs must be a multiple of replicas with at least four distinct candidates")
    if goals and (args.num_envs % (goal_count * args.replicas) or args.num_envs // (goal_count * args.replicas) < 4):
        parser.error("Batch num_envs must be divisible by goals * replicas, with at least four samples per goal")
    if args.video and not args.replay:
        parser.error("Video is only supported for an independent replay")
    if not args.replay and args.sparse_refinement and args.num_envs // (goal_count * args.replicas) < 6:
        parser.error("Sparse refinement needs at least six candidates per goal")
    if args.replay:
        args.num_envs = 1
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    failed = False
    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.math import quat_apply

        import locolab.source.locomotion.tasks  # noqa: F401 -- Gym registration
        from locolab.source.locomotion.assets.robot import GO2W_CFG
        from locolab.source.locomotion.tasks.Go2W.jump_env_cfg import Go2WJumpPlayEnvCfg
        from locolab.source.locomotion.terrains.grid_plane import MovingJumpPlaneTerrainImporter

        class TargetPlaneImporter(MovingJumpPlaneTerrainImporter):
            """Identical flat mesh dimensions in batch and single-env replay."""

            def import_ground_plane(self, name, size=(100.0, 100.0)):
                from isaaclab.terrains.trimesh.utils import make_plane

                self.import_mesh(name, make_plane(size, height=0.0))

        class TargetGapImporter(MovingJumpPlaneTerrainImporter):
            def import_ground_plane(self, name, size=(100.0, 100.0)):
                import omni.usd
                from pxr import UsdPhysics

                self.import_mesh(name, (tilted_ramp_mesh if ramp else gap_mesh)(gap_target))
                prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.cfg.prim_path + f"/{name}/mesh")
                # A convex hull would close the hole. Keep the actual static
                # triangle mesh as the collider, not just the visual geometry.
                UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("none")

        torch.manual_seed(args.seed)
        target = JumpTarget(
            args.jump_height, args.landing_x, args.landing_y, args.height_tolerance, args.landing_tolerance
        )
        seed_payload = None
        goal_seeds = {
            entry["name"]: json.loads(Path(entry["seed_candidate"]).read_text())
            for entry in (target_entries or [])
            if entry.get("seed_candidate")
        }
        goal_seed_hashes = {
            entry["name"]: hashlib.sha256(Path(entry["seed_candidate"]).read_bytes()).hexdigest()
            for entry in (target_entries or [])
            if entry.get("seed_candidate")
        }
        if args.replay or args.resume:
            seed_payload = json.loads((args.replay or args.resume).read_text())
            if args.replay:
                args.seed = seed_payload.get("seed", args.seed)
                args.motion_mode = seed_payload.get("motion_mode", "point_jump")
                args.performance_refinement = seed_payload.get("performance_refinement", False)
                args.sparse_refinement = seed_payload.get("sparse_refinement", False)
                if args.performance_refinement:
                    performance_rows = {"single": seed_payload["performance_baseline"]}
                target_type = {
                    "point_jump": JumpTarget,
                    "forward_leap": ForwardLeapTarget,
                    "parkour_leap": ParkourLeapTarget,
                    "gap_leap": GapLeapTarget,
                    "tilted_ramp": TiltedRampTarget,
                }[args.motion_mode]
                target = (
                    load_forward_target(seed_payload["target"])
                    if args.motion_mode == "forward_leap"
                    else target_type(**seed_payload["target"])
                )
                args.forward_speed = getattr(target, "forward_speed", 0.0)
                args.actuator_contract = seed_payload["actuator_contract"]
                args.warmup_seconds = seed_payload.get("warmup_seconds", 1.0)
                args.balanced_warmup = seed_payload.get("balanced_warmup", False)
                args.position_iterations = seed_payload.get("position_iterations", 4)
                args.velocity_iterations = seed_payload.get("velocity_iterations", 1)
                args.canonical_origins = seed_payload.get("canonical_origins", False)
                args.replicas = seed_payload.get("optimization_replicas_per_candidate", 1)
                args.refine_launch = seed_payload.get("launch_only_refinement", False)
                if args.noise_scale is None:
                    args.noise_scale = seed_payload.get("search_noise_scale", 0.3)
                if args.guarded_warmup is None:
                    args.guarded_warmup = seed_payload.get("guarded_warmup", False)
        ramp = args.motion_mode == "tilted_ramp"
        gap = args.motion_mode in ("gap_leap", "tilted_ramp")
        parkour = args.motion_mode in ("parkour_leap", "gap_leap", "tilted_ramp")
        moving = args.motion_mode != "point_jump"
        knot_times = (0.0, 0.12, 0.22, 0.36, 0.52, 0.72, 1.10, 1.60, 1.80) if moving else KNOT_TIMES
        if moving:
            knot_times = tuple(t * args.launch_time_scale if i <= 5 else t for i, t in enumerate(knot_times))
        recovery_from_knot = 6 if moving else None
        rear_delay = args.rear_delay if moving else 0.0
        if args.replay:
            knot_times = seed_payload.get("knot_times", KNOT_TIMES)
            recovery_from_knot = seed_payload.get("recovery_from_knot")
            rear_delay = seed_payload.get("rear_delay", 0.0)
        if gap and not args.replay:
            extra = (
                dict(
                    side=args.ramp_side,
                    tilt_degrees=args.ramp_tilt,
                    platform_length=args.ramp_length,
                    platform_width=args.ramp_width,
                    offset_y=args.ramp_offset_y,
                )
                if ramp
                else {}
            )
            target = (TiltedRampTarget if ramp else GapLeapTarget)(
                width=args.gap_width,
                near_x=args.gap_near_x,
                depth=args.gap_depth,
                trigger_distance=args.gap_trigger_distance,
                forward_speed=args.forward_speed,
                **extra,
            )
        elif moving and not args.replay:
            target = (command_target if parkour else ForwardLeapTarget)(
                height=args.jump_height,
                landing_x=args.landing_x,
                landing_y=args.landing_y,
                height_tolerance=args.height_tolerance,
                landing_tolerance=args.landing_tolerance,
                forward_speed=args.forward_speed,
            )
        if parkour:
            # Candidate-dependent schedules are saved separately. Legacy fixed
            # timing CLI flags do not override the optimized v2 intervals.
            knot_times, recovery_from_knot, rear_delay = None, None, None
        gap_target = target if gap else None
        if gap:
            args.canonical_origins = True
        if args.guarded_warmup is None:
            args.guarded_warmup = True
        if args.noise_scale is None:
            args.noise_scale = 0.3 if seed_payload else 1.0
        torch.manual_seed(args.seed)
        cfg = Go2WJumpPlayEnvCfg()
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.scene.env_spacing = 2.5
        cfg.scene.terrain.num_envs = args.num_envs
        cfg.scene.terrain.env_spacing = cfg.scene.env_spacing
        if gap:
            cfg.scene.terrain.class_type = TargetGapImporter
            cfg.scene.terrain.visual_material = None  # retain bank/pit vertex colours
        elif not args.replay or seed_payload.get("fixed_ground_mesh", False):
            cfg.scene.terrain.class_type = TargetPlaneImporter
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
        if args.actuator_contract == "shared":
            cfg.scene.robot.actuators = GO2W_CFG.actuators.copy()
        cfg.commands.jump_command.jump_probability = 0.0
        cfg.commands.jump_command.resampling_time_range = (1000.0, 1000.0)
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        cfg.events.reset_base.params.update(
            jump_probability=0.0,
            pose_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
            velocity_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
            joint_position_range=(1.0, 1.0),
            joint_velocity_range=(0.0, 0.0),
        )
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.7, 2.0, 1.4)
        cfg.viewer.lookat = (0.1, 0.1, 0.35)
        cfg.viewer.resolution = (1280, 720)
        env = gym.make("OmniBot-Go2W-Jump-MLP-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
        native = env.unwrapped
        gap_geometry = None
        if gap:
            import omni.usd
            from pxr import UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            path = cfg.scene.terrain.prim_path + "/terrain/mesh"
            prim = stage.GetPrimAtPath(path)
            mesh = UsdGeom.Mesh(prim)
            other_colliders = [
                str(p.GetPath())
                for p in stage.Traverse()
                if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith("/World/envs/")
            ]
            gap_geometry = dict(
                target=asdict(gap_target),
                mesh_sha256=geometry_fingerprint(mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()),
                collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                collision_approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                no_extra_static_colliders=other_colliders == [path],
                static_collider_paths=other_colliders,
            )
            (validate_ramp_geometry if ramp else validate_gap_geometry)(gap_geometry, gap_target)
        if args.canonical_origins:
            # Each clone retains Isaac Lab's inter-environment collision filter.
            # Identical world placement removes mesh-contact/float-coordinate
            # differences from nominal TO optimization. This is not DR testing.
            native.scene.env_origins.zero_()
        robot = native.scene["robot"]
        sensor = native.scene.sensors["contact_forces"]
        ids = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)[0]
        wheels = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)[0]
        foot_names = [name.removesuffix("_joint") for name in WHEEL_JOINT_NAMES]
        feet = sensor.find_bodies(foot_names, preserve_order=True)[0]
        foot_bodies = robot.find_bodies(foot_names, preserve_order=True)[0]
        hip_bodies = robot.find_bodies([name.replace("foot", "hip") for name in foot_names], preserve_order=True)[0]
        body_masses = robot.root_physx_view.get_masses().to(native.device)
        mass_weights = body_masses / body_masses.sum(-1, keepdim=True)
        nonfeet = [i for i in range(len(sensor.body_names)) if i not in feet]
        dt, sim_dt = native.step_dt, cfg.sim.dt
        default = robot.data.default_joint_pos[0, ids].clone()
        limits = robot.data.joint_pos_limits[0, ids].clone()
        speed_limits = robot.data.joint_vel_limits[0].clone()
        n, device = args.num_envs, native.device
        if goals:
            per_goal = n // goal_count
            if parkour:
                target = BatchedParkourTargets([g for _, g in goals], per_goal, device)
            else:
                target = torch.tensor([list(asdict(g).values()) for _, g in goals], device=device)
                target = target.repeat_interleave(per_goal, dim=0)
        offsets = torch.tensor([[0.0, s * 0.048, 0.0] for s in (-1, 1, -1, 1)], device=device)
        performance_baseline = None
        if args.performance_refinement:
            rows = (
                [performance_rows[name]["metrics"] for name, _ in goals]
                if goals
                else [performance_rows["single"]["metrics"]]
            )
            performance_baseline = {
                key: torch.tensor([row[key] for row in rows], device=device).repeat_interleave(n // len(rows))
                for key, value in rows[0].items()
                if isinstance(value, (int, float, bool))
            }
        axes = torch.tensor([[0.0, 1.0, 0.0]] * 4, device=device)
        base_target = torch.zeros(n, 16, device=device)
        base_target[:, :12] = default
        writer = None
        actual_warmup_seconds = args.warmup_seconds
        actual_physics_steps = 0

        def step(targets, render=False):
            robot.set_joint_position_target(targets[:, :12], joint_ids=ids)
            robot.set_joint_velocity_target(targets[:, 12:], joint_ids=wheels)
            native.scene.write_data_to_sim()
            step_dynamics(native.sim, native.scene, sim_dt, render)

        def sample():
            quat = robot.data.root_quat_w
            w, x, y, z = quat.unbind(-1)
            foot_quat = robot.data.body_link_quat_w[:, foot_bodies]
            center = robot.data.body_link_pos_w[:, foot_bodies] + quat_apply(foot_quat, offsets.expand(n, -1, -1))
            wheel_axis = quat_apply(foot_quat, axes.expand(n, -1, -1))
            az = wheel_axis[:, :, 2].clamp(-1, 1)
            extent = 0.0225 * az.abs() + 0.086 * (1 - az.square()).clamp_min(0).sqrt()
            ax, ay = wheel_axis[:, :, 0].clamp(-1, 1), wheel_axis[:, :, 1].clamp(-1, 1)
            return {
                "position": (robot.data.root_pos_w - native.scene.env_origins).clone(),
                "velocity": robot.data.root_lin_vel_w.clone(),
                "quaternion": quat.clone(),
                "omega": robot.data.root_ang_vel_w.clone(),
                "yaw": torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square())),
                "roll": torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square())),
                "pitch": torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
                "tilt": torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)),
                "contacts": sensor.data.net_forces_w[:, feet].norm(dim=-1) > 1.0,
                "nonfoot_force": sensor.data.net_forces_w[:, nonfeet].norm(dim=-1).max(-1).values,
                "speed_violation": (robot.data.joint_vel.abs() - speed_limits).clamp_min(0).max(-1).values,
                "limit_violation": torch.maximum(
                    limits[:, 0] - robot.data.joint_pos[:, ids], robot.data.joint_pos[:, ids] - limits[:, 1]
                )
                .clamp_min(0)
                .max(-1)
                .values,
                "clearance": (center[:, :, 2] - extent - native.scene.env_origins[:, 2, None]).min(-1).values,
                "wheel_bottom_height": (center[:, :, 2] - extent - native.scene.env_origins[:, 2, None]).clone(),
                "wheel_forward_radius": (0.0225 * ax.abs() + 0.086 * (1 - ax.square()).clamp_min(0).sqrt()).clone(),
                "wheel_lateral_radius": (0.0225 * ay.abs() + 0.086 * (1 - ay.square()).clamp_min(0).sqrt()).clone(),
                "wheel_axis": wheel_axis.clone(),
                "wheel_center_position": (center - native.scene.env_origins[:, None]).clone(),
                "hip_position": (robot.data.body_link_pos_w[:, hip_bodies] - native.scene.env_origins[:, None]).clone(),
                "com_position": (robot.data.body_com_pos_w * mass_weights[..., None]).sum(1) - native.scene.env_origins,
                "com_velocity": (robot.data.body_com_lin_vel_w * mass_weights[..., None]).sum(1),
                "wheel_contact_force": sensor.data.net_forces_w[:, feet].clone(),
                "joint_position": robot.data.joint_pos[:, ids].clone(),
                "joint_velocity": robot.data.joint_vel[:, ids].clone(),
                "wheel_position": robot.data.joint_pos[:, wheels].clone(),
                "wheel_velocity": robot.data.joint_vel[:, wheels].clone(),
            }

        def rollout(parameters):
            nonlocal actual_warmup_seconds, actual_physics_steps
            # Only ordinary grounded resets, followed by identical standing PD
            # warmup. No simulated state is prescribed during the trajectory.
            env.reset()
            base_target[:, :12] = default
            base_target[:, 12:] = 0.0
            stable_count = torch.zeros(n, device=device, dtype=torch.long)
            minimum_steps = round(args.warmup_seconds / sim_dt)
            maximum_steps = minimum_steps + (round(3.0 / sim_dt) if args.guarded_warmup else 0)
            if gap:
                maximum_steps = round(12.0 / sim_dt)
            for warmup_step in range(maximum_steps):
                if moving:
                    # Accelerate from the ordinary zero-speed reset through
                    # wheel actuators. No root-velocity initialization shortcut.
                    desired_vx = args.forward_speed * min(1.0, warmup_step * sim_dt / 1.5)
                    speed_error = desired_vx - robot.data.root_lin_vel_b[:, 0]
                    base_target[:, 12:] = ((desired_vx + 0.5 * speed_error) / 0.086)[:, None].clamp(-25, 25)
                if args.balanced_warmup:
                    # Establish a stationary support pose with actuator PD only.
                    # No root wrench/state edits. Feedback ends at trigger time.
                    gravity = robot.data.projected_gravity_b
                    pitch = torch.asin(gravity[:, 0].clamp(-1, 1))
                    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                    omega = robot.data.root_ang_vel_b
                    fb = (2.0 * pitch + 0.2 * omega[:, 1]).clamp(-0.35, 0.35)
                    lr = (-2.0 * roll - 0.2 * omega[:, 0]).clamp(-0.25, 0.25)
                    calf_delta = fb[:, None] * parameters.new_tensor([1, 1, -1, -1])
                    calf_delta += lr[:, None] * parameters.new_tensor([-1, 1, -1, 1])
                    base_target[:, 1:12:3] = 0.78 - 0.5 * calf_delta
                    base_target[:, 2:12:3] = -1.30 + calf_delta
                step(base_target)
                if args.guarded_warmup or gap:
                    target_velocity = torch.zeros_like(robot.data.root_lin_vel_w)
                    if moving:
                        target_velocity[:, 0] = args.forward_speed
                    ready = (
                        (sensor.data.net_forces_w[:, feet].norm(dim=-1) > 1.0).all(-1)
                        & ((robot.data.root_lin_vel_w - target_velocity).norm(dim=-1) < 0.1)
                        & (robot.data.projected_gravity_b[:, 2] < -0.99)
                        & (robot.data.joint_vel[:, ids].abs().max(-1).values < 0.3)
                    )
                    stable_count = torch.where(ready, stable_count + 1, 0)
                    if gap:
                        at_marker = robot.data.root_pos_w[:, 0] >= gap_target.near_x - gap_target.trigger_distance
                        if warmup_step + 1 >= minimum_steps and at_marker.all() and ready.float().mean() >= 0.9:
                            break
                        if (robot.data.root_pos_w[:, 0] > gap_target.near_x - 0.28).any():
                            raise RuntimeError("Could not obtain a grounded rolling trigger before the physical gap")
                    elif warmup_step + 1 >= minimum_steps and (stable_count >= 8).float().mean() >= 0.9:
                        break
            if gap and not bool(at_marker.all()):
                raise RuntimeError("Rolling approach did not reach the fixed launch marker")
            actual_warmup_seconds = (warmup_step + 1) * sim_dt
            if parkour:
                tape = parkour_control_tape(
                    parameters,
                    base_target[:, :12],
                    limits[:, 0],
                    limits[:, 1],
                    dt,
                    args.forward_speed,
                    end_speed=0.0 if ramp else None,
                )
            else:
                tape = control_tape(
                    parameters,
                    base_target[:, :12],
                    limits[:, 0],
                    limits[:, 1],
                    dt,
                    endpoint_wheel_speed=args.forward_speed / 0.086 if moving else 0.0,
                    knot_times=knot_times,
                    recovery_from_knot=recovery_from_knot,
                    rear_delay=rear_delay,
                )
            if args.video and moving:
                start = robot.data.root_pos_w[0].tolist()
                native.sim.set_camera_view(
                    eye=(
                        start[0] + 0.8,
                        start[1] + (2.0 if abs(target.landing_y) > 1e-6 else 2.7),
                        2.2 if abs(target.landing_y) > 1e-6 else 1.1,
                    ),
                    target=(start[0] + 0.8, start[1], 0.4),
                )
                if gap:
                    middle = 0.5 * (gap_target.near_x + gap_target.far_x)
                    native.sim.set_camera_view(eye=(middle + 1.5, 3.0, 1.8), target=(middle, 0.0, 0.05))
                if ramp:
                    native.sim.set_camera_view(eye=(target.near_x - 1.0, 0.0, 2.7), target=(target.center_x, 0.0, 0.0))
            initial_physics_step = native.sim.current_time_step_index
            record = [sample()]
            for action in tape.unbind(1):
                for substep in range(cfg.decimation):
                    step(action, render=bool(writer and substep == cfg.decimation - 1))
                    record.append(sample())
                if writer:
                    before_render = native.sim.current_time_step_index
                    writer.append_data(env.render())
                    if native.sim.current_time_step_index != before_render:
                        raise RuntimeError("RGB frame capture advanced physics")
            actual_physics_steps = native.sim.current_time_step_index - initial_physics_step
            states = {key: torch.stack([row[key] for row in record], dim=1) for key in record[0]}
            metric_function = (
                tilted_ramp_metrics
                if ramp
                else (
                    gap_leap_metrics
                    if gap
                    else (parkour_leap_metrics if parkour else (forward_leap_metrics if moving else trajectory_metrics))
                )
            )
            metrics = metric_function(states, target, sim_dt)
            if args.performance_refinement:
                metrics = performance_refinement_metrics(metrics, target, performance_baseline)
            return metrics, states, tape

        def row(values, index):
            return {key: value[index].detach().cpu().tolist() for key, value in values.items()}

        metadata = {
            "method": "CEM_direct_shooting_full_IsaacLab_Go2W_dynamics",
            "seed": args.seed,
            "target": None if goals else asdict(target),
            "target_batch": [{"name": name, "target": asdict(g)} for name, g in goals] if goals else None,
            "target_frame": (
                "fixed physical bank coordinates; per-wheel landing, no point-displacement command"
                if gap
                else "fixed settled-trigger yaw; landing at first contact after sustained airborne period"
            ),
            "height_definition": "maximum base rise from settled trigger height before touchdown",
            "motion_mode": args.motion_mode,
            "acceptance_profile": (
                "flat_goal_first_v4" if parkour and not gap and target.criteria_version == 4 else "versioned_original"
            ),
            "parameterization": "parkour_v2_time_and_shape" if parkour else "legacy_70_mode_fixed_schedule",
            "search_method": (
                "tilted_ramp_v1_finite_surface_landing_and_stop"
                if ramp
                else (
                    "physical_gap_v1_per_wheel_crossing"
                    if gap
                    else (
                        "parkour_v5_guarded_performance"
                        if args.performance_refinement
                        else (
                            "parkour_v7_goal_first_soft_style"
                            if parkour and target.criteria_version == 4
                            else (
                                "parkour_v6_sparse_coordinates"
                                if args.sparse_refinement
                                else ("parkour_v4_budget_and_load_transfer" if parkour else "legacy_scalar_cem")
                            )
                        )
                    )
                )
            ),
            "performance_refinement": args.performance_refinement,
            "sparse_refinement": args.sparse_refinement,
            "gap_geometry": gap_geometry,
            "landing_surface": (
                {
                    "kind": "cross_sloped_finite_rectangle",
                    "side": target.side,
                    "tilt_degrees": target.side * target.tilt_degrees,
                    "exit_speed": 0.0,
                }
                if ramp
                else None
            ),
            "performance_baseline": performance_rows.get("single") if args.replay else None,
            "control_landmarks": LANDMARKS if parkour else None,
            "lateral_control_modes": "enabled only for nonzero landing_y goals" if parkour else None,
            "command_spatial_constraints": (
                "each wheel first touches the selected inclined rectangle; stop inside all four boundaries"
                if ramp
                else (
                    "all four wheels cross the missing deck and first touch the far bank after flight"
                    if gap
                    else (
                        "height/first-touchdown XY and real flight hard; airborne fraction/virtual crossing soft"
                        if parkour and target.criteria_version == 4
                        else ("minimum airborne dx=2/3*dx; virtual base crossing [1/2*dx,2/3*dx]" if parkour else None)
                    )
                )
            ),
            "knot_times": knot_times,
            "recovery_from_knot": recovery_from_knot,
            "rear_delay": rear_delay,
            "initial_state": (
                "ordinary reset plus actuator-only rolling warmup"
                if moving
                else "ordinary standing reset plus zero-wheel-speed PD warmup"
            ),
            "crossing_scope": (
                "actual mirrored cross slopes, cylinder-to-surface contact and bounded landing footprint"
                if ramp
                else (
                    "actual missing deck, per-wheel far-bank touchdown and whole-robot stable recovery"
                    if gap
                    else (
                        "virtual BASE-progress interval on flat ground, not physical gap certification"
                        if moving
                        else None
                    )
                )
            ),
            "warmup_seconds": args.warmup_seconds,
            "balanced_warmup": args.balanced_warmup,
            "guarded_warmup": args.guarded_warmup,
            "fixed_ground_mesh": cfg.scene.terrain.class_type is TargetPlaneImporter,
            "position_iterations": args.position_iterations,
            "velocity_iterations": args.velocity_iterations,
            "canonical_origins": args.canonical_origins,
            "launch_only_refinement": args.refine_launch,
            "search_noise_scale": args.noise_scale,
            "optimization_replicas_per_candidate": args.replicas,
            "actuator_contract": args.actuator_contract,
            "mass_kg": float(robot.root_physx_view.get_masses()[0].sum()),
            "gravity": float(sum(v * v for v in cfg.sim.gravity) ** 0.5),
            "leg_effort_limit_nm": robot.data.joint_effort_limits[0, ids].tolist(),
            "leg_velocity_limit_rad_s": speed_limits[ids].tolist(),
            "leg_joint_names": LEG_JOINT_NAMES,
            "wheel_joint_names": WHEEL_JOINT_NAMES,
            "leg_action_scale": cfg.actions.joint_pos.scale,
            "wheel_action_scale": cfg.actions.joint_vel.scale,
            "self_collisions_enabled": cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
            "simulation_dt": sim_dt,
            "control_dt": dt,
            "policy_used": False,
            "external_wrenches_used": False,
            "rl_training_started": False,
            "production_reference_replaced": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(Path(__file__).with_name("jump_target_to.py").read_bytes()).hexdigest(),
            "forward_leap_sha256": hashlib.sha256(
                Path(__file__).with_name("forward_leap_to.py").read_bytes()
            ).hexdigest(),
            "parkour_leap_sha256": hashlib.sha256(
                Path(__file__).with_name("parkour_leap_to.py").read_bytes()
            ).hexdigest(),
            "gap_leap_sha256": hashlib.sha256(Path(__file__).with_name("gap_leap_to.py").read_bytes()).hexdigest(),
            "tilted_ramp_sha256": hashlib.sha256(
                Path(__file__).with_name("tilted_ramp_to.py").read_bytes()
            ).hexdigest(),
            "optimization_seed_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        }
        print("TO_CONTRACT " + json.dumps(metadata), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        # Freeze the exact solver used, independently of later local revisions.
        import shutil

        for filename in (
            "solve_jump_target_to.py",
            "jump_target_to.py",
            "forward_leap_to.py",
            "parkour_leap_to.py",
            "gap_leap_to.py",
            "tilted_ramp_to.py",
        ):
            shutil.copyfile(Path(__file__).with_name(filename), args.output / filename)
        if args.replay:
            if args.video:
                import imageio.v2 as imageio

                writer = imageio.get_writer(str(args.output / "replay.mp4"), fps=round(1 / dt), codec="libx264")
            parameters = torch.tensor(seed_payload["parameters"], device=device).unsqueeze(0)
            metrics, states, tape = rollout(parameters)
            if writer:
                writer.close()
                writer = None
            result = dict(
                metadata,
                accepted=bool(metrics["accepted"][0]),
                metrics=row(metrics, 0),
                input_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                states=row(states, 0),
                control_tape=tape[0].tolist(),
                independent_replay=True,
                actual_warmup_seconds=actual_warmup_seconds,
                actual_physics_steps=actual_physics_steps,
                control_schedule=knot_schedule(parameters)[0].tolist() if parkour else None,
            )
            write_json(args.output / "replay.json", result)
            if result["accepted"]:
                write_json(args.output / "reference.json", export_reference(result))
            print("TO_REPLAY " + json.dumps(result["metrics"]), flush=True)
            return 0 if result["accepted"] else 2
        if parkour:
            distributions = [
                parameter_distribution(device, args.forward_speed, goal_seeds.get(name, seed_payload), g.landing_y)
                for name, g in (goals if goals else [("single", target)])
            ]
            mean = torch.stack([d[0] for d in distributions])
            initial_std = torch.stack([d[1] for d in distributions])
            low, high = distributions[0][2:4]
            parkour_mask = torch.stack([d[4] for d in distributions])
        else:
            mean, initial_std = initial_distribution(device)
            low, high = parameter_bounds(device)
        if seed_payload and not parkour:
            mean = torch.tensor(seed_payload["parameters"], device=device)
        if moving and not parkour:
            # Wheel controls have the same units/action scale as before, but
            # cannot be bounded to +/-8 rad/s when rolling at 1 m/s.
            low.reshape(7, 10)[:, 9] = -25.0
            high.reshape(7, 10)[:, 9] = 25.0
            if not seed_payload or seed_payload.get("motion_mode") != "forward_leap":
                mean.reshape(7, 10)[:, 9] = args.forward_speed / 0.086
            mean.reshape(7, 10)[:, [0, 2, 5, 8]] = 0.0
        mean = mean.expand(goal_count, -1).clone()
        initial_std = initial_std.expand_as(mean).clone()
        if ramp and (not seed_payload or seed_payload.get("motion_mode") != "tilted_ramp"):
            poses = mean[:, :70].reshape(goal_count, 7, 10)
            poses[:, :2, 0] = -0.14 * target.side
            poses[:, 4:, 9] = mean.new_tensor([4.0, 1.0, 0.0])
        if gap:
            initial_std[:, :70].reshape(goal_count, 7, 10)[:, :, 9] *= 3.0
        if args.performance_refinement:
            initial_std[:, 30:70] *= 0.4
            initial_std[:, 73:77] *= 0.4
            initial_std[:, 80:84] *= 0.4
        std = initial_std * args.noise_scale
        if goals and seed_payload and not parkour:
            source_y = seed_payload["target"]["landing_y"]
            if abs(source_y) > 1e-6:
                for i, (_, goal) in enumerate(goals):
                    direction = 0 if abs(goal.landing_y) < 1e-6 else (1 if goal.landing_y * source_y > 0 else -1)
                    mean[i] = lateral_seed(mean[i], direction)
        search_mask = (
            parkour_mask
            if parkour
            else (
                launch_parameter_mask(device) if args.refine_launch else torch.ones(70, dtype=torch.bool, device=device)
            )
        )
        if moving and not parkour:
            # Nominal straight leap: retain sagittal front/rear differences,
            # remove asymmetric left/right modes rather than chasing yaw.
            search_mask.reshape(7, 10)[:, [0, 2, 5, 8]] = False
            if recovery_from_knot is not None:
                search_mask.reshape(7, 10)[recovery_from_knot - 1 :] = False
        if search_mask.ndim == 1:
            search_mask = search_mask.expand(goal_count, -1)
        initial_std *= search_mask
        std *= search_mask
        frozen_parameters = mean.clone()
        best_scores = [float("inf")] * goal_count
        best_priorities = [(float("inf"),) * 5 for _ in range(goal_count)]
        best_parameters, history = mean.clone(), []
        best_accepted = [False] * goal_count
        population = n // (goal_count * args.replicas)
        for iteration in range(args.iterations):
            samples = mean[:, None] + torch.randn(goal_count, population, mean.shape[-1], device=device) * std[:, None]
            if parkour:
                # Maintain a broad proposal tail so discrete contact changes
                # remain searchable after the local elite spread contracts.
                tail = max(1, population // 5)
                samples[:, -tail:] = best_parameters[:, None] + torch.randn_like(samples[:, -tail:]) * initial_std[
                    :, None
                ] * (0.25 if args.performance_refinement else 0.6)
                # Keep local/broad unconstrained samples as well as structured
                # launch proposals; seed/mean slots below remain unmodified.
                if args.performance_refinement:
                    samples = performance_proposals(samples, best_parameters, low, high)
                elif ramp:
                    samples[:, 2 : 2 + tail] = ramp_proposals(samples[:, 2 : 2 + tail], low, high, target.side)
                elif gap:
                    samples[:, 2 : 2 + tail] = gap_proposals(samples[:, 2 : 2 + tail], low, high)
                elif target.criteria_version < 4:
                    samples[:, 2 : 2 + tail] = launch_proposals(
                        samples[:, 2 : 2 + tail],
                        [goal for _, goal in goals] if goals else [target],
                        low,
                        high,
                    )
            if args.sparse_refinement:
                count = max(4, min(population - 2, population * 3 // 4))
                samples[:, 2 : 2 + count] = sparse_control_proposals(
                    best_parameters, search_mask, initial_std, count, iteration, low, high
                )
            samples = samples.clamp(low, high)
            samples[:, 0] = best_parameters
            samples[:, 1] = mean
            if goals:
                for i, done in enumerate(best_accepted):
                    if done and (not parkour or target.criteria_version < 4):
                        samples[i] = best_parameters[i]
            samples = torch.where(search_mask[:, None], samples, frozen_parameters[:, None])
            with torch.inference_mode():
                metrics, states, tape = rollout(samples.flatten(0, 1).repeat_interleave(args.replicas, dim=0))
            per_replica_score = torch.nan_to_num(metrics["score"], nan=1e9, posinf=1e9, neginf=1e9)
            per_replica_score = per_replica_score.reshape(goal_count, population, args.replicas)
            score = per_replica_score.max(-1).values
            all_accepted = metrics["accepted"].reshape(goal_count, population, args.replicas).all(-1)
            keys = search_priority(metrics, args.replicas).reshape(goal_count, population, -1) if parkour else None
            reports = []
            for g in range(goal_count):
                elite = (priority_order(keys[g]) if parkour else score[g].argsort())[: max(4, population // 10)]
                best = int(elite[0])
                accepted_ids = torch.where(all_accepted[g])[0]
                if len(accepted_ids) and not parkour:
                    best = int(accepted_ids[score[g, accepted_ids].argmin()])
                is_accepted = bool(all_accepted[g, best])
                base_index = (g * population + best) * args.replicas
                worst_replica = base_index + int(per_replica_score[g, best].argmax())
                improved = (is_accepted and not best_accepted[g]) or (
                    is_accepted == best_accepted[g] and float(score[g, best]) < best_scores[g]
                )
                if parkour:
                    priority = tuple(keys[g, best].tolist())
                    improved = priority < best_priorities[g]
                    replica_keys = search_priority(
                        {k: v[base_index : base_index + args.replicas] for k, v in metrics.items()}
                    )
                    worst_replica = base_index + int(priority_order(replica_keys)[-1])
                if improved:
                    if parkour:
                        best_priorities[g] = priority
                    best_scores[g], best_parameters[g] = float(score[g, best]), samples[g, best].clone()
                    best_accepted[g] = is_accepted
                    destination = args.output / goals[g][0] if goals else args.output
                    candidate_metadata = dict(metadata, target=asdict(goals[g][1])) if goals else metadata
                    if args.performance_refinement:
                        candidate_metadata = dict(
                            candidate_metadata, performance_baseline=performance_rows[goals[g][0]]
                        )
                    if goals and goals[g][0] in goal_seed_hashes:
                        candidate_metadata["optimization_seed_sha256"] = goal_seed_hashes[goals[g][0]]
                    write_json(
                        destination / "candidate.json",
                        dict(
                            candidate_metadata,
                            iteration=iteration,
                            parameters=best_parameters[g].tolist(),
                            control_schedule=knot_schedule(best_parameters[g : g + 1])[0].tolist() if parkour else None,
                            metrics=row(metrics, worst_replica),
                            all_replicas_accepted=is_accepted,
                            search_priority=list(best_priorities[g]) if parkour else None,
                            actual_warmup_seconds=actual_warmup_seconds,
                            replica_metrics=[row(metrics, base_index + i) for i in range(args.replicas)],
                        ),
                    )
                    write_json(destination / "candidate_states.json", row(states, worst_replica))
                if not (goals and best_accepted[g]) or (parkour and target.criteria_version == 4):
                    mean[g] = 0.2 * mean[g] + 0.8 * samples[g, elite].mean(0)
                    std[g] = torch.maximum(
                        0.2 * std[g] + 0.8 * samples[g, elite].std(0, unbiased=False),
                        initial_std[g]
                        * (
                            0.05
                            if args.sparse_refinement
                            else (0.10 if args.performance_refinement else (0.20 if parkour else 0.08))
                        ),
                    )
                reports.append(
                    {
                        "name": goals[g][0] if goals else "single",
                        "best_score": best_scores[g],
                        "search_priority": list(best_priorities[g]) if parkour else None,
                        "all_replicas_accepted": best_accepted[g],
                        "batch_best": row(metrics, worst_replica),
                    }
                )
            report = dict(iteration=iteration, goals=reports, accepted_count=sum(best_accepted))
            history.append(report)
            write_json(args.output / "progress.json", dict(metadata, history=history, complete=False))
            print("TO_ITERATION " + json.dumps(report), flush=True)
            if args.stop_on_success and all(best_accepted):
                break
        write_json(args.output / "progress.json", dict(metadata, history=history, complete=True))
    except BaseException:
        import traceback

        traceback.print_exc()
        failed = True
    finally:
        if env:
            env.close()
        try:
            app.close()
        except SystemExit as error:
            if not failed and error.code not in (None, 0):
                raise
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
