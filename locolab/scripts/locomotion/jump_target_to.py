"""CPU-testable parameterization and acceptance criteria for targeted jump TO.

The optimization variables are actuator targets, not root poses. Every scored
state must come from a free-base, contact-enabled dynamics rollout. Targets are
relative to the settled start pose in its fixed yaw frame, in metres.
"""

from dataclasses import asdict, dataclass

import torch

LEGS = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
WHEEL_JOINT_NAMES = [f"{leg}_foot_joint" for leg in LEGS]
KNOT_TIMES = (0.0, 0.20, 0.32, 0.46, 0.62, 0.82, 1.10, 1.60, 1.80)


def step_dynamics(sim, scene, dt, render=False):
    """Advance exactly one physics step, independently of the rendering rate.

    Isaac Sim's step(render=True) calls app.update(), which can advance several
    physics substeps when rendering_dt != physics_dt. Follow Isaac Lab's native
    environment loop instead and audit the actual physics callback counter.
    """
    before = sim.current_time_step_index
    sim.step(render=False)
    if render:
        sim.render()
    scene.update(dt)
    if sim.current_time_step_index - before != 1:
        raise RuntimeError("Rendering changed the physical step count")


@dataclass(frozen=True)
class JumpTarget:
    height: float = 0.20
    landing_x: float = 0.10
    landing_y: float = 0.15
    height_tolerance: float = 0.01
    landing_tolerance: float = 0.03

    def __post_init__(self):
        import math

        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Target values must be finite")
        if self.height <= 0 or self.height_tolerance <= 0 or self.landing_tolerance <= 0:
            raise ValueError("Height and tolerances must be positive")


def parse_target_batch(entries, target_factory=JumpTarget):
    """Read named scalar goals; names are safe output-directory components."""
    import re

    if not isinstance(entries, list) or not entries:
        raise ValueError("A target batch must be a nonempty list")
    goals = []
    seen = set()
    for entry in entries:
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) or name in seen:
            raise ValueError(f"Invalid or duplicate target name: {name!r}")
        seen.add(name)
        goals.append((name, target_factory(**entry["target"])))
    return goals


def lateral_seed(parameters, direction):
    """Modify a control initial guess only; never mirror simulated states.

    -1 mirrors the left/right spatial control modes; 0 removes their odd part;
    +1 preserves them. Each result still needs optimization and dynamic replay.
    """
    if direction not in (-1, 0, 1):
        raise ValueError("Direction must be -1, 0, or 1")
    p = parameters.clone().reshape(7, 10)
    p[:, [0, 2, 5, 8]] *= direction
    return p.flatten()


def launch_parameter_mask(device="cpu"):
    """Sagittal launch/tuck controls; keep lateral and recovery targets fixed."""
    mask = torch.zeros(7, 10, dtype=torch.bool, device=device)
    mask[:3, [3, 4, 6, 7, 9]] = True
    return mask.flatten()


def initial_distribution(device="cpu"):
    """Seven knots, ten spatial modes each; restore a standing endpoint.

    Modes: hip common/LR/FB; thigh common/FB/LR; calf common/FB/LR;
    wheel speed common. Lateral motion is simulated, never a rotated replay.
    """
    mean = torch.zeros(7, 10, device=device)
    mean[:, 3] = torch.tensor([1.12, 0.1, 1.12, 1.12, 0.85, 0.8, 0.8], device=device)
    mean[:, 6] = torch.tensor([-2.24, -0.55, -2.24, -2.24, -1.70, -1.5, -1.5], device=device)
    std = torch.tensor([0.13, 0.05, 0.06, 0.20, 0.12, 0.08, 0.26, 0.15, 0.10, 1.0], device=device)
    return mean.flatten(), std.expand_as(mean).clone().flatten()


def parameter_bounds(device="cpu"):
    lower = [-0.5, -0.15, -0.2, -0.2, -0.55, -0.35, -2.7, -0.6, -0.4, -8.0]
    upper = [0.5, 0.15, 0.2, 1.8, 0.55, 0.35, -0.5, 0.6, 0.4, 8.0]
    return torch.tensor(lower, device=device).repeat(7), torch.tensor(upper, device=device).repeat(7)


def control_tape(
    parameters,
    default_position,
    lower_limits,
    upper_limits,
    dt=0.02,
    endpoint_wheel_speed=0.0,
    knot_times=KNOT_TIMES,
    recovery_from_knot=None,
    rear_delay=0.0,
):
    """Produce smooth PD targets with no changes to Go2W action scales/gains."""
    p = parameters.reshape(-1, 7, 10)
    n = len(p)
    fb = p.new_tensor([1, 1, -1, -1])
    lr = p.new_tensor([-1, 1, -1, 1])
    knots = p.new_zeros(n, 9, 16)
    for index, base in enumerate((0, 3, 6)):
        # Hip uses LR, FB; thigh/calf use FB, LR.
        a, b = (lr, fb) if index == 0 else (fb, lr)
        knots[:, 1:8, index:12:3] = p[:, :, base, None] + p[:, :, base + 1, None] * a
        knots[:, 1:8, index:12:3] += p[:, :, base + 2, None] * b
    knots[:, 1:8, 12:] = p[:, :, 9, None]
    knots[:, 0, :12] = default_position
    knots[:, -1, :12] = default_position
    knots[:, 0, 12:] = endpoint_wheel_speed
    knots[:, -1, 12:] = endpoint_wheel_speed
    if recovery_from_knot is not None:
        if not 1 <= recovery_from_knot <= 8:
            raise ValueError("Recovery knot must be between one and eight")
        knots[:, recovery_from_knot:, :12] = torch.as_tensor(default_position, device=p.device).expand(n, 12)[:, None]
        knots[:, recovery_from_knot:, 12:] = endpoint_wheel_speed
    knots[:, :, :12] = knots[:, :, :12].clamp(lower_limits + 0.02, upper_limits - 0.02)
    times = p.new_tensor(knot_times)
    if len(times) != 9 or not (times[1:] > times[:-1]).all() or times[0] != 0:
        raise ValueError("Expected nine increasing knot times starting at zero")
    query = torch.arange(round(float(times[-1]) / dt), device=p.device) * dt
    right = torch.searchsorted(times, query, right=True).clamp(1, len(times) - 1)
    left = right - 1
    u = (query - times[left]) / (times[right] - times[left])
    alpha = u**3 * (10 - 15 * u + 6 * u**2)
    tape = knots[:, left] * (1 - alpha[None, :, None]) + knots[:, right] * alpha[None, :, None]
    if rear_delay:
        rear_times = times.clone()
        rear_times[1:6] += rear_delay
        if not (rear_times[1:] > rear_times[:-1]).all():
            raise ValueError("Rear delay must preserve increasing control knots")
        right = torch.searchsorted(rear_times, query, right=True).clamp(1, len(times) - 1)
        left = right - 1
        u = (query - rear_times[left]) / (rear_times[right] - rear_times[left])
        alpha = u**3 * (10 - 15 * u + 6 * u**2)
        tape[:, :, 6:12] = knots[:, left, 6:12] * (1 - alpha[None, :, None])
        tape[:, :, 6:12] += knots[:, right, 6:12] * alpha[None, :, None]
    return tape


def trajectory_metrics(states, target, dt=0.005):
    """Evaluate physical flight and the first contact after a sustained flight.

    Inputs are N x T tensors. Landing location is frozen at contact, so rolling
    to the target afterwards cannot earn landing accuracy. A missing landing is
    always rejected. All tolerances are explicit and saved with each candidate.
    """
    pos, vel, contacts = states["position"], states["velocity"], states["contacts"]
    n, steps, _ = pos.shape
    device = pos.device
    if isinstance(target, JumpTarget):
        goals = pos.new_tensor(list(asdict(target).values())).expand(n, -1)
    else:
        goals = torch.as_tensor(target, device=device, dtype=pos.dtype)
        if goals.shape != (n, 5):
            raise ValueError("Batched goals must be N x 5: height, x, y, height tolerance, XY tolerance")
        if not torch.isfinite(goals).all() or (goals[:, [0, 3, 4]] <= 0).any():
            raise ValueError("Invalid batched target values")
    height_goal, height_tolerance, landing_tolerance = goals[:, 0], goals[:, 3], goals[:, 4]
    yaw0 = states["yaw"][:, 0]
    relative = pos - pos[:, :1]
    x = relative[:, :, 0] * yaw0.cos()[:, None] + relative[:, :, 1] * yaw0.sin()[:, None]
    y = -relative[:, :, 0] * yaw0.sin()[:, None] + relative[:, :, 1] * yaw0.cos()[:, None]
    heading = states["yaw"] - yaw0[:, None]
    heading = torch.atan2(heading.sin(), heading.cos())
    all_air = ~contacts.any(-1)
    run = torch.zeros(n, device=device, dtype=torch.long)
    longest = run.clone()
    landing_index = torch.full_like(run, steps - 1)
    found = torch.zeros(n, device=device, dtype=torch.bool)
    for i in range(steps):
        landing_now = ~all_air[:, i] & (run * dt >= 0.08) & ~found
        landing_index = torch.where(landing_now, i, landing_index)
        found |= landing_now
        run = torch.where(all_air[:, i], run + 1, 0)
        longest = torch.maximum(longest, run)
    idx = torch.arange(n, device=device)
    touch_xy = torch.stack((x[idx, landing_index], y[idx, landing_index]), dim=-1)
    goal = goals[:, 1:3]
    xy_error = (touch_xy - goal).norm(dim=-1)
    time = torch.arange(steps, device=device)[None, :]
    pre_touch = time <= landing_index[:, None]
    rise = relative[:, :, 2].masked_fill(~pre_touch, -torch.inf).max(-1).values
    height_error = (rise - height_goal).abs()
    stable = (
        contacts.all(-1)
        & (vel[:, :, 2].abs() < 0.2)
        & (vel[:, :, :2].norm(dim=-1) < 0.3)
        & (states["tilt"] < 0.25)
        & (time > landing_index[:, None])
    )
    stable_run, stable_max = torch.zeros_like(run), torch.zeros_like(run)
    for i in range(steps):
        stable_run = torch.where(stable[:, i], stable_run + 1, 0)
        stable_max = torch.maximum(stable_max, stable_run)
    tilt = states["tilt"].max(-1).values
    yaw = heading.abs().max(-1).values
    nonfoot = states["nonfoot_force"].max(-1).values
    speed_violation = states["speed_violation"].max(-1).values
    limit_violation = states["limit_violation"].max(-1).values
    final_speed = vel[:, -1].norm(dim=-1)
    final_xy_error = (torch.stack((x[:, -1], y[:, -1]), -1) - goal).norm(dim=-1)
    initial_stable = contacts[:, 0].all(-1) & (vel[:, 0].norm(dim=-1) < 0.1) & (states["tilt"][:, 0] < 0.15)
    clearance = states["clearance"].max(-1).values
    terminal_stable = stable_run * dt
    score = (
        1500 * height_error.square()
        + 1000 * xy_error.square()
        + 80 * final_xy_error.square()
        + 10 * final_speed.square()
        + 40 * tilt.square()
        + 40 * yaw.square()
        + 500 * (nonfoot > 5).float()
        + 50 * speed_violation.square()
        + 5000 * limit_violation.square()
        + 60 * (~found).float()
        + 100 * (0.16 - longest * dt).clamp_min(0)
        + 40 * (0.12 - terminal_stable).clamp_min(0)
        + 50 * (~initial_stable).float()
    )
    # Do not trade a missed height/landing goal against a slightly smaller
    # attitude cost. These penalties enforce the same tolerances as acceptance.
    score += 100 * (height_error / height_tolerance - 1).clamp_min(0).square()
    score += 100 * (xy_error / landing_tolerance - 1).clamp_min(0).square()
    # Attitude violations must not be a cheap substitute for a feasible launch.
    # This changes search ranking only; the 0.35 rad acceptance limit is fixed.
    score += 1000 * (tilt / 0.35 - 1).clamp_min(0).square()
    score += 100 * (yaw / 0.15 - 1).clamp_min(0).square()
    score += 100 * (1 - terminal_stable / 0.1).clamp_min(0).square()
    accepted = (
        initial_stable
        & found
        & (longest * dt >= 0.16)
        & (height_error <= height_tolerance)
        & (xy_error <= landing_tolerance)
        & (tilt <= 0.35)
        & (yaw <= 0.15)
        & (nonfoot <= 5)
        & (speed_violation <= 0.05)
        & (limit_violation <= 0.002)
        & (terminal_stable >= 0.1 - 1e-6)
    )
    return {
        "score": score,
        "accepted": accepted,
        "height": rise,
        "height_error": height_error,
        "landing_xy": touch_xy,
        "landing_xy_error": xy_error,
        "landing_time": landing_index * dt,
        "landing_observed": found,
        "max_all_wheels_air_time": longest * dt,
        "peak_min_wheel_clearance": clearance,
        "max_tilt": tilt,
        "max_yaw_error": yaw,
        "max_nonfoot_contact_force": nonfoot,
        "max_joint_speed_excess": speed_violation,
        "max_joint_limit_violation": limit_violation,
        "stable_landing_duration": stable_max * dt,
        "terminal_stable_duration": terminal_stable,
        "final_speed": final_speed,
        "final_xy_error": final_xy_error,
        "initial_stable": initial_stable,
    }


def export_reference(replay):
    """Export actual independently replayed states, without retiming or IK.

    This is a candidate reference, not an automatic production replacement.
    The requested height/landing goal and their achieved values remain separate.
    """
    import math

    if not replay.get("accepted") or not replay.get("independent_replay"):
        raise ValueError("Only a physically accepted independent replay may be exported")
    if replay.get("policy_used") or replay.get("external_wrenches_used"):
        raise ValueError("Expected free-base open-loop actuator-only dynamics")
    s = replay["states"]
    if replay.get("actual_physics_steps") != len(s["position"]) - 1:
        raise ValueError("Physics steps and exported state timestamps disagree")
    dt = replay["simulation_dt"]
    tensors = {k: torch.tensor(v).unsqueeze(0) for k, v in s.items()}
    if replay.get("motion_mode") == "tilted_ramp":
        from tilted_ramp_to import TiltedRampTarget, tilted_ramp_metrics, validate_ramp_geometry

        ramp_target = TiltedRampTarget(**replay["target"])
        validate_ramp_geometry(replay.get("gap_geometry", {}), ramp_target)
        measured = tilted_ramp_metrics(tensors, ramp_target, dt)
    elif replay.get("motion_mode") == "gap_leap":
        from gap_leap_to import GapLeapTarget, gap_leap_metrics, validate_gap_geometry

        gap_target = GapLeapTarget(**replay["target"])
        validate_gap_geometry(replay.get("gap_geometry", {}), gap_target)
        measured = gap_leap_metrics(tensors, gap_target, dt)
    elif replay.get("motion_mode") == "parkour_leap":
        from parkour_leap_to import ParkourLeapTarget, parkour_leap_metrics

        measured = parkour_leap_metrics(tensors, ParkourLeapTarget(**replay["target"]), dt)
    elif replay.get("motion_mode") == "forward_leap":
        from forward_leap_to import forward_leap_metrics, load_forward_target

        measured = forward_leap_metrics(tensors, load_forward_target(replay["target"]), dt)
    else:
        measured = trajectory_metrics(tensors, JumpTarget(**replay["target"]), dt)
    if not measured["accepted"].item():
        raise ValueError("Stored acceptance disagrees with recomputed trajectory checks")
    landing = round(measured["landing_time"].item() / dt)
    takeoff = landing - 1
    while takeoff > 0 and not any(s["contacts"][takeoff - 1]):
        takeoff -= 1
    apex = max(range(takeoff, landing + 1), key=lambda i: s["position"][i][2])
    if not 0 < takeoff < apex < landing < len(s["position"]) - 1:
        raise ValueError("Invalid observed main-flight schedule")
    origin = s["position"][0]
    yaw0 = s["yaw"][0]
    c, sn = math.cos(yaw0), math.sin(yaw0)

    def rotate(v):
        return [c * v[0] + sn * v[1], -sn * v[0] + c * v[1], v[2]]

    orientations, pitches, pitch_rates = [], [], []
    cw, sz = math.cos(yaw0 / 2), -math.sin(yaw0 / 2)
    for q, omega in zip(s["quaternion"], s["omega"]):
        w, x, y, z = q
        q = [cw * w - sz * z, cw * x - sz * y, cw * y + sz * x, cw * z + sz * w]
        norm = math.sqrt(sum(v * v for v in q))
        q = [v / norm for v in q]
        if orientations and sum(a * b for a, b in zip(q, orientations[-1])) < 0:
            q = [-v for v in q]
        orientations.append(q)
        w, x, y, z = q
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        omega = rotate(omega)
        pitches.append(pitch)
        pitch_rates.append(-math.sin(yaw) * omega[0] + math.cos(yaw) * omega[1])
    return {
        "metadata": {
            "model": "isaac_lab_full_go2w_rigid_body_dynamics",
            "generator": "solve_jump_target_to.py --replay",
            "state_source": "independent_replay_actual_simulated_state",
            "joint_targets_used_for_reward": False,
            "coordinate_frame": "trigger_yaw; trigger XY is zero; absolute ground-relative Z",
            "command_target": replay["target"],
            **(
                {"gap_geometry_world": replay["gap_geometry"], "trigger_origin_world": origin}
                if replay.get("motion_mode") == "gap_leap"
                else {}
            ),
            **(
                {
                    "tilted_ramp_geometry_world": replay["gap_geometry"],
                    "trigger_origin_world": origin,
                    "contact_measure": "signed wheel-cylinder separation along the selected ramp normal",
                }
                if replay.get("motion_mode") == "tilted_ramp"
                else {}
            ),
            "motion_mode": replay.get("motion_mode", "point_jump"),
            "control_schedule": replay.get("control_schedule"),
            "parameterization": replay.get("parameterization", "legacy_70_mode_fixed_schedule"),
            "achieved_landing_xy": measured["landing_xy"][0].tolist(),
            "target_jump_height": measured["height"].item(),
            "target_takeoff_time": takeoff * dt,
            "target_apex_time": apex * dt,
            "target_landing_time": landing * dt,
            "flight_start_time": takeoff * dt,
            "flight_end_time": landing * dt,
            "mass": replay["mass_kg"],
            "gravity": replay["gravity"],
            "sequence_duration": (len(s["position"]) - 1) * dt,
            "time_step": dt,
            "leg_joint_names": replay["leg_joint_names"],
            "wheel_joint_names": replay["wheel_joint_names"],
            "actuator_contract": replay["actuator_contract"],
            "leg_effort_limit_nm": replay["leg_effort_limit_nm"],
            "leg_velocity_limit_rad_s": replay["leg_velocity_limit_rad_s"],
            "self_collisions_enabled": replay["self_collisions_enabled"],
            "control_candidate_sha256": replay["input_sha256"],
            "simulation_contract": {
                key: replay[key]
                for key in (
                    "seed",
                    "position_iterations",
                    "velocity_iterations",
                    "fixed_ground_mesh",
                    "balanced_warmup",
                    "warmup_seconds",
                    "guarded_warmup",
                    "canonical_origins",
                )
                if key in replay
            },
            "validation": {"success": True, "metrics": replay["metrics"], "robustness_certified": False},
        },
        "time": [i * dt for i in range(len(s["position"]))],
        "base_position": [rotate([p[0] - origin[0], p[1] - origin[1], p[2]]) for p in s["position"]],
        "base_linear_velocity": [rotate(v) for v in s["velocity"]],
        "base_orientation": orientations,
        "base_angular_velocity_world": [rotate(v) for v in s["omega"]],
        "base_pitch": pitches,
        "base_pitch_rate": pitch_rates,
        **{key: s[key] for key in ("joint_position", "joint_velocity", "wheel_position", "wheel_velocity")},
    }
