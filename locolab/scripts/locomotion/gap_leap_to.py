"""Physical-bank gap TO: terrain geometry and measured wheel-by-wheel crossing.

Unlike the flat-ground point command, width is an actual missing strip of deck.
Each wheel's first contact after the main flight must be on the far bank. A
front-only hop, rear return to the near bank, wall brace or pit-floor step fails.
"""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GapLeapTarget:
    width: float = 0.4
    near_x: float = 6.0
    depth: float = 1.0
    bank_half_width: float = 4.0
    bank_length: float = 12.0
    trigger_distance: float = 0.40
    forward_speed: float = 1.0
    speed_tolerance: float = 0.25
    edge_clearance: float = 0.025
    landing_margin: float = 0.03
    minimum_air_time: float = 0.08
    maximum_roll: float = 0.20
    maximum_pitch: float = 0.55
    maximum_yaw_error: float = 0.15
    maximum_landing_sink: float = 0.15
    lateral_corridor: float = 0.5
    recovery_contact_memory: float = 0.02
    recovery_ground_tolerance: float = 0.002
    criteria_version: int = 1

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Gap parameters must be finite and positive")
        if self.criteria_version != 1 or type(self.criteria_version) is not int:
            raise ValueError("Unsupported physical-gap criteria version")
        if not 0.30 <= self.trigger_distance <= 0.7 or self.near_x <= self.trigger_distance + 2:
            raise ValueError("Gap needs a safe grounded trigger and a rolling approach")
        if self.lateral_corridor >= self.bank_half_width or self.speed_tolerance >= self.forward_speed:
            raise ValueError("Invalid corridor or forward speed range")

    @property
    def far_x(self):
        return self.near_x + self.width

    @property
    def landing_y(self):
        return 0.0


def gap_mesh(target):
    """Two same-height solid banks and a lower pit floor; NEVER a top plane."""
    import trimesh

    parts = []
    for lo, hi, top, thickness, color in (
        (target.near_x - target.bank_length, target.near_x, 0, target.depth, [155, 165, 170, 255]),
        (target.far_x, target.far_x + target.bank_length, 0, target.depth, [125, 150, 135, 255]),
        (target.near_x, target.far_x, -target.depth, 0.10, [45, 48, 52, 255]),
    ):
        box = trimesh.creation.box(extents=(hi - lo, 2 * target.bank_half_width, thickness))
        box.apply_translation(((lo + hi) / 2, 0, top - thickness / 2))
        box.visual.vertex_colors = color
        parts.append(box)
    return trimesh.util.concatenate(parts)


def geometry_fingerprint(vertices, faces):
    """Hash actual USD float32 points and triangle indices, not rendered pixels."""
    return hashlib.sha256(
        np.asarray(vertices, dtype="<f4").tobytes() + np.asarray(faces, dtype="<i4").reshape(-1, 3).tobytes()
    ).hexdigest()


def validate_gap_geometry(certificate, target):
    mesh = gap_mesh(target)
    if (
        certificate.get("mesh_sha256") != geometry_fingerprint(mesh.vertices, mesh.faces)
        or certificate.get("target") != asdict(target)
        or certificate.get("collision_approximation") != "none"
        or not certificate.get("collision_enabled")
        or not certificate.get("no_extra_static_colliders")
    ):
        raise ValueError("Physical-gap collider geometry is missing or does not match the requested banks")


def gap_proposals(samples, low, high):
    """Control guesses for delayed rear landing and stronger grounded drive.

    There is no retiming of simulated states. Front/rear joint targets and
    durations remain bounded search variables; dynamics decide contact events.
    """
    result = samples.clone()
    count = result.shape[1]
    strength = torch.linspace(0, 1, count, device=result.device)[None, :, None]
    poses = result[..., :70].reshape(result.shape[0], count, 7, 10)
    # Keep rear legs tucked during front touchdown instead of dropping them
    # back onto the near bank / down into the hole as in a flat-ground leap.
    for common, difference, desired in ((3, 4, 1.1), (6, 7, -2.4)):
        front = poses[:, :, 2:5, common] + poses[:, :, 2:5, difference]
        rear = poses[:, :, 2:5, common] - poses[:, :, 2:5, difference]
        rear = rear * (1 - strength) + desired * strength
        poses[:, :, 2:5, common] = (front + rear) / 2
        poses[:, :, 2:5, difference] = (front - rear) / 2
    poses[:, :, :5, 9] += 6.0 * strength
    durations = result[..., 70:].reshape(result.shape[0], count, 2, 7)
    durations[:, :, 1, 3:5] += 0.06 * strength
    return result.clamp(low, high)


def gap_leap_metrics(states, target, dt=0.005):
    """All criteria use the physical bank coordinates and actual contact states."""
    pos, vel, contact = states["position"], states["velocity"], states["contacts"]
    wheels, bottom = states["wheel_center_position"], states["wheel_bottom_height"]
    extent = states["wheel_forward_radius"]
    n, steps, _ = pos.shape
    ids = torch.arange(n, device=pos.device)
    t = torch.arange(steps, device=pos.device)[None]
    x = wheels[..., 0]
    run = torch.zeros(n, dtype=torch.long, device=pos.device)
    takeoff, landing = run.clone(), torch.full_like(run, steps - 1)
    found = torch.zeros_like(run, dtype=torch.bool)
    all_air = ~contact.any(-1)
    for i in range(steps):
        now = ~all_air[:, i] & (run * dt >= target.minimum_air_time - 1e-6) & ~found
        takeoff = torch.where(now, i - run, takeoff)
        landing = torch.where(now, i, landing)
        found |= now
        run = torch.where(all_air[:, i], run + 1, 0)
    flight = (t >= takeoff[:, None]) & (t < landing[:, None]) & found[:, None]
    air_time = (landing - takeoff) * dt * found
    first_touch, touch_found, last_support = [], [], []
    for foot in range(4):
        before = contact[..., foot] & (t < takeoff[:, None])
        last_support.append(t.expand(n, -1).masked_fill(~before, 0).amax(-1))
        mask = contact[..., foot] & (t >= takeoff[:, None]) & found[:, None]
        first_touch.append(t.expand(n, -1).masked_fill(~mask, steps - 1).amin(-1))
        touch_found.append(mask.any(-1))
    first_touch = torch.stack(first_touch, -1)
    last_support = torch.stack(last_support, -1)
    touch_found = torch.stack(touch_found, -1)
    foot_ids = torch.arange(4, device=pos.device)[None]
    touch_x = x[ids[:, None], first_touch, foot_ids]
    touch_z = bottom[ids[:, None], first_touch, foot_ids]
    support_x = x[ids[:, None], last_support, foot_ids]
    support_z = bottom[ids[:, None], last_support, foot_ids]
    # Otherwise a robot could bridge/walk across first and perform a small hop
    # on the far bank afterwards. Every wheel must depart the NEAR bank for
    # this main flight, not merely end somewhere beyond the hole.
    near_departure = (support_x <= target.near_x + 0.005) & (support_z.abs() <= 0.02) & found[:, None]
    far_touch = touch_found & (touch_x >= target.far_x + target.landing_margin) & (touch_z.abs() <= 0.02)
    near_return = (contact & (x < target.near_x) & (t >= takeoff[:, None])[..., None] & found[:, None, None]).any(
        (1, 2)
    )
    over_gap = (x > target.near_x + 0.005) & (x < target.far_x - 0.005)
    gap_contact = (contact & over_gap).any((1, 2))
    clearance_deficit = (target.edge_clearance - bottom).clamp_min(0).masked_fill(~over_gap, 0).amax((1, 2))
    each_crossed = over_gap.any(1) & (x[:, -1] - extent[:, -1] >= target.far_x + target.landing_margin)
    # Brief rolling-force dropouts may be bridged only with wheel-bottom
    # geometry on the far deck, never with a wheel above the pit or its floor.
    recent = contact.clone()
    for lag in range(1, round(target.recovery_contact_memory / dt) + 1):
        recent[:, lag:] |= contact[:, :-lag]
    supported = recent & (bottom.abs() <= target.recovery_ground_tolerance)
    on_far = (x - extent >= target.far_x + target.landing_margin).all(-1)
    stable = (
        supported.all(-1)
        & on_far
        & ((vel[..., 0] - target.forward_speed).abs() <= target.speed_tolerance)
        & (vel[..., 1].abs() < 0.15)
        & (vel[..., 2].abs() < 0.2)
        & (states["tilt"] < 0.25)
        & (states["omega"].norm(dim=-1) < 1.0)
        & (t > landing[:, None])
        & found[:, None]
    )
    run = torch.zeros_like(run)
    for i in range(steps):
        run = torch.where(stable[:, i], run + 1, 0)
    terminal_duration = run * dt
    heading = states["yaw"] - states["yaw"][:, :1]
    yaw = torch.atan2(heading.sin(), heading.cos()).abs().amax(-1)
    roll, pitch = states["roll"].abs().amax(-1), states["pitch"].abs().amax(-1)
    nonfoot = states["nonfoot_force"].amax(-1)
    speed = states["speed_violation"].amax(-1)
    limit = states["limit_violation"].amax(-1)
    lateral = wheels[..., 1].abs().amax((1, 2))
    initial_stable = (
        contact[:, 0].all(-1)
        & (bottom[:, 0].abs() <= 0.01).all(-1)
        & (x[:, 0] + extent[:, 0] < target.near_x - 0.005).all(-1)
        & ((vel[:, 0, 0] - target.forward_speed).abs() <= target.speed_tolerance)
        & (vel[:, 0, 1:].norm(dim=-1) < 0.1)
        & (states["tilt"][:, 0] < 0.15)
    )
    after = (t >= landing[:, None]) & found[:, None]
    sink = (pos[:, :1, 2] - pos[..., 2]).clamp_min(0).masked_fill(~after, 0).amax(-1)
    height = (pos[..., 2] - pos[:, :1, 2]).masked_fill(t > landing[:, None], -torch.inf).amax(-1)
    air_distance = (pos[ids, landing, 0] - pos[ids, takeoff, 0]) * found
    # Continuous deficits give useful information even before any complete
    # crossing exists. In particular, late rear touchdown is scored per wheel.
    touch_shortfall = (target.far_x + target.landing_margin - touch_x).clamp_min(0)
    final_shortfall = (target.far_x + target.landing_margin - (x[:, -1] - extent[:, -1])).clamp_min(0)
    safety = torch.stack(
        ((nonfoot - 5).clamp_min(0) / 200, (speed - 0.05).clamp_min(0), (limit - 0.002).clamp_min(0) / 0.02), -1
    )
    violations = torch.stack(
        (
            (~initial_stable).float(),
            (~found).float(),
            (~near_departure).float().mean(-1),
            (support_x - target.near_x).clamp_min(0).amax(-1) / 0.15,
            (~touch_found).float().mean(-1),
            touch_shortfall.amax(-1) / 0.15,
            final_shortfall.amax(-1) / 0.15,
            (touch_z.abs() - 0.02).clamp_min(0).amax(-1) / 0.1,
            clearance_deficit / 0.1,
            gap_contact.float(),
            near_return.float(),
            (0.15 - terminal_duration).clamp_min(0) / 0.15,
            (roll / target.maximum_roll - 1).clamp_min(0),
            (pitch / target.maximum_pitch - 1).clamp_min(0),
            (yaw / target.maximum_yaw_error - 1).clamp_min(0),
            (lateral - target.lateral_corridor).clamp_min(0) / 0.2,
            (sink - target.maximum_landing_sink).clamp_min(0) / 0.1,
        ),
        -1,
    )
    accepted = (
        initial_stable
        & found
        & near_departure.all(-1)
        & far_touch.all(-1)
        & each_crossed.all(-1)
        & ~gap_contact
        & ~near_return
        & (clearance_deficit <= 1e-6)
        & (terminal_duration >= 0.15 - 1e-6)
        & (roll <= target.maximum_roll)
        & (pitch <= target.maximum_pitch)
        & (yaw <= target.maximum_yaw_error)
        & (lateral <= target.lateral_corridor)
        & (sink <= target.maximum_landing_sink)
        & (nonfoot <= 5)
        & (speed <= 0.05)
        & (limit <= 0.002)
    )
    # During search balance all infeasibilities so "never approach the gap"
    # does not dominate every near-success with a minor simulated edge touch.
    # Actual acceptance above ALWAYS requires all safety thresholds to pass.
    violations = torch.cat((violations, safety), -1)
    total = violations.square().sum(-1)
    score = 100 * total + 1000 * safety.square().sum(-1) + 10 * pitch.square()
    return dict(
        score=score,
        accepted=accepted,
        height=height,
        landing_xy=pos[ids, landing, :2] - pos[:, 0, :2],
        takeoff_time=takeoff * dt,
        landing_time=landing * dt,
        landing_observed=found,
        airborne_forward_distance=air_distance,
        peak_min_wheel_clearance=bottom.amin(-1).masked_fill(~flight, 0).amax(-1),
        max_all_wheels_air_time=air_time,
        first_touchdown_x=touch_x,
        last_support_x=support_x,
        near_bank_takeoff=near_departure,
        first_touchdown_bottom_height=touch_z,
        first_touchdown_time=first_touch * dt,
        far_bank_touchdown=far_touch,
        wheel_fully_crossed=each_crossed,
        near_bank_return=near_return,
        gap_contact=gap_contact,
        gap_clearance_deficit=clearance_deficit,
        terminal_stable_duration=terminal_duration,
        maximum_landing_sink=sink,
        initial_stable=initial_stable,
        initial_distance_to_edge=target.near_x - pos[:, 0, 0],
        initial_vx=vel[:, 0, 0],
        final_vx=vel[:, -1, 0],
        final_base_x=pos[:, -1, 0],
        final_rearmost_wheel_x=(x[:, -1] - extent[:, -1]).amin(-1),
        max_nonfoot_contact_force=nonfoot,
        max_joint_speed_excess=speed,
        max_joint_limit_violation=limit,
        max_roll=roll,
        max_pitch=pitch,
        max_yaw_error=yaw,
        maximum_lateral_excursion=lateral,
        physical_safety_violation=safety.square().sum(-1),
        search_safety_violation=torch.zeros_like(total),
        search_max_violation=violations.amax(-1),
        search_total_violation=total,
    )
