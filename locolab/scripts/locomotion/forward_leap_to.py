"""Moving-entry, moving-exit forward leap objectives, independent of RL.

The virtual crossing zone is a BASE-progress interval on flat ground. All four
wheel bottoms must be clear while the base crosses it. It prevents rolling to
the landing command, but is NOT a physical gap or a whole-body obstacle test.
"""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ForwardLeapTarget:
    height: float = 0.20
    landing_x: float = 0.60
    landing_y: float = 0.0
    height_tolerance: float = 0.02
    landing_tolerance: float = 0.05
    forward_speed: float = 1.0
    speed_tolerance: float = 0.25
    minimum_air_distance: float = 0.30
    minimum_air_time: float = 0.20
    crossing_start: float = 0.30
    crossing_end: float = 0.40
    crossing_clearance: float = 0.08
    maximum_roll: float = 0.20
    maximum_pitch: float = 0.55
    maximum_yaw_error: float = 0.15
    recovery_contact_memory: float = 0.02
    recovery_ground_tolerance: float = 0.002
    minimum_rear_push_delay: float = 0.01
    maximum_front_landing_delay: float = 0.02
    maximum_preparatory_airtime: float = 0.03
    criteria_version: int = 3

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Forward leap targets must be finite")
        positive = [v for k, v in asdict(self).items() if k != "landing_y"]
        if min(positive) <= 0 or not self.crossing_start < self.crossing_end < self.landing_x:
            raise ValueError("Expected positive limits and a crossing interval before touchdown")
        if self.speed_tolerance >= self.forward_speed or self.minimum_air_distance > self.landing_x:
            raise ValueError("Forward speed must stay positive; airborne distance must fit the landing goal")
        if type(self.criteria_version) is not int or self.criteria_version not in (1, 2, 3, 4):
            raise ValueError("Supported forward-leap acceptance versions are 1, 2, 3 and 4")


def load_forward_target(values):
    """Preserve the acceptance rules of frozen pre-versioned experiments.

    v1 used strict rolling contacts; v2 introduced geometric contact bridging;
    v3 added the fore/hind contact order. v4 explicitly separates flat-ground
    goal accuracy from style. Unversioned/new targets still default to v3.
    """
    values = dict(values)
    if "criteria_version" not in values:
        values["criteria_version"] = (
            3 if "minimum_rear_push_delay" in values else (2 if "recovery_contact_memory" in values else 1)
        )
    return ForwardLeapTarget(**values)


def forward_leap_metrics(states, target, dt=0.005):
    """First sustained flight only; do not reward a later hop or rolling endpoint."""
    pos, vel, contacts = (states[k] for k in ("position", "velocity", "contacts"))
    n, steps, _ = pos.shape
    ids = torch.arange(n, device=pos.device)
    t = torch.arange(steps, device=pos.device)[None]
    yaw0 = states["yaw"][:, 0]
    c, s = yaw0.cos()[:, None], yaw0.sin()[:, None]
    delta = pos - pos[:, :1]
    x, y = c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]
    vx, vy = c * vel[..., 0] + s * vel[..., 1], -s * vel[..., 0] + c * vel[..., 1]
    heading = states["yaw"] - yaw0[:, None]
    yaw = torch.atan2(heading.sin(), heading.cos()).abs().amax(-1)
    all_air = ~contacts.any(-1)
    run = torch.zeros(n, dtype=torch.long, device=pos.device)
    landing = torch.full_like(run, steps - 1)
    takeoff = torch.zeros_like(run)
    found = torch.zeros_like(run, dtype=torch.bool)
    for i in range(steps):
        now = ~all_air[:, i] & (run * dt >= 0.08) & ~found
        takeoff = torch.where(now, i - run, takeoff)
        landing = torch.where(now, i, landing)
        found |= now
        run = torch.where(all_air[:, i], run + 1, 0)
    flight = (t >= takeoff[:, None]) & (t < landing[:, None]) & found[:, None]
    air_time = (landing - takeoff) * dt * found
    air_distance = (x[ids, landing] - x[ids, takeoff]) * found
    # Parkour-like contact sequence: forelegs leave first, hindlegs finish the
    # push. On touchdown the forelegs must reach the ground no later than the
    # hindlegs (20 ms tolerance). Timing comes from actual contacts, not targets.
    last_support = []
    first_touch = []
    for group in (contacts[:, :, :2], contacts[:, :, 2:]):
        group_contact = group.any(-1)
        last_support.append(t.expand(n, -1).masked_fill(~group_contact | (t >= takeoff[:, None]), 0).amax(-1))
        first_touch.append(t.expand(n, -1).masked_fill(~group_contact | (t < landing[:, None]), steps - 1).amin(-1))
    rear_push_delay = (last_support[1] - last_support[0]) * dt
    front_landing_delay = (first_touch[0] - first_touch[1]) * dt
    pre_run = torch.zeros_like(run)
    pre_max = torch.zeros_like(run)
    for i in range(steps):
        pre_run = torch.where(all_air[:, i] & (i < takeoff), pre_run + 1, 0)
        pre_max = torch.maximum(pre_max, pre_run)
    preparatory_airtime = pre_max * dt
    touchdown = torch.stack((x[ids, landing], y[ids, landing]), -1)
    landing_goal = torch.stack(
        (
            torch.as_tensor(target.landing_x, device=pos.device, dtype=pos.dtype).expand(n),
            torch.as_tensor(target.landing_y, device=pos.device, dtype=pos.dtype).expand(n),
        ),
        -1,
    )
    landing_error = (touchdown - landing_goal).norm(dim=-1)
    height = delta[..., 2].masked_fill(t > landing[:, None], -torch.inf).amax(-1)
    height_error = (height - target.height).abs()
    entry_speed_error = (vx[:, 0] - target.forward_speed).abs()
    initial_stable = (
        contacts[:, 0].all(-1)
        & (entry_speed_error <= target.speed_tolerance)
        & (vy[:, 0].abs() < 0.1)
        & (vel[:, 0, 2].abs() < 0.1)
        & (states["tilt"][:, 0] < 0.15)
    )
    # Recovery may keep travelling beyond the touchdown goal; never penalize
    # distance from that goal after the first contact or demand zero XY speed.
    # Rolling contact-force samples can momentarily be zero even with the
    # wheel bottom on the plane. Bridge <=20 ms only when geometry confirms
    # <=2 mm clearance; never count an elevated, unforced wheel as support.
    recent_contact = contacts.clone()
    for lag in range(1, round(target.recovery_contact_memory / dt) + 1):
        recent_contact[:, lag:] |= contacts[:, :-lag]
    supported = recent_contact & (states["wheel_bottom_height"] <= target.recovery_ground_tolerance)
    if target.criteria_version == 1:
        supported = contacts
    stable = (
        supported.all(-1)
        & ((vx - target.forward_speed).abs() <= target.speed_tolerance)
        & (vy.abs() < 0.15)
        & (vel[..., 2].abs() < 0.2)
        & (states["tilt"] < 0.25)
        & (states["omega"].norm(dim=-1) < 1.0)
        & (t > landing[:, None])
    )
    run = torch.zeros_like(run)
    for i in range(steps):
        run = torch.where(stable[:, i], run + 1, 0)
    terminal_duration = run * dt
    exit_error = (vx[:, -1] - target.forward_speed).abs()
    roll, pitch = states["roll"].abs().amax(-1), states["pitch"].abs().amax(-1)
    nonfoot = states["nonfoot_force"].amax(-1)
    overspeed = states["speed_violation"].amax(-1)
    overlimit = states["limit_violation"].amax(-1)
    # A sampled body-progress interval, not a declared traversable gap width.
    crossing_start = torch.as_tensor(target.crossing_start, device=pos.device, dtype=pos.dtype).expand(n)
    crossing_end = torch.as_tensor(target.crossing_end, device=pos.device, dtype=pos.dtype).expand(n)
    crossing = (x >= crossing_start[:, None]) & (x <= crossing_end[:, None])
    crossed = (x[:, 0] < crossing_start) & (x[:, -1] > crossing_end) & crossing.any(-1)
    crossing_deficit = (target.crossing_clearance - states["clearance"]).clamp_min(0)
    crossing_deficit = crossing_deficit.masked_fill(~crossing, 0).amax(-1)
    crossing_supported = (crossing & ~all_air).any(-1)
    crossing_in_main_flight = (~crossing | flight).all(-1)
    clearance = states["clearance"].masked_fill(~flight, 0).amax(-1)
    # Continuous gradients for CEM selection; acceptance below remains explicit.
    score = (
        1200 * height_error.square()
        + 1000 * landing_error.square()
        + 100 * (height_error / target.height_tolerance - 1).clamp_min(0).square()
        + 100 * (landing_error / target.landing_tolerance - 1).clamp_min(0).square()
        + 500 * (target.minimum_air_distance - air_distance).clamp_min(0).square()
        + 500 * (target.minimum_air_time - air_time).clamp_min(0).square()
        + 60 * (crossing_supported | ~crossing_in_main_flight | ~crossed).float()
        + 100 * entry_speed_error.square()
        + 100 * exit_error.square()
        + 40 * vy[:, -1].square()
        + 40 * vel[:, -1, 2].square()
        + 100 * (0.15 - terminal_duration).clamp_min(0) / 0.15
        + 100 * (~found).float()
        + 100 * (~initial_stable).float()
        + 1000 * (roll / target.maximum_roll - 1).clamp_min(0).square()
        + 1000 * (pitch / target.maximum_pitch - 1).clamp_min(0).square()
        + 1000 * (yaw / target.maximum_yaw_error - 1).clamp_min(0).square()
        + 10 * roll.square()
        + 10 * yaw.square()
        + 500 * (nonfoot > 5).float()
        + 50 * overspeed.square()
        + 5000 * overlimit.square()
    )
    if target.criteria_version >= 2:
        score += 100 * (crossing_deficit / 0.04).square()
        score += 100 * ((target.minimum_air_distance - air_distance).clamp_min(0) / 0.05).square()
    else:
        score += 500 * crossing_deficit.square()
    if target.criteria_version >= 3:
        score += 100 * ((target.minimum_rear_push_delay - rear_push_delay).clamp_min(0) / 0.04).square()
        score += 100 * ((front_landing_delay - target.maximum_front_landing_delay).clamp_min(0) / 0.04).square()
        score += 100 * ((preparatory_airtime - target.maximum_preparatory_airtime).clamp_min(0) / 0.02).square()
    accepted = (
        initial_stable
        & found
        & (height_error <= target.height_tolerance)
        & (landing_error <= target.landing_tolerance)
        & (air_time >= target.minimum_air_time)
        & (roll <= target.maximum_roll)
        & (pitch <= target.maximum_pitch)
        & (yaw <= target.maximum_yaw_error)
        & (nonfoot <= 5)
        & (overspeed <= 0.05)
        & (overlimit <= 0.002)
        & (terminal_duration >= 0.15 - 1e-6)
    )
    if target.criteria_version < 4:
        accepted &= (
            crossed
            & ~crossing_supported
            & crossing_in_main_flight
            & (crossing_deficit <= 1e-6)
            & (air_distance >= target.minimum_air_distance)
        )
    else:
        # A point command includes grounded preload displacement. A virtual
        # crossing interval / fixed airborne fraction is style, not a real gap.
        # Still require actual simultaneous wheel clearance during main flight.
        accepted &= clearance >= target.crossing_clearance
    if target.criteria_version >= 3:
        accepted &= preparatory_airtime <= target.maximum_preparatory_airtime + 1e-6
        if target.criteria_version == 3:
            accepted &= (rear_push_delay >= target.minimum_rear_push_delay - 1e-6) & (
                front_landing_delay <= target.maximum_front_landing_delay + 1e-6
            )
    return dict(
        score=score,
        accepted=accepted,
        height=height,
        height_error=height_error,
        landing_xy=touchdown,
        landing_xy_error=landing_error,
        landing_time=landing * dt,
        takeoff_time=takeoff * dt,
        landing_observed=found,
        airborne_forward_distance=air_distance,
        max_all_wheels_air_time=air_time,
        peak_min_wheel_clearance=clearance,
        initial_vx=vx[:, 0],
        final_vx=vx[:, -1],
        touchdown_vx=vx[ids, landing],
        initial_stable=initial_stable,
        terminal_stable_duration=terminal_duration,
        max_roll=roll,
        max_pitch=pitch,
        max_yaw_error=yaw,
        max_nonfoot_contact_force=nonfoot,
        max_joint_speed_excess=overspeed,
        max_joint_limit_violation=overlimit,
        crossing_observed=crossed,
        crossing_clearance_deficit=crossing_deficit,
        crossing_supported=crossing_supported,
        crossing_in_main_flight=crossing_in_main_flight,
        rear_push_delay=rear_push_delay,
        front_landing_delay=front_landing_delay,
        preparatory_airtime=preparatory_airtime,
        final_support_contact_fraction=contacts[:, -round(0.15 / dt) :].float().mean(1),
        final_max_wheel_bottom_height=states["wheel_bottom_height"][:, -round(0.15 / dt) :].amax((1, 2)),
    )
