"""Versioned time-and-shape direct-shooting TO for a rolling forward leap.

Variables are 70 actuator-target modes plus 14 positive segment durations:
seven independently optimized intervals for each leg pair. These are spline
landmarks, NOT prescribed contact events or a policy command state machine.
Actual takeoff, apex and touchdown are measured from free-base dynamics.
"""

import math
from dataclasses import asdict, dataclass

import torch
from forward_leap_to import ForwardLeapTarget, forward_leap_metrics
from jump_target_to import initial_distribution, parameter_bounds

PARAMETER_COUNT = 84
DURATION = 1.8
LANDMARKS = ("start", "preload", "drive", "fold", "reach", "cushion", "recover", "settle", "end")
DEFAULT_FRONT_INTERVALS = (0.10, 0.09, 0.13, 0.14, 0.19, 0.20, 0.35)
DEFAULT_REAR_INTERVALS = (0.12, 0.10, 0.12, 0.16, 0.18, 0.19, 0.33)
INTERVAL_LOWER = (0.035, 0.025, 0.04, 0.07, 0.08, 0.12, 0.14)
INTERVAL_UPPER = (0.16, 0.13, 0.18, 0.22, 0.22, 0.30, 0.40)


@dataclass(frozen=True)
class ParkourLeapTarget(ForwardLeapTarget):
    minimum_air_distance: float = 0.40
    minimum_takeoff_com_vx: float = 0.80
    maximum_landing_sink: float = 0.10
    maximum_landing_pitch: float = 0.30
    maximum_rebound_airtime: float = 0.04
    minimum_front_reach: float = 0.15
    minimum_landing_leg_ratio: float = 0.75
    maximum_apex_leg_ratio: float = 0.88
    minimum_push_leg_ratio: float = 0.85
    foot_path_tolerance: float = 0.06

    def __post_init__(self):
        super().__post_init__()
        if self.criteria_version not in (3, 4):
            raise ValueError("Parkour requires current contact criteria (3) or explicit goal-first criteria (4)")
        ratios = (self.minimum_landing_leg_ratio, self.maximum_apex_leg_ratio, self.minimum_push_leg_ratio)
        if not all(0 < r <= 1 for r in ratios):
            raise ValueError("Leg-length ratios must be between zero and one")


def command_target(**values):
    """Scale spatial constraints, never scale already-simulated trajectories.

    Height remains base rise; XY remains trigger-to-first-touchdown displacement.
    The nominal x=0.6 case retains its original 0.4 m airborne requirement and
    [0.3,0.4] m virtual BASE crossing interval. Physics/attitude limits stay fixed.
    """
    x = values.get("landing_x", 0.60)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("Parkour forward displacement must be positive and finite")
    values.setdefault("minimum_air_distance", x * 2 / 3)
    values.setdefault("crossing_start", x / 2)
    values.setdefault("crossing_end", x * 2 / 3)
    return ParkourLeapTarget(**values)


class BatchedParkourTargets:
    """GPU-vectorized goals with common dynamics/style limits, no cross-talk."""

    variable_fields = frozenset(
        ("height", "landing_x", "landing_y", "minimum_air_distance", "crossing_start", "crossing_end")
    )

    def __init__(self, goals, repeats=1, device="cpu"):
        if not goals or repeats < 1:
            raise ValueError("Batched targets need at least one goal and one repeat")
        rows = [asdict(goal) for goal in goals]
        for name, value in rows[0].items():
            if name in self.variable_fields:
                tensor = torch.tensor([row[name] for row in rows], device=device, dtype=torch.float32)
                setattr(self, name, tensor.repeat_interleave(repeats))
            else:
                if any(row[name] != value for row in rows[1:]):
                    raise ValueError(f"All goals must share {name}; split batches to change dynamics/style limits")
                setattr(self, name, value)


def parameter_distribution(device="cpu", speed=1.0, seed=None, lateral_goal=0.0):
    """Fresh interpretable pose seed, or explicitly upgrade a 70-D control seed.

    Old simulated states/certificates are never reused. Legacy landing poses
    are not copied: their folded-under landing was the reason for this change.
    """
    mean, std = initial_distribution(device)
    low, high = parameter_bounds(device)
    p = mean.reshape(7, 10)
    if seed is not None and seed.get("motion_mode") in ("parkour_leap", "gap_leap", "tilted_ramp"):
        mean = torch.tensor(seed["parameters"], device=device)
        if mean.shape != (PARAMETER_COUNT,):
            raise ValueError("Parkour v2 candidates require 84 parameters")
    else:
        if seed is not None:
            old = torch.as_tensor(seed["parameters"], device=device)
            if old.shape != (70,):
                raise ValueError("Legacy control seeds require 70 parameters")
            p[:2] = old.reshape(7, 10)[:2]
        # Front/rear pose differences remain search variables. Landing reach,
        # compression and recovery now have separate controllable landmarks.
        p[2:, :] = 0
        front_thigh = p.new_tensor([0.90, 0.40, 0.85, 0.78, 0.78])
        rear_thigh = p.new_tensor([0.95, 0.80, 0.85, 0.78, 0.78])
        front_calf = p.new_tensor([-2.30, -1.25, -1.85, -1.30, -1.30])
        rear_calf = p.new_tensor([-2.30, -1.90, -1.85, -1.30, -1.30])
        for values_f, values_r, common, difference in ((front_thigh, rear_thigh, 3, 4), (front_calf, rear_calf, 6, 7)):
            p[2:, common] = (values_f + values_r) / 2
            p[2:, difference] = (values_f - values_r) / 2
        p[:, [0, 2, 5, 8]] = 0
        p[:, 9] = speed / 0.086
        mean = torch.cat((mean, mean.new_tensor(DEFAULT_FRONT_INTERVALS + DEFAULT_REAR_INTERVALS)))
    low.reshape(7, 10)[:, 9] = -25
    high.reshape(7, 10)[:, 9] = 25
    low = torch.cat((low, low.new_tensor(INTERVAL_LOWER * 2)))
    high = torch.cat((high, high.new_tensor(INTERVAL_UPPER * 2)))
    std = torch.cat((std, std.new_tensor([0.025, 0.02, 0.03, 0.035, 0.035, 0.04, 0.05] * 2)))
    mask = torch.ones(PARAMETER_COUNT, dtype=torch.bool, device=device)
    if not math.isfinite(lateral_goal):
        raise ValueError("Lateral goal must be finite")
    if abs(lateral_goal) < 1e-8:
        mask[:70].reshape(7, 10)[:, [0, 2, 5, 8]] = False
        mean[:70].reshape(7, 10)[:, [0, 2, 5, 8]] = 0
    else:
        # Hip common/FB, thigh LR and calf LR enable genuine lateral impulses.
        # Only a control initial guess is mirrored, never states or footage.
        source_y = seed.get("target", {}).get("landing_y", 0.0) if seed else 0.0
        if seed and seed.get("motion_mode") == "tilted_ramp":
            source_y = seed["target"]["side"] * seed["target"]["offset_y"]
        p = mean[:70].reshape(7, 10)
        if source_y * lateral_goal < 0:
            p[:, [0, 2, 5, 8]] *= -1
        elif abs(source_y) < 1e-8:
            p[:2, 0] = -math.copysign(0.08, lateral_goal)
    return mean.clamp(low, high), std, low, high, mask


def knot_schedule(parameters):
    if parameters.ndim != 2 or parameters.shape[-1] != PARAMETER_COUNT:
        raise ValueError("Expected N x 84 control-and-timing parameters")
    intervals = parameters[:, 70:].reshape(-1, 2, 7)
    if not torch.isfinite(parameters).all() or (intervals <= 0).any():
        raise ValueError("Parameters must be finite and segment durations positive")
    cumulative = intervals.cumsum(-1)
    if (cumulative[..., -1] >= DURATION - 0.05).any():
        raise ValueError("Must leave at least 50 ms for the final segment")
    return torch.cat(
        (torch.zeros_like(cumulative[..., :1]), cumulative, torch.full_like(cumulative[..., :1], DURATION)), -1
    )


def launch_proposals(samples, goals, low, high):
    """Explore earlier front unloading / later hind push, without retiming states.

    Only control proposals change. Reach/cushion timestamps are kept where
    possible; actual contact timing and joint speeds still decide feasibility.
    The remainder of the population keeps unconstrained timing exploration.
    """
    proposals = samples.clone()
    count = samples.shape[1]
    phase = torch.linspace(0, 1, count, device=samples.device)
    for g, goal in enumerate(goals):
        poses = proposals[g, :, :70].reshape(count, 7, 10)
        # A timing-only proposal can preserve the old front-leg stomp. Explore
        # partial front extension too, while leaving the rear drive unchanged.
        # Common +/- FB modes address the front pair without changing the rear.
        poses[:, 1, 3] += 0.4 * phase
        poses[:, 1, 4] += 0.4 * phase
        poses[:, 1, 6] -= 0.6 * phase
        poses[:, 1, 7] -= 0.6 * phase
        intervals = proposals[g, :, 70:].reshape(count, 2, 7)
        reach_time = intervals[:, :, :4].sum(-1)
        budget_seconds = (goal.landing_x - goal.minimum_air_distance) / goal.forward_speed
        # Leave a 40 ms actuator-response allowance, not a claimed flight event.
        push_budget = max(0.06, budget_seconds - 0.04)
        scale = (push_budget / intervals[:, :, :2].sum(-1).amax(-1)).clamp(0.5, 1.0)
        intervals[:, :, :2] *= scale[:, None, None]
        intervals[:, 0, 1] -= 0.015 + 0.035 * phase
        intervals[:, 1, 1] += 0.025 * phase
        intervals[:, 0, 2] *= 0.65 + 0.25 * phase
        intervals[:] = intervals.clamp(low[70:].reshape(2, 7), high[70:].reshape(2, 7))
        intervals[:, :, 3] = reach_time - intervals[:, :, :3].sum(-1)
    return proposals.clamp(low, high)


def sparse_control_proposals(anchor, active, initial_std, count, iteration, low, high):
    """Perturb a few actuator/timing variables; preserve the other control landmarks."""
    if anchor.ndim != 2 or anchor.shape[-1] != PARAMETER_COUNT or count < 4:
        raise ValueError("Expected G x 84 anchors and at least four proposals")
    output = anchor[:, None].expand(-1, count, -1).clone()
    singles = count // 2
    rows = torch.arange(singles, device=anchor.device)
    j = torch.arange(count - singles, device=anchor.device)
    scales = anchor.new_tensor([0.15, 0.35, 0.70])
    for g in range(len(anchor)):
        columns = active[g].nonzero(as_tuple=False).flatten()
        if not len(columns):
            raise ValueError("A local search needs active control variables")
        chosen = columns[(rows // 2 + iteration * max(1, singles // 2)) % len(columns)]
        delta = (2 * (rows % 2) - 1) * scales[(rows // 2 + iteration) % 3] * initial_std[g, chosen]
        output[g, rows, chosen] += delta

        pose = j[j % 3 == 0]
        common = ((pose // 6 + iteration) % 7) * 10 + torch.where((pose // 3) % 2 == 0, 3, 6)
        delta = (2 * (pose % 2) - 1) * scales[(pose // 2 + iteration) % 3] * initial_std[g, common]
        side = 2 * ((pose // 2) % 2) - 1
        output[g, singles + pose, common] += delta / 2
        output[g, singles + pose, common + 1] += side * delta / 2

        timing = j[j % 3 == 1]
        column = 70 + ((timing // 3) % 2) * 7 + (timing // 6 + iteration) % 6
        delta = (2 * (timing % 2) - 1) * scales[(timing // 2 + iteration) % 3] * initial_std[g, column]
        output[g, singles + timing, column] += delta
        output[g, singles + timing, column + 1] -= delta

        # Opposite drive-end shifts preserve fold times where bounds permit.
        launch = j[j % 3 == 2]
        delta = (2 * (launch % 2) - 1) * anchor.new_tensor([0.0025, 0.005, 0.01, 0.02])[(launch // 6 + iteration) % 4]
        output[g, singles + launch, 71] -= delta
        output[g, singles + launch, 72] += delta
        output[g, singles + launch, 78] += delta
        output[g, singles + launch, 79] -= delta
    output = torch.where(active[:, None], output, anchor[:, None])
    return output.clamp(low, high)


def search_priority(metrics, replicas=1):
    """Lexicographic keys; every replica must pass, no scalar safety trade-off.

    Historical metrics retain accepted-first/score ordering. New search keys
    never alter the versioned acceptance test or the historical score.
    """
    accepted = metrics["accepted"].reshape(-1, replicas).all(-1)
    columns = [(~accepted).float()]
    fields = ("search_safety_violation", "search_max_violation", "search_total_violation", "score")
    if "search_target_violation" in metrics:
        # v4: never sacrifice flight/safety or goal tolerance for contact order
        # and tucking. Inside tolerance, style has a bounded, lower-weight cost.
        fields = (
            "search_safety_violation",
            "search_motion_violation",
            "search_target_violation",
            "search_target_style_cost",
            "score",
        )
    elif "performance_guard_violation" in metrics:
        fields = (
            "search_safety_violation",
            "performance_guard_violation",
            "performance_loss",
            "search_total_violation",
        )
    for key in fields:
        value = metrics.get(key, torch.zeros_like(metrics["score"]))
        value = torch.nan_to_num(value, nan=1e9, posinf=1e9, neginf=1e9)
        columns.append(value.reshape(-1, replicas).amax(-1))
    return torch.stack(columns, -1)


def performance_refinement_metrics(metrics, target, baseline):
    """Improve height AND true air distance without sacrificing the approved seed.

    This is an opt-in search objective, never a replacement acceptance test.
    Baselines are independently replayed measurements, fixed throughout search.
    Tolerances allow small contact-sampling differences, not a new motion style.
    """
    result = dict(metrics)
    template = metrics["height"]

    def ref(key):
        return torch.as_tensor(baseline[key], device=template.device, dtype=template.dtype).expand_as(template)

    def above(key, bound, scale):
        return (metrics[key] - bound).clamp_min(0) / scale

    def below(key, bound, scale):
        return (bound - metrics[key]).clamp_min(0) / scale

    guards = [
        (~metrics["landing_observed"]).float(),
        (~metrics["initial_stable"]).float(),
        below("height", ref("height") - 0.002, 0.02),
        below("airborne_forward_distance", ref("airborne_forward_distance") - 0.003, 0.05),
        above("height", target.height + target.height_tolerance, 0.02),
        above("preflight_forward_distance", ref("preflight_forward_distance") + 0.01, 0.05),
        above("landing_xy_error", ref("landing_xy_error").clamp_min(target.landing_tolerance) + 0.005, 0.05),
        above("maximum_landing_sink", ref("maximum_landing_sink") + 0.01, 0.05),
        above("maximum_landing_pitch", ref("maximum_landing_pitch") + 0.03, 0.15),
        above("max_roll", (ref("max_roll") + 0.03).clamp_max(target.maximum_roll), 0.1),
        above("max_pitch", (ref("max_pitch") + 0.04).clamp_max(target.maximum_pitch), 0.15),
        above("max_yaw_error", (ref("max_yaw_error") + 0.03).clamp_max(target.maximum_yaw_error), 0.1),
        below("rear_push_delay", ref("rear_push_delay") - 0.005, 0.04),
        above("front_landing_delay", ref("front_landing_delay").clamp_min(target.maximum_front_landing_delay), 0.04),
        above("preparatory_airtime", ref("preparatory_airtime").clamp_min(target.maximum_preparatory_airtime), 0.02),
        above("foot_path_rms_error", ref("foot_path_rms_error") + 0.005, 0.03),
        above("maximum_apex_leg_ratio", ref("maximum_apex_leg_ratio") + 0.03, 0.15),
        below("minimum_landing_leg_ratio", ref("minimum_landing_leg_ratio") - 0.03, 0.15),
        below("minimum_front_touchdown_reach", ref("minimum_front_touchdown_reach") - 0.015, 0.05),
        below("takeoff_com_vx", ref("takeoff_com_vx") - 0.02, 0.2),
        below("terminal_stable_duration", ref("terminal_stable_duration").clamp_max(0.15), 0.15),
        above(
            "maximum_rebound_airtime", ref("maximum_rebound_airtime").clamp_min(target.maximum_rebound_airtime), 0.04
        ),
        above("crossing_clearance_deficit", ref("crossing_clearance_deficit") + 0.005, 0.04),
        above("lateral_air_distance_deficit", ref("lateral_air_distance_deficit") + 0.005, 0.05),
    ]
    height_shortfall = (target.height - metrics["height"]).clamp_min(0) / 0.02
    air_shortfall = (target.minimum_air_distance - metrics["airborne_forward_distance"]).clamp_min(0) / 0.05
    result.update(
        performance_guard_violation=torch.stack(guards, -1).square().sum(-1),
        performance_height_shortfall=height_shortfall,
        performance_air_distance_shortfall=air_shortfall,
        performance_loss=height_shortfall.square()
        + air_shortfall.square()
        + 0.05 * (metrics["landing_xy_error"] / target.landing_tolerance).square(),
    )
    return result


def performance_proposals(samples, anchor, low, high):
    """Small launch-local changes; leave recovery controls and times unchanged.

    Avoid the v4 large front-flexion bias that removed useful takeoff impulse.
    Half the proposals vary only launch; the remainder can adapt landing too.
    No actuator gains, force limits, reset states or simulated paths are scaled.
    """
    proposals = samples.clone()
    count = samples.shape[1] // 2
    if count:
        p = proposals[:, :count]
        p[:, :, 30:70] = anchor[:, None, 30:70]
        durations = p[:, :, 70:].reshape(len(samples), count, 2, 7)
        original = anchor[:, 70:].reshape(len(samples), 2, 7)
        durations[..., 3:] = original[:, None, :, 3:]
        # Keep each pair's fold landmark fixed while searching preload/drive
        # timing. This preserves the seed's subsequent reach/cushion schedule.
        durations[..., 2] = original[:, None, :, :3].sum(-1) - durations[..., :2].sum(-1)
    return proposals.clamp(low, high)


def priority_order(keys):
    """Stable lexicographic GPU sort; no fragile weighted big-M encoding."""
    order = torch.arange(len(keys), device=keys.device)
    for column in reversed(range(keys.shape[-1])):
        order = order[keys[order, column].argsort(stable=True)]
    return order


def candidate_priority(candidate):
    """Use the solver's robust key also when choosing batch vs refinement."""
    if candidate.get("search_priority") is not None:
        return tuple(candidate["search_priority"])
    metrics = {k: torch.as_tensor(v).unsqueeze(0) for k, v in candidate["metrics"].items()}
    return tuple(search_priority(metrics)[0].tolist())


def monotone_spline(times, values, query):
    """Batched C1 cubic with no overshoot and no artificial stop at every knot.

    End tangents are zero for continuity with the warmup/recovered hold. Only
    genuine turning points have zero interior tangents, unlike stop-to-stop
    quintic interpolation. Physical joint limits still bound every target.
    """
    h = times.diff(dim=-1)
    secant = values.diff(dim=1) / h[..., None]
    left, right = secant[:, :-1], secant[:, 1:]
    same_sign = left * right > 0
    # Weighted harmonic mean; safe denominators avoid NaNs in zero segments.
    w1, w2 = 2 * h[:, 1:] + h[:, :-1], h[:, 1:] + 2 * h[:, :-1]
    safe_left, safe_right = torch.where(same_sign, left, 1), torch.where(same_sign, right, 1)
    interior = (w1 + w2)[..., None] / (w1[..., None] / safe_left + w2[..., None] / safe_right)
    slope = torch.zeros_like(values)
    slope[:, 1:-1] = torch.where(same_sign, interior, 0)
    q = query.expand(len(times), -1).contiguous()
    index = torch.searchsorted(times.contiguous(), q, right=True).clamp(1, times.shape[1] - 1)
    batch = torch.arange(len(times), device=times.device)[:, None]
    start, end = times[batch, index - 1], times[batch, index]
    width = end - start
    u = ((q - start) / width).clamp(0, 1)[..., None]
    a, b = values[batch, index - 1], values[batch, index]
    da, db = slope[batch, index - 1] * width[..., None], slope[batch, index] * width[..., None]
    return (2 * u**3 - 3 * u**2 + 1) * a + (u**3 - 2 * u**2 + u) * da + (-2 * u**3 + 3 * u**2) * b + (u**3 - u**2) * db


def parkour_control_tape(parameters, default_position, lower_limits, upper_limits, dt=0.02, speed=1.0, end_speed=None):
    times = knot_schedule(parameters)
    p = parameters[:, :70].reshape(-1, 7, 10)
    knots = p.new_zeros(len(p), 9, 16)
    fb, lr = p.new_tensor([1, 1, -1, -1]), p.new_tensor([-1, 1, -1, 1])
    for joint, mode in enumerate((0, 3, 6)):
        a, b = (lr, fb) if joint == 0 else (fb, lr)
        knots[:, 1:8, joint:12:3] = p[..., mode, None] + p[..., mode + 1, None] * a + p[..., mode + 2, None] * b
    knots[:, 1:8, 12:] = p[..., 9, None]
    knots[:, 0, :12] = default_position
    knots[:, -1, :12] = default_position
    knots[:, (0, -1), 12:] = speed / 0.086
    if end_speed is not None:
        if not math.isfinite(end_speed):
            raise ValueError("End speed must be finite")
        knots[:, -1, 12:] = end_speed / 0.086
    knots[:, :, :12] = knots[:, :, :12].clamp(lower_limits + 0.02, upper_limits - 0.02)
    query = torch.arange(round(DURATION / dt), device=p.device, dtype=p.dtype)[None] * dt
    front = monotone_spline(times[:, 0], knots[:, :, :6], query)
    rear = monotone_spline(times[:, 1], knots[:, :, 6:12], query)
    wheel = monotone_spline(times.mean(1), knots[:, :, 12:], query)
    return torch.cat((front, rear, wheel), -1)


def parkour_leap_metrics(states, target, dt=0.005):
    """Task-space paths and full landing transient, measured from actual states."""
    result = forward_leap_metrics(states, target, dt)
    pos, feet, hips = states["position"], states["wheel_center_position"], states["hip_position"]
    n, steps, _ = pos.shape
    ids = torch.arange(n, device=pos.device)
    time = torch.arange(steps, device=pos.device)[None]
    takeoff = (result["takeoff_time"] / dt).round().long()
    landing = (result["landing_time"] / dt).round().long()
    valid = result["landing_observed"]
    in_flight = (time >= takeoff[:, None]) & (time < landing[:, None]) & valid[:, None]
    apex = pos[..., 2].masked_fill(~in_flight, -torch.inf).argmax(-1)
    # Lengths measured from each hip, so torso pitch alone cannot fake tucking.
    length = (feet - hips).norm(dim=-1)
    initial_length = length[:, 0].clamp_min(0.1)
    ratio = length / initial_length[:, None]
    apex_ratio = ratio[ids, apex].amax(-1)
    push_window = (time >= (takeoff - round(0.06 / dt)).clamp_min(0)[:, None]) & (time < takeoff[:, None])
    push_ratio = ratio.masked_fill(~push_window[..., None], 0).amax(1).amin(-1)
    # Each forewheel must reach forward at its OWN first contact, not one
    # convenient sample where only one foot is well positioned.
    contact = states["contacts"]
    foot_touch = []
    for foot in range(4):
        mask = contact[..., foot] & (time >= landing[:, None])
        foot_touch.append(time.expand(n, -1).masked_fill(~mask, steps - 1).amin(-1))
    foot_touch = torch.stack(foot_touch, -1)
    foot_ids = torch.arange(4, device=pos.device)[None]
    landing_ratios = ratio[ids[:, None], foot_touch, foot_ids]
    touchdown_relative = feet[ids[:, None], foot_touch, foot_ids] - pos[ids[:, None], foot_touch]
    yaw0 = states["yaw"][:, 0]
    reach = touchdown_relative[..., 0] * yaw0.cos()[:, None] + touchdown_relative[..., 1] * yaw0.sin()[:, None]
    front_reach = reach[:, :2].amin(-1)
    landing_leg_ratio = landing_ratios.amin(-1)
    # Continuous mid-flight contraction/late-flight extension envelope. Its
    # phase is measured from actual contact events, never guessed knot times.
    phase = ((time - takeoff[:, None]) / (landing - takeoff).clamp_min(1)[:, None]).clamp(0, 1)
    fold = torch.sin(torch.pi * phase).square()
    desired_ratio = 1.0 - 0.25 * fold
    leg_path_error = ((ratio - desired_ratio[..., None]).square().mean(-1) * in_flight).sum(-1)
    leg_path_error = (leg_path_error / in_flight.sum(-1).clamp_min(1)).sqrt() * initial_length.mean(-1)
    relative = feet - pos[:, :, None]
    forward = relative[..., 0] * yaw0.cos()[:, None, None] + relative[..., 1] * yaw0.sin()[:, None, None]
    late_phase = ((phase - 0.55) / 0.45).clamp(0, 1)
    reach_blend = late_phase.square() * (3 - 2 * late_phase)
    desired_front_x = forward[:, :1, :2] + 0.04 * reach_blend[..., None]
    late = in_flight & (phase >= 0.55)
    reach_path_error = ((forward[..., :2] - desired_front_x).square().mean(-1) * late).sum(-1)
    reach_path_error = (reach_path_error / late.sum(-1).clamp_min(1)).sqrt()
    foot_path_error = (leg_path_error.square() + reach_path_error.square()).sqrt()
    after = (time >= landing[:, None]) & valid[:, None]
    sink = (pos[:, :1, 2] - pos[..., 2]).masked_fill(~after, 0).amax(-1).clamp_min(0)
    landing_pitch = states["pitch"].abs().masked_fill(~after, 0).amax(-1)
    run = torch.zeros(n, device=pos.device, dtype=torch.long)
    rebound = run.clone()
    for i in range(steps):
        run = torch.where(~contact[:, i].any(-1) & after[:, i], run + 1, 0)
        rebound = torch.maximum(rebound, run)
    rebound_time = rebound * dt
    # True COM translation separates thrust from root-link velocity changes
    # due merely to swinging heavy limbs in flight.
    com_velocity = states["com_velocity"][ids, takeoff]
    com_vx = com_velocity[:, 0] * yaw0.cos() + com_velocity[:, 1] * yaw0.sin()
    relative_y = pos[..., 1] * yaw0.cos()[:, None] - pos[..., 0] * yaw0.sin()[:, None]
    air_y = relative_y[ids, landing] - relative_y[ids, takeoff]
    goal_y = torch.as_tensor(target.landing_y, device=pos.device, dtype=pos.dtype).expand(n)
    lateral_air_deficit = (goal_y.abs() * (2 / 3) - air_y * goal_y.sign()).clamp_min(0)
    deficits = {
        "takeoff_forward_speed": (target.minimum_takeoff_com_vx - com_vx).clamp_min(0) / 0.20,
        "landing_sink": (sink - target.maximum_landing_sink).clamp_min(0) / 0.05,
        "landing_pitch": (landing_pitch - target.maximum_landing_pitch).clamp_min(0) / 0.15,
        "front_reach": (target.minimum_front_reach - front_reach).clamp_min(0) / 0.05,
        "landing_extension": (target.minimum_landing_leg_ratio - landing_leg_ratio).clamp_min(0) / 0.15,
        "apex_tuck": (apex_ratio - target.maximum_apex_leg_ratio).clamp_min(0) / 0.15,
        "push_extension": (target.minimum_push_leg_ratio - push_ratio).clamp_min(0) / 0.15,
        "rebound": (rebound_time - target.maximum_rebound_airtime).clamp_min(0) / 0.04,
        "foot_path": (foot_path_error - target.foot_path_tolerance).clamp_min(0) / 0.03,
        "lateral_air_distance": lateral_air_deficit / 0.05,
    }
    violations = torch.stack(list(deficits.values()), -1)
    result["score"] += 100 * violations.square().sum(-1) + 30 * (foot_path_error / 0.05).square()
    if target.criteria_version == 4:
        # Compression depth / leg extension are style proxies. Actual torso
        # collisions, attitude, rebound and sustained loaded recovery stay hard.
        hard_names = ("takeoff_forward_speed", "landing_pitch", "rebound")
        hard_landing = torch.stack([deficits[name] for name in hard_names], -1)
        result["accepted"] &= (hard_landing <= 1e-5).all(-1) & valid
    else:
        result["accepted"] &= (violations <= 1e-5).all(-1) & valid
    result.update(
        takeoff_com_vx=com_vx,
        maximum_landing_sink=sink,
        maximum_landing_pitch=landing_pitch,
        minimum_front_touchdown_reach=front_reach,
        minimum_landing_leg_ratio=landing_leg_ratio,
        maximum_apex_leg_ratio=apex_ratio,
        minimum_push_extension_ratio=push_ratio,
        foot_path_rms_error=foot_path_error,
        radial_foot_path_error=leg_path_error,
        forward_foot_path_error=reach_path_error,
        maximum_rebound_airtime=rebound_time,
        airborne_lateral_distance=air_y,
        lateral_air_distance_deficit=lateral_air_deficit,
        style_violation_count=(violations > 1e-5).sum(-1),
    )
    # Search-only diagnostics and priorities: do not reinterpret old certificates
    # or relax any existing acceptance condition to obtain a better-looking score.
    displacement = pos[ids, takeoff] - pos[:, 0]
    preflight_x = displacement[:, 0] * yaw0.cos() + displacement[:, 1] * yaw0.sin()
    budget = torch.as_tensor(target.landing_x - target.minimum_air_distance, device=pos.device).expand(n)
    budget_excess = (preflight_x - budget).clamp_min(0)
    rear_fraction = torch.zeros(n, device=pos.device)
    impulse_available = "wheel_contact_force" in states
    if impulse_available:
        # Reject force above the plane (e.g. wheel-wheel contact) as support.
        grounded = contact & (states["wheel_bottom_height"] <= target.recovery_ground_tolerance)
        fz = states["wheel_contact_force"][..., 2].clamp_min(0) * grounded
        impulse = (fz * push_window[..., None]).sum(1) * dt
        rear_fraction = impulse[:, 2:].sum(-1) / impulse.sum(-1).clamp_min(1e-6)
        result.update(front_push_impulse=impulse[:, :2].sum(-1), rear_push_impulse=impulse[:, 2:].sum(-1))
    # 45% hind share is a dense search guide, NOT a new pass threshold. The
    # original measured rear-last contact criterion remains authoritative.
    transfer = ((0.45 - rear_fraction).clamp_min(0) / 0.25).square() if impulse_available else 0
    guide = (budget_excess / 0.05).square() + transfer
    safety = torch.stack(
        (
            (result["max_nonfoot_contact_force"] - 5).clamp_min(0) / 20,
            (result["max_joint_speed_excess"] - 0.05).clamp_min(0) / 1.0,
            (result["max_joint_limit_violation"] - 0.002).clamp_min(0) / 0.02,
        ),
        -1,
    )
    task = torch.stack(
        (
            (~valid).float(),
            (~result["initial_stable"]).float(),
            # Without this primary guide, a contact-order improvement could
            # win the minimax rank by delaying takeoff even further. Nominal
            # budget is a search target only; original acceptance is unchanged.
            budget_excess / 0.05,
            (result["height_error"] / target.height_tolerance - 1).clamp_min(0),
            (result["landing_xy_error"] / target.landing_tolerance - 1).clamp_min(0),
            (target.minimum_air_distance - result["airborne_forward_distance"]).clamp_min(0) / 0.05,
            (target.minimum_air_time - result["max_all_wheels_air_time"]).clamp_min(0) / 0.05,
            (0.15 - result["terminal_stable_duration"]).clamp_min(0) / 0.15,
            (result["max_roll"] / target.maximum_roll - 1).clamp_min(0),
            (result["max_pitch"] / target.maximum_pitch - 1).clamp_min(0),
            (result["max_yaw_error"] / target.maximum_yaw_error - 1).clamp_min(0),
            (~result["crossing_observed"]).float(),
            result["crossing_supported"].float(),
            (~result["crossing_in_main_flight"]).float(),
            result["crossing_clearance_deficit"] / 0.04,
            (target.minimum_rear_push_delay - result["rear_push_delay"]).clamp_min(0) / 0.04,
            (result["front_landing_delay"] - target.maximum_front_landing_delay).clamp_min(0) / 0.04,
            (result["preparatory_airtime"] - target.maximum_preparatory_airtime).clamp_min(0) / 0.02,
        ),
        -1,
    )
    all_violations = torch.cat((task, violations), -1)
    result.update(
        preflight_forward_distance=preflight_x,
        nominal_preflight_distance_budget=budget,
        preflight_distance_excess=budget_excess,
        rear_push_impulse_fraction=rear_fraction,
        push_impulse_observed=torch.full_like(valid, impulse_available),
        search_launch_guidance=guide,
        search_safety_violation=safety.square().sum(-1),
        search_max_violation=all_violations.amax(-1),
        search_total_violation=all_violations.square().sum(-1) + 0.2 * guide,
    )
    if target.criteria_version == 4:
        # Explicit motion gates, separate from goal error and stylistic guides.
        # These mirror acceptance without height/XY and remain dense for CEM.
        motion = torch.stack(
            (
                (~valid).float(),
                (~result["initial_stable"]).float(),
                (target.minimum_air_time - result["max_all_wheels_air_time"]).clamp_min(0) / 0.05,
                (target.crossing_clearance - result["peak_min_wheel_clearance"]).clamp_min(0) / 0.04,
                (0.15 - result["terminal_stable_duration"]).clamp_min(0) / 0.15,
                (result["max_roll"] / target.maximum_roll - 1).clamp_min(0),
                (result["max_pitch"] / target.maximum_pitch - 1).clamp_min(0),
                (result["max_yaw_error"] / target.maximum_yaw_error - 1).clamp_min(0),
                (result["preparatory_airtime"] - target.maximum_preparatory_airtime).clamp_min(0) / 0.02,
                *(deficits[name] for name in hard_names),
            ),
            -1,
        )
        target_error = torch.stack(
            (result["height_error"] / target.height_tolerance, result["landing_xy_error"] / target.landing_tolerance),
            -1,
        )
        style = torch.stack(
            (
                *(value for name, value in deficits.items() if name not in hard_names),
                (target.minimum_rear_push_delay - result["rear_push_delay"]).clamp_min(0) / 0.04,
                (result["front_landing_delay"] - target.maximum_front_landing_delay).clamp_min(0) / 0.04,
                (target.minimum_air_distance - result["airborne_forward_distance"]).clamp_min(0) / 0.05,
                result["crossing_clearance_deficit"] / 0.04,
                result["crossing_supported"].float(),
                (~result["crossing_in_main_flight"]).float(),
                (~result["crossing_observed"]).float(),
                budget_excess / 0.05,
            ),
            -1,
        )
        # Cap each preference's contribution so a large shape mismatch cannot
        # overwhelm accuracy. This does not change any hard acceptance gate.
        style_cost = style.clamp_max(2).square().mean(-1) + 0.1 * (foot_path_error / 0.05).clamp_max(2).square()
        target_style_cost = target_error.square().sum(-1) + 0.05 * style_cost
        result.update(
            search_motion_violation=motion.square().sum(-1),
            search_target_violation=(target_error - 1).clamp_min(0).square().sum(-1),
            search_target_error=target_error.square().sum(-1),
            search_style_cost=style_cost,
            search_target_style_cost=target_style_cost,
            score=1000 * safety.square().sum(-1) + 1000 * motion.square().sum(-1) + target_style_cost,
        )
    return result
