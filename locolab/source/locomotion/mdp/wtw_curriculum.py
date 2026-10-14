"""Iteration-based command expansion with replay; no robot or reward changes."""

import math

import torch


DEFAULT_SCHEDULE = (
    (0, 1.0, 0.10),
    (2000, 1.5, 0.12),
    (6000, 2.0, 0.15),
    (10000, 3.0, 0.15),
    (15000, 4.0, 0.15),
    (20000, 5.0, 0.15),
)


def validate_schedule(schedule):
    if not schedule or schedule[0][0] != 0:
        raise ValueError("Curriculum must start at iteration zero.")
    previous = (-1, 0.0, 0.0)
    for point in schedule:
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("Expected finite (iteration, speed, swing height) points.")
        iteration, speed, swing = point
        if iteration <= previous[0] or speed < previous[1] or swing < previous[2] or speed <= 0 or swing < 0:
            raise ValueError("Curriculum iterations must increase and limits must not decrease.")
        previous = point


def curriculum_limits(iteration, schedule=DEFAULT_SCHEDULE):
    """Linearly expand command limits between milestones; hold after the last."""
    if not math.isfinite(iteration) or iteration < 0:
        raise ValueError("Curriculum iteration must be finite and nonnegative.")
    for left, right in zip(schedule, schedule[1:]):
        if iteration < right[0]:
            alpha = (iteration - left[0]) / (right[0] - left[0])
            return tuple(left[i] + alpha * (right[i] - left[i]) for i in (1, 2))
    return tuple(schedule[-1][1:])


def sample_curriculum_commands(count, device, iteration, schedule=DEFAULT_SCHEDULE):
    """50% original envelope, 30% intermediate (speed <=2), 20% expanding envelope.

    Bands are chosen per command resampling, not permanently assigned to robots.
    Standing and rolling masks are applied by WTWCommand afterwards.
    """
    speed, swing = curriculum_limits(iteration, schedule)
    selector = torch.rand(count, device=device)
    replay = selector < 0.5
    intermediate = (selector >= 0.5) & (selector < 0.8)
    speed_limits = torch.full((count,), speed, device=device)
    swing_limits = torch.full((count,), swing, device=device)
    speed_limits[replay] = min(speed, schedule[0][1])
    swing_limits[replay] = min(swing, schedule[0][2])
    speed_limits[intermediate] = min(speed, 2.0)
    vx = (2 * torch.rand(count, device=device) - 1) * speed_limits
    clearance = torch.rand(count, device=device) * swing_limits
    return vx, clearance
