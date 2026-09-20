"""Tensor-only Walk These Ways contact schedules (paper Appendix D).

Canonical foot order is FR, FL, RR, RL. Command order is documented in
``COMMAND_NAMES``; height is absolute height above the ground as in the paper.
Full-stance rolling (duration=1) is a Go2W extension; its frequency is independent
of contact state and can use the same continuous range as stepping. Its internal
phase advances, but the four clock observations are zero.
"""

from __future__ import annotations

import math

import torch

COMMAND_NAMES = (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "body_height",
    "gait_frequency",
    "gait_phase",
    "gait_offset",
    "gait_bound",
    "gait_duration",
    "footswing_height",
)
FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
PHASE_PRESETS = {"rolling": (0.0, 0.0, 0.0), "trot": (0.5, 0.0, 0.0), "bound": (0.0, 0.0, 0.5)}


def foot_phase_offsets(parameters: torch.Tensor) -> torch.Tensor:
    """Appendix D offsets in FR, FL, RR, RL order, before taking modulo one."""
    theta1, theta2, theta3 = parameters.unbind(-1)
    return torch.stack((theta2 + theta3, theta1 + theta3, theta1, theta2), dim=-1)


def validate_phase_neighborhood(jitter_deg: float, max_pair_error_deg: float) -> None:
    if not math.isfinite(jitter_deg) or not 0 <= jitter_deg < 90:
        raise ValueError("Phase jitter must be finite and in [0, 90) degrees.")
    if not math.isfinite(max_pair_error_deg) or not 0 <= max_pair_error_deg <= 180:
        raise ValueError("Pairwise phase-error limit must be finite and in [0, 180] degrees.")
    if jitter_deg > 0 and max_pair_error_deg == 0:
        raise ValueError("Nonzero phase jitter requires a positive pairwise phase-error limit.")


def sample_phase_neighborhood(centers: torch.Tensor, jitter_deg: float, max_pair_error_deg: float) -> torch.Tensor:
    """Uniform box proposals, rejected if any wheel-pair phase error is too large.

    Return unwrapped center-plus-offset parameters so a neighborhood of zero is
    continuous in the command observation. Only periodic schedules take modulo
    one. The wheel-pair limit measures circular error relative to the preset,
    not the absolute separation (which can be half a cycle for trot/bound).
    """
    validate_phase_neighborhood(jitter_deg, max_pair_error_deg)
    result = centers.clone()
    if jitter_deg == 0:
        return result
    radius, limit = jitter_deg / 360.0, max_pair_error_deg / 360.0
    pending = torch.arange(len(centers), device=centers.device)
    # Bound retries to fail explicitly for an impractically tight configuration,
    # instead of hanging or silently accepting an out-of-neighborhood sample.
    for _ in range(32):
        if pending.numel() == 0:
            return result
        noise = (2 * torch.rand_like(centers[pending]) - 1) * radius
        foot_error = foot_phase_offsets(noise)
        pair_error = foot_error[:, :, None] - foot_error[:, None, :]
        pair_error = torch.remainder(pair_error + 0.5, 1.0) - 0.5
        accepted = pair_error.abs().amax(dim=(1, 2)) <= limit
        accepted_ids = pending[accepted]
        result[accepted_ids] = centers[accepted_ids] + noise[accepted]
        pending = pending[~accepted]
    if pending.numel():
        raise RuntimeError("Phase rejection sampling exhausted retries; reduce jitter or relax the pairwise limit.")
    return result


def validate_commands(command: torch.Tensor) -> None:
    """Reject ambiguous schedules rather than silently clipping user commands."""
    if command.ndim != 2 or command.shape[1] != 10:
        raise ValueError("WTW commands must have shape (N, 10).")
    if not torch.isfinite(command).all():
        raise ValueError("WTW commands must be finite.")
    if (command[:, 3] <= 0).any() or (command[:, 4] < 0).any() or (command[:, 9] < 0).any():
        raise ValueError("Height must be positive; frequency and swing height must be non-negative.")
    if ((command[:, 5:8] < -1) | (command[:, 5:8] >= 1)).any():
        raise ValueError("Phase parameters must be signed cycle offsets in [-1, 1).")
    if ((command[:, 8] <= 0) | (command[:, 8] > 1)).any():
        raise ValueError("gait_duration must be in (0, 1].")
    full_stance = command[:, 8] == 1
    if ((command[:, 4] == 0) & ~full_stance).any():
        raise ValueError("Periodic stepping requires positive frequency; zero frequency is only valid for full stance.")
    if (full_stance & (command[:, 9] != 0)).any():
        raise ValueError("Full stance requires footswing_height=0; its frequency may be positive.")


def contact_schedule(
    phase: torch.Tensor, command: torch.Tensor, kappa: float = 0.07
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw phase, duration-warped phase, four sine clocks and contact probabilities.

    Appendix D: [FR, FL, RR, RL] = [t+theta2+theta3,
    t+theta1+theta3, t+theta1, t+theta2] modulo one. Use the normal CDF
    (not the PDF printed as Phi in the paper), following the released code.
    """
    if kappa <= 0:
        raise ValueError("Contact smoothing kappa must be positive.")
    offsets = foot_phase_offsets(command[:, 5:8])
    raw = torch.remainder(phase[:, None] + offsets, 1.0)
    full_stance = command[:, 8:9] == 1
    # Safe denominators also on rolling rows; torch.where evaluates both branches.
    duration = torch.where(full_stance, 0.5, command[:, 8:9])
    warped = torch.where(raw < duration, 0.5 * raw / duration, 0.5 + 0.5 * (raw - duration) / (1 - duration))
    # A full-stance command has no swing interval to warp. Retain its internal
    # phase for gait switching, independently of the masked observation clocks.
    warped = torch.where(full_stance, raw, warped)

    def cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.erf(x / (kappa * math.sqrt(2))))

    contact = cdf(warped) * (1 - cdf(warped - 0.5)) + cdf(warped - 1) * (1 - cdf(warped - 1.5))
    # Rolling/standing must not expose an oscillatory leg cue to actor or critic.
    # Zero swing height alone is not rolling: stepping clocks stay active.
    clocks = torch.where(~full_stance & (command[:, 4:5] > 0), torch.sin(2 * math.pi * warped), 0.0)
    return raw, warped, clocks, torch.where(full_stance, 1.0, contact)


def swing_height_target(warped: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """Released-code triangular swing trajectory, measured from the wheel bottom."""
    shape = 1 - torch.abs(1 - 2 * torch.clamp(2 * warped - 1, 0, 1))
    return command[:, 9:10] * shape


def contact_force_cost(forces: torch.Tensor, desired: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    return ((1 - desired) * (1 - torch.exp(-forces.square().sum(-1) / sigma))).mean(-1)


def contact_velocity_cost(velocities: torch.Tensor, desired: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    return (desired * (1 - torch.exp(-velocities.square().sum(-1) / sigma))).mean(-1)


def wheel_contact_geometry(
    axis_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
    radius: float,
    half_width: float,
    center_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest point and material velocity of a finite cylindrical wheel on a plane.

    Axis is the wheel link's +Y in world coordinates. Link linear velocity is
    measured at the link origin (not COM); angular velocity includes wheel spin.
    Radius, width and signed axial offsets come from the robot collision geometry.
    """
    down = torch.zeros_like(axis_w)
    down[..., 2] = -1
    along_axis = (down * axis_w).sum(-1, keepdim=True)
    radial = down - along_axis * axis_w
    radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    displacement = (center_offsets[None, :, None] + half_width * along_axis.sign()) * axis_w + radius * radial
    return displacement, lin_vel_w + torch.cross(ang_vel_w, displacement, dim=-1)
