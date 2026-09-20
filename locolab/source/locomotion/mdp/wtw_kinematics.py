"""Go2W level-body height IK for WTW reward pose targets (no simulator dependency).

Geometry is from the Go2W URDF: thigh-to-knee 0.213 m, knee-to-wheel-axis
0.2264 m, and wheel radius 0.086 m. Hip joint origins have zero base-frame z.
Keeping hip=0 preserves the wheel-axis direction and the existing stance width.
"""

from __future__ import annotations

import math

import torch

LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf"))
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.2264


def height_reference_pose(
    height: torch.Tensor, default_pose: torch.Tensor, wheel_radius: float = 0.086
) -> torch.Tensor:
    """Return (N, 4, 3) hip/thigh/calf angles for level, four-wheel support.

    Height is base-link height above the plane. Preserve each wheel center's x/y
    from the immutable asset pose, choosing the existing negative-knee branch.
    This is a geometric nominal pose, not gravity compensation or a swing path.
    Unreachable requests are rejected; only acos roundoff is clamped.
    """
    if height.ndim != 1 or default_pose.shape != (len(height), 4, 3):
        raise ValueError("Expected N heights and an (N, 4, 3) default leg pose.")
    if not math.isfinite(wheel_radius) or wheel_radius <= 0:
        raise ValueError("Wheel radius must be finite and positive.")
    if not torch.isfinite(height).all() or not torch.isfinite(default_pose).all():
        raise ValueError("Height and default pose must be finite.")
    if (default_pose[..., 0].abs() > 1e-6).any() or (default_pose[..., 2] >= 0).any():
        raise ValueError("Go2W height IK requires zero nominal hips and the negative-knee branch.")
    down = height[:, None] - wheel_radius
    if (down <= 0).any():
        raise ValueError("Target base height must exceed the wheel radius.")
    thigh, calf = default_pose[..., 1], default_pose[..., 2]
    x = -THIGH_LENGTH * torch.sin(thigh) - CALF_LENGTH * torch.sin(thigh + calf)
    cosine = (x.square() + down.square() - THIGH_LENGTH**2 - CALF_LENGTH**2) / (2 * THIGH_LENGTH * CALF_LENGTH)
    if ((cosine < -1 - 1e-6) | (cosine > 1 + 1e-6)).any():
        raise ValueError("Requested Go2W base height is outside the leg workspace.")
    knee = -torch.acos(cosine.clamp(-1.0, 1.0))
    upper = torch.atan2(-x, down) - torch.atan2(
        CALF_LENGTH * torch.sin(knee), THIGH_LENGTH + CALF_LENGTH * torch.cos(knee)
    )
    return torch.stack((torch.zeros_like(upper), upper, knee), dim=-1)
