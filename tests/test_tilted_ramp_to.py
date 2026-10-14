"""Cross-slope geometry, normal contact distances and selected-pad stopping."""

import math
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_parkour_leap_to import PARKOUR, TO

RAMP = sys.modules["tilted_ramp_to"]


def landing(side=1):
    target = RAMP.TiltedRampTarget(side=side)
    t = torch.arange(361) * 0.005
    progress = ((t - 0.12) / 0.5).clamp(0, 1)
    blend = progress.square() * (3 - 2 * progress)
    roll_blend = ((t - 0.18) / 0.60).clamp(0, 1)
    roll = side * math.radians(target.tilt_degrees) * roll_blend
    phase = ((t - 0.16) / 0.54).clamp(0, 1)
    pos = torch.zeros(1, len(t), 3)
    pos[0, :, 0] = (5.6 + t + 0.8 * (t - 0.16).clamp(0, 0.54)).clamp_max(7.05)
    pos[0, :, 1] = target.landing_y * blend
    pos[0, :, 2] = (
        0.413 + 0.20 * torch.sin(torch.pi * phase) + roll_blend * (0.413 / math.cos(math.radians(20)) - 0.413)
    )
    wheels = pos[:, :, None].repeat(1, 1, 4, 1)
    wheels[..., 0] += torch.tensor([0.18, 0.18, -0.20, -0.20])
    wheels[..., 1] += torch.tensor([-0.16, 0.16, -0.16, 0.16])
    axes = torch.zeros_like(wheels)
    axes[..., 1] = roll.cos()[None, :, None]
    axes[..., 2] = roll.sin()[None, :, None]
    n = torch.tensor(target.plane()[1])
    cosine = (axes * n).sum(-1).clamp(-1, 1)
    radius_n = 0.0225 * cosine.abs() + 0.086 * (1 - cosine.square()).clamp_min(0).sqrt()
    az = axes[..., 2]
    radius_z = 0.0225 * az.abs() + 0.086 * (1 - az.square()).sqrt()
    ramp_z = (radius_n - n[1] * (wheels[..., 1] - target.landing_y)) / n[2]
    contacts = torch.ones(1, len(t), 4, dtype=torch.bool)
    for feet, leave, touch in ((slice(0, 2), 0.14, 0.70), (slice(2, 4), 0.16, 0.84)):
        p = ((t - leave) / (touch - leave)).clamp(0, 1)[None, :, None]
        b = p.square() * (3 - 2 * p)
        wheels[:, :, feet, 2] = radius_z[:, :, feet] * (1 - b) + ramp_z[:, :, feet] * b + 0.30 * torch.sin(torch.pi * p)
        contacts[:, round(leave / 0.005) : round(touch / 0.005), feet] = False
    vel = torch.zeros_like(pos)
    vel[..., 0] = 1.0
    vel[:, t >= 1.1] = 0.0
    quat = torch.zeros(1, len(t), 4)
    quat[..., 0] = (roll / 2).cos()
    quat[..., 1] = (roll / 2).sin()
    states = dict(
        position=pos,
        velocity=vel,
        quaternion=quat,
        contacts=contacts,
        wheel_center_position=wheels,
        wheel_axis=axes,
        wheel_bottom_height=wheels[..., 2] - radius_z,
        wheel_forward_radius=torch.full_like(radius_z, 0.086),
        wheel_lateral_radius=0.0225 * axes[..., 1].abs() + 0.086 * (1 - axes[..., 1].square()).clamp_min(0).sqrt(),
        omega=torch.zeros_like(pos),
        roll=roll[None],
        tilt=roll.abs()[None],
    )
    for key in ("pitch", "yaw", "nonfoot_force", "speed_violation", "limit_violation"):
        states[key] = torch.zeros(1, len(t))
    return states, target


def certificate(target):
    mesh = RAMP.tilted_ramp_mesh(target)
    return dict(
        target=asdict(target),
        mesh_sha256=RAMP.geometry_fingerprint(mesh.vertices, mesh.faces),
        collision_enabled=True,
        collision_approximation="none",
        no_extra_static_colliders=True,
    )


def test_both_ramps_are_opposite_cross_slopes_not_flat_or_forward_inclined():
    a, b = RAMP.TiltedRampTarget(side=1), RAMP.TiltedRampTarget(side=-1)
    mesh = RAMP.tilted_ramp_mesh(a)
    assert RAMP.geometry_fingerprint(mesh.vertices, mesh.faces) == certificate(b)["mesh_sha256"]
    top = (mesh.triangles_center[:, 0] > a.far_x) & (mesh.triangles_center[:, 0] < a.end_x)
    top &= (mesh.face_normals[:, 2] > 0.5) & (mesh.triangles_center[:, 2] > -0.5)
    assert top.sum() == 4
    np.testing.assert_allclose(mesh.face_normals[top, 0], 0, atol=1e-7)
    np.testing.assert_allclose(np.abs(mesh.face_normals[top, 1]), math.sin(math.radians(20)), atol=1e-7)
    assert set(np.sign(mesh.face_normals[top, 1])) == {-1, 1}
    assert a.rectangle(-1)[3] < a.rectangle(1)[2]  # real empty space separates them
    RAMP.validate_ramp_geometry(certificate(a), a)


def test_lateral_proposals_preserve_braking_and_mirror_controls_only():
    mean, _, low, high, _ = PARKOUR.parameter_distribution(lateral_goal=0.4)
    samples = mean[None, None].expand(1, 11, -1).clone()
    samples[..., :70].reshape(1, 11, 7, 10)[..., :, [0, 2, 5, 8]] = 0
    left = RAMP.ramp_proposals(samples, low, high, 1)
    right = RAMP.ramp_proposals(samples, low, high, -1)
    poses = left[..., :70].reshape(1, 11, 7, 10)
    source = samples[..., :70].reshape_as(poses)
    torch.testing.assert_close(poses[..., 4:, 9], source[..., 4:, 9])
    assert (poses[..., :2, 1] >= source[..., :2, 1]).all()
    mirrored = left.clone()
    mirrored[..., :70].reshape_as(poses)[..., :, [0, 2, 5, 8]] *= -1
    torch.testing.assert_close(right, mirrored)
    assert ((left >= low) & (left <= high)).all()


@pytest.mark.parametrize("side", [-1, 1])
def test_true_inclined_contacts_are_accepted_even_when_world_z_bottoms_are_not_zero(side):
    states, goal = landing(side)
    m = RAMP.tilted_ramp_metrics(states, goal)
    assert m["accepted"].item(), {k: v.tolist() for k, v in m.items()}
    assert m["chosen_platform_touchdown"].all()
    assert states["wheel_bottom_height"][:, -1].abs().max() > 0.04
    assert m["first_touchdown_surface_clearance"].abs().max() < 1e-5
    assert m["terminal_stable_duration"].item() >= 0.25


def test_landing_goal_tracks_contact_footprint_not_torso_offset_on_cross_slope():
    states, goal = landing()
    states["position"][:, 220:, 1] -= 0.12
    result = RAMP.tilted_ramp_metrics(states, goal)
    assert result["accepted"].item()
    assert result["final_center_error_y"].item() < goal.center_tolerance_y


def test_a_level_torso_with_leg_adaptation_is_also_allowed_on_the_slope():
    states, goal = landing()
    states["quaternion"][:, 200:] = torch.tensor([1.0, 0, 0, 0])
    states["roll"][:, 200:] = 0
    states["tilt"][:, 200:] = 0
    assert RAMP.tilted_ramp_metrics(states, goal)["accepted"].item()


@pytest.mark.parametrize("axis", [[0, 1, 0], [0, 0, 1]])
def test_cylinder_support_distance_and_parallel_axis_degeneracy(axis):
    center = torch.tensor([[[0.0, 0.0, 0.3]]])
    a = torch.tensor(axis, dtype=torch.float32).expand_as(center)
    sep, point = RAMP.cylinder_plane_contact(center, a, (0, 0, 0), (0, 0, 1))
    expected = 0.3 - (0.0225 if axis[2] else 0.086)
    assert sep.item() == pytest.approx(expected)
    assert point[..., 2].item() == pytest.approx(expected)
    assert torch.isfinite(point).all()


@pytest.mark.parametrize(
    "bad", ["wrong_ramp", "flat_contact", "pit", "runoff", "moving_exit", "ground_lateral", "edge_overhang"]
)
def test_rejects_wrong_surface_or_unstable_landing(bad):
    states, goal = landing()
    if bad == "wrong_ramp":
        goal = RAMP.TiltedRampTarget(side=-1)
    elif bad == "flat_contact":
        states["wheel_center_position"][:, 140:, :, 2] = 0.086
    elif bad == "pit":
        states["contacts"][:, 80, :2] = True
        states["wheel_center_position"][:, 80, :2, 2] = -0.914
    elif bad == "runoff":
        states["wheel_center_position"][:, -40:, :, 0] = goal.end_x + 0.2
    elif bad == "moving_exit":
        states["velocity"][:, -60:, 0] = 0.5
    elif bad == "ground_lateral":
        states["position"][:, 30:, 1] = goal.landing_y
    else:
        states["wheel_center_position"][:, -40:, 1, 1] = goal.rectangle()[3] - 0.04
    assert not RAMP.tilted_ramp_metrics(states, goal)["accepted"].item()


def test_rolling_entry_and_zero_speed_endpoint_keep_original_default_tape_unchanged():
    mean, _, low, high, _ = PARKOUR.parameter_distribution()
    default = torch.tensor([0.0, 0.78, -1.3] * 4)
    lo, hi = torch.tensor([-1, -1.5, -2.7] * 4), torch.tensor([1, 3.5, -0.8] * 4)
    ordinary = PARKOUR.parkour_control_tape(mean[None], default, lo, hi)
    stop = PARKOUR.parkour_control_tape(mean[None], default, lo, hi, end_speed=0)
    torch.testing.assert_close(stop[:, 0], ordinary[:, 0])
    assert stop[:, -1, 12:].abs().max() < 0.1
    assert ordinary[:, -1, 12:].min() > 11
    seed = dict(motion_mode="tilted_ramp", parameters=mean.tolist(), target=dict(side=1, offset_y=0.4))
    mean[:70].reshape(7, 10)[:2, 0] = -0.14
    seed["parameters"] = mean.tolist()
    positive, *_ = PARKOUR.parameter_distribution(seed=seed, lateral_goal=0.4)
    negative, *_ = PARKOUR.parameter_distribution(seed=seed, lateral_goal=-0.4)
    torch.testing.assert_close(positive[70:], negative[70:])
    torch.testing.assert_close(positive[:70].reshape(7, 10)[:, 0], -negative[:70].reshape(7, 10)[:, 0])
    proposed = RAMP.ramp_proposals(mean.repeat(1, 8, 1), low, high, 1)
    assert (proposed >= low).all() and (proposed <= high).all()


def test_tilted_reference_requires_actual_ramp_geometry():
    states, goal = landing()
    for key, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[key] = torch.zeros(1, 361, width)
    m = RAMP.tilted_ramp_metrics(states, goal)
    replay = dict(
        accepted=True,
        independent_replay=True,
        motion_mode="tilted_ramp",
        target=asdict(goal),
        gap_geometry=certificate(goal),
        actual_physics_steps=360,
        simulation_dt=0.005,
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in m.items()},
        mass_kg=19.523,
        gravity=9.81,
        leg_joint_names=TO.LEG_JOINT_NAMES,
        wheel_joint_names=TO.WHEEL_JOINT_NAMES,
        actuator_contract="shared",
        leg_effort_limit_nm=[23.5, 23.5, 35] * 4,
        leg_velocity_limit_rad_s=[30, 30, 20] * 4,
        self_collisions_enabled=True,
        input_sha256="test",
    )
    assert TO.export_reference(replay)["metadata"]["motion_mode"] == "tilted_ramp"
    replay["gap_geometry"]["collision_approximation"] = "convexHull"
    with pytest.raises(ValueError, match="collider geometry"):
        TO.export_reference(replay)
