"""Physical geometry, wheel-specific crossing and false-success rejection."""

import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_parkour_leap_to import PARKOUR, TO

GAP = sys.modules["gap_leap_to"]


def crossing(width=0.4):
    target = GAP.GapLeapTarget(width=width)
    t = torch.arange(361) * 0.005
    pos = torch.zeros(1, len(t), 3)
    pos[..., 0] = 5.6 + t + 0.6 * (t - 0.16).clamp(0, 0.5)
    phase = ((t - 0.16) / 0.5).clamp(0, 1)
    pos[..., 2] = 0.4 + 0.2 * torch.sin(torch.pi * phase)
    wheels = pos[:, :, None].repeat(1, 1, 4, 1)
    wheels[..., 0] += torch.tensor([0.18, 0.18, -0.22, -0.22])
    wheels[..., 1] += torch.tensor([-0.18, 0.18, -0.18, 0.18])
    bottom = torch.zeros(1, len(t), 4)
    for feet, leave, touch in ((slice(0, 2), 0.14, 0.66), (slice(2, 4), 0.16, 0.86)):
        phi = ((t - leave) / (touch - leave)).clamp(0, 1)
        bottom[:, :, feet] = (0.22 * torch.sin(torch.pi * phi))[None, :, None]
    wheels[..., 2] = bottom + 0.086
    contact = torch.ones(1, len(t), 4, dtype=torch.bool)
    contact[:, 28:132, :2] = False
    contact[:, 32:172, 2:] = False
    vel = torch.zeros_like(pos)
    vel[..., 0] = 1.0
    states = dict(
        position=pos,
        velocity=vel,
        contacts=contact,
        wheel_center_position=wheels,
        wheel_bottom_height=bottom,
        wheel_forward_radius=torch.full_like(bottom, 0.086),
        omega=torch.zeros_like(pos),
    )
    for key in ("roll", "pitch", "yaw", "tilt", "nonfoot_force", "speed_violation", "limit_violation"):
        states[key] = torch.zeros(1, len(t))
    return states, target


def certificate(target):
    mesh = GAP.gap_mesh(target)
    return dict(
        target=asdict(target),
        mesh_sha256=GAP.geometry_fingerprint(mesh.vertices, mesh.faces),
        collision_enabled=True,
        collision_approximation="none",
        no_extra_static_colliders=True,
    )


@pytest.mark.parametrize("width", [0.4, 0.6])
def test_gap_is_missing_deck_not_a_visual_rectangle(width):
    target = GAP.GapLeapTarget(width=width)
    mesh = GAP.gap_mesh(target)
    triangles = mesh.vertices[mesh.faces]
    top = triangles[np.all(np.isclose(triangles[..., 2], 0), axis=-1)]
    assert len(top) == 4
    assert np.all((top[..., 0].max(-1) <= target.near_x + 1e-7) | (top[..., 0].min(-1) >= target.far_x - 1e-7))
    pit = triangles[np.all(np.isclose(triangles[..., 2], -target.depth), axis=-1)]
    assert np.any(np.isclose(pit[..., 0].min(-1), target.near_x) & np.isclose(pit[..., 0].max(-1), target.far_x))
    GAP.validate_gap_geometry(certificate(target), target)


@pytest.mark.parametrize(
    "bad", ["collision_enabled", "no_extra_static_colliders", "mesh_sha256", "collision_approximation"]
)
def test_export_geometry_guard_rejects_invisible_floor_and_convex_hull(bad):
    target = GAP.GapLeapTarget()
    cert = certificate(target)
    cert[bad] = False if isinstance(cert[bad], bool) else "wrong"
    with pytest.raises(ValueError, match="collider geometry"):
        GAP.validate_gap_geometry(cert, target)


def test_accepts_only_full_wheel_crossing_and_stable_far_bank_recovery():
    states, target = crossing()
    m = GAP.gap_leap_metrics(states, target)
    assert m["accepted"].item(), {k: v.tolist() for k, v in m.items()}
    assert m["far_bank_touchdown"].all() and m["wheel_fully_crossed"].all()
    assert m["first_touchdown_time"][0, 2] > m["first_touchdown_time"][0, 0]
    assert m["terminal_stable_duration"].item() >= 0.15
    # The same movement is NOT a 60 cm certificate.
    assert not GAP.gap_leap_metrics(states, GAP.GapLeapTarget(width=0.6))["accepted"].item()


@pytest.mark.parametrize(
    "bad", ["near_rear", "pit", "wall", "no_flight", "lateral", "nonfoot", "limit", "unstable", "edge_balance"]
)
def test_rejects_partial_crossing_and_contact_shortcuts(bad):
    states, target = crossing()
    if bad == "near_rear":
        states["contacts"][:, 140, 2:] = True
        states["wheel_center_position"][:, 140, 2:, 0] = target.near_x - 0.1
        states["wheel_bottom_height"][:, 140, 2:] = 0
    elif bad in ("pit", "wall"):
        states["contacts"][:, 75, :2] = True
        states["wheel_center_position"][:, 75, :2, 0] = target.near_x + target.width / 2
        states["wheel_bottom_height"][:, 75, :2] = -target.depth if bad == "pit" else -0.04
    elif bad == "no_flight":
        states["contacts"][:] = True
    elif bad == "lateral":
        states["wheel_center_position"][:, 100, :, 1] = target.bank_half_width + 0.1
    elif bad == "nonfoot":
        states["nonfoot_force"][:, 100] = 6
    elif bad == "limit":
        states["limit_violation"][:, 100] = 0.003
    elif bad == "unstable":
        states["omega"][:, -30:] = 2
    else:
        states["wheel_center_position"][:, -40:, 2:, 0] = target.far_x + 0.06
    assert not GAP.gap_leap_metrics(states, target)["accepted"].item()


def test_gap_control_guesses_do_not_change_limited_robot_or_seed():
    mean, _, low, high, _ = PARKOUR.parameter_distribution()
    restored, *_ = PARKOUR.parameter_distribution(seed=dict(motion_mode="gap_leap", parameters=mean.tolist()))
    torch.testing.assert_close(restored, mean)
    samples = mean.repeat(1, 16, 1)
    proposed = GAP.gap_proposals(samples, low, high)
    torch.testing.assert_close(samples, mean.repeat(1, 16, 1))
    assert (proposed >= low).all() and (proposed <= high).all()
    assert torch.isfinite(PARKOUR.knot_schedule(proposed[0])).all()


def test_walking_onto_far_bank_before_hopping_is_not_a_gap_jump():
    states, target = crossing()
    # A forewheel has already touched the far bank before the main flight.
    states["contacts"][:, 30, :2] = True
    states["wheel_center_position"][:, 30, :2, 0] = target.far_x + 0.1
    states["wheel_bottom_height"][:, 30, :2] = 0
    result = GAP.gap_leap_metrics(states, target)
    assert result["far_bank_touchdown"].all()
    assert not result["near_bank_takeoff"].all()
    assert not result["accepted"].item()


def test_reference_export_requires_real_gap_geometry_certificate():
    states, target = crossing()
    for key, size in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[key] = torch.zeros(1, 361, size)
    states["quaternion"] = torch.tensor([1.0, 0, 0, 0]).repeat(1, 361, 1)
    m = GAP.gap_leap_metrics(states, target)
    replay = dict(
        accepted=True,
        independent_replay=True,
        motion_mode="gap_leap",
        target=asdict(target),
        actual_physics_steps=360,
        simulation_dt=0.005,
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in m.items()},
        gap_geometry=certificate(target),
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
    assert TO.export_reference(replay)["metadata"]["motion_mode"] == "gap_leap"
    replay["gap_geometry"]["no_extra_static_colliders"] = False
    with pytest.raises(ValueError, match="collider geometry"):
        TO.export_reference(replay)
