"""Moving leap acceptance must reject rolling, stop-to-hop and late recovery tricks."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "locolab/scripts/locomotion"
for name in ("jump_target_to", "forward_leap_to"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
TO = sys.modules["jump_target_to"]
LEAP = sys.modules["forward_leap_to"]


def flight():
    t = torch.arange(361) * 0.005
    position = torch.zeros(1, len(t), 3)
    position[0, :, 0] = t
    position[0, :, 2] = 0.4
    position[0, 40:121, 2] += 0.2 * torch.sin(torch.linspace(0, torch.pi, 81))
    contacts = torch.ones(1, len(t), 4, dtype=torch.bool)
    contacts[:, 40:120] = False
    contacts[:, 38:40, :2] = False
    velocity = torch.zeros_like(position)
    velocity[..., 0] = 1.0
    states = dict(position=position, velocity=velocity, contacts=contacts)
    for key in ("yaw", "tilt", "roll", "pitch", "nonfoot_force", "speed_violation", "limit_violation", "clearance"):
        states[key] = torch.zeros(1, len(t))
    states["clearance"][:, 40:120] = 0.12
    states["omega"] = torch.zeros_like(position)
    states["wheel_bottom_height"] = torch.zeros(1, len(t), 4)
    states["wheel_bottom_height"][:, 40:120] = 0.12
    return states


def test_forward_flight_recovers_while_moving_not_at_landing_point():
    result = LEAP.forward_leap_metrics(flight(), LEAP.ForwardLeapTarget())
    assert result["accepted"].item()
    assert result["airborne_forward_distance"].item() == pytest.approx(0.4)
    assert result["landing_xy"][0, 0].item() == pytest.approx(0.6)
    assert result["final_vx"].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "failure",
    [
        "static_start",
        "stopped_exit",
        "rolling",
        "short_air",
        "scrape",
        "yaw",
        "roll",
        "pitch",
        "collision",
        "overspeed",
        "joint_limit",
        "no_landing",
    ],
)
def test_rejects_invalid_motion(failure):
    states = flight()
    if failure == "static_start":
        states["velocity"][:, 0, 0] = 0
    elif failure == "stopped_exit":
        states["velocity"][:, -10:, 0] = 0
    elif failure == "rolling":
        states["contacts"][:] = True
    elif failure == "short_air":
        states["contacts"][:, :85] = True
    elif failure == "scrape":
        states["clearance"][:, 70] = 0.02
    elif failure in ("yaw", "roll", "pitch"):
        states[failure][:, 70] = 0.7
    elif failure == "collision":
        states["nonfoot_force"][:, 70] = 6
    elif failure == "overspeed":
        states["speed_violation"][:, 70] = 0.2
    elif failure == "joint_limit":
        states["limit_violation"][:, 70] = 0.01
    else:
        states["contacts"][:, 40:] = False
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()


def test_pitch_is_not_conflated_with_roll_or_required_to_stay_level():
    states = flight()
    states["pitch"][:, 45:70] = -0.45
    states["tilt"][:, 45:70] = 0.45
    assert LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()


def test_crossing_cannot_be_deferred_to_a_second_flight():
    states = flight()
    states["contacts"][:, 60:65] = True
    result = LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())
    assert not result["accepted"].item()
    assert result["takeoff_time"].item() == pytest.approx(0.2)
    assert result["landing_time"].item() == pytest.approx(0.3)
    assert not result["crossing_in_main_flight"].item()


def test_fixed_yaw_frame_includes_velocity():
    states = flight()
    for key in ("position", "velocity"):
        xy = states[key][..., :2].clone()
        states[key][..., 0] = -xy[..., 1]
        states[key][..., 1] = xy[..., 0]
    states["position"] += torch.tensor([2, 3, 0])
    states["yaw"][:] = torch.pi / 2
    assert LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()


def test_recovery_contact_memory_requires_geometry_and_recent_load():
    states = flight()
    states["contacts"][:, -2:, 0] = False
    assert LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()
    states["wheel_bottom_height"][:, -2:, 0] = 0.01
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()


def test_rear_legs_must_finish_push_and_forelegs_reach_landing():
    states = flight()
    states["contacts"][:, 38:40] = states["contacts"][:, 38:40].flip(-1)
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()
    states = flight()
    states["contacts"][:, 120:130, :2] = False
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()
    states = flight()
    states["contacts"][:, 10:18] = False
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()
    states = flight()
    states["contacts"][:, -6:, 0] = False
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()


def test_moving_control_endpoints():
    mean, _ = TO.initial_distribution()
    default = torch.tensor([0.0, 0.8, -1.5] * 4)
    tape = TO.control_tape(mean[None], default, default - 2, default + 2, endpoint_wheel_speed=1 / 0.086)
    assert tape[0, 0, 12].item() == pytest.approx(1 / 0.086)
    assert tape[0, -1, 12].item() > 11
    tape = TO.control_tape(
        mean[None].expand(2, -1),
        default,
        default - 2,
        default + 2,
        endpoint_wheel_speed=1 / 0.086,
        recovery_from_knot=6,
    )
    torch.testing.assert_close(tape[:, 55:, :12], default.expand(2, 35, 12))
    torch.testing.assert_close(tape[:, 55:, 12:], torch.full((2, 35, 4), 1 / 0.086))


def test_rear_delay_moves_only_rear_leg_controls_not_simulated_states():
    mean, _ = TO.initial_distribution()
    default = torch.tensor([0.0, 0.8, -1.5] * 4)
    tape = TO.control_tape(mean[None], default, default - 2, default + 2)
    delayed = TO.control_tape(mean[None], default, default - 2, default + 2, rear_delay=0.04)
    torch.testing.assert_close(delayed[..., :6], tape[..., :6])
    torch.testing.assert_close(delayed[..., 12:], tape[..., 12:])
    torch.testing.assert_close(delayed[:, 0], tape[:, 0])
    assert not torch.allclose(delayed[..., 6:12], tape[..., 6:12])
    with pytest.raises(ValueError, match="increasing"):
        TO.control_tape(mean[None], default, default - 2, default + 2, rear_delay=1.0)


def test_export_rechecks_moving_acceptance_and_preserves_motion_mode():
    states = flight()
    for name, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[name] = torch.zeros(1, 361, width)
    states["quaternion"] = torch.tensor([1.0, 0, 0, 0]).repeat(1, 361, 1)
    target = LEAP.ForwardLeapTarget()
    metrics = LEAP.forward_leap_metrics(states, target)
    replay = dict(
        accepted=True,
        independent_replay=True,
        actual_physics_steps=360,
        simulation_dt=0.005,
        motion_mode="forward_leap",
        target=asdict(target),
        mass_kg=19.523,
        gravity=9.81,
        states={k: v[0].tolist() for k, v in states.items()},
        metrics={k: v[0].tolist() for k, v in metrics.items()},
        leg_joint_names=TO.LEG_JOINT_NAMES,
        wheel_joint_names=TO.WHEEL_JOINT_NAMES,
        actuator_contract="shared",
        leg_effort_limit_nm=[23.5, 23.5, 35] * 4,
        leg_velocity_limit_rad_s=[30, 30, 20] * 4,
        self_collisions_enabled=True,
        input_sha256="test",
    )
    reference = TO.export_reference(replay)
    assert reference["metadata"]["motion_mode"] == "forward_leap"
    assert reference["base_linear_velocity"][-1][0] == 1.0
    replay["states"]["velocity"][-1][0] = 0.0
    with pytest.raises(ValueError, match="acceptance"):
        TO.export_reference(replay)


@pytest.mark.parametrize(
    "values", [dict(forward_speed=0), dict(crossing_start=0.5), dict(height=float("nan")), dict(speed_tolerance=2)]
)
def test_invalid_targets(values):
    with pytest.raises(ValueError):
        LEAP.ForwardLeapTarget(**values)


def test_pre_versioned_target_infers_original_acceptance_not_latest_defaults():
    values = asdict(LEAP.ForwardLeapTarget())
    values.pop("criteria_version")
    assert LEAP.load_forward_target(values).criteria_version == 3
    for key in ("minimum_rear_push_delay", "maximum_front_landing_delay", "maximum_preparatory_airtime"):
        values.pop(key)
    old_target = LEAP.load_forward_target(values)
    assert old_target.criteria_version == 2
    states = flight()
    states["contacts"][:, 38:40] = states["contacts"][:, 38:40].flip(-1)
    assert LEAP.forward_leap_metrics(states, old_target)["accepted"].item()
    assert not LEAP.forward_leap_metrics(states, LEAP.ForwardLeapTarget())["accepted"].item()
    values.pop("recovery_contact_memory")
    values.pop("recovery_ground_tolerance")
    oldest_target = LEAP.load_forward_target(values)
    assert oldest_target.criteria_version == 1
    states["contacts"][:, -1, 0] = False
    assert LEAP.forward_leap_metrics(states, old_target)["accepted"].item()
    assert not LEAP.forward_leap_metrics(states, oldest_target)["accepted"].item()
