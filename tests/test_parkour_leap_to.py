"""Variable schedules, bounded smooth control and actual task-space acceptance."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "locolab/scripts/locomotion"
for name in ("jump_target_to", "forward_leap_to", "parkour_leap_to", "gap_leap_to", "tilted_ramp_to"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
TO, PARKOUR = sys.modules["jump_target_to"], sys.modules["parkour_leap_to"]


def flight():
    t = torch.arange(361) * 0.005
    position = torch.zeros(1, len(t), 3)
    position[0, :, 0] = 1.02 * t
    position[..., 2] = 0.4
    phase = ((t - 0.2) / 0.4).clamp(0, 1)
    position[0, :, 2] += 0.2 * torch.sin(torch.pi * phase)
    contact = torch.ones(1, len(t), 4, dtype=torch.bool)
    contact[:, 40:120] = False
    contact[:, 38:40, :2] = False
    velocity = torch.zeros_like(position)
    velocity[..., 0] = 1.02
    states = dict(position=position, velocity=velocity, contacts=contact, com_velocity=velocity.clone())
    for key in ("yaw", "tilt", "roll", "pitch", "nonfoot_force", "speed_violation", "limit_violation", "clearance"):
        states[key] = torch.zeros(1, len(t))
    states["clearance"][:, 40:120] = 0.12
    states["omega"] = torch.zeros_like(position)
    states["wheel_bottom_height"] = torch.zeros(1, len(t), 4)
    states["wheel_bottom_height"][:, 40:120] = 0.12
    offsets = torch.tensor([[0.19, -0.1, 0], [0.19, 0.1, 0], [-0.19, -0.1, 0], [-0.19, 0.1, 0]])
    hips = position[:, :, None] + offsets
    feet = hips.clone()
    late = ((phase - 0.55) / 0.45).clamp(0, 1)
    reach = 0.04 * late.square() * (3 - 2 * late)
    feet[..., :2, 0] += reach[None, :, None]
    length = 0.314 * (1 - 0.25 * torch.sin(torch.pi * phase).square())
    feet[..., 2] -= torch.sqrt(length[None, :, None].square() - (feet[..., 0] - hips[..., 0]).square())
    states.update(hip_position=hips, wheel_center_position=feet)
    return states


def test_independent_timing_and_recovery_variables():
    mean, std, low, high, mask = PARKOUR.parameter_distribution()
    assert mean.shape == std.shape == low.shape == high.shape == mask.shape == (84,)
    assert mask[70:].all() and mask[50:70].any()  # recovery is no longer frozen
    times = PARKOUR.knot_schedule(mean[None])
    changed = mean.clone()
    changed[71] += 0.02
    new = PARKOUR.knot_schedule(changed[None])
    torch.testing.assert_close(new[:, 1], times[:, 1])
    assert new[0, 0, 2] == pytest.approx(times[0, 0, 2].item() + 0.02)
    assert (PARKOUR.knot_schedule(low[None]).diff() > 0).all()
    assert (PARKOUR.knot_schedule(high[None]).diff() > 0).all()


def test_monotone_spline_has_no_overshoot_and_does_not_stop_at_straight_knots():
    times = torch.tensor([[0.0, 1, 2, 3]])
    values = torch.tensor([[[0.0], [1.0], [2.0], [1.0]]])
    q = torch.linspace(0, 3, 3001)[None]
    out = PARKOUR.monotone_spline(times, values, q)[0, :, 0]
    assert out.min() >= 0 and out.max() <= 2
    assert (out[1001] - out[999]).item() / 0.002 == pytest.approx(1.0, abs=0.01)
    assert abs((out[2001] - out[1999]).item() / 0.002) < 0.01


def test_batched_tape_matches_single_and_respects_target_limits():
    mean, _, low, high, _ = PARKOUR.parameter_distribution()
    p = torch.stack((mean, (mean + low) / 2, (mean + high) / 2))
    default = torch.tensor([0.0, 0.78, -1.3] * 4)
    lo, hi = torch.tensor([-1, -1.5, -2.7] * 4), torch.tensor([1, 3.5, -0.8] * 4)
    tape = PARKOUR.parkour_control_tape(p, default, lo, hi)
    assert tape.shape == (3, 90, 16)
    assert torch.isfinite(tape).all()
    assert (tape[..., :12] >= lo + 0.0199).all() and (tape[..., :12] <= hi - 0.0199).all()
    torch.testing.assert_close(tape[:, 0, :12], default.expand(3, 12))
    for i in range(3):
        single = PARKOUR.parkour_control_tape(p[i : i + 1], default, lo, hi)
        torch.testing.assert_close(single[0], tape[i])


def test_legacy_seed_import_keeps_launch_but_replaces_bad_landing_and_adds_times():
    old, _ = TO.initial_distribution()
    old.reshape(7, 10)[3:, 3] = 1.8
    mean, *_ = PARKOUR.parameter_distribution(seed={"motion_mode": "forward_leap", "parameters": old.tolist()})
    torch.testing.assert_close(mean[:20].reshape(2, 10)[:, 3], old[:20].reshape(2, 10)[:, 3])
    assert mean.reshape(-1)[33] < 1.0
    restored, *_ = PARKOUR.parameter_distribution(seed={"motion_mode": "parkour_leap", "parameters": mean.tolist()})
    torch.testing.assert_close(restored, mean)


def test_valid_task_space_path():
    m = PARKOUR.parkour_leap_metrics(flight(), PARKOUR.ParkourLeapTarget())
    assert m["accepted"].item(), {k: v.tolist() for k, v in m.items()}
    assert m["foot_path_rms_error"].item() < 1e-5
    assert m["minimum_front_touchdown_reach"].item() == pytest.approx(0.23)


@pytest.mark.parametrize(
    "failure",
    ["sink", "late_rebound", "no_tuck", "underbody_landing", "com_braking", "bad_touch_pitch", "missing_front_touch"],
)
def test_rejects_bad_style_even_if_final_recovery_is_good(failure):
    states = flight()
    if failure == "sink":
        states["position"][:, 140:150, 2] = 0.18
    elif failure == "late_rebound":
        states["contacts"][:, 200:220] = False
    elif failure == "no_tuck":
        states["wheel_center_position"][:, 78:83, :, 2] -= 0.15
    elif failure == "underbody_landing":
        states["wheel_center_position"][:, 120:140, :2, 0] -= 0.16
    elif failure == "com_braking":
        states["com_velocity"][:, 40, 0] = 0.3
    elif failure == "bad_touch_pitch":
        states["pitch"][:, 120:140] = 0.4
    else:
        states["contacts"][:, 120:, :2] = False
    assert not PARKOUR.parkour_leap_metrics(states, PARKOUR.ParkourLeapTarget())["accepted"].item()


def test_metrics_use_fixed_trigger_yaw_for_com_and_foot_reach():
    states = flight()
    for key in ("position", "velocity", "com_velocity", "hip_position", "wheel_center_position"):
        xy = states[key][..., :2].clone()
        states[key][..., 0], states[key][..., 1] = -xy[..., 1], xy[..., 0]
    states["yaw"][:] = torch.pi / 2
    assert PARKOUR.parkour_leap_metrics(states, PARKOUR.ParkourLeapTarget())["accepted"].item()


def test_export_rechecks_v2_not_legacy_criteria():
    states = flight()
    for name, width in (("joint_position", 12), ("joint_velocity", 12), ("wheel_position", 4), ("wheel_velocity", 4)):
        states[name] = torch.zeros(1, 361, width)
    states["quaternion"] = torch.tensor([1.0, 0, 0, 0]).repeat(1, 361, 1)
    target = PARKOUR.ParkourLeapTarget()
    metrics = PARKOUR.parkour_leap_metrics(states, target)
    replay = dict(
        accepted=True,
        independent_replay=True,
        actual_physics_steps=360,
        simulation_dt=0.005,
        motion_mode="parkour_leap",
        parameterization="parkour_v2_time_and_shape",
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
    assert TO.export_reference(replay)["metadata"]["motion_mode"] == "parkour_leap"
    replay["states"]["position"][140][2] = 0.18
    with pytest.raises(ValueError, match="acceptance"):
        TO.export_reference(replay)


@pytest.mark.parametrize(
    "bad",
    [dict(maximum_apex_leg_ratio=1.2), dict(maximum_landing_sink=-0.1), dict(minimum_takeoff_com_vx=float("nan"))],
)
def test_invalid_style_parameters(bad):
    with pytest.raises(ValueError):
        PARKOUR.ParkourLeapTarget(**bad)


def test_invalid_schedules():
    mean, *_ = PARKOUR.parameter_distribution()
    with pytest.raises(ValueError):
        PARKOUR.knot_schedule(mean[:70][None])
    mean[70] = 0
    with pytest.raises(ValueError):
        PARKOUR.knot_schedule(mean[None])


def test_new_parkour_cannot_disable_contact_criteria_using_legacy_version():
    with pytest.raises(ValueError, match="current contact"):
        PARKOUR.ParkourLeapTarget(criteria_version=2)


@pytest.mark.parametrize("dx", [0.4, 0.6, 0.8])
def test_command_distance_scales_spatial_constraints_not_states(dx):
    target = PARKOUR.command_target(height=0.2, landing_x=dx, landing_y=0)
    assert target.minimum_air_distance == pytest.approx(dx * 2 / 3)
    assert target.crossing_start == pytest.approx(dx / 2)
    assert target.crossing_end == pytest.approx(dx * 2 / 3)
    assert target.maximum_landing_sink == 0.1
    assert target.maximum_roll == 0.2
    assert target.crossing_clearance == 0.08


def test_side_goals_unlock_only_control_modes_and_keep_nominal_symmetry():
    center, _, _, _, center_mask = PARKOUR.parameter_distribution()
    left, _, _, _, left_mask = PARKOUR.parameter_distribution(lateral_goal=0.15)
    right, _, _, _, right_mask = PARKOUR.parameter_distribution(lateral_goal=-0.15)
    assert left_mask.all() and right_mask.all()
    assert not center_mask[:70].reshape(7, 10)[:, [0, 2, 5, 8]].any()
    assert torch.count_nonzero(center[:70].reshape(7, 10)[:, [0, 2, 5, 8]]) == 0
    assert left[0] == -right[0] and left[0] < 0
    torch.testing.assert_close(left[70:], right[70:])


def test_batched_parkour_metrics_match_independent_scalar_targets():
    goals = [
        PARKOUR.command_target(height=h, landing_x=x, landing_y=y)
        for h, x, y in [(0.15, 0.4, 0), (0.2, 0.6, 0), (0.25, 0.8, 0), (0.2, 0.6, 0.15)]
    ]
    original = flight()
    states = {k: v.repeat((8,) + (1,) * (v.ndim - 1)) for k, v in original.items()}
    batched = PARKOUR.BatchedParkourTargets(goals, repeats=2)
    metrics = PARKOUR.parkour_leap_metrics(states, batched)
    for i in range(8):
        scalar = PARKOUR.parkour_leap_metrics(original, goals[i // 2])
        for key in metrics:
            torch.testing.assert_close(metrics[key][i], scalar[key][0], atol=1e-4, rtol=1e-5)
    assert metrics["accepted"].tolist() == [False, False, True, True, False, False, False, False]


def test_side_displacement_must_happen_in_main_flight():
    target = PARKOUR.command_target(landing_y=0.15)
    states = flight()
    # Slide to the commanded Y before takeoff, then leap straight. Accurate
    # touchdown alone must not certify this as an airborne diagonal leap.
    shift = torch.linspace(0, 0.15, 40)
    for key in ("position", "hip_position", "wheel_center_position"):
        states[key][:, :40, ..., 1] += shift.reshape((1, 40) + (1,) * (states[key].ndim - 3))
        states[key][:, 40:, ..., 1] += 0.15
    metrics = PARKOUR.parkour_leap_metrics(states, target)
    assert metrics["landing_xy_error"].item() < 0.05
    assert metrics["lateral_air_distance_deficit"].item() == pytest.approx(0.1)
    assert not metrics["accepted"].item()


def test_batch_disallows_silently_mixing_robot_style_limits():
    a = PARKOUR.command_target()
    b = PARKOUR.command_target(maximum_roll=0.5)
    with pytest.raises(ValueError, match="maximum_roll"):
        PARKOUR.BatchedParkourTargets([a, b])


def test_safe_names_and_target_factory_in_library_parser():
    values = [{"name": "h20_x40", "target": {"height": 0.2, "landing_x": 0.4, "landing_y": 0}}]
    goals = TO.parse_target_batch(values, PARKOUR.command_target)
    assert goals[0][1].minimum_air_distance == pytest.approx(0.4 * 2 / 3)
    with pytest.raises(ValueError):
        TO.parse_target_batch(values + values, PARKOUR.command_target)


def test_preflight_budget_is_measured_in_trigger_frame_without_changing_acceptance():
    states = flight()
    target = PARKOUR.command_target()
    m = PARKOUR.parkour_leap_metrics(states, target)
    assert m["preflight_forward_distance"].item() == pytest.approx(0.204)
    assert m["nominal_preflight_distance_budget"].item() == pytest.approx(0.2)
    assert m["preflight_distance_excess"].item() == pytest.approx(0.004, abs=1e-6)
    assert m["accepted"].item()  # nominal guide is not a new pass criterion
    for key in ("position", "velocity", "com_velocity", "hip_position", "wheel_center_position"):
        xy = states[key][..., :2].clone()
        states[key][..., 0], states[key][..., 1] = -xy[..., 1], xy[..., 0]
    states["yaw"][:] = torch.pi / 2
    rotated = PARKOUR.parkour_leap_metrics(states, target)
    torch.testing.assert_close(rotated["preflight_forward_distance"], m["preflight_forward_distance"])


def test_impulse_guide_excludes_airborne_force_and_does_not_redefine_acceptance():
    states = flight()
    force = torch.zeros(1, 361, 4, 3)
    force[..., :2, 2] = 100
    force[..., 2:, 2] = 40
    states["wheel_contact_force"] = force
    m = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target())
    assert m["front_push_impulse"].item() == pytest.approx(10)
    assert m["rear_push_impulse"].item() == pytest.approx(4.8)
    assert m["rear_push_impulse_fraction"].item() == pytest.approx(4.8 / 14.8)
    assert m["accepted"].item()
    states["wheel_bottom_height"][:, 28:40, 2:] = 0.2
    m = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target())
    assert m["rear_push_impulse"].item() == 0


def test_priority_keeps_safety_and_feasibility_ahead_of_scalar_score():
    metrics = dict(
        accepted=torch.tensor([False, False, False, True]),
        score=torch.tensor([1.0, 100, 200, 1000]),
        search_safety_violation=torch.tensor([0.1, 0, 0, 0]),
        search_max_violation=torch.tensor([0.1, 1, 0.5, 0]),
        search_total_violation=torch.tensor([0.1, 1, 2, 0]),
    )
    keys = PARKOUR.search_priority(metrics)
    assert PARKOUR.priority_order(keys).tolist() == [3, 2, 1, 0]
    assert PARKOUR.search_priority(metrics, replicas=2)[1, 0] == 1  # one good replica is insufficient
    assert PARKOUR.search_priority(metrics, replicas=2)[0, 1].item() == pytest.approx(0.1)
    metrics["search_max_violation"][2] = float("nan")
    assert torch.isfinite(PARKOUR.search_priority(metrics)).all()


def test_structured_launch_search_shortens_short_goal_and_preserves_landing_schedule():
    mean, _, low, high, _ = PARKOUR.parameter_distribution()
    samples = mean.repeat(2, 8, 1)
    before = samples.clone()
    goals = [PARKOUR.command_target(landing_x=x) for x in (0.4, 0.8)]
    proposals = PARKOUR.launch_proposals(samples, goals, low, high)
    torch.testing.assert_close(samples, before)
    old_pose, new_pose = samples[..., :70].reshape(2, 8, 7, 10), proposals[..., :70].reshape(2, 8, 7, 10)
    # Only the front drive is flexed; hind drive and all other landmarks remain.
    torch.testing.assert_close(new_pose[..., 1, 3] - new_pose[..., 1, 4], old_pose[..., 1, 3] - old_pose[..., 1, 4])
    torch.testing.assert_close(new_pose[..., 1, 6] - new_pose[..., 1, 7], old_pose[..., 1, 6] - old_pose[..., 1, 7])
    torch.testing.assert_close(new_pose[..., 2:, :], old_pose[..., 2:, :])
    assert (new_pose[:, -1, 1, 3] + new_pose[:, -1, 1, 4] > old_pose[:, -1, 1, 3] + old_pose[:, -1, 1, 4]).all()
    assert (proposals >= low).all() and (proposals <= high).all()
    schedule = PARKOUR.knot_schedule(proposals.reshape(-1, 84)).reshape(2, 8, 2, 9)
    assert schedule[0, :, :, 2].mean() < schedule[1, :, :, 2].mean()
    assert (schedule[..., 0, 2] < schedule[..., 1, 2]).all()
    # Compensation is bounded: landing landmarks never move later than before.
    old = PARKOUR.knot_schedule(samples.reshape(-1, 84)).reshape(2, 8, 2, 9)
    assert (schedule[..., 4] <= old[..., 4] + 1e-6).all()


def test_distance_budget_cannot_be_traded_for_later_takeoff_in_primary_rank():
    states = flight()
    # Translate the entire trajectory after trigger by 10 cm, retaining flight
    # height and duration: pre-takeoff travel must now be a primary violation.
    states["position"][:, 1:, 0] += 0.1
    metrics = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target())
    assert metrics["search_max_violation"].item() >= 2.07


def test_performance_refinement_keeps_seed_safe_and_does_not_redefine_acceptance():
    target = PARKOUR.command_target(height=0.24, landing_x=0.7)
    baseline = PARKOUR.parkour_leap_metrics(flight(), target)
    before = PARKOUR.performance_refinement_metrics(baseline, target, baseline)
    assert before["performance_guard_violation"].item() == 0
    assert not before["accepted"].item()
    improved = dict(
        baseline,
        height=baseline["height"] + 0.02,
        airborne_forward_distance=baseline["airborne_forward_distance"] + 0.03,
    )
    after = PARKOUR.performance_refinement_metrics(improved, target, baseline)
    assert after["performance_guard_violation"].item() == 0
    assert after["performance_loss"].item() < before["performance_loss"].item()
    assert not after["accepted"].item()
    assert "performance_loss" not in baseline


@pytest.mark.parametrize(
    "bad", ["height", "airborne_forward_distance", "max_pitch", "maximum_landing_sink", "rear_push_delay"]
)
def test_performance_improvement_cannot_buy_regression_in_approved_motion(bad):
    target = PARKOUR.command_target(height=0.24, landing_x=0.7)
    baseline = PARKOUR.parkour_leap_metrics(flight(), target)
    changed = dict(
        baseline,
        height=baseline["height"] + 0.02,
        airborne_forward_distance=baseline["airborne_forward_distance"] + 0.03,
    )
    delta = -0.04 if bad in ("height", "airborne_forward_distance", "rear_push_delay") else 0.1
    changed[bad] = baseline[bad] + delta
    after = PARKOUR.performance_refinement_metrics(changed, target, baseline)
    assert after["performance_guard_violation"].item() > 0
    before = PARKOUR.performance_refinement_metrics(baseline, target, baseline)
    keys = torch.cat((PARKOUR.search_priority(after), PARKOUR.search_priority(before)))
    assert PARKOUR.priority_order(keys).tolist() == [1, 0]


def test_performance_proposals_preserve_recovery_and_fold_times_for_local_half():
    mean, _, low, high, _ = PARKOUR.parameter_distribution()
    anchor = mean.repeat(2, 1)
    samples = anchor[:, None].repeat(1, 8, 1) + 0.002
    saved = samples.clone()
    proposals = PARKOUR.performance_proposals(samples, anchor, low, high)
    torch.testing.assert_close(samples, saved)
    torch.testing.assert_close(proposals[:, :4, 30:70], anchor[:, None, 30:70].expand(2, 4, 40))
    schedule = PARKOUR.knot_schedule(proposals[:, :4].reshape(-1, 84)).reshape(2, 4, 2, 9)
    reference = PARKOUR.knot_schedule(anchor)
    torch.testing.assert_close(schedule[..., 3:], reference[:, None, :, 3:].expand(2, 4, 2, 6))


def test_batched_performance_guards_keep_each_commands_own_baseline():
    goals = [PARKOUR.command_target(height=h, landing_x=x) for h, x in ((0.22, 0.6), (0.24, 0.7))]
    scalar = [PARKOUR.parkour_leap_metrics(flight(), goal) for goal in goals]
    metrics = {key: torch.cat([row[key] for row in scalar]) for key in scalar[0]}
    batched = PARKOUR.performance_refinement_metrics(metrics, PARKOUR.BatchedParkourTargets(goals), metrics)
    for i, goal in enumerate(goals):
        expected = PARKOUR.performance_refinement_metrics(scalar[i], goal, scalar[i])
        for key in expected:
            torch.testing.assert_close(batched[key][i], expected[key][0])
