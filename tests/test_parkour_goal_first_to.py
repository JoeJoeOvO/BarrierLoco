"""Explicit flat v4: accurate real jumps first, shape preferences second."""

import pytest
import torch
from test_parkour_leap_to import PARKOUR, flight


@pytest.mark.parametrize("preference", ["rear_order", "air_fraction", "tuck", "reach", "sink", "crossing"])
def test_v4_shape_is_soft_but_original_v3_certificate_is_unchanged(preference):
    states = flight()
    values = {}
    if preference == "rear_order":
        states["contacts"][:, 38:40] = states["contacts"][:, 38:40].flip(-1)
    elif preference == "air_fraction":
        values["minimum_air_distance"] = 0.5
    elif preference == "tuck":
        states["wheel_center_position"][:, 78:83, :, 2] -= 0.15
    elif preference == "reach":
        states["wheel_center_position"][:, 120:140, :2, 0] -= 0.16
    elif preference == "sink":
        states["position"][:, 140:150, 2] = 0.28
    else:
        states["clearance"][:, 70] = 0.02
    old = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target(**values))
    new = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target(criteria_version=4, **values))
    assert not old["accepted"].item()
    assert new["accepted"].item()
    assert new["search_style_cost"].item() > 0
    assert "search_target_violation" not in old


@pytest.mark.parametrize(
    "failure",
    [
        "height",
        "landing",
        "rolling",
        "clearance",
        "short_flight",
        "no_landing",
        "unstable_recovery",
        "collision",
        "speed",
        "joint_limit",
        "roll",
        "pitch",
        "yaw",
        "touch_pitch",
        "rebound",
        "braking",
    ],
)
def test_v4_does_not_relax_accuracy_flight_or_safety(failure):
    states = flight()
    if failure == "height":
        states["position"][:, :, 2] = 0.4 + (states["position"][:, :, 2] - 0.4) * 0.75
    elif failure == "landing":
        states["position"][:, 120, 1] = 0.1
    elif failure == "rolling":
        states["contacts"][:] = True
    elif failure == "clearance":
        states["clearance"][:] = 0.001
    elif failure == "short_flight":
        states["contacts"][:, 40:85] = True
    elif failure == "no_landing":
        states["contacts"][:, 40:] = False
    elif failure == "unstable_recovery":
        states["wheel_bottom_height"][:, -50:, 0] = 0.01
    elif failure == "collision":
        states["nonfoot_force"][:, 70] = 6
    elif failure == "speed":
        states["speed_violation"][:, 70] = 0.2
    elif failure == "joint_limit":
        states["limit_violation"][:, 70] = 0.01
    elif failure in ("roll", "pitch", "yaw"):
        states[failure][:, 70] = 0.7
    elif failure == "touch_pitch":
        states["pitch"][:, 120:140] = 0.4
    elif failure == "rebound":
        states["contacts"][:, 200:220] = False
    else:
        states["com_velocity"][:, 40, 0] = 0.3
    m = PARKOUR.parkour_leap_metrics(states, PARKOUR.command_target(criteria_version=4))
    assert not m["accepted"].item()


def test_goal_first_priority_cannot_trade_accuracy_for_pretty_style():
    metrics = dict(
        accepted=torch.tensor([False, False, False, False, True]),
        search_safety_violation=torch.tensor([0.1, 0, 0, 0, 0]),
        search_motion_violation=torch.tensor([0.0, 0.1, 0, 0, 0]),
        search_target_violation=torch.tensor([0.0, 0, 0.5, 0.1, 0]),
        search_target_style_cost=torch.tensor([0.0, 0, 0, 100, 100]),
        score=torch.tensor([0.0, 0, 0, 1000, 2000]),
    )
    assert PARKOUR.priority_order(PARKOUR.search_priority(metrics)).tolist() == [4, 3, 2, 1, 0]
    assert PARKOUR.search_priority(metrics, replicas=5)[0, 0] == 1


def test_v4_batched_goals_match_independent_replay_metrics():
    goals = [PARKOUR.command_target(height=h, landing_x=x, criteria_version=4) for h, x in ((0.15, 0.4), (0.2, 0.6))]
    original = flight()
    states = {k: v.repeat((4,) + (1,) * (v.ndim - 1)) for k, v in original.items()}
    metrics = PARKOUR.parkour_leap_metrics(states, PARKOUR.BatchedParkourTargets(goals, repeats=2))
    for i in range(4):
        scalar = PARKOUR.parkour_leap_metrics(original, goals[i // 2])
        for key in scalar:
            torch.testing.assert_close(metrics[key][i], scalar[key][0], atol=1e-4, rtol=1e-5)


def test_v4_is_opt_in_not_a_reinterpretation_of_unversioned_targets():
    assert PARKOUR.command_target().criteria_version == 3
    with pytest.raises(ValueError, match="share criteria_version"):
        PARKOUR.BatchedParkourTargets([PARKOUR.command_target(), PARKOUR.command_target(criteria_version=4)])
