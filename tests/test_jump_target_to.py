"""Target definitions, smooth control splines and non-gameable landing checks."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "locolab/scripts/locomotion/jump_target_to.py"
SPEC = importlib.util.spec_from_file_location("jump_target_to_test", PATH)
TO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TO
SPEC.loader.exec_module(TO)


def flight():
    n = 361
    position = torch.zeros(1, n, 3)
    position[:, :, 2] = 0.4
    position[:, 80:161, 2] += 0.2 * torch.sin(torch.linspace(0, torch.pi, 81))
    position[:, 80:161, 0] = torch.linspace(0, 0.1, 81)
    position[:, 80:161, 1] = torch.linspace(0, 0.15, 81)
    position[:, 161:, :2] = torch.tensor([0.1, 0.15])
    contacts = torch.ones(1, n, 4, dtype=torch.bool)
    contacts[:, 80:160] = False
    return {
        "position": position,
        "velocity": torch.zeros(1, n, 3),
        "contacts": contacts,
        **{
            key: torch.zeros(1, n)
            for key in ("yaw", "tilt", "nonfoot_force", "speed_violation", "limit_violation", "clearance")
        },
    }


def test_valid_synthetic_flight_and_landing_location():
    result = TO.trajectory_metrics(flight(), TO.JumpTarget())
    assert result["accepted"].item()
    assert result["landing_time"].item() == pytest.approx(0.8)
    assert result["height"].item() == pytest.approx(0.2)
    assert result["landing_xy_error"].item() < 1e-6


def test_landing_cannot_be_improved_by_rolling_after_contact():
    states = flight()
    states["position"][:, 160:200, 0] += 0.15
    result = TO.trajectory_metrics(states, TO.JumpTarget())
    assert result["landing_xy_error"].item() == pytest.approx(0.15)
    assert not result["accepted"].item()


@pytest.mark.parametrize(
    "failure", ["never_airborne", "never_landed", "overspeed", "collision", "not_settled", "bad_initial_pose"]
)
def test_physical_failure_rejected(failure):
    states = flight()
    if failure == "never_airborne":
        states["contacts"][:] = True
    elif failure == "never_landed":
        states["contacts"][:, 80:] = False
    elif failure == "overspeed":
        states["speed_violation"][:, 83] = 1.0
    elif failure == "collision":
        states["nonfoot_force"][:, 83] = 10.0
    elif failure == "bad_initial_pose":
        states["tilt"][:, 0] = 0.2
    else:
        states["velocity"][:, -5:, 2] = 0.5
    assert not TO.trajectory_metrics(states, TO.JumpTarget())["accepted"].item()


def test_target_frame_rotates_with_trigger_yaw_not_current_heading():
    states = flight()
    xy = states["position"][..., :2].clone()
    states["position"][..., 0] = -xy[..., 1]
    states["position"][..., 1] = xy[..., 0]
    states["yaw"][:] = torch.pi / 2
    states["position"] += torch.tensor([3.0, 4.0, 0.0])
    result = TO.trajectory_metrics(states, TO.JumpTarget())
    assert result["landing_xy_error"].item() < 1e-6
    assert result["accepted"].item()


def test_control_parameterization_starts_at_default_and_respects_physical_limits():
    mean, std = TO.initial_distribution()
    lower, upper = TO.parameter_bounds()
    assert mean.shape == std.shape == lower.shape == upper.shape == (70,)
    default = torch.tensor([0.0, 0.8, -1.5] * 4)
    lo = torch.tensor([-1.0, -1.5, -2.7] * 4)
    hi = torch.tensor([1.0, 3.5, -0.8] * 4)
    tape = TO.control_tape(mean.unsqueeze(0), default, lo, hi)
    assert tape.shape == (1, 90, 16)
    torch.testing.assert_close(tape[0, 0, :12], default)
    assert (tape[:, :, :12] >= lo).all() and (tape[:, :, :12] <= hi).all()
    assert torch.isfinite(tape).all()


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        TO.JumpTarget(height=0)
    with pytest.raises(ValueError):
        TO.JumpTarget(landing_y=float("nan"))


def test_export_keeps_actual_joint_states_and_rejects_unvalidated_rollouts():
    from dataclasses import asdict

    states = flight()
    for name, width in (
        ("omega", 3),
        ("joint_position", 12),
        ("joint_velocity", 12),
        ("wheel_position", 4),
        ("wheel_velocity", 4),
    ):
        states[name] = torch.zeros(1, 361, width)
    states["quaternion"] = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 361, 1)
    metrics = TO.trajectory_metrics(states, TO.JumpTarget())
    replay = {
        "accepted": True,
        "independent_replay": True,
        "simulation_dt": 0.005,
        "actual_physics_steps": 360,
        "mass_kg": 19.523,
        "gravity": 9.81,
        "target": asdict(TO.JumpTarget()),
        "states": {k: v[0].tolist() for k, v in states.items()},
        "metrics": {k: v[0].tolist() for k, v in metrics.items()},
        "leg_joint_names": TO.LEG_JOINT_NAMES,
        "wheel_joint_names": TO.WHEEL_JOINT_NAMES,
        "actuator_contract": "shared",
        "leg_effort_limit_nm": [23.5, 23.5, 35] * 4,
        "leg_velocity_limit_rad_s": [30, 30, 20] * 4,
        "self_collisions_enabled": True,
        "input_sha256": "test",
    }
    reference = TO.export_reference(replay)
    assert reference["joint_position"] == replay["states"]["joint_position"]
    assert reference["metadata"]["target_takeoff_time"] == 0.4
    assert reference["metadata"]["target_landing_time"] == 0.8
    assert reference["metadata"]["command_target"]["height"] == 0.2
    assert not reference["metadata"]["validation"]["robustness_certified"]
    ref_path = PATH.parents[2] / "source/locomotion/mdp/jump_reference.py"
    spec = importlib.util.spec_from_file_location("target_to_compatible_loader", ref_path)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    trajectory = loader.JumpReferenceTrajectory(reference)
    assert torch.isfinite(trajectory.sample(torch.tensor([0.0, 0.6, 1.8])).joint_position).all()
    replay["independent_replay"] = False
    with pytest.raises(ValueError):
        TO.export_reference(replay)
    replay["independent_replay"] = True
    replay["states"]["speed_violation"][85] = 2.0
    with pytest.raises(ValueError):
        TO.export_reference(replay)


@pytest.mark.parametrize("render", [False, True])
def test_rendering_cannot_change_the_physics_clock(render):
    class Sim:
        current_time_step_index = 0
        renders = 0

        def step(self, render):
            assert render is False
            self.current_time_step_index += 1

        def render(self):
            self.renders += 1

    class Scene:
        def update(self, dt):
            assert dt == 0.005

    sim = Sim()
    TO.step_dynamics(sim, Scene(), 0.005, render=render)
    assert sim.current_time_step_index == 1
    assert sim.renders == int(render)


def test_extra_physics_step_is_detected():
    class Sim:
        current_time_step_index = 0

        def step(self, render):
            self.current_time_step_index += 1

        def render(self):
            self.current_time_step_index += 1

    class Scene:
        def update(self, dt):
            pass

    with pytest.raises(RuntimeError, match="step count"):
        TO.step_dynamics(Sim(), Scene(), 0.005, render=True)


def test_batch_goals_match_scalar_scores_and_do_not_share_targets():
    from dataclasses import asdict

    goals = [TO.JumpTarget(), TO.JumpTarget(landing_y=-0.15), TO.JumpTarget(height=0.25)]
    batch = {k: v.repeat(3, *([1] * (v.ndim - 1))) for k, v in flight().items()}
    matrix = torch.tensor([list(asdict(goal).values()) for goal in goals])
    results = TO.trajectory_metrics(batch, matrix)
    for i, goal in enumerate(goals):
        scalar = TO.trajectory_metrics(flight(), goal)
        for key in results:
            torch.testing.assert_close(results[key][i], scalar[key][0])
    assert results["accepted"].tolist() == [True, False, False]
    with pytest.raises(ValueError, match="N x 5"):
        TO.trajectory_metrics(batch, matrix[:1])


def test_target_batch_names_are_unique_and_path_safe():
    entries = [{"name": "h20_x10_yp15", "target": {"height": 0.2}}]
    assert TO.parse_target_batch(entries)[0][1].height == 0.2
    for invalid in ([], entries * 2, [{"name": "../escape", "target": {}}]):
        with pytest.raises(ValueError):
            TO.parse_target_batch(invalid)


def test_lateral_mirroring_is_only_a_reversible_control_initial_guess():
    parameters = torch.arange(70).float()
    mirrored = TO.lateral_seed(parameters, -1)
    torch.testing.assert_close(TO.lateral_seed(mirrored, -1), parameters)
    unchanged = [i for i in range(70) if i % 10 not in (0, 2, 5, 8)]
    torch.testing.assert_close(mirrored[unchanged], parameters[unchanged])
    centered = TO.lateral_seed(parameters, 0).reshape(7, 10)
    assert not centered[:, [0, 2, 5, 8]].any()


def test_launch_refinement_does_not_change_lateral_or_recovery_modes():
    mask = TO.launch_parameter_mask()
    assert mask.shape == (70,) and mask.dtype == torch.bool
    assert mask.sum() == 15
    assert not mask.reshape(7, 10)[3:].any()
    assert not mask.reshape(7, 10)[:, [0, 1, 2, 5, 8]].any()
