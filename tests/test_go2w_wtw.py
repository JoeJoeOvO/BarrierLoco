"""Numerical contracts for the WTW paper port, runnable without Isaac Sim."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "locolab/source/locomotion/mdp"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wtw = load_module("wtw_math_test", MDP / "wtw_math.py")
kinematics = load_module("wtw_kinematics_test", MDP / "wtw_kinematics.py")


def commands(kind="trot", count=1):
    result = torch.tensor([0.5, 0, 0, 0.42, 2, *wtw.PHASE_PRESETS[kind], 0.5, 0.08]).repeat(count, 1)
    if kind == "rolling":
        result[:, 8] = 1
        result[:, 9] = 0
    return result


def urdf_wheel_centers_and_bottoms(leg_pose):
    """Independent FK through the URDF joint frames, including wheel collision offsets."""
    root = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    n = len(leg_pose)
    centers, bottoms = [], []
    for leg_index, leg in enumerate(kinematics.LEG_NAMES):
        position = leg_pose.new_zeros(n, 3)
        rotation = torch.eye(3, dtype=leg_pose.dtype).expand(n, 3, 3).clone()
        parent = "base"
        for joint_index, name in enumerate(("hip", "thigh", "calf", "foot")):
            joint = root.find(f"joint[@name='{leg}_{name}_joint']")
            assert joint.find("parent").attrib["link"] == parent
            parent = joint.find("child").attrib["link"]
            assert joint.find("origin").attrib["rpy"] == "0 0 0"
            translation = leg_pose.new_tensor([float(x) for x in joint.find("origin").attrib["xyz"].split()])
            position += torch.einsum("nij,j->ni", rotation, translation)
            if name == "foot":
                break  # wheel rotation does not change its axial collision center
            axis = leg_pose.new_tensor([float(x) for x in joint.find("axis").attrib["xyz"].split()])
            angle = leg_pose[:, leg_index, joint_index]
            # Rodrigues rotation, independent of the planar IK equations.
            x, y, z = axis.tolist()
            cross = leg_pose.new_tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            local_rotation = (
                torch.eye(3, dtype=leg_pose.dtype)[None] * angle.cos()[:, None, None]
                + (1 - angle.cos())[:, None, None] * torch.outer(axis, axis)[None]
                + angle.sin()[:, None, None] * cross[None]
            )
            rotation = torch.bmm(rotation, local_rotation)
        collision = root.find(f"link[@name='{leg}_foot']/collision")
        center = leg_pose.new_tensor([float(x) for x in collision.find("origin").attrib["xyz"].split()])
        position += torch.einsum("nij,j->ni", rotation, center)
        cylinder = collision.find("geometry/cylinder")
        radius, width = float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])
        axis_z = rotation[:, 2, 1]  # cylinder axis is joint/local Y
        bottom = position[:, 2] - radius * (1 - axis_z.square()).sqrt() - width / 2 * axis_z.abs()
        centers.append(position)
        bottoms.append(bottom)
    return torch.stack(centers, dim=1), torch.stack(bottoms, dim=1)


def test_height_ik_matches_urdf_fk_over_entire_command_range():
    height = torch.linspace(0.22, 0.46, 1001, dtype=torch.float64)
    original = torch.tensor([0.0, 0.8, -1.5], dtype=torch.float64).repeat(len(height), 4, 1)
    pose = kinematics.height_reference_pose(height, original)
    original_centers, _ = urdf_wheel_centers_and_bottoms(original)
    centers, bottoms = urdf_wheel_centers_and_bottoms(pose)
    torch.testing.assert_close(centers[..., :2], original_centers[..., :2], atol=1e-10, rtol=0)
    torch.testing.assert_close(bottoms + height[:, None], torch.zeros_like(bottoms), atol=1e-10, rtol=0)
    assert (pose[..., 0] == 0).all()
    assert (pose[..., 2] < 0).all()


def test_height_ik_recovers_original_pose_at_its_geometric_height():
    original = torch.tensor([0.0, 0.8, -1.5], dtype=torch.float64).repeat(1, 4, 1)
    _, bottoms = urdf_wheel_centers_and_bottoms(original)
    pose = kinematics.height_reference_pose(-bottoms[:, 0], original)
    torch.testing.assert_close(pose, original, atol=1e-10, rtol=0)


@pytest.mark.parametrize("height", [0.05, 0.086, 0.60, float("nan"), float("inf")])
def test_height_ik_rejects_infeasible_or_nonfinite_requests(height):
    original = torch.tensor([0.0, 0.8, -1.5]).repeat(1, 4, 1)
    with pytest.raises(ValueError):
        kinematics.height_reference_pose(torch.tensor([height]), original)


@pytest.mark.parametrize("index,value", [(0, 0.1), (2, 1.5)])
def test_height_ik_rejects_unsupported_default_pose(index, value):
    original = torch.tensor([0.0, 0.8, -1.5]).repeat(1, 4, 1)
    original[0, 0, index] = value
    with pytest.raises(ValueError):
        kinematics.height_reference_pose(torch.tensor([0.34]), original)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("trot", [True, False, False, True]),
        ("bound", [False, False, True, True]),
    ],
)
def test_paper_named_foot_pairing(kind, expected):
    cmd = commands(kind)
    for time, expected_contact in ((0.1, expected), (0.6, [not x for x in expected])):
        _, _, _, contact = wtw.contact_schedule(torch.tensor([time]), cmd)
        assert (contact[0] > 0.5).tolist() == expected_contact
    assert wtw.FOOT_NAMES == ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


def test_contact_smoothing_matches_scalar_normal_cdf_reference():
    cmd = commands(count=11)
    cmd[:, 8] = 0.6
    times = torch.linspace(0, 0.999, 11)
    raw, warped, clocks, contact = wtw.contact_schedule(times, cmd)
    expected = []

    def cdf(x):
        return (1 + math.erf(x / (0.07 * math.sqrt(2)))) / 2

    for row in raw.tolist():
        expected.append([])
        for phi in row:
            phi = phi / 1.2 if phi < 0.6 else 0.5 + (phi - 0.6) / 0.8
            expected[-1].append(cdf(phi) * (1 - cdf(phi - 0.5)) + cdf(phi - 1) * (1 - cdf(phi - 1.5)))
    torch.testing.assert_close(contact, torch.tensor(expected), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(clocks, torch.sin(2 * math.pi * warped))


def test_duration_and_triangular_clearance_have_correct_timing():
    cmd = commands(count=3)
    cmd[:, 5:8] = 0
    cmd[:, 8] = 0.6
    _, warped, _, _ = wtw.contact_schedule(torch.tensor([0.6, 0.8, 0.999999]), cmd)
    height = wtw.swing_height_target(warped, cmd)
    torch.testing.assert_close(height[0], torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(height[1], torch.full((4,), 0.08))
    assert height[2].max() < 1e-6


def test_rolling_is_finite_at_every_clock_phase():
    cmd = commands("rolling", 100)
    cmd[:, 4] = torch.linspace(1.5, 4.0, 100)
    cmd[:, 5:8] = wtw.sample_phase_neighborhood(cmd[:, 5:8], 5.0, 10.0)
    wtw.validate_commands(cmd)
    raw, warped, clocks, contact = wtw.contact_schedule(torch.rand(100), cmd)
    assert all(torch.isfinite(x).all() for x in (raw, warped, clocks, contact))
    assert torch.all(contact == 1)
    torch.testing.assert_close(warped, raw)
    assert (clocks == 0).all()
    assert torch.all(wtw.swing_height_target(warped, cmd) == 0)


@pytest.mark.parametrize(
    "index,value", [(3, 0), (4, -1), (5, 1), (6, -1.01), (8, 0), (8, 1), (9, -0.1), (0, float("nan"))]
)
def test_invalid_commands_rejected(index, value):
    cmd = commands()
    cmd[0, index] = value
    with pytest.raises(ValueError):
        wtw.validate_commands(cmd)


def test_rolling_contract_rejects_frozen_swing_requests():
    cmd = commands()
    cmd[0, 4] = 0
    with pytest.raises(ValueError, match="Periodic stepping requires positive frequency"):
        wtw.validate_commands(cmd)


def test_legacy_zero_frequency_full_stance_is_still_valid():
    cmd = commands("rolling")
    cmd[:, 4] = 0
    wtw.validate_commands(cmd)
    _, _, clocks, contact = wtw.contact_schedule(torch.tensor([0.7]), cmd)
    assert (clocks == 0).all() and (contact == 1).all()


@pytest.mark.parametrize("kind", ["trot", "bound"])
def test_zero_clearance_does_not_turn_stepping_into_full_stance(kind):
    cmd = commands(kind)
    cmd[:, 9] = 0
    wtw.validate_commands(cmd)
    _, warped, clocks, contact = wtw.contact_schedule(torch.tensor([0.1]), cmd)
    assert (contact < 0.5).sum() == 2
    assert (wtw.swing_height_target(warped, cmd) == 0).all()
    torch.testing.assert_close(clocks, torch.sin(2 * math.pi * warped))
    assert clocks.abs().max() > 0


@pytest.mark.parametrize("kind", ["rolling", "trot", "bound"])
def test_phase_neighborhood_limits_parameters_and_all_wheel_pairs(kind):
    torch.manual_seed(42)
    center = torch.tensor(wtw.PHASE_PRESETS[kind])
    samples = wtw.sample_phase_neighborhood(center.repeat(8192, 1), 5.0, 10.0)
    delta = samples - center
    assert (delta.abs() <= 5.0 / 360.0 + 1e-7).all()
    assert (delta.min(dim=0).values < -0.01).all()
    assert (delta.max(dim=0).values > 0.01).all()
    assert (samples[:, center == 0] < 0).any()  # no wrapping negative samples up to ~1

    # Independent Appendix D mapping, including the half-cycle preset offsets.
    def offsets(parameters):
        a, b, c = parameters.unbind(-1)
        return torch.stack((b + c, a + c, a, b), dim=-1)

    actual, nominal = offsets(samples), offsets(center)
    pair_error = (actual[:, :, None] - actual[:, None, :]) - (nominal[:, None] - nominal[None, :])
    circular_error_deg = (torch.remainder(pair_error + 0.5, 1) - 0.5).abs() * 360
    assert circular_error_deg.max() <= 10.0 + 1e-4
    cmd = commands(kind, len(samples))
    cmd[:, 5:8] = samples
    wtw.validate_commands(cmd)


def test_phase_neighborhood_rejects_bad_pairs_without_resampling_accepted_rows(monkeypatch):
    centers = torch.tensor([wtw.PHASE_PRESETS["trot"], wtw.PHASE_PRESETS["bound"]])
    draws = []

    def draw(template):
        draws.append(template.shape)
        if len(draws) == 1:
            # +/-5 deg on each parameter, but row 0 has a 15 deg FR/RR error.
            return template.new_tensor([[0.0, 1.0, 1.0], [0.5, 0.75, 0.5]])
        return torch.full_like(template, 0.5)

    monkeypatch.setattr(torch, "rand_like", draw)
    samples = wtw.sample_phase_neighborhood(centers, 5.0, 10.0)
    assert draws == [torch.Size([2, 3]), torch.Size([1, 3])]
    torch.testing.assert_close(samples[0], centers[0])
    torch.testing.assert_close(samples[1], centers[1] + torch.tensor([0, 2.5 / 360, 0]))


def test_phase_neighborhood_has_bounded_retries_and_never_accepts_bad_pairs(monkeypatch):
    centers = torch.tensor([wtw.PHASE_PRESETS["trot"]])
    draws = []

    def draw(template):
        draws.append(1)
        return template.new_tensor([[0.0, 1.0, 1.0]]).expand_as(template)

    monkeypatch.setattr(torch, "rand_like", draw)
    with pytest.raises(RuntimeError, match="exhausted retries"):
        wtw.sample_phase_neighborhood(centers, 5.0, 10.0)
    assert len(draws) == 32


@pytest.mark.parametrize(
    "jitter,pair_limit", [(-1, 10), (90, 10), (float("nan"), 10), (5, -1), (5, 181), (5, 0), (5, float("inf"))]
)
def test_invalid_phase_neighborhood_rejected(jitter, pair_limit):
    with pytest.raises(ValueError):
        wtw.sample_phase_neighborhood(torch.zeros(2, 3), jitter, pair_limit)


def test_zero_phase_jitter_and_empty_batch():
    centers = torch.tensor(list(wtw.PHASE_PRESETS.values()))
    torch.testing.assert_close(wtw.sample_phase_neighborhood(centers, 0.0, 0.0), centers)
    assert wtw.sample_phase_neighborhood(centers[:0], 5.0, 10.0).shape == (0, 3)


@pytest.mark.parametrize("kind", ["trot", "bound"])
def test_signed_phase_parameters_match_wrapped_schedule(kind):
    signed = commands(kind, 2)
    signed[:, 5:8] += torch.tensor([-3.0, 2.0, -1.0]) / 360
    wrapped = signed.clone()
    wrapped[:, 5:8] = torch.remainder(wrapped[:, 5:8], 1)
    wtw.validate_commands(signed)
    wtw.validate_commands(wrapped)
    times = torch.tensor([0.07, 0.57])
    for actual, expected in zip(wtw.contact_schedule(times, signed), wtw.contact_schedule(times, wrapped)):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_force_and_velocity_costs_have_correct_direction_and_gate():
    zeros = torch.zeros(1, 4, 3)
    nonzero = torch.ones_like(zeros) * 10
    stance = torch.ones(1, 4)
    swing = torch.zeros_like(stance)
    assert wtw.contact_force_cost(nonzero, swing).item() > wtw.contact_force_cost(zeros, swing).item()
    assert wtw.contact_force_cost(nonzero, stance).item() == 0
    assert wtw.contact_velocity_cost(nonzero, stance).item() > wtw.contact_velocity_cost(zeros, stance).item()
    assert wtw.contact_velocity_cost(nonzero, swing).item() == 0


def test_wheel_material_contact_velocity_cancels_pure_rolling():
    axis = torch.tensor([0.0, 1, 0]).expand(2, 4, 3)
    v = torch.zeros(2, 4, 3)
    v[..., 0] = 0.86
    omega = torch.zeros_like(v)
    omega[..., 1] = 10
    offsets = torch.tensor([-0.048, 0.048, -0.048, 0.048])
    displacement, material_v = wtw.wheel_contact_geometry(axis, v, omega, 0.086, 0.0225, offsets)
    torch.testing.assert_close(material_v, torch.zeros_like(v), atol=1e-6, rtol=0)
    torch.testing.assert_close(displacement[..., 2], torch.full((2, 4), -0.086))
    v[..., 1] = 0.5
    _, material_v = wtw.wheel_contact_geometry(axis, v, omega, 0.086, 0.0225, offsets)
    assert wtw.contact_velocity_cost(material_v, torch.ones(2, 4)).min() > 0


def test_wheel_collision_dimensions_and_axes_match_go2w_urdf():
    root = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    for name, offset in zip(wtw.FOOT_NAMES, (-0.048, 0.048, -0.048, 0.048)):
        collision = root.find(f"link[@name='{name}']/collision")
        assert float(collision.find("geometry/cylinder").attrib["radius"]) == 0.086
        assert float(collision.find("geometry/cylinder").attrib["length"]) == 0.045
        assert float(collision.find("origin").attrib["xyz"].split()[1]) == offset
        assert root.find(f"joint[@name='{name}_joint']/axis").attrib["xyz"] == "0 1 0"


@pytest.fixture
def command_term(monkeypatch):
    # Only the Isaac manager boundary is replaced; use real implementation and tensors.
    class CommandTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device
            self.metrics = {}
            self.time_left = torch.zeros(env.num_envs)

        def reset(self, env_ids):
            self._resample_command(env_ids)
            return {}

    manager = ModuleType("isaaclab.managers")
    manager.CommandTerm = CommandTerm
    manager.SceneEntityCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab.managers", manager)
    package = ModuleType("wtw_under_test")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load_module("wtw_under_test.command.wtw_command", MDP / "command/wtw_command.py")
    n = 12
    joint_names = list(kinematics.LEG_JOINT_NAMES) + [name + "_joint" for name in wtw.FOOT_NAMES]
    default = torch.tensor([0.0, 0.8, -1.5] * 4 + [0.0] * 4).repeat(n, 1)
    urdf = ET.parse(ROOT / "locolab/source/locomotion/assets/Go2W/urdf/go2w.urdf").getroot()
    limits = []
    for name in joint_names:
        joint = urdf.find(f"joint[@name='{name}']")
        if joint.attrib["type"] == "continuous":
            limits.append([-float("inf"), float("inf")])
        else:
            lower, upper = (float(joint.find("limit").attrib[key]) for key in ("lower", "upper"))
            middle, half = (lower + upper) / 2, 0.9 * (upper - lower) / 2
            limits.append([middle - half, middle + half])
    robot = SimpleNamespace(
        find_bodies=lambda names, preserve_order: (list(range(4)), names),
        find_joints=lambda names, preserve_order: ([joint_names.index(name) for name in names], names),
        data=SimpleNamespace(
            root_lin_vel_b=torch.zeros(n, 3),
            root_pos_w=torch.zeros(n, 3),
            heading_w=torch.zeros(n),
            default_joint_pos=default,
            joint_pos=default.clone(),
            soft_joint_pos_limits=torch.tensor(limits).repeat(n, 1, 1),
        ),
    )
    sensor = SimpleNamespace(
        find_bodies=robot.find_bodies,
        data=SimpleNamespace(net_forces_w=torch.zeros(n, 4, 3)),
    )

    class Scene(dict):
        sensors = {"contact_forces": sensor}
        env_origins = torch.zeros(n, 3)

    cfg = SimpleNamespace(
        debug_vis=False,
        sampling_weights=(1.0, 1.0, 1.0),
        standing_probability=0.1,
        kappa=0.07,
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-math.pi, math.pi),
        heading_command=True,
        heading=(-math.pi, math.pi),
        rel_heading_envs=0.8,
        heading_control_stiffness=2.0,
        body_height=(0.22, 0.46),
        height_conditioned_reward_reference=True,
        frequency=(1.5, 4.0),
        phase_jitter_deg=5.0,
        max_pair_phase_error_deg=10.0,
        footswing_height=(0.0, 0.1),
        gait_duration=0.5,
        asset_name="robot",
        contact_sensor_name="contact_forces",
        wheel_radius=0.086,
        wheel_center_offsets=(-0.048, 0.048, -0.048, 0.048),
    )
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, max_episode_length=1000, scene=Scene(robot=robot))
    return module.WTWCommand(cfg, env)


def test_command_reset_subset_and_read_does_not_advance_clock(command_term):
    term = command_term
    term.set_command(commands()[0])
    term._update_command()
    torch.testing.assert_close(term.phase, torch.full((12,), 0.04))
    before = term.phase.clone()
    _ = term.command
    _ = term.clock_inputs
    torch.testing.assert_close(term.phase, before)
    term.reset([2, 5])
    assert term.phase[2] == term.phase[5] == 0
    assert term.phase[0] == before[0]
    wtw.validate_commands(term.command)
    term._update_command()
    assert term.phase[2] == term.phase[5] == 0
    assert term.phase[0] > before[0]


def test_switch_retains_global_clock_and_recomputes_contacts(command_term):
    term = command_term
    term.phase[:] = 0.1
    term.set_command(commands("trot")[0])
    first = term.desired_contact.clone()
    term.set_command(commands("bound")[0])
    torch.testing.assert_close(term.phase, torch.full((12,), 0.1))
    assert not torch.equal(first, term.desired_contact)
    assert torch.isinf(term.time_left).all()
    term.set_command(commands("rolling")[0], [1, 3])
    assert (term.desired_contact[[1, 3]] == 1).all()
    assert (term.clock_inputs[[1, 3]] == 0).all()
    term._update_command()
    torch.testing.assert_close(term.phase[[1, 3]], torch.full((2,), 0.14))
    assert (term.desired_contact[[1, 3]] == 1).all()
    assert (term.clock_inputs[[1, 3]] == 0).all()
    term.set_command(commands("trot")[0], [1, 3])
    torch.testing.assert_close(term.phase[[1, 3]], torch.full((2,), 0.14))
    torch.testing.assert_close(term.clock_inputs[[1, 3]], torch.sin(2 * math.pi * term.leg_phase[[1, 3]]))


@pytest.mark.parametrize("category", [0, 1, 2])
def test_every_category_samples_continuous_positive_frequency(command_term, category):
    term = command_term
    term.cfg.sampling_weights = tuple(float(i == category) for i in range(3))
    term.cfg.standing_probability = 0.0
    term._resample_command(torch.arange(term.num_envs))
    wtw.validate_commands(term.command)
    frequency = term.command[:, 4]
    assert ((frequency >= 1.5) & (frequency <= 4.0)).all()
    assert frequency.unique().numel() > 1
    before = term.phase.clone()
    term._update_command()
    torch.testing.assert_close(term.phase, before + frequency * term._env.step_dt)
    if category == 0:
        assert (term.command[:, 8] == 1).all()
        assert (term.command[:, 9] == 0).all()
        assert (term.desired_contact == 1).all()
        assert (term.clock_inputs == 0).all()
    else:
        assert term.clock_inputs.abs().max() > 0


def test_standing_keeps_positive_frequency_and_full_support(command_term):
    term = command_term
    term.cfg.standing_probability = 1.0
    term._resample_command(torch.arange(term.num_envs))
    term._update_command()
    assert (term.command[:, :3] == 0).all()
    assert (term.command[:, 4] >= 1.5).all()
    assert (term.desired_contact == 1).all()
    assert (term.clock_inputs == 0).all()


@pytest.mark.parametrize("category", [0, 1, 2])
def test_sampled_phase_parameters_stay_fixed_between_command_changes(command_term, category):
    term = command_term
    term.cfg.sampling_weights = tuple(float(i == category) for i in range(3))
    term.cfg.standing_probability = 0.0
    term._resample_command(torch.arange(term.num_envs))
    before = term.command[:, 5:8].clone()
    center = torch.tensor(list(wtw.PHASE_PRESETS.values())[category])
    assert ((before - center).abs() <= 5.0 / 360.0 + 1e-7).all()
    assert not torch.equal(before, center.expand_as(before))
    for _ in range(20):
        term._update_command()
    torch.testing.assert_close(term.command[:, 5:8], before)
    term._resample_command([1, 3])
    assert not torch.equal(term.command[[1, 3], 5:8], before[[1, 3]])
    torch.testing.assert_close(term.command[[0, 2, 4], 5:8], before[[0, 2, 4]])


def test_manual_signed_phase_preserves_command_and_changes_schedule(command_term):
    term = command_term
    term.phase[:] = 0.48
    term.set_command(commands()[0])
    before = term.desired_contact.clone()
    manual = commands()[0]
    manual[5:8] += torch.tensor([2.0, -3.0, 1.0]) / 360
    term.set_command(manual)
    torch.testing.assert_close(term.command, manual.expand(term.num_envs, -1))
    assert not torch.equal(term.desired_contact, before)
    torch.testing.assert_close(term.clock_inputs, torch.sin(2 * math.pi * term.leg_phase))


def test_height_reward_reference_updates_without_changing_robot_defaults_or_state(command_term):
    term = command_term
    asset_default = term.robot.data.default_joint_pos.clone()
    joint_pos_before = term.robot.data.joint_pos.clone()
    root_pos_before = term.robot.data.root_pos_w.clone()
    cmd = commands("rolling", term.num_envs)
    cmd[:, 3] = torch.linspace(0.22, 0.46, term.num_envs)
    term.set_command(cmd)
    pose = term.reward_joint_pos[:, term.leg_joint_ids].reshape(-1, 4, 3)
    _, bottoms = urdf_wheel_centers_and_bottoms(pose)
    torch.testing.assert_close(bottoms + cmd[:, 3:4], torch.zeros_like(bottoms), atol=1e-6, rtol=0)
    limits = term.robot.data.soft_joint_pos_limits
    assert ((term.reward_joint_pos >= limits[..., 0]) & (term.reward_joint_pos <= limits[..., 1])).all()
    torch.testing.assert_close(term.reward_joint_pos[:, term.wheel_joint_ids], asset_default[:, term.wheel_joint_ids])
    before = term.reward_joint_pos.clone()
    term.reset([2, 5])
    unchanged = [i for i in range(term.num_envs) if i not in (2, 5)]
    torch.testing.assert_close(term.reward_joint_pos[unchanged], before[unchanged])
    _, bottoms = urdf_wheel_centers_and_bottoms(term.reward_joint_pos[:, term.leg_joint_ids].reshape(-1, 4, 3))
    torch.testing.assert_close(bottoms + term.command[:, 3:4], torch.zeros_like(bottoms), atol=1e-6, rtol=0)
    torch.testing.assert_close(term.robot.data.default_joint_pos, asset_default)
    torch.testing.assert_close(term.robot.data.joint_pos, joint_pos_before)
    torch.testing.assert_close(term.robot.data.root_pos_w, root_pos_before)
    before = term.reward_joint_pos.clone()
    term._update_command()
    torch.testing.assert_close(term.reward_joint_pos, before)  # frequency does not drive the reward pose


@pytest.mark.parametrize("height,error", [(0.6, "workspace"), (0.18, "soft joint limits")])
def test_bad_height_request_does_not_commit_partial_command(command_term, height, error):
    term = command_term
    command_before, reference_before = term.command.clone(), term.reward_joint_pos.clone()
    cmd = commands()[0]
    cmd[3] = height
    with pytest.raises(ValueError, match=error):
        term.set_command(cmd, [1, 3])
    torch.testing.assert_close(term.command, command_before)
    torch.testing.assert_close(term.reward_joint_pos, reference_before)


def test_disabling_height_reward_reference_restores_fixed_reward_targets(command_term):
    term = command_term
    term.cfg.height_conditioned_reward_reference = False
    cmd = commands("rolling", term.num_envs)
    cmd[:, 3] = torch.linspace(0.22, 0.46, term.num_envs)
    term.set_command(cmd)
    torch.testing.assert_close(term.reward_joint_pos, term.robot.data.default_joint_pos)


def test_height_ik_stays_out_of_action_and_joint_observation_configuration():
    # Inspect real configuration inheritance without importing the simulator.
    tasks = ROOT / "locolab/source/locomotion/tasks"
    gait = ast.parse((tasks / "Go2W/gait_env_cfg.py").read_text())
    flat = ast.parse((tasks / "Go2W/flat_env_cfg.py").read_text())
    base = ast.parse((tasks / "velocity_env_cfg.py").read_text())

    def child_class(node, name):
        return next(child for child in node.body if isinstance(child, ast.ClassDef) and child.name == name)

    def assigned_value(node, name):
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.target.id == name:
                return child.value
            if isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in child.targets
            ):
                return child.value
        return None

    env_cfg = child_class(gait, "Go2WGaitEnvCfg")
    assert [ast.unparse(parent) for parent in env_cfg.bases] == ["Go2WFlatEnvCfg"]
    assert assigned_value(env_cfg, "actions") is None  # unchanged Flat actions
    assert not any(isinstance(node, ast.Attribute) and node.attr == "actions" for node in ast.walk(env_cfg))
    flat_env = child_class(flat, "Go2WFlatEnvCfg")
    assert ast.unparse(assigned_value(flat_env, "actions")) == "Go2WActionsCfg()"
    action_cfg = child_class(flat, "Go2WActionsCfg")
    for name, class_name, scale in (
        ("joint_pos", "JointPositionActionCfg", 0.25),
        ("joint_vel", "JointVelocityActionCfg", 10.0),
    ):
        action = assigned_value(action_cfg, name)
        assert ast.unparse(action.func) == f"mdp.{class_name}"
        keywords = {item.arg: item.value for item in action.keywords}
        assert ast.literal_eval(keywords["scale"]) == scale
        assert ast.literal_eval(keywords["use_default_offset"]) is True

    gait_obs = child_class(gait, "Go2WGaitObservationsCfg")
    flat_obs = child_class(flat, "Go2WObservationsCfg")
    base_obs = child_class(base, "LocomotionObservationsCfg")
    for group, function in (("PolicyCfg", "mdp.joint_pos_rel"), ("CriticCfg", "mdp.joint_pos")):
        gait_group = child_class(gait_obs, group)
        assert [ast.unparse(parent) for parent in gait_group.bases] == [f"Go2WObservationsCfg.{group}"]
        assert assigned_value(gait_group, "joint_pos") is None
        assert not any(isinstance(node, ast.Attribute) and node.attr == "joint_pos" for node in ast.walk(gait_group))
        flat_group = child_class(flat_obs, group)
        assert assigned_value(flat_group, "joint_pos") is None
        assert [ast.unparse(parent) for parent in flat_group.bases] == [f"LocomotionObservationsCfg.{group}"]
        term = assigned_value(child_class(base_obs, group), "joint_pos")
        assert ast.unparse(next(item.value for item in term.keywords if item.arg == "func")) == function


def test_sampler_accepts_zero_clearance_for_periodic_gaits(command_term):
    term = command_term
    term.cfg.sampling_weights = (0.0, 1.0, 0.0)
    term.cfg.standing_probability = 0.0
    term.cfg.footswing_height = (0.0, 0.0)
    term._resample_command(torch.arange(term.num_envs))
    wtw.validate_commands(term.command)
    assert (term.command[:, 9] == 0).all()
    assert (term.command[:, 8] == 0.5).all()
    assert (term.command[:, 4] > 0).all()


def test_height_reward_weight_is_five_times_original_wtw_cost():
    source = ROOT / "locolab/source/locomotion/tasks/Go2W/gait_env_cfg.py"
    config = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Go2WGaitRewardsCfg"
    )
    terms = {node.targets[0].id: node.value for node in config.body if isinstance(node, ast.Assign)}
    height = {item.arg: item.value for item in terms["base_height"].keywords}
    assert ast.unparse(height["func"]) == "wtw_rewards.wtw_height"
    weight = ast.literal_eval(height["weight"])
    assert weight == -50.0
    assert weight * 0.1**2 == pytest.approx(-0.5)  # 10 cm error, before reward-manager dt scaling
    for name in ("stand_still", "hip_default"):
        assert ast.literal_eval(next(item.value for item in terms[name].keywords if item.arg == "weight")) == -0.5
    assert ast.literal_eval(terms["thigh_calf_default"]) is None


def test_training_sampling_and_manual_input_validation(command_term):
    term = command_term
    for _ in range(10):
        term._resample_command(torch.arange(12))
        wtw.validate_commands(term.command)
    with pytest.raises(ValueError):
        term.set_command(torch.zeros(9))
    bad = commands()[0]
    bad[4] = float("inf")
    with pytest.raises(ValueError):
        term.set_command(bad)


def test_heading_wrap_gain_clipping_and_direct_yaw_rows(command_term):
    term = command_term
    term.set_command(commands("rolling")[0])
    term.is_heading_env[:3] = True
    term.heading_target[:3] = torch.tensor([-math.pi + 0.1, 2.0, -2.0])
    term.robot.data.heading_w[0] = math.pi - 0.1
    term._command[3, 2] = 0.7
    term._update_command()
    torch.testing.assert_close(term.command[:4, 2], torch.tensor([0.4, math.pi, -math.pi, 0.7]))
    term.robot.data.heading_w[0] = term.heading_target[0]
    term._update_command()
    assert term.command[0, 2] == 0
    assert term.command.shape == (12, 10)


def test_heading_does_not_override_standing_or_manual_commands(command_term):
    term = command_term
    term.cfg.standing_probability = 1.0
    term.cfg.rel_heading_envs = 1.0
    term._resample_command(torch.arange(12))
    term.heading_target[:] = 2.0
    term._update_command()
    assert term.is_heading_env.all() and term.is_standing_env.all()
    assert (term.command[:, :3] == 0).all()
    manual = commands("rolling")[0]
    manual[2] = -0.6
    term.set_command(manual, [1, 4])
    term._update_command()
    torch.testing.assert_close(term.command[[1, 4], 2], torch.full((2,), -0.6))
    assert not term.is_heading_env[[1, 4]].any()
    assert not term.is_standing_env[[1, 4]].any()
    assert term.command[0, 2] == 0


def test_heading_false_keeps_sampled_yaw_rate(command_term):
    term = command_term
    term.cfg.heading_command = False
    term.cfg.standing_probability = 0.0
    term._resample_command(torch.arange(12))
    before = term.command[:, 2].clone()
    assert not term.is_heading_env.any()
    term.robot.data.heading_w[:] = 1.5
    term._update_command()
    torch.testing.assert_close(term.command[:, 2], before)


def test_reward_adapters_score_height_clearance_and_rolling_without_slip(monkeypatch, command_term):
    # Identity orientation is enough for this fixture; test the actual reward
    # adapters, ordered indexing, ground offset, and input/output shapes.
    math_module = ModuleType("isaaclab.utils.math")
    math_module.quat_apply = lambda q, v: v
    math_module.quat_apply_inverse = lambda q, v: v
    math_module.yaw_quat = lambda q: q
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    rewards = load_module("wtw_under_test.wtw_rewards", MDP / "wtw_rewards.py")
    term = command_term
    n = term.num_envs
    term.cfg.wheel_radius = 0.086
    term.cfg.wheel_half_width = 0.0225
    robot = term.robot.data
    robot.body_link_pos_w = torch.zeros(n, 4, 3)
    robot.body_link_pos_w[..., 2] = 0.086
    robot.body_link_quat_w = torch.zeros(n, 4, 4)
    robot.body_link_quat_w[..., 0] = 1
    robot.body_link_lin_vel_w = torch.zeros(n, 4, 3)
    robot.body_link_lin_vel_w[..., 0] = 0.86
    robot.body_ang_vel_w = torch.zeros(n, 4, 3)
    robot.body_ang_vel_w[..., 1] = 10
    robot.joint_vel = torch.full((n, 16), 10.0)
    robot.root_pos_w[:, 2] = 0.42
    robot.root_quat_w = torch.zeros(n, 4)
    robot.root_quat_w[:, 0] = 1
    env = term._env
    env.command_manager = SimpleNamespace(get_term=lambda _: term, get_command=lambda _: term.command)
    term.set_command(commands("rolling")[0])
    for func in (
        rewards.wtw_height,
        rewards.wtw_stance_velocity,
        rewards.wtw_clearance,
        rewards.wtw_swing_force,
        rewards.wtw_swing_wheel_velocity,
        rewards.wtw_raibert,
    ):
        value = func(env)
        assert value.shape == (n,)
        torch.testing.assert_close(value, torch.zeros(n), atol=1e-6, rtol=0)
    assert rewards.wtw_commands(env).shape == (n, 10)
    assert rewards.wtw_clocks(env).shape == (n, 4)
    assert (rewards.wtw_clocks(env) == 0).all()
    robot.root_pos_w[:, 2] += 0.1
    torch.testing.assert_close(rewards.wtw_height(env), torch.full((n,), 0.01))
    # A translated plane must not change relative-height costs.
    env.scene.env_origins[:, 2] = 2
    robot.root_pos_w[:, 2] += 2
    robot.body_link_pos_w[..., 2] += 2
    torch.testing.assert_close(rewards.wtw_height(env), torch.full((n,), 0.01))
    term.phase[:] = 0.25
    term.set_command(commands("trot")[0])
    # Two feet are in mid-swing; keeping them on the ground must incur cost.
    assert (rewards.wtw_clearance(env) > 0).all()
    assert (rewards.wtw_swing_wheel_velocity(env) > 0).all()
    # Reward and observation adapters use the same perturbed command schedule.
    manual = commands("trot")[0]
    manual[5:8] += torch.tensor([2.0, -3.0, 1.0]) / 360
    term.set_command(manual)
    torch.testing.assert_close(rewards.wtw_commands(env)[:, 5:8], manual[5:8].expand(n, -1))
    target = wtw.swing_height_target(term.leg_phase, term.command)
    torch.testing.assert_close(rewards.wtw_clearance(env), (target.square() * (1 - term.desired_contact)).sum(-1))
    torch.testing.assert_close(rewards.wtw_clocks(env), torch.sin(2 * math.pi * term.leg_phase))
    term._update_metrics()
    assert all(torch.isfinite(metric).all() for metric in term.metrics.values())

    # Standstill must ignore height/duration, exclude wheels, and not fight
    # explicitly requested in-place stepping or turning.
    pose_cfg = SimpleNamespace(name="robot", joint_ids=list(range(12)))
    cmd = commands("rolling", n)
    cmd[:, :3] = 0
    cmd[0, 3] = 0.22
    term.set_command(cmd)
    robot.joint_pos = term.reward_joint_pos.clone()
    torch.testing.assert_close(rewards.wtw_stand_still(env, pose_cfg), torch.zeros(n))
    torch.testing.assert_close(rewards.wtw_joint_deviation_l1(env, pose_cfg), torch.zeros(n))
    assert not torch.equal(robot.joint_pos[0, :12], robot.default_joint_pos[0, :12])
    # Height changes the reward target, not the asset-default-relative joint observation.
    fixed_relative_pos = (robot.joint_pos - robot.default_joint_pos).clone()
    cmd[0, 3] = 0.46
    term.set_command(cmd)
    assert rewards.wtw_stand_still(env, pose_cfg)[0] > 0
    torch.testing.assert_close(robot.joint_pos - robot.default_joint_pos, fixed_relative_pos)
    cmd[0, 3] = 0.22
    term.set_command(cmd)
    torch.testing.assert_close(rewards.wtw_stand_still(env, pose_cfg), torch.zeros(n))
    robot.joint_pos[:, :12] += 0.1
    robot.joint_pos[:, 12:] = 100  # wheel angles are not pose targets
    torch.testing.assert_close(rewards.wtw_stand_still(env, pose_cfg), torch.full((n,), 1.2))
    torch.testing.assert_close(rewards.wtw_joint_deviation_l1(env, pose_cfg), torch.full((n,), 1.2))
    hip_cfg = SimpleNamespace(name="robot", joint_ids=[0, 3, 6, 9])
    torch.testing.assert_close(rewards.wtw_joint_deviation_l1(env, hip_cfg), torch.full((n,), 0.4))
    cmd[0, 0] = 0.5
    cmd[1, 2] = 0.5
    cmd[2, 4] = 2.0
    cmd[2, 8] = 0.5
    term.set_command(cmd)
    torch.testing.assert_close(rewards.wtw_stand_still(env, pose_cfg)[:3], torch.zeros(3))
