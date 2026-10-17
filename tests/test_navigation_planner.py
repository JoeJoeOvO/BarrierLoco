"""Navigation observation geometry and conditional-action probability contracts."""

import pytest
import torch

from locolab.planner.network import NavigationActorCritic
from locolab.planner.occupancy import OccupancySpec, voxelize_rays, yaw_rotate
from locolab.planner.skills import decode_command


def test_first_return_preserves_occlusion_and_unknown():
    spec = OccupancySpec(minimum=(0.0, -0.1, -0.1), shape_xyz=(20, 1, 1), resolution=0.2, max_range=4.0)
    directions = torch.tensor([[1.0, 0.0, 0.0]])
    result = voxelize_rays(directions, torch.tensor([[1.1], [float("inf")], [float("nan")]]), spec)
    assert result[0, 0, 0, 0, 5] == 1
    assert result[0, 1, 0, 0, :6].all()
    assert not result[0, 1, 0, 0, 6:].any()  # behind surface remains unknown
    assert not result[1, 0].any() and result[1, 1].all()  # valid no-return free ray
    assert not result[2].any()  # invalid measurement is not evidence of free space


def test_yaw_frame_round_trip():
    quaternion = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    point = torch.tensor([[[1.0, 0.0, 2.0]]])
    world = yaw_rotate(point, quaternion)
    torch.testing.assert_close(world, torch.tensor([[[0.0, 1.0, 2.0]]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(yaw_rotate(world, quaternion, inverse=True), point, atol=1e-6, rtol=0)


def test_conditional_likelihood_ignores_unselected_head():
    torch.manual_seed(3)
    network = NavigationActorCritic()
    observation = {"occ": torch.zeros(2, 2, 16, 24, 32), "proprio": torch.zeros(2, 58), "goal": torch.zeros(2, 3)}
    allowed = torch.tensor([[True, False, False], [True, False, False]])
    output = network.sample(observation, allowed)
    before, _ = network.evaluate(observation, output["skill"], output["raw_command"], allowed)
    with torch.no_grad():
        network.cmd_heads[2].bias.add_(10)
    after, _ = network.evaluate(observation, output["skill"], output["raw_command"], allowed)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(before, output["log_prob"])
    (-after.mean()).backward()
    assert network.cmd_heads[0].weight.grad.abs().sum() > 0
    assert network.cmd_heads[2].weight.grad.abs().sum() == 0
    assert (output["command"].abs() <= 1).all()


def test_command_bounds_and_rolling_convention():
    command = torch.ones(2, 10)
    result = decode_command(0, command)
    assert (result[:, 8] == 1).all() and (result[:, 9] == 0).all()
    jump = decode_command(2, -command)
    torch.testing.assert_close(jump[0], torch.tensor([0.2, 0.15, 0.4, -0.15]))


def test_busy_jump_does_not_credit_ignored_commands():
    network = NavigationActorCritic()
    observation = {"occ": torch.zeros(1, 2, 16, 24, 32), "proprio": torch.zeros(1, 58), "goal": torch.zeros(1, 3)}
    observation["proprio"][:, -1] = 1
    allowed = torch.tensor([[False, False, True]])
    action = network.sample(observation, allowed)
    likelihood, _ = network.evaluate(observation, action["skill"], action["raw_command"] + 10, allowed)
    torch.testing.assert_close(likelihood, torch.zeros(1))


def test_skill_buffers_remain_resettable_after_inference_mode():
    from types import SimpleNamespace

    from locolab.planner.skills import FrozenSkills

    skill = FrozenSkills.__new__(FrozenSkills)
    skill.env = SimpleNamespace(step_dt=0.02)
    skill.phase = torch.zeros(2)
    skill.timer = torch.ones(2)
    skill.command = torch.ones(2, 10)
    skill.skill = torch.zeros(2, dtype=torch.long)
    skill.previous = torch.zeros(2, 16)
    with torch.inference_mode():
        skill._advance(torch.ones(2, 16))
    assert not any(tensor.is_inference() for tensor in (skill.phase, skill.timer, skill.previous))
    skill.reset(torch.arange(2))
    assert not skill.previous.any() and not skill.phase.any() and not skill.timer.any()


@pytest.mark.parametrize("gap_only,directional", [(False, False), (True, False), (False, True)])
def test_cross_terrain_goals_follow_surface_and_leave_spawn_tile(gap_only, directional):
    from types import SimpleNamespace

    from locolab.planner.navigation import NavigationRollout

    torch.manual_seed(42)
    n = 64
    rollout = NavigationRollout.__new__(NavigationRollout)
    rollout.cross_terrain = True
    rollout.gap_only = gap_only
    rollout.all_gap_terrain = False
    rollout.forward_only = False
    rollout.directional_goals = directional
    rollout.route_heading = torch.zeros(n)
    rollout.goal_w = torch.zeros(n, 3)
    rollout.goal_valid = torch.zeros(n, dtype=torch.bool)
    rollout.arrival_steps = torch.zeros(n, dtype=torch.long)
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([0.0, 0.0, 0.3]).repeat(n, 1)))

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    scene.terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_rows=6, num_cols=10, size=(16, 16))),
        terrain_origins=torch.zeros(6, 10, 3),
    )
    rollout.env = SimpleNamespace(scene=scene, device="cpu")

    def raycast(starts, directions, distance):
        assert (starts[..., 2] == 10).all()
        result = starts.clone()
        result[..., 2] = 0.02 * result[..., 0]  # supported sloping surface
        return result

    rollout.raycast = raycast
    rollout.sample_goals(torch.arange(n))
    assert rollout.goal_valid.all()
    torch.testing.assert_close(rollout.goal_w[:, 2], 0.02 * rollout.goal_w[:, 0])
    assert (rollout.goal_w[:, :2].abs().amax(-1) > 8).any()
    if gap_only:
        assert (rollout.goal_w[:, 0] > 6.6).all()  # Opposite bank of the gap at x=6.


def test_varied_course_geometry_is_reproducible_and_not_constant():
    from types import SimpleNamespace

    import numpy as np

    from locolab.source.locomotion.terrains.planner_course import planner_course_terrain

    cfg = SimpleNamespace(family="mixed", size=(16.0, 16.0), vary_geometry=True)
    first, _ = planner_course_terrain(0.42, cfg)
    repeated, _ = planner_course_terrain(0.42, cfg)
    other, _ = planner_course_terrain(0.43, cfg)
    np.testing.assert_array_equal(first[-1].vertices, repeated[-1].vertices)
    assert not np.allclose(first[-1].vertices, other[-1].vertices)
    assert 0.15 <= first[-1].bounds[1, 2] <= 0.65
    assert 1.0 <= first[-1].extents[0] <= 3.0
    assert 0.2 <= first[1].extents[0] <= 0.6


@pytest.mark.parametrize("family", ["single_gap", "double_gap", "diagonal_gap", "zigzag_gap", "variable_gap"])
def test_diverse_gap_meshes_have_closed_pits_and_safe_spawn(family):
    from types import SimpleNamespace

    import numpy as np

    from locolab.source.locomotion.terrains.planner_gaps import diverse_gap_terrain, gap_bands

    cfg = SimpleNamespace(family=family, size=(16.0, 16.0))
    for difficulty in (0.0, 0.2, 0.7, 1.0):
        meshes, origin = diverse_gap_terrain(difficulty, cfg)
        bands, depth = gap_bands(difficulty, family)
        assert meshes[0].is_watertight
        assert np.isclose(meshes[0].bounds[1, 2], 0)
        assert 0.6 <= depth <= 1.4
        for _, _, intervals in bands:
            assert len(intervals) == (2 if family == "double_gap" else 1)
            for left, widths in intervals:
                assert (left > origin[0] + 0.5).all()
                assert ((widths >= 0.15) & (widths <= 0.9)).all()
                assert (left + widths < 16).all()


def test_jump_probe_rejects_falling_and_detects_supported_upward_flight():
    from types import SimpleNamespace

    from locolab.planner.jump_probe import JumpProbe

    data = SimpleNamespace(
        root_pos_w=torch.zeros(1, 3), root_lin_vel_b=torch.zeros(1, 3), root_lin_vel_w=torch.zeros(1, 3)
    )
    forces = torch.ones(1, 4, 3)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces), find_bodies=lambda _: (list(range(4)), []))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data), "contact_forces": sensor}, step_dt=0.02)
    probe = JumpProbe(env)
    probe.begin(torch.tensor([0]), torch.tensor([[0.8, 0.2, 0.6, 0.0]]), torch.zeros(1, 3), 0)
    forces.zero_()
    data.root_pos_w[:, 2] = -0.2
    data.root_lin_vel_w[:, 2] = -1
    for _ in range(5):
        probe.tick()
    assert probe.takeoffs == 0
    data.root_pos_w[:, 2] = 0.1
    data.root_lin_vel_w[:, 2] = 0.5
    probe.tick()
    assert probe.takeoffs == 1
    forces.fill_(1)
    probe.tick()
    assert probe.landed == 1
    probe.finish(torch.zeros(1), torch.tensor([False]))
    assert not probe.active
