from types import SimpleNamespace

import pytest
import torch
from test_go2w_m20_mdp import _entity, _env
from test_go2w_m20_mdp import port as _port_fixture

port = _port_fixture


def _scan_env(n):
    env = _env(n)
    env.commands[:, 0] = 0.6
    robot = env.scene["robot"]
    robot.data.joint_pos.fill_(1.0)
    robot.data.root_pos_w[:, 2] = 0.45
    robot.data.heading_w = torch.zeros(n)
    x, y = torch.meshgrid(torch.linspace(-0.8, 0.8, 17), torch.linspace(-0.5, 0.5, 11), indexing="ij")
    hits = torch.stack([x.flatten(), y.flatten(), torch.zeros(x.numel())], dim=1).repeat(n, 1, 1)
    env.scene["height_scanner"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
    return env, hits, x.flatten(), y.flatten()


def _reward(port, env):
    return port.rewards.joint_pos_penalty_local_terrain(
        env, "base_velocity", _entity("robot", [0, 1, 2, 3]), _entity("height_scanner")
    )


def test_local_terrain_gate_anticipates_steps_and_rejects_irrelevant_or_invalid_rays(port):
    env, hits, x, y = _scan_env(19)
    hits[1, :, 2] = 5.0  # A high but flat platform must not weaken posture.
    env.scene["robot"].data.root_pos_w[1, 2] += 5.0
    hits[2, x > 0.35, 2] = 0.12  # Step ahead; the patch directly under the base is flat.
    hits[3, x > 0.35, 2] = -0.12
    hits[4, x > 0.65, 2] = 0.12  # Beyond the preview distance.
    hits[5, y.abs() > 0.35, 2] = 0.12  # Outside the wheel corridor.
    hits[6:8, x < -0.35, 2] = 0.12
    env.commands[7, 0] = -0.6  # A backward command reverses the preview.
    hits[8, x > -0.01, 2] = 0.12
    hits[9, (x.abs() < 0.01) & (y.abs() < 0.01), 2] = 0.4  # Isolated outlier.
    hits[10] = float("inf")
    selected = (x.abs() < 0.21) & (y.abs() < 0.01)
    hits[11, ~selected] = float("nan")
    hits[11, selected, 2] = torch.linspace(0.0, 0.2, int(selected.sum()))
    hits[12, x > -0.01, 2] = 0.06
    hits[13, x > -0.01, 2] = 0.061
    hits[14:16, x > 0.35, 2] = 0.12
    env.commands[14] = 0.0  # Standing inspects only the footprint, with the original x5 penalty.
    env.commands[15] = torch.tensor([0.0, 0.0, 0.4])
    env.commands[16] = torch.tensor([0.0, 0.4, 0.0])
    hits[17, (x.abs() < 0.01) & (y.abs() < 0.01), 0] = float("nan")
    hits[18, :, 2] = 0.2 * x  # A slope with sufficient local height variation also qualifies.
    env.scene.env_origins[:, 2] = 100.0  # Must have no influence on the local criterion.
    expected = torch.tensor([2, 2, 0.2, 0.2, 2, 2, 2, 0.2, 0.2, 2, 2, 2, 2, 0.2, 10, 0, 0, 2, 0.2])
    torch.testing.assert_close(_reward(port, env), expected)
    assert (hits[2, (x.abs() < 0.051) & (y.abs() < 0.051), 2] == 0).all()


@pytest.mark.parametrize("yaw", [0.0, 0.7, 1.5707963267948966, 3.141592653589793])
def test_local_terrain_gate_is_invariant_to_world_translation_and_heading(port, yaw):
    env, hits, x, y = _scan_env(1)
    hits[0, x > 0.35, 2] = 0.12
    expected = _reward(port, env)
    angle = torch.tensor(yaw)
    hits[0, :, 0] = angle.cos() * x - angle.sin() * y + 70.0
    hits[0, :, 1] = angle.sin() * x + angle.cos() * y - 30.0
    hits[0, :, 2] += 4.0
    env.scene["robot"].data.root_pos_w[:] = torch.tensor([70.0, -30.0, 4.45])
    env.scene["robot"].data.heading_w[:] = yaw
    del env.scene.env_origins
    env._m20_gait_level = torch.tensor(float("nan"))
    torch.testing.assert_close(_reward(port, env), expected)
