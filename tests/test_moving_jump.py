"""Ground-start, repeated-trigger contracts without an Isaac Sim dependency."""

import ast
import math
from types import SimpleNamespace

import pytest
import torch
from test_go2w_jump_stack import GO2W, REFERENCE, _load_jump_command, _load_jump_rewards


def make_command(monkeypatch, count=1):
    module = _load_jump_command(monkeypatch)

    class Sensor:
        data = SimpleNamespace(net_forces_w=torch.zeros(count, 4, 3))

        def find_bodies(self, names):
            return list(range(4)), ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

    sensor = Sensor()
    sensor.data.net_forces_w[:, :, 2] = 10.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.42]]).repeat(count, 1),
            root_lin_vel_w=torch.tensor([[0.35, 0.0, 0.0]]).repeat(count, 1),
            root_lin_vel_b=torch.tensor([[0.35, 0.0, 0.0]]).repeat(count, 1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
    )

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=0.02, common_step_counter=0, scene=Scene(robot=robot))
    cfg = SimpleNamespace(
        asset_name="robot",
        contact_sensor_name="contact_forces",
        foot_body_names=".*_foot",
        jump_probability=1.0,
        contact_force_threshold=1.0,
        minimum_jump_height=0.18,
        landing_velocity_threshold=0.2,
        landing_recovery_time=0.06,
        landing_attitude_limit=0.5,
        takeoff_time=0.36,
        apex_time=0.54,
        landing_time=0.72,
        sequence_duration=1.2,
        initial_trigger_delay_range=(1.0, 1.0),
        resampling_time_range=(2.0, 2.0),
        require_grounded_trigger=True,
        minimum_forward_speed=0.1,
    )
    term = module.BinaryJumpCommand(cfg, env)
    term.reset(torch.arange(count))
    return term, env, robot, sensor


def step(term, env, count=1):
    for _ in range(count):
        env.common_step_counter += 1
        term.compute(env.step_dt)


def test_walk_first_then_repeated_trigger_preserves_momentum_and_clock(monkeypatch):
    term, env, robot, sensor = make_command(monkeypatch)
    assert not term.active.any()
    before = {k: v.clone() for k, v in vars(robot.data).items()}
    step(term, env, 40)
    assert not term.active.any()
    step(term, env, 12)
    assert term.active.item()
    for key, value in before.items():
        torch.testing.assert_close(getattr(robot.data, key), value)
    assert term.metrics["ground_attempts"].item() == 1

    clock = term.reference_time.clone()
    term._resample_command(torch.tensor([0]))  # Cannot interrupt/restart active jump.
    torch.testing.assert_close(term.reference_time, clock)
    step(term, env, 60)
    assert not term.active.item()
    assert term.metrics["ground_completions"].item() == 1
    term._resample_command(torch.tensor([0]))
    assert term.active.item()
    assert term.reference_time.item() == 0
    assert term.metrics["ground_attempts"].item() == 2
    assert term.metrics["ground_completions"].item() == 1


def test_external_flag_is_rising_edge_and_zero_does_not_abort(monkeypatch):
    term, env, _, _ = make_command(monkeypatch)
    term.cfg.jump_probability = 0
    assert term.set_trigger(1).item()
    step(term, env, 5)
    clock = term.reference_time.clone()
    assert not term.set_trigger(1).item()
    torch.testing.assert_close(term.reference_time, clock)
    term.set_trigger(0)
    assert term.active.item()
    assert not term.set_trigger(1).item()  # Rising while busy is not queued.
    step(term, env, 60)
    assert not term.set_trigger(1).item()  # Holding 1 across completion does not retrigger.
    term.set_trigger(0)
    assert term.set_trigger(1).item()
    assert term.metrics["ground_attempts"].item() == 2


def test_only_upright_grounded_moving_environments_accept_trigger(monkeypatch):
    term, _, robot, sensor = make_command(monkeypatch, count=5)
    robot.data.root_lin_vel_b[1, 0] = 0.0
    sensor.data.net_forces_w[2, 1] = 0.0
    robot.data.projected_gravity_b[3, 2] = 0.0
    robot.data.root_lin_vel_w[4, 2] = 0.5
    assert term.set_trigger(1).tolist() == [True, False, False, False, False]


def test_physical_late_landing_uses_precontact_speed_and_recovery_window(monkeypatch):
    term, env, robot, sensor = make_command(monkeypatch)
    term.cfg.jump_probability = 0.0
    term.set_trigger(1)
    step(term, env, 19)
    assert not term.all_wheels_airborne_achieved.item()
    sensor.data.net_forces_w.zero_()
    robot.data.root_pos_w[:, 2] = 0.62
    robot.data.root_lin_vel_w[:, 2] = -1.1
    step(term, env, 22)  # Actual landing will occur after the old 0.8 s cutoff.
    assert term.active.item() and term.metrics["ground_takeoffs"].item() == 1
    sensor.data.net_forces_w[:, :, 2] = 10.0
    robot.data.root_lin_vel_w[:, 2] = 0.1
    step(term, env)
    assert term.touchdown_event.item()
    assert term.touchdown_velocity.item() == pytest.approx(1.1)
    step(term, env, 20)
    assert not term.active.item()
    assert term.metrics["ground_jump_success"].item() == 1.0
    assert term.metrics["ground_successes"].item() == 1.0
    term._resample_command(torch.tensor([0]))  # Probability zero retains outcome.
    assert term.metrics["ground_successes"].item() == 1.0


def test_reference_reset_is_not_counted_as_ground_start(monkeypatch):
    term, env, robot, _ = make_command(monkeypatch)
    env._jump_reset_pending = torch.tensor([True])
    env._jump_episode_mask = torch.tensor([True])
    env._jump_initial_phase_time = torch.tensor([0.5])
    env._jump_trigger_position = robot.data.root_pos_w.clone()
    env._jump_target_height = torch.tensor([0.2])
    term.reset(torch.tensor([0]))
    assert term.active.item()
    assert term.reference_time.item() == 0.5
    assert term.metrics["jump_attempts"].item() == 1
    assert term.metrics["ground_attempts"].item() == 0
    assert not term._ground_attempt.item()


def test_success_requires_still_upright_and_moving_at_completion(monkeypatch):
    term, env, robot, sensor = make_command(monkeypatch)
    term.cfg.jump_probability = 0
    term.set_trigger(1)
    sensor.data.net_forces_w.zero_()
    robot.data.root_pos_w[:, 2] = 0.62
    step(term, env, 30)
    sensor.data.net_forces_w[:, :, 2] = 10.0
    step(term, env, 10)
    assert term.stable_landing_achieved.item()
    robot.data.root_lin_vel_b[:, 0] = 0.0
    step(term, env, 22)
    assert term.metrics["ground_jump_success"].item() == 0


def test_terminal_to_pose_is_held_but_velocity_settles_to_zero(monkeypatch):
    rewards = _load_jump_rewards(monkeypatch)
    trajectory = rewards.load_jump_reference(REFERENCE)
    times = torch.tensor([0.78, 0.9, 1.18, 1.2])
    sample = trajectory.sample(times)
    velocity = sample.joint_velocity.clone()
    velocity[times > 0.8] = 0
    term = SimpleNamespace(
        active=torch.tensor([True, True, True, False]), reference_time=times, update_state=lambda: None
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=sample.base_position, joint_pos=sample.joint_position, joint_vel=velocity)
    )
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_term=lambda _: term))
    kwargs = dict(
        command_name="jump_command",
        reference_path=str(REFERENCE),
        asset_cfg=SimpleNamespace(name="robot", joint_ids=list(range(12))),
    )
    torch.testing.assert_close(
        rewards.jump_reference_joint_position_reward(env, std=0.26, **kwargs), torch.tensor([1.0, 1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        rewards.jump_reference_joint_velocity_reward(env, start_time=0.36, std=3.0, **kwargs),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )


def test_horizontal_tracking_is_yaw_only_and_stronger_during_jump(monkeypatch):
    rewards = _load_jump_rewards(monkeypatch)
    yaw, pitch = math.pi / 2, 0.4
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=torch.tensor([[cy * cp, -sy * sp, cy * sp, sy * cp]]).repeat(2, 1),
            root_lin_vel_w=torch.tensor([[0.0, 0.35, 3.0], [0.0, 0.35, -2.0]]),
        )
    )
    term = SimpleNamespace(active=torch.tensor([False, True]), reference_time=torch.zeros(2), update_state=lambda: None)
    command = torch.tensor([[0.35, 0.0, 0.0]]).repeat(2, 1)
    env = SimpleNamespace(
        scene={"robot": robot}, command_manager=SimpleNamespace(get_term=lambda _: term, get_command=lambda _: command)
    )
    reward = rewards.jump_track_lin_vel_xy_exp(env, "base_velocity", "jump_command", std=0.5)
    torch.testing.assert_close(reward, torch.tensor([1.0, 2.0]))
    robot.data.root_lin_vel_w[:, :2] = 0.0
    assert (rewards.jump_track_lin_vel_xy_exp(env, "base_velocity", "jump_command", std=0.5) < reward).all()


def test_standstill_does_not_penalize_moving_or_jumping(monkeypatch):
    rewards = _load_jump_rewards(monkeypatch)
    term = SimpleNamespace(
        active=torch.tensor([False, False, True]), reference_time=torch.zeros(3), update_state=lambda: None
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.ones(3, 12), default_joint_pos=torch.zeros(3, 12)))
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(
            get_term=lambda _: term,
            get_command=lambda _: torch.tensor([[0.35, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ),
    )
    result = rewards.jump_inactive_joint_deviation_l1(
        env,
        "jump_command",
        SimpleNamespace(name="robot", joint_ids=list(range(12))),
        velocity_command_name="base_velocity",
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 12.0, 0.0]))


def test_moving_training_and_ground_start_play_configuration():
    tree = ast.parse((GO2W / "jump_env_cfg.py").read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WJumpCommandsCfg")
    terms = {n.targets[0].id: n.value for n in cfg.body if isinstance(n, ast.Assign)}
    velocity = {k.arg: k.value for k in terms["base_velocity"].keywords}
    jump = {k.arg: k.value for k in terms["jump_command"].keywords}
    assert ast.literal_eval(velocity["resampling_time_range"]) == (10.0, 10.0)
    assert ast.literal_eval(jump["resampling_time_range"]) == (2.0, 4.0)
    assert ast.literal_eval(jump["initial_trigger_delay_range"]) == (1.0, 2.0)
    assert ast.literal_eval(jump["require_grounded_trigger"]) is True
    assert ast.literal_eval(jump["minimum_forward_speed"]) == 0.1
    events = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WJumpEventsCfg")
    event = next(n.value for n in events.body if isinstance(n, ast.Assign))
    params = next(k.value for k in event.keywords if k.arg == "params")
    values = {ast.literal_eval(k): v for k, v in zip(params.keys, params.values)}
    assert ast.literal_eval(values["jump_probability"]) == 0.2
    play = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2WJumpPlayEnvCfg")
    assert "self.events.reset_base.params['jump_probability'] = 0.0" in ast.unparse(play)
