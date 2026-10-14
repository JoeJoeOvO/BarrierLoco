"""Clock/trigger contracts with real bank data and minimal manager stubs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

MDP = Path(__file__).resolve().parents[1] / "locolab/source/locomotion/mdp"


@pytest.fixture
def term(monkeypatch):
    class Base:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.num_envs, self.device = cfg, env, env.num_envs, env.device
            self.metrics = {}

        def reset(self, ids):
            self._resample_command(ids)
            return {}

    managers = ModuleType("isaaclab.managers")
    managers.CommandTerm = Base
    managers.CommandTermCfg = object
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    package = ModuleType("parkour_cmd_under_test")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    modules = {}
    for name in ("parkour_reference", "parkour_jump"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", MDP / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    bank = modules["parkour_reference"].ParkourReferenceBank()
    sample = bank.sample(torch.arange(7), torch.zeros(7))
    data = SimpleNamespace(
        root_pos_w=sample["position"].clone(),
        root_quat_w=sample["quaternion"].clone(),
        root_lin_vel_w=sample["velocity"].clone(),
        root_lin_vel_b=sample["velocity"].clone(),
        root_ang_vel_w=sample["omega"].clone(),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(7, 1),
        body_pos_w=torch.zeros(7, 4, 3),
    )
    robot = SimpleNamespace(
        data=data,
        find_joints=lambda names, **kw: (list(range(len(names))), names),
        find_bodies=lambda names, **kw: (list(range(len(names))), names),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=torch.cat((torch.tensor([0.0, 0.0, 50.0]).expand(7, 4, 3).clone(), torch.zeros(7, 1, 3)), 1)
        ),
        find_bodies=lambda names, **kw: (list(range(len(names))), names),
        body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot", "base"],
    )

    class Scene(dict):
        env_origins = torch.zeros(7, 3)

    env = SimpleNamespace(
        num_envs=7,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        scene=Scene(robot=robot, contact_forces=sensor),
        _parkour_bank=bank,
        _parkour_reset_time=torch.zeros(7),
        _parkour_reset_reference=torch.zeros(7, dtype=torch.bool),
        _parkour_reset_mode=torch.zeros(7, dtype=torch.long),
        episode_length_buf=torch.zeros(7, dtype=torch.long),
    )
    term = modules["parkour_jump"].ParkourJumpCommand(SimpleNamespace(), env)
    env.command_manager = SimpleNamespace(get_term=lambda _name: term)
    return term


def test_ground_trigger_does_not_teleport_and_clock_advances_once(term):
    position = term.robot.data.root_pos_w.clone()
    velocity = term.robot.data.root_lin_vel_w.clone()
    term.update_state()
    assert term.active.all() and term.ground_attempt.all()
    torch.testing.assert_close(term.reference_time, torch.zeros(7))
    torch.testing.assert_close(term.robot.data.root_pos_w, position)
    torch.testing.assert_close(term.robot.data.root_lin_vel_w, velocity)
    term._env.common_step_counter += 1
    term.update_state()
    term.update_state()
    torch.testing.assert_close(term.reference_time, torch.full((7,), 0.02))


def test_grid_recipe_rejects_retained_bank_as_a_silent_fallback(term):
    module = sys.modules[type(term).__module__]
    term._env.cfg = SimpleNamespace(required_reference_bank_version=2)
    with pytest.raises(ValueError, match="goal-grid reference bank"):
        module.bank_for(term._env)


def test_fixed_timeout_completes_even_without_landing_and_cannot_retrigger(term):
    term.update_state()
    term.reference_time[:] = 2.19
    term.sensor.data.net_forces_w.zero_()
    term._env.common_step_counter += 1
    term.update_state()
    assert term.finished.all() and not term.active.any() and not term.ground_success.any()
    term.sensor.data.net_forces_w[:] = 50
    term._env.common_step_counter += 1
    term.update_state()
    assert not term.active.any()


def test_reference_reset_never_counts_as_a_ground_success(term):
    term._env._parkour_reset_time[:] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.8])
    term._env._parkour_reset_reference[:] = True
    term.reset()
    assert term.active.all() and not term.ground_attempt.any() and not term.air_seen.any()
    torch.testing.assert_close(term.normalized_phase, term.reference_time / 2.2)
    assert term.command.shape == (7, 7)
    torch.testing.assert_close(term.command[:, 4], term.bank.descriptors[:, 0])


def test_trigger_requires_ground_support_and_launch_marker(term):
    term.robot.data.root_pos_w[:, 0] = -0.1
    term.update_state()
    assert not term.active.any()
    term.robot.data.root_pos_w[:, 0] = 0.0
    term.sensor.data.net_forces_w.zero_()
    term._env.common_step_counter += 1
    term.update_state()
    assert not term.active.any()


def test_goal_command_is_remaining_displacement_in_current_yaw_frame(term):
    term.robot.data.root_pos_w[:, :2] = torch.tensor([0.1, 0.2])
    q = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    term.robot.data.root_quat_w[:] = q
    remaining = term.bank.descriptors[:, 1:] - torch.tensor([0.1, 0.2])
    torch.testing.assert_close(term.command[:, 5], remaining[:, 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(term.command[:, 6], -remaining[:, 0], atol=1e-6, rtol=0)
    torch.testing.assert_close(term.command[:, 4], term.bank.descriptors[:, 0])


def test_fixed_target_actor_keeps_height_and_displacement_under_motion_and_turning(term):
    term.cfg.landing_command_mode = "fixed"
    original = term.command[:, 4:].clone()
    torch.testing.assert_close(original, term.bank.descriptors)
    term.robot.data.root_pos_w[:, :2] += torch.tensor([0.35, -0.20])
    term.robot.data.root_quat_w[:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    term.reference_time[:] = 0.7
    term.started[:] = True
    term.active[:] = True
    torch.testing.assert_close(term.command[:, 4:], original)
    torch.testing.assert_close(term.command[:, 0], torch.zeros(7), atol=1e-6, rtol=0)
    torch.testing.assert_close(term.command[:, 1], -term.forward_speed, atol=1e-6, rtol=0)
    assert (term.command[:, 3] == 1).all()


def test_fixed_actor_does_not_read_root_position(term):
    term.cfg.landing_command_mode = "fixed"
    del term.robot.data.root_pos_w
    torch.testing.assert_close(term.command[:, 4:], term.bank.descriptors)


def test_privileged_critic_command_matches_legacy_stream_exactly(term):
    term.robot.data.root_pos_w[:, :2] += torch.tensor([0.35, -0.20])
    term.robot.data.root_quat_w[:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    old = term.command.clone()
    term.cfg.landing_command_mode = "fixed"
    module = sys.modules[type(term).__module__]
    torch.testing.assert_close(module.parkour_critic_commands(term._env), old, rtol=0, atol=0)
    torch.testing.assert_close(term.command[:, :5], old[:, :5], rtol=0, atol=0)
    assert not torch.allclose(term.command[:, 5:], old[:, 5:])


def test_fixed_targets_clear_for_rolling_and_exit_without_relocating_world_goals(term):
    term.cfg.landing_command_mode = "fixed"
    term.cfg.rolling_exit_duration = 1.0
    descriptors = term.bank.descriptors.clone()
    term.locomotion_only[0] = True
    assert not term.command[0, 3:].any()
    term.started[:] = True
    term.finished[:] = True
    term.task_success[:] = True
    assert not term.command[:5, 4:].any()
    torch.testing.assert_close(term.command[5:, 4:], descriptors[5:])  # ramp tasks still end on the platform
    torch.testing.assert_close(term.bank.descriptors, descriptors, rtol=0, atol=0)


def test_fixed_targets_survive_reference_phase_resets(term):
    term.cfg.landing_command_mode = "fixed"
    term._env._parkour_reset_reference[:] = True
    for time in (0.0, 0.2, 0.6, 1.4):
        term._env._parkour_reset_time[:] = time
        term.reset()
        torch.testing.assert_close(term.command[:, 4:], term.bank.descriptors)
        torch.testing.assert_close(term.reference_time, torch.full((7,), time))


def test_invalid_landing_command_mode_is_rejected(term):
    with pytest.raises(ValueError, match="landing_command_mode"):
        type(term)(SimpleNamespace(landing_command_mode="bad"), term._env)


def test_endpoint_mixture_keeps_continuous_coverage_and_samples_both_limits(term):
    term.num_envs = 20000
    term.forward_speed = torch.zeros(term.num_envs)
    term.cfg.forward_speed_range = (0.2, 1.5)
    term.cfg.speed_endpoint_probability = 0.30
    torch.manual_seed(17)
    term._resample_command(torch.arange(term.num_envs))
    values = term.forward_speed
    assert ((values >= 0.2) & (values <= 1.5)).all()
    assert (values == 0.2).float().mean().item() == pytest.approx(0.15, abs=0.01)
    assert (values == 1.5).float().mean().item() == pytest.approx(0.15, abs=0.01)
    assert ((values > 0.2) & (values < 1.5)).float().mean().item() == pytest.approx(0.70, abs=0.015)


@pytest.mark.parametrize(
    "name,value",
    [
        ("speed_endpoint_probability", -0.1),
        ("speed_endpoint_probability", 1.1),
        ("approach_velocity_std", 0.0),
        ("approach_speed_error_floor", -0.1),
        ("reference_xy_feedback_gain", float("nan")),
        ("reference_xy_feedback_limit", 0.0),
    ],
)
def test_tuning_parameters_reject_invalid_values(term, name, value):
    with pytest.raises(ValueError, match=name):
        type(term)(SimpleNamespace(**{name: value}), term._env)


def test_low_speed_overshoot_loses_reward_and_recovery_uses_same_tracking(term):
    module = sys.modules[type(term).__module__]
    term._last_step = term._env.common_step_counter
    term.cfg.approach_velocity_std = 0.25
    term.cfg.approach_speed_error_floor = 0.5
    term.forward_speed[:] = 0.2
    term.robot.data.root_lin_vel_w[:] = torch.tensor([0.2, 0.0, 0.0])
    matched = 20 * module.parkour_reward(term._env, "approach_velocity") - 10 * module.parkour_reward(
        term._env, "approach_speed_error"
    )
    torch.testing.assert_close(matched, torch.full((7,), 20.0))
    term.robot.data.root_lin_vel_w[:, 0] = 0.55
    overspeed = 20 * module.parkour_reward(term._env, "approach_velocity") - 10 * module.parkour_reward(
        term._env, "approach_speed_error"
    )
    assert (overspeed < 0).all()
    term.started[:] = True
    assert not module.parkour_reward(term._env, "approach_speed_error").any()
    term.cfg.rolling_exit_duration = 1.0
    term.finished[:] = True
    term.task_success[:] = True
    term.exit_time[:] = 0.1
    assert (module.parkour_reward(term._env, "approach_speed_error")[:5] > 0).all()


def test_position_feedback_requests_braking_without_mutating_teacher(term):
    module = sys.modules[type(term).__module__]
    term._last_step = term._env.common_step_counter
    term.active[:] = True
    term.started[:] = True
    term.cfg.reference_xy_feedback_gain = 2.0
    term.cfg.reference_xy_feedback_limit = 0.75
    ref = term.sample()
    old = ref["velocity"].clone()
    term.robot.data.root_pos_w[:] = ref["position"]
    term.robot.data.root_pos_w[:, 0] += 0.2
    term.robot.data.root_lin_vel_w[:] = old
    term.robot.data.root_lin_vel_w[:, 0] -= 0.4
    braking = module.parkour_reward(term._env, "base_velocity")
    torch.testing.assert_close(braking, torch.ones(7))
    term.robot.data.root_lin_vel_w[:, 0] += 0.8
    assert (module.parkour_reward(term._env, "base_velocity") < braking).all()
    torch.testing.assert_close(ref["velocity"], old, rtol=0, atol=0)


def test_landing_lead_and_recovery_costs_are_zero_on_the_measured_teacher(term):
    module = sys.modules[type(term).__module__]
    term._last_step = term._env.common_step_counter
    term.active[:] = True
    for phase in (0.0, 0.2, 0.4, 0.8, 1.8):
        term.reference_time[:] = phase
        term._sample = None
        sample = term.sample()
        term.robot.data.root_pos_w[:] = sample["position"]
        term.robot.data.root_lin_vel_w[:] = sample["velocity"]
        assert not module.parkour_reward(term._env, "landing_lead_error").any()
        assert not module.parkour_reward(term._env, "recovery_position_error").any()
    term.reference_time[:] = 0.2
    term._sample = None
    ref = term.sample()
    term.robot.data.root_pos_w[:] = ref["position"]
    term.robot.data.root_lin_vel_w[:] = ref["velocity"]
    term.robot.data.root_lin_vel_w[:, 0] += 0.5
    assert (module.parkour_reward(term._env, "landing_lead_error") > 0).all()
    term.touchdown_seen[:] = True
    assert not module.parkour_reward(term._env, "landing_lead_error").any()
    term.robot.data.root_pos_w[:, 0] += 0.1
    torch.testing.assert_close(module.parkour_reward(term._env, "recovery_position_error"), torch.full((7,), 0.01))


def test_first_landing_error_charged_once_with_rsi_scaling_and_no_inf_nan(term):
    module = sys.modules[type(term).__module__]
    term._last_step = term._env.common_step_counter
    assert torch.isinf(term.touchdown_error).all()
    assert not module.parkour_reward(term._env, "landing_position_error").any()
    term.touchdown_event[:] = True
    term.touchdown_error[:] = 0.1
    term.reset_mode[:] = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    cost = -5000 * 0.02 * module.parkour_reward(term._env, "landing_position_error")
    torch.testing.assert_close(cost, torch.tensor([-1.0, -1.0, -1.0, -1 / 6, -1 / 6, -1.0, -1.0]))
    term.touchdown_event[:] = False
    assert not module.parkour_reward(term._env, "landing_position_error").any()


def test_real_takeoff_accounting_emits_only_one_first_touchdown_event(term):
    module = sys.modules[type(term).__module__]
    term.update_state()
    term.sensor.data.net_forces_w.zero_()
    term.robot.data.root_lin_vel_w[:, 2] = 1.0
    term.robot.data.root_pos_w[:, 2] = term.launch_height + 0.1
    for _ in range(6):
        term._env.common_step_counter += 1
        term.update_state()
    assert term.air_seen.all()
    term.sensor.data.net_forces_w[:, :4, 2] = 50
    term.robot.data.root_pos_w[:, :2] = term.bank.descriptors[:, 1:] + torch.tensor([0.1, 0.0])
    term._env.common_step_counter += 1
    term.update_state()
    assert term.touchdown_event.all()
    assert (module.parkour_reward(term._env, "landing_position_error") > 0).all()
    term._env.common_step_counter += 1
    term.update_state()
    assert not term.touchdown_event.any() and term.touchdown_seen.all()
    assert not module.parkour_reward(term._env, "landing_position_error").any()


def test_falling_airtime_never_becomes_takeoff_and_peak_survives_landing(term):
    term.update_state()
    term.sensor.data.net_forces_w.zero_()
    term.robot.data.root_lin_vel_w[:, 2] = -1.0
    term.robot.data.root_pos_w[:, 2] = term.launch_height + 0.12
    for _ in range(6):
        term._env.common_step_counter += 1
        term.update_state()
    assert (term.air_time >= 0.1).all()
    assert not term.air_seen.any() and not term.upward_takeoff.any()
    term.robot.data.root_pos_w[:, 2] = term.launch_height
    term._env.common_step_counter += 1
    term.update_state()
    torch.testing.assert_close(term.peak_height, torch.full((7,), 0.12))
    torch.testing.assert_close(term.metrics["jump_height"], torch.zeros(7))
    term.reference_time[:] = 2.19
    term._env.common_step_counter += 1
    term.update_state()
    assert term.finished.all() and not term.task_success.any()
    module = sys.modules[type(term).__module__]
    assert module.parkour_deadline_failed(term._env).all()
    assert not module.parkour_succeeded(term._env).any()


def test_all_seven_recorded_state_traces_can_satisfy_the_new_jump_accounting(term):
    # A pure accounting regression, not a replacement for native physics replay.
    ids = torch.arange(7)
    for tick in range(111):
        t = tick * 0.02
        sample = term.bank.sample(ids, torch.full((7,), t))
        term.robot.data.root_pos_w[:] = sample["position"]
        term.robot.data.root_lin_vel_w[:] = sample["velocity"]
        term.robot.data.root_lin_vel_b[:] = sample["velocity"]
        term.robot.data.root_ang_vel_w[:] = sample["omega"]
        term.robot.data.root_quat_w[:] = sample["quaternion"]
        term.sensor.data.net_forces_w.zero_()
        term.sensor.data.net_forces_w[:, :4, 2] = sample["contacts"] * 50
        index = min(round(t / 0.005), 360)
        for i, entry in enumerate(term.bank.entries):
            wheel = torch.tensor(entry["replay"]["states"]["wheel_center_position"][index])
            wheel[:, :2] -= term.bank.origins[i, :2]
            term.robot.data.body_pos_w[i] = wheel
        term._env.common_step_counter = tick
        term.update_state()
    assert term.air_seen.all()
    assert term.finished.all()
    assert term.task_success.all(), (term.stable_time, term.touchdown_error, term.peak_height)
    assert term.ground_success.all()
    assert (term.metrics["jump_height"] > 0.14).all()


def test_success_and_failure_have_distinct_terminal_rewards(term):
    module = sys.modules[type(term).__module__]
    term._env.common_step_counter = term._last_step
    term.finished[:] = True
    term.task_success[:] = True
    term.reset_mode[:] = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    out = module.parkour_reward(term._env, "success") * 1500 * 0.02
    torch.testing.assert_close(out, torch.tensor([30.0, 30.0, 30.0, 5.0, 5.0, 30.0, 30.0]))
    zeros = torch.zeros(7, dtype=torch.bool)
    term._env.termination_manager = SimpleNamespace(get_term=lambda _name: zeros, dones=~zeros)
    assert not module.parkour_failure_cost(term._env).any()
    # A simultaneous collision cannot be hidden by the success flag.
    term._env.termination_manager = SimpleNamespace(
        get_term=lambda name: ~zeros if name == "base_contact" else zeros, dones=~zeros
    )
    assert (module.parkour_failure_cost(term._env) > 0).all()
    term._env.termination_manager.dones = zeros
    assert not module.parkour_failure_cost(term._env).any()


def test_approach_deadline_is_failure_not_unlimited_survival(term):
    module = sys.modules[type(term).__module__]
    term.robot.data.root_pos_w[:, 0] = -0.8
    term._env.episode_length_buf[:] = 151
    assert module.parkour_failed(term._env).all()


@pytest.mark.parametrize("bad_field", ["height", "speed"])
def test_crawling_or_overspeed_entry_cannot_start_the_teacher_clock(term, bad_field):
    if bad_field == "height":
        term.robot.data.root_pos_w[:, 2] = 0.12
    else:
        term.robot.data.root_lin_vel_b[:, 0] = 2.2
    term.update_state()
    assert not term.started.any()


def test_late_high_jump_still_receives_height_error_feedback(term):
    module = sys.modules[type(term).__module__]
    term.update_state()
    term.reference_time[:] = 0.95
    term._sample = None
    term.robot.data.root_pos_w[:, 2] = 0.9
    value = module.parkour_reward(term._env, "height_error")
    assert (value > 0.10).all()


@pytest.mark.parametrize("speed", [0.2, 0.5, 1.0, 1.5])
def test_speed_stage_exposes_and_accepts_requested_speed_not_fixed_one(term, speed):
    term.forward_speed[:] = speed
    term.robot.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    term.robot.data.root_lin_vel_b[:, 0] = speed
    term.robot.data.root_lin_vel_w[:, 0] = speed
    torch.testing.assert_close(term.command[:, 0], torch.full((7,), speed))
    term.update_state()
    assert term.active.all()
    module = sys.modules[type(term).__module__]
    term.started[:] = False
    term.robot.data.root_lin_vel_w[:, 1] = 0
    torch.testing.assert_close(module.parkour_reward(term._env, "approach_velocity"), torch.ones(7))
    assert not module.parkour_reward(term._env, "approach_speed_error").any()


def test_ordinary_rolling_has_zero_jump_goals_and_no_marker_or_deadline_failure(term):
    term.locomotion_only[:] = True
    term.robot.data.root_pos_w[:, 0] = 2.0
    term._env.episode_length_buf[:] = 450
    term.update_state()
    assert not term.active.any() and not term.started.any()
    assert not term.command[:, 3:].any()
    module = sys.modules[type(term).__module__]
    assert not module.parkour_failed(term._env).any()
    term._env.termination_manager = SimpleNamespace(
        get_term=lambda name: torch.full((7,), name == "time_out"), dones=torch.ones(7, dtype=torch.bool)
    )
    assert not module.parkour_failure_cost(term._env).any()
    term.locomotion_only[:] = False
    assert (module.parkour_failure_cost(term._env) > 0).all()


def test_slow_approach_gets_six_seconds_but_cannot_live_forever(term):
    term.cfg.approach_timeout = 6.0
    term.robot.data.root_pos_w[:, 0] = -0.5
    module = sys.modules[type(term).__module__]
    term._env.episode_length_buf[:] = 251
    assert not module.parkour_failed(term._env).any()
    term._env.episode_length_buf[:] = 301
    assert module.parkour_failed(term._env).all()


def test_speed_sampler_and_nonjump_posture_rewards_are_gated(term):
    term.cfg.forward_speed_range = (0.2, 1.5)
    term._resample_command(torch.arange(7))
    assert ((term.forward_speed >= 0.2) & (term.forward_speed <= 1.5)).all()
    assert term.forward_speed.unique().numel() > 1
    term.robot.data.joint_pos = term.bank.states["joint_position"][:, 0].clone() + 0.1
    term.robot.data.projected_gravity_b[:, 0] = 0.2
    term.robot.data.root_ang_vel_w[:, 0] = 0.2
    term._last_step = term._env.common_step_counter
    module = sys.modules[type(term).__module__]
    for kind in ("approach_joint_position", "inactive_orientation", "inactive_angular_velocity", "inactive_thigh_calf"):
        term.started[:] = False
        assert (module.parkour_reward(term._env, kind) > 0).all()
        term.started[:] = True
        assert not module.parkour_reward(term._env, kind).any()


def test_speed_reset_uses_supported_reference_pose_but_zero_ground_velocity(term):
    module = sys.modules[type(term).__module__]
    robot = term.robot
    robot.find_joints = lambda names, **kw: (list(range(12, 16)) if len(names) == 4 else list(range(12)), names)
    robot.data.default_root_state = torch.zeros(7, 13)
    robot.data.default_root_state[:, 2:4] = torch.tensor([0.45, 1.0])
    robot.data.default_joint_pos = torch.zeros(7, 16)
    robot.data.joint_pos_limits = torch.tensor([-10.0, 10.0]).expand(7, 16, 2)
    robot.data.joint_vel_limits = torch.full((7, 16), 30.0)
    saved = {}
    robot.write_root_pose_to_sim = lambda pose, **kw: saved.update(pose=pose)
    robot.write_root_velocity_to_sim = lambda vel, **kw: saved.update(velocity=vel)
    robot.write_joint_state_to_sim = lambda q, qd, **kw: saved.update(q=q, qd=qd)
    module.reset_parkour_state(term._env, torch.arange(7), reference_probability=0.0, ground_reference_pose=True)
    torch.testing.assert_close(saved["pose"][:, 2], term.bank.states["position"][:, 0, 2])
    torch.testing.assert_close(saved["pose"][:, 3:], term.bank.states["quaternion"][:, 0])
    torch.testing.assert_close(saved["q"][:, :12], term.bank.states["joint_position"][:, 0])
    assert not saved["velocity"].any() and not saved["qd"].any()
    assert not term._env._parkour_reset_reference.any()


def test_flat_and_gap_return_to_normal_motion_but_ramps_stop_on_the_platform(term):
    module = sys.modules[type(term).__module__]
    term.cfg.rolling_exit_duration = 1.0
    term._last_step = term._env.common_step_counter
    term.finished[:] = True
    term.started[:] = True
    term.task_success[:] = True
    assert term.rolling_exit.tolist() == [True] * 5 + [False] * 2
    assert module.parkour_succeeded(term._env).tolist() == [False] * 5 + [True] * 2
    assert not term.command[:5, 3:].any()
    term.exit_time[:] = 0.2
    assert term.locomotion_active.tolist() == [True] * 5 + [False] * 2
    assert not module.parkour_reward(term._env, "success")[:5].any()
    assert not module.parkour_reward(term._env, "base_position")[:5].any()
    assert not module.parkour_reward(term._env, "height_error")[:5].any()
    term.exit_time[:] = 1.0
    assert module.parkour_succeeded(term._env).all()
    assert (module.parkour_reward(term._env, "success") > 0).all()
    term.task_success[:] = False
    assert module.parkour_deadline_failed(term._env).all()  # failed jumps never receive a free rolling tail
