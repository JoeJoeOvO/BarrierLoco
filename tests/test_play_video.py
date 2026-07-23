from types import SimpleNamespace

from locolab.source.locomotion.utils.video import configure_robot_follow_view, make_video_recording_plan


def test_video_recording_plan_produces_exact_twenty_second_target() -> None:
    plan = make_video_recording_plan(step_dt=0.02, duration_s=20.0)

    assert plan.fps == 50
    assert plan.frame_count == 1000
    assert plan.simulation_steps == 1001


def test_explicit_frame_count_overrides_duration() -> None:
    plan = make_video_recording_plan(step_dt=0.02, duration_s=-1.0, frame_count=250)

    assert plan.fps == 50
    assert plan.frame_count == 250
    assert plan.simulation_steps == 251


def test_robot_follow_view_preserves_task_specific_offsets() -> None:
    default_view = SimpleNamespace(
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="world",
        asset_name=None,
        env_index=3,
    )
    configure_robot_follow_view(default_view)

    assert default_view.eye == (2.2, 2.2, 1.3)
    assert default_view.lookat == (0.0, 0.0, 0.15)
    assert default_view.origin_type == "asset_root"
    assert default_view.asset_name == "robot"
    assert default_view.env_index == 0

    task_view = SimpleNamespace(
        eye=(-3.0, 3.5, 2.2),
        lookat=(1.0, 0.0, 0.4),
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
    )
    configure_robot_follow_view(task_view)

    assert task_view.eye == (-3.0, 3.5, 2.2)
    assert task_view.lookat == (1.0, 0.0, 0.4)
