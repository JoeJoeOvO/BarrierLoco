from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from locolab.source.locomotion.utils.video import (
    TerrainVideoView,
    configure_robot_follow_view,
    configure_terrain_view,
    make_video_recording_plan,
)


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


def _terrain_video_cfg(num_envs=4096, num_cols=20):
    generator = SimpleNamespace(
        curriculum=True,
        num_cols=num_cols,
        sub_terrains={
            name: SimpleNamespace(proportion=value)
            for name, value in [
                ("rough", 0.1),
                ("slope", 0.1),
                ("boxes", 0.3),
                ("pyramid_stairs", 0.25),
                ("pyramid_stairs_inv", 0.25),
            ]
        },
    )
    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=num_envs, terrain=SimpleNamespace(terrain_generator=generator)),
        viewer=SimpleNamespace(),
    )


@pytest.mark.parametrize("num_envs,num_cols", [(4096, 20), (8192, 20), (32, 5)])
def test_video_selection_uses_stair_columns_without_changing_terrain(num_envs, num_cols):
    cfg = _terrain_video_cfg(num_envs, num_cols)
    before = repr(cfg.scene)
    index = configure_terrain_view(cfg, "pyramid_stairs_inv")
    column = int(index // (num_envs / num_cols))
    assert column / num_cols + 0.001 >= 0.75
    assert cfg.viewer.origin_type == "env"
    assert cfg.viewer.env_index == index
    assert repr(cfg.scene) == before


def test_video_selection_rejects_missing_or_unoccupied_terrain():
    cfg = _terrain_video_cfg(num_envs=1)
    with pytest.raises(ValueError, match="No environment occupies"):
        configure_terrain_view(cfg, "pyramid_stairs_inv")
    with pytest.raises(ValueError, match="Unknown video terrain"):
        configure_terrain_view(cfg, "missing")
    cfg.scene.terrain.terrain_generator.curriculum = False
    with pytest.raises(ValueError, match="curriculum"):
        configure_terrain_view(cfg, "pyramid_stairs_inv")


def test_fixed_terrain_camera_refreshes_on_reset_and_curriculum_moves_only():
    class Env(gym.Env):
        def __init__(self):
            self.cfg = SimpleNamespace(viewer=SimpleNamespace(env_index=1))
            self.scene = SimpleNamespace(env_origins=torch.zeros(2, 3))
            self.refreshes = 0
            self.viewport_camera_controller = SimpleNamespace(update_view_to_env=self.refresh)

        def refresh(self):
            self.refreshes += 1

        def reset(self, **kwargs):
            self.scene.env_origins[1, 0] += 8
            return {}, {}

        def step(self, action):
            self.scene.env_origins[1, 0] += action
            return {}, 0, False, False, {}

    base = Env()
    env = TerrainVideoView(base)
    assert base.refreshes == 1
    env.step(0)
    assert base.refreshes == 1
    env.step(8)
    assert base.refreshes == 2
    env.reset()
    assert base.refreshes == 3
