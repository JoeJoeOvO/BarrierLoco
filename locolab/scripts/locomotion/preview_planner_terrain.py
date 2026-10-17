"""Inspect Planner courses without loading policies or running PPO."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=0, help="Zero keeps the desktop preview open.")
parser.add_argument("--report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import PreviewSurfaceCfg, SphereCfg
    from isaaclab.utils.warp import raycast_mesh

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.source.locomotion.mdp.planner_terrain import goal_distance
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainPlayEnvCfg
    from locolab.source.locomotion.terrains.planner_course_cfg import TERRAIN_FAMILIES as FAMILIES

    cfg = Go2WPlannerTerrainPlayEnvCfg()
    cfg.sim.device = args.device
    cfg.viewer.origin_type = "env"
    cfg.viewer.eye = (6.0, -10.0, 9.0)
    cfg.viewer.lookat = (6.0, 0.0, 0.0)
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg)
    native = env.unwrapped
    terrain = native.scene.terrain
    ids = torch.arange(native.num_envs, device=native.device)
    # One robot per tile, no duplicated/colliding preview robots.
    terrain.terrain_levels[:] = ids // len(FAMILIES)
    terrain.terrain_types[:] = ids % len(FAMILIES)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    observations, _ = env.reset()
    assert native.action_manager.total_action_dim == 16
    assert torch.isfinite(observations["policy"]).all()
    assert native.command_manager.active_terms == []
    assert native.reward_manager.active_terms == []
    local = native.scene["robot"].data.root_pos_w - native.scene.env_origins
    assert (local[:, :2].abs() < 0.2).all(), "Reset must stay in the flat start area"

    # Inspect the actual combined raycast mesh, not only the analytic generator.
    ray_origins, expected = [], []
    for i in range(native.num_envs):
        family = FAMILIES[i % len(FAMILIES)]
        for x, z in [(0.0, 0.0), (13.0, 0.0)] if i % len(FAMILIES) < 4 else []:
            ray_origins.append(native.scene.env_origins[i] + local.new_tensor([x, 0, 5]))
            expected.append(z)
        if family in ("gap", "mixed"):
            x = (10.1 if family == "mixed" else 6.1) - 1.5
            ray_origins.append(native.scene.env_origins[i] + local.new_tensor([x, 0, 5]))
            expected.append(-1.0)
    starts = torch.stack(ray_origins)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1
    sensor = native.scene["height_scanner"]
    hits = raycast_mesh(starts, directions, sensor.meshes["/World/ground"], max_dist=10)[0]
    torch.testing.assert_close(hits[:, 2], starts.new_tensor(expected), atol=1e-4, rtol=0)

    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PlannerEndpoints",
            markers={
                "start": SphereCfg(radius=0.12, visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3))),
                "goal": SphereCfg(radius=0.18, visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05))),
            },
        )
    )
    starts = native.scene.env_origins.clone()
    goals = starts.clone()
    goals[:, 0] += goal_distance(native)
    endpoints = torch.cat((starts, goals))
    rays = endpoints.clone()
    rays[:, 2] += 5.0
    directions = torch.zeros_like(rays)
    directions[:, 2] = -1
    surfaces = raycast_mesh(rays, directions, sensor.meshes["/World/ground"], max_dist=20)[0]
    assert torch.isfinite(surfaces).all(), "Every family needs grounded start/goal markers"
    endpoints[:, 2] = surfaces[:, 2] + 0.15
    starts, goals = endpoints.chunk(2)
    markers.visualize(torch.cat((starts, goals)), marker_indices=torch.cat((ids * 0, ids * 0 + 1)))
    if not args.headless:
        import omni.ui as ui

        window = ui.Window("Planner Terrain", width=340, height=170)
        selection = [0]

        def show(delta):
            selection[0] = (selection[0] + delta) % native.num_envs
            index = selection[0]
            native.viewport_camera_controller.set_view_env_index(index)
            label.text = (
                f"{index + 1}/{native.num_envs} | {FAMILIES[index % len(FAMILIES)]}"
                f" | difficulty row {index // len(FAMILIES) + 1}/6"
            )

        with window.frame, ui.VStack(spacing=8):
            label = ui.Label("", word_wrap=True)
            ui.Label("Green: start | Orange: destination", word_wrap=True)
            ui.Label("Terrain preview only. No Planner or skill policy running.", word_wrap=True)
            with ui.HStack(height=28):
                ui.Button("Previous", clicked_fn=lambda: show(-1))
                ui.Button("Next", clicked_fn=lambda: show(1))
        show(0)
    actions = torch.zeros(native.num_envs, 16, device=native.device)
    step = 0
    while app.is_running() and (not args.steps or step < args.steps):
        tick = time.time()
        with torch.inference_mode():
            observations, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(observations["policy"]).all() and torch.isfinite(reward).all()
        step += 1
        if not args.headless:
            time.sleep(max(0, native.step_dt - (time.time() - tick)))
    report = dict(
        success=True,
        tiles=native.num_envs,
        families=list(FAMILIES),
        difficulty_rows=6,
        steps=step,
        finite_observations=True,
        grounded_spawn_and_goal=True,
        actual_gap_depth_verified=1.0,
        planner_policy_loaded=False,
        ppo_updates=0,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PLANNER_TERRAIN_OK " + json.dumps(report), flush=True)
finally:
    if "env" in globals():
        env.close()
    app.close()
