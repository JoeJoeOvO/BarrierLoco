"""Replay a trained OCC navigation Planner with docked telemetry; no PPO updates."""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--difficulty", type=int, choices=range(6), default=2)
parser.add_argument("--terrain-index", type=int, choices=range(10), default=4)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--forward-course", action="store_true", help="Continuous forward gap route across six tiles")
parser.add_argument("--follow-jumps", action="store_true", help="Observe natural Jump choices and physical takeoff")
parser.add_argument("--stochastic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import PreviewSurfaceCfg, SphereCfg

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.planner.navigation import NavigationRollout
    from locolab.planner.network import SKILL_NAMES, NavigationActorCritic
    from locolab.planner.occupancy import yaw_rotate
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg
    from locolab.source.locomotion.terrains.planner_course_cfg import TERRAIN_FAMILIES

    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    gap_only = saved["config"].get("gap_only", False)
    diverse_gaps = saved["config"].get("diverse_gaps", False)
    families = ("gap",) * len(TERRAIN_FAMILIES) if gap_only else TERRAIN_FAMILIES
    if diverse_gaps:
        from locolab.source.locomotion.terrains.planner_gaps import GAP_FAMILIES

        families = GAP_FAMILIES
    if args.forward_course and not gap_only:
        raise ValueError("Forward course requires a gap-only checkpoint")
    cfg = Go2WPlannerTerrainEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = len(families)
    if gap_only:
        from locolab.source.locomotion.terrains.planner_course_cfg import PlannerCourseCfg

        cfg.scene.terrain.terrain_generator.sub_terrains = {"gap": PlannerCourseCfg(proportion=1.0, family="gap")}
        cfg.events.reset_base.params["pose_range"]["x"] = (2.0, 3.0)
        if args.forward_course:
            # Six consecutive 16 m tiles at the selected difficulty band.
            cfg.scene.terrain.terrain_generator.difficulty_range = (args.difficulty / 6, (args.difficulty + 1) / 6)
    if saved["config"].get("continuous_mixed", False):
        from locolab.source.locomotion.terrains.planner_course_cfg import configure_continuous_terrain

        configure_continuous_terrain(cfg)
    if diverse_gaps:
        from locolab.source.locomotion.terrains.planner_course_cfg import configure_diverse_gap_terrain

        configure_diverse_gap_terrain(cfg)
    cfg.sim.device = args.device
    cfg.scene.terrain.max_init_terrain_level = 5
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (-5.0, -6.0, 4.0) if gap_only or diverse_gaps else (-9.0, -10.0, 9.0)
    cfg.viewer.lookat = (1.2, 0.0, 0.0)
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg).unwrapped
    navigation = NavigationRollout(env, ROOT, manual_reset=True, falls_only=True, gap_only=gap_only)
    navigation.all_gap_terrain = diverse_gaps
    navigation.log_resets = True
    navigation.skills.jump_always_active = saved["config"].get("jump_always_active", False)
    if saved["config"].get("continuous_mixed", False):
        navigation.directional_goals = True
        navigation.arrival_radius = saved["config"]["goal_threshold_m"]
        navigation.arrival_speed = saved["config"]["goal_speed_threshold_m_s"]
        navigation.arrival_hold_steps = saved["config"]["goal_hold_steps"]
    navigation.forward_only = args.forward_course
    navigation.realtime = not args.headless
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_types[:] = ids
    terrain.terrain_levels[:] = 0 if args.forward_course else args.difficulty
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    observation = navigation.reset()
    # Audit the actual collision mesh, not just the configured terrain names.
    axis = torch.linspace(-7.0, 7.0, 29, device=env.device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), dim=-1)
    centers = terrain.env_origins.clone()
    centers[
        : (env.num_envs if gap_only or diverse_gaps else 4), 0
    ] += 6.5  # Custom courses spawn near the tile's left edge.
    starts = centers[:, None] + points[None]
    starts[..., 2] = 15.0
    down = torch.zeros_like(starts)
    down[..., 2] = -1.0
    heights = navigation.raycast(starts, down, distance=30.0)[..., 2]
    for index, name in enumerate(families):
        finite = heights[index][torch.isfinite(heights[index])]
        print(
            "TERRAIN_AUDIT "
            + json.dumps(
                {
                    "family": name,
                    "z_min": float(finite.min()),
                    "z_max": float(finite.max()),
                    "origin": terrain.env_origins[index].tolist(),
                }
            ),
            flush=True,
        )
    model = NavigationActorCritic().to(env.device).eval()
    model.load_state_dict(saved["model"], strict=True)
    assert saved["config"]["proprio_dim"] == 58
    iteration = saved["iteration"]
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/NavigationGoals",
            markers={
                "goal": SphereCfg(radius=0.08, visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)))
            },
        )
    )
    state = {
        "index": args.terrain_index,
        "difficulty": args.difficulty,
        "reset": False,
        "clear": True,
        "stochastic": args.stochastic,
    }
    failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    timeouts = torch.zeros_like(failures)
    if not args.headless:
        import omni.ui as ui

        from locolab.source.locomotion.utils.velocity_panel import VelocityPanel

        window = ui.Window("Planner Navigation", width=480, height=760)

        def select(delta):
            state["index"] = (state["index"] + delta) % env.num_envs
            env.viewport_camera_controller.set_view_env_index(state["index"])
            state["clear"] = True

        def level(delta):
            if args.forward_course:
                return  # The whole route uses the band selected at launch.
            state["difficulty"] = max(0, min(5, state["difficulty"] + delta))
            state["reset"] = True

        def toggle_sampling():
            state["stochastic"] = not state["stochastic"]

        with window.frame, ui.VStack(spacing=6):
            ui.Label(f"Planner checkpoint {iteration} | Orange sphere = goal", height=24, word_wrap=True)
            with ui.HStack(height=28):
                ui.Button("Previous terrain", clicked_fn=lambda: select(-1))
                ui.Button("Next terrain", clicked_fn=lambda: select(1))
            with ui.HStack(height=28):
                ui.Button("Easier", clicked_fn=lambda: level(-1))
                ui.Button("Harder", clicked_fn=lambda: level(1))
                ui.Button("New goals", clicked_fn=lambda: state.update(new_goals=True))
            ui.Button("Toggle mean / sampled policy", height=26, clicked_fn=toggle_sampling)
            frame = ui.Frame()
        panel = VelocityPanel(window=window, frame=frame)
        select(0)
    probe = None
    if args.follow_jumps:
        from locolab.planner.jump_probe import JumpProbe
        from locolab.planner.skills import decode_command

        probe = JumpProbe(env)
        navigation.step_observer = probe.tick
    step = 0
    while app.is_running() and (not args.steps or step < args.steps):
        if state.pop("new_goals", False):
            navigation.sample_goals(ids)
            observation = navigation.observe()
        if state["reset"]:
            terrain.terrain_levels[:] = state["difficulty"]
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            observation = navigation.reset()
            failures.zero_()
            timeouts.zero_()
            state.update(reset=False, clear=True)
        if args.forward_course and env.scene["robot"].data.root_pos_w[state["index"], 0] >= 46.5:
            if not args.headless:
                panel.details.text = "Forward route complete (~90 m, 6 gaps). Simulation held for inspection."
            env.sim.render()
            continue
        with torch.no_grad():
            decision = model.sample(observation, navigation.skills.allowed(), deterministic=not state["stochastic"])
            if probe is not None:
                new_jumps = ((decision["skill"] == 2) & ~navigation.skills.busy).nonzero().flatten()
                if len(new_jumps):
                    probe.begin(new_jumps, decode_command(2, decision["command"][new_jumps]), navigation.goal_w, step)
                    if not args.headless and not navigation.skills.busy[state["index"]]:
                        state["index"] = int(new_jumps[0])
                        env.viewport_camera_controller.set_view_env_index(state["index"])
                        state["clear"] = True
            observation, reward, done, info = navigation.step(decision["skill"], decision["command"])
            if probe is not None:
                probe.finish(navigation.skills.timer, done)
        assert all(torch.isfinite(value).all() for value in observation.values())
        failures += info["failed"]
        timeouts += info["timeout"]
        # Show only the tracked robot's target; its sphere touches the surface.
        goal = navigation.goal_w[state["index"] : state["index"] + 1].clone()
        goal[:, 2] += 0.08
        markers.set_visibility(bool(navigation.goal_valid[state["index"]]))
        markers.visualize(goal)
        index = state["index"]
        if not args.headless:
            robot = env.scene["robot"].data
            skill = int(navigation.skills.skill[index])
            command = navigation.skills.command[index]
            if skill == 2:
                velocity = torch.zeros(1, 3, device=env.device)
                velocity[:, 0] = command[0]
                velocity = yaw_rotate(
                    yaw_rotate(velocity, navigation.skills.jump_heading[index : index + 1]),
                    robot.root_quat_w[index : index + 1],
                    inverse=True,
                )[0]
                velocity[2] = 0
            else:
                velocity = command[:3]
            actual = yaw_rotate(
                robot.root_lin_vel_w[index : index + 1], robot.root_quat_w[index : index + 1], inverse=True
            )[0]
            actual[2] = robot.root_ang_vel_w[index, 2]
            distance = float((navigation.goal_w[index, :2] - robot.root_pos_w[index, :2]).norm())
            mode = "sampled" if state["stochastic"] else "mean"
            panel.update(
                step * 0.1,
                f"{families[index]} | level {state['difficulty']+1}/6 | {mode}",
                velocity.tolist(),
                actual.tolist(),
                int(failures[index] + timeouts[index]),
                clear=state["clear"] or bool(done[index]),
            )
            dimensions = (10, 3, 4)[skill]
            panel.details.text = (
                f"Skill: {SKILL_NAMES[skill]} | goal distance {distance:.2f} m\n"
                + (
                    f"Forward route: {float(robot.root_pos_w[index, 0]) + 44.0:.1f} / ~90 m | 6 gaps\n"
                    if args.forward_course
                    else ""
                )
                + f"Reached: {int(navigation.goal_count[index])} | failed: {int(failures[index])}"
                f" | timeout: {int(timeouts[index])}\n"
                f"Goal world XYZ: {[round(v,2) for v in navigation.goal_w[index].tolist()]}\n"
                f"Skill cmd: {[round(v,3) for v in command[:dimensions].tolist()]}\n"
                f"Jump remaining: {float(navigation.skills.timer[index]):.2f} s"
            )
            if probe is not None:
                panel.details.text += (
                    f"\nJump calls / upward takeoffs / contact after flight: "
                    f"{probe.selected} / {probe.takeoffs} / {probe.landed}"
                )
            state["clear"] = False
        step += 1
        if step == 1 or step % 100 == 0:
            print(
                "PLANNER_PLAY_ALIVE "
                + json.dumps(
                    {
                        "checkpoint_iteration": iteration,
                        "steps": step,
                        "arrivals": int(navigation.goal_count.sum()),
                        "failures": int(failures.sum()),
                        "ppo_updates": 0,
                    }
                ),
                flush=True,
            )
    print("PLANNER_PLAY_OK", step, flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if "panel" in globals():
        panel.close()
    if "env" in globals():
        env.close()
    app.close()
