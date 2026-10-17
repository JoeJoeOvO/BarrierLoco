"""Manual GoalGrid mode-input inspection; no Planner or automatic jump timing."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "locolab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--initial-flag", type=int, choices=(0, 1), default=0)
parser.add_argument("--steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch

    import locolab.source.locomotion.tasks  # noqa: F401
    from locolab.planner.occupancy import yaw_rotate
    from locolab.planner.skills import FrozenSkills
    from locolab.source.locomotion.tasks.Go2W.planner_terrain_env_cfg import Go2WPlannerTerrainEnvCfg

    cfg = Go2WPlannerTerrainEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.observations.policy.goal_xy = None
    cfg.sim.device = args.device
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (-4.0, -4.0, 2.5)
    cfg.viewer.lookat = (1.0, 0.0, 0.0)
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    env = gym.make("BarrierLoco-Go2W-PlannerTerrain-v0", cfg=cfg).unwrapped
    env.reset(seed=42)
    adapter = FrozenSkills(env, ROOT)
    adapter.reset(torch.tensor([0], device=env.device))
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    base_ids, _ = sensor.find_bodies("base")
    state = {"flag": args.initial_flag, "speed": 1.0, "reset": False}
    descriptors = torch.tensor([[0.2, 0.6, 0.0]], device=env.device)
    resets = 0
    low_time = 0.0
    if not args.headless:
        import omni.ui as ui

        from locolab.source.locomotion.utils.velocity_panel import VelocityPanel

        window = ui.Window("Jump 0 / 1 Control", width=480, height=760)

        def set_flag(value):
            state["flag"] = int(value)
            flag_label.text = f"Jump input = {state['flag']} (held until you change it)"
            print("JUMP_SWITCH " + json.dumps(state), flush=True)

        with window.frame, ui.VStack(spacing=6):
            flag_label = ui.Label(f"Jump input = {state['flag']} (held until you change it)", height=28)
            ui.Button("Toggle Jump 0 / 1", height=48, clicked_fn=lambda: set_flag(1 - state["flag"]))
            with ui.HStack(height=32):
                ui.Button("Set 0", clicked_fn=lambda: set_flag(0))
                ui.Button("Set 1", clicked_fn=lambda: set_flag(1))
            ui.Label("Speed command (m/s)", height=22)
            with ui.HStack(height=30):
                for speed in (0.0, 0.5, 1.0, 1.5):
                    ui.Button(str(speed), clicked_fn=lambda value=speed: state.update(speed=value))
            ui.Label("Fixed targets: H=0.20 m, X=0.60 m, Y=0.00 m", height=28)
            ui.Button("Reset robot (keeps selected 0 / 1)", height=28, clicked_fn=lambda: state.update(reset=True))
            frame = ui.Frame()
        panel = VelocityPanel(window=window, frame=frame)
    step = 0
    while app.is_running() and (not args.steps or step < args.steps):
        tick = time.monotonic()
        if state["reset"]:
            env.reset()
            adapter.reset(torch.tensor([0], device=env.device))
            state["reset"] = False
            low_time = 0.0
            resets += 1
        data = robot.data
        velocity_w = torch.tensor([[state["speed"], 0.0, 0.0]], device=env.device)
        velocity = yaw_rotate(velocity_w, data.root_quat_w, inverse=True)
        velocity[:, 2] = 0
        # Match the original pre-trigger contract: target descriptors remain
        # visible at flag=0. No timer masks, zeros, or automatic trigger changes.
        obs = torch.cat(
            (
                data.root_ang_vel_b * 0.2,
                data.projected_gravity_b,
                velocity,
                torch.full((1, 1), float(state["flag"]), device=env.device),
                descriptors,
                data.joint_pos[:, adapter.legs] - data.default_joint_pos[:, adapter.legs],
                (data.joint_vel[:, adapter.joints] - data.default_joint_vel[:, adapter.joints]) * 0.05,
                adapter.previous,
            ),
            -1,
        )
        assert obs.shape[-1] == 57
        with torch.no_grad():
            action = adapter.models[2](obs.clamp(-100, 100)).clamp(-100, 100)
            adapter.previous.copy_(action)
            env.step(action)
        low = float(data.root_pos_w[0, 2]) < 0.12
        contact = float(sensor.data.net_forces_w[:, base_ids].norm(dim=-1).amax()) > 10
        low_time = low_time + env.step_dt if low or contact else 0.0
        if float(data.projected_gravity_b[0, 2]) > -0.5403023 or low_time >= 0.3:
            state["reset"] = True
        if not args.headless:
            actual = yaw_rotate(data.root_lin_vel_w, data.root_quat_w, inverse=True)[0]
            actual[2] = data.root_ang_vel_w[0, 2]
            panel.update(
                step * env.step_dt,
                f"GoalGrid only | Jump flag = {state['flag']}",
                velocity[0].tolist(),
                actual.tolist(),
                resets,
            )
            panel.details.text = (
                f"Manual jump flag: {state['flag']} | no automatic timer\n"
                f"H / X / Y = 0.20 / 0.60 / 0.00 m (also supplied at flag=0)\n"
                f"Body height: {float(data.root_pos_w[0, 2]):.3f} m | "
                f"vertical speed: {float(data.root_lin_vel_w[0, 2]):.3f} m/s\n"
                f"Resets: {resets}"
            )
            time.sleep(max(0, env.step_dt - (time.monotonic() - tick)))
        if step % 250 == 0:
            print(
                "JUMP_SWITCH_ALIVE "
                + json.dumps(
                    {
                        "step": step,
                        **state,
                        "height": float(data.root_pos_w[0, 2]),
                        "vx": float(data.root_lin_vel_b[0, 0]),
                    }
                ),
                flush=True,
            )
        step += 1
finally:
    if "panel" in globals():
        panel.close()
    if "env" in globals():
        env.close()
    app.close()
