"""Docked Kit panel for command and measured body-frame velocities."""

import asyncio
import math
from collections import deque

import omni.kit.app
import omni.ui as ui

# Kit colors use ABGR.
TEXT = 0xFFE8DFD5
MUTED = 0xFFA99A88
CMD = 0xFF64C6FA
ACTUAL = 0xFFE6C45E
GRID = 0xFF42382C


class VelocityPanel:
    def __init__(self, window=None, frame=None):
        # Embed in the V1 command inspector, or retain the standalone play panel.
        self.owns_window = window is None
        self.window = window or ui.Window("Velocity Tracking", width=430, height=440)
        self.history = [[deque(maxlen=250), deque(maxlen=250)] for _ in range(3)]
        self.plots = []
        self.values = []
        self.ticks = []
        self.last_update = -1.0
        with (frame if frame is not None else self.window.frame):
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=3, height=0):
                    ui.Label("VELOCITY TRACKING", height=20, style={"color": TEXT, "font_size": 16})
                    self.case = ui.Label("Waiting for simulation", height=22, style={"color": MUTED})
                    with ui.HStack(height=18):
                        ui.Label("  CMD", style={"color": CMD})
                        ui.Label("  ACTUAL", style={"color": ACTUAL})
                        ui.Label("Body frame | 25 s", style={"color": MUTED})
                    for name, unit in (("vx", "m/s"), ("vy", "m/s"), ("wz", "rad/s")):
                        with ui.HStack(height=21, spacing=4):
                            ui.Label(f"{name}  {unit}", width=80, style={"color": TEXT, "font_size": 14})
                            command = ui.Label("+0.000", style={"color": CMD, "font_size": 14})
                            actual = ui.Label("+0.000", style={"color": ACTUAL, "font_size": 14})
                            error = ui.Label("err 0.000", width=95, style={"color": MUTED})
                            self.values.append((command, actual, error))
                        with ui.HStack(height=60, spacing=4):
                            with ui.VStack(width=38):
                                upper = ui.Label("+1.0", height=12, style={"color": MUTED, "font_size": 10})
                                ui.Spacer()
                                ui.Label("0", height=12, style={"color": MUTED, "font_size": 10})
                                ui.Spacer()
                                lower = ui.Label("-1.0", height=12, style={"color": MUTED, "font_size": 10})
                                self.ticks.append((upper, lower))
                            with ui.ZStack():
                                ui.Rectangle(style={"background_color": 0xFF241C14, "border_radius": 4})
                                with ui.VStack():
                                    ui.Line(height=1, style={"color": GRID})
                                    ui.Spacer()
                                    ui.Line(height=1, style={"color": GRID})
                                    ui.Spacer()
                                    ui.Line(height=1, style={"color": GRID})
                                pair = []
                                for color in (CMD, ACTUAL):
                                    pair.append(ui.Plot(ui.Type.LINE, -1.0, 1.0, 0.0, 0.0,
                                                        style={"color": color, "background_color": 0x0}))
                                self.plots.append(pair)
                        with ui.HStack(height=12):
                            ui.Spacer(width=42)
                            ui.Label("-25 s", style={"color": MUTED, "font_size": 10})
                            ui.Label("-12.5 s", style={"color": MUTED, "font_size": 10})
                            ui.Label("now", width=30, style={"color": MUTED, "font_size": 10})
                    self.status = ui.Label("", height=24, word_wrap=True, style={"color": MUTED})
        self.docking = asyncio.ensure_future(self._dock())

    async def _dock(self):
        # Wait for Kit's layout, then share the existing right-hand tab group.
        # SAME preserves the viewport width instead of adding another split pane.
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
            target = ui.Workspace.get_window("Stage") or ui.Workspace.get_window("Render Settings")
            if target is not None:
                self.window.dock_in(target, ui.DockPosition.SAME, 1.0)
                self.window.focus()
                print("[INFO] Velocity Tracking docked with Stage / Render Settings", flush=True)
                return

    def update(self, timestamp, case, cmd, actual, resets, clear=False):
        if clear:
            # Never draw a continuous trajectory across a teleport / environment reset.
            for pair in self.history:
                for history in pair:
                    history.clear()
            self.last_update = -1.0
        if timestamp - self.last_update < 0.099:
            return
        self.last_update = timestamp
        self.case.text = case
        for index, (pair, plots) in enumerate(zip(self.history, self.plots)):
            measured = actual[index]
            labels = self.values[index]
            labels[0].text = f"{cmd[index]:+.3f}"
            labels[1].text = f"{measured:+.3f}"
            labels[2].text = f"err {measured - cmd[index]:+.3f}"
            if not math.isfinite(measured):
                continue
            pair[0].append(cmd[index])
            pair[1].append(measured)
            maximum = max(0.2, *(abs(v) for series in pair for v in series)) * 1.1
            # Round axes to readable steps and avoid jitter from tiny fluctuations.
            increment = 0.1 if maximum <= 1 else 0.5
            bound = math.ceil(maximum / increment) * increment
            for plot, series in zip(plots, pair):
                plot.scale_min = -bound
                plot.scale_max = bound
                # Left padding holds the first sample until the 25-second buffer fills.
                plot.set_data(*([series[0]] * (250 - len(series)) + list(series)))
            self.ticks[index][0].text = f"+{bound:.1f}"
            self.ticks[index][1].text = f"-{bound:.1f}"
        self.status.text = f"t {timestamp:.1f}s  |  resets {resets}  |  history clears on reset / new case"

    def close(self):
        self.docking.cancel()
        if self.owns_window:
            self.window.destroy()
