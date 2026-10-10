"""Deterministic body-frame command cases for interactive Go2W inspection."""


def velocity_cases():
    """31 body-frame cases per gait; yaw is rad/s, translation is m/s.

    These are representative probes, not a Cartesian product of all velocities.
    Playback resets between cases and holds each command for six seconds by default.
    """
    cases = [("stand", (0.0, 0.0, 0.0))]
    for speed in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        for sign in (1, -1):
            cases.append((f"vx {sign * speed:+g}", (sign * speed, 0.0, 0.0)))
    for speed in (0.25, 0.5):
        for sign in (1, -1):
            cases.append((f"vy {sign * speed:+g}", (0.0, sign * speed, 0.0)))
    for vx in (1.0, -1.0):
        for vy in (0.5, -0.5):
            cases.append((f"diagonal {vx:+g}/{vy:+g}", (vx, vy, 0.0)))
    for rate in (0.5, 1.5, 3.0):
        for sign in (1, -1):
            cases.append((f"turn {sign * rate:+g}", (0.0, 0.0, sign * rate)))
    for vx in (1.0, -1.0):
        for rate in (1.0, -1.0):
            cases.append((f"arc {vx:+g}/{rate:+g}", (vx, 0.0, rate)))
    return [(gait, name, velocity) for gait in ("rolling", "trot", "bound") for name, velocity in cases]


def gait_command(gait, velocity, phase_presets):
    """Keep height/frequency fixed to isolate velocity tracking in each gait."""
    rolling = gait == "rolling"
    return [*velocity, 0.34, 2.5, *phase_presets[gait], 1.0 if rolling else 0.5, 0.0 if rolling else 0.06]
