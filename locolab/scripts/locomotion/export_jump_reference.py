"""Convert an accepted full-dynamics replay to a named, full-state training reference.

No retiming, smoothing, or pose reconstruction is performed. The only coordinate
change is translating XY to the trigger point and removing its initial yaw.
Rewards use simulated joint positions, never PD targets or raw policy actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def rotate_z(value, angle):
    x, y, z = value
    c, s = math.cos(angle), math.sin(angle)
    return [c * x - s * y, s * x + c * y, z]


def quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    ]


def rotate_inverse(q, vector):
    v = quat_multiply(quat_multiply([q[0], -q[1], -q[2], -q[3]], [0.0, *vector]), q)
    return v[1:]


def convert(candidate: dict, replay: dict, *, candidate_sha: str, replay_sha: str) -> dict:
    metadata = candidate["metadata"]
    if not replay.get("accepted") or not metadata.get("independent_validation_success"):
        raise ValueError("An independently accepted trajectory is required")
    if replay["input_sha256"] != metadata["independent_validation_input_sha256"]:
        raise ValueError("Replay does not correspond to the accepted control tape")
    if replay.get("policy_used") or not replay.get("normal_ground_reset_and_warmup"):
        raise ValueError("Expected open-loop dynamics after an ordinary grounded warmup")
    states = replay["record"]["states"]
    time = [round(float(s["time"]), 8) for s in states]
    if time[0] != 0.0 or any(b <= a for a, b in zip(time, time[1:])):
        raise ValueError("Reference times must increase from zero")
    airborne_segments, segment = [], []
    for i, state in enumerate(states):
        if not any(state["wheel_contact"]):
            segment.append(i)
        elif segment:
            airborne_segments.append(segment)
            segment = []
    if segment:
        airborne_segments.append(segment)
    flight = max(airborne_segments, key=len)
    if flight[-1] + 1 >= len(time):
        raise ValueError("Reference never touches down")
    takeoff, landing = time[flight[0]], time[flight[-1] + 1]
    apex_index = max(range(flight[0], flight[-1] + 2), key=lambda i: states[i]["base_position"][2])
    apex = time[apex_index]
    if not 0 < takeoff < apex < landing < time[-1]:
        raise ValueError("Invalid main-flight schedule")
    origin = states[0]["base_position"]
    q0 = states[0]["base_orientation_wxyz"]
    w, x, y, z = q0
    heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw_inverse = [math.cos(heading / 2), 0.0, 0.0, -math.sin(heading / 2)]
    positions, velocities, quaternions, angular_velocities, pitches, pitch_rates = [], [], [], [], [], []
    for state in states:
        p = state["base_position"]
        positions.append(rotate_z([p[0] - origin[0], p[1] - origin[1], p[2]], -heading))
        velocities.append(rotate_z(state["base_linear_velocity"], -heading))
        q = quat_multiply(yaw_inverse, state["base_orientation_wxyz"])
        norm = math.sqrt(sum(v * v for v in q))
        q = [v / norm for v in q]
        if quaternions and sum(a * b for a, b in zip(quaternions[-1], q)) < 0:
            q = [-v for v in q]
        quaternions.append(q)
        omega = rotate_z(state["base_angular_velocity_world"], -heading)
        angular_velocities.append(omega)
        w, x, y, z = q
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        body_omega = rotate_inverse(q, omega)
        pitches.append(pitch)
        pitch_rates.append(math.cos(roll) * body_omega[1] - math.sin(roll) * body_omega[2])
    height = max(p[2] for p in positions) - positions[0][2]
    result = {
        "metadata": {
            "task": "OmniBot-Go2W-Jump-MLP-v0",
            "model": "isaac_lab_full_go2w_rigid_body_dynamics",
            "generator": "scripts/locomotion/export_jump_reference.py",
            "source_method": metadata["method"],
            "candidate_sha256": candidate_sha,
            "replay_sha256": replay_sha,
            "control_tape_sha256": replay["input_sha256"],
            "seed_policy_sha256": metadata["seed_checkpoint_sha256"],
            "state_source": "independent_replay_actual_simulated_state",
            "joint_targets_used_for_reward": False,
            "coordinate_frame": "trigger_yaw; trigger XY is zero; absolute ground-relative Z",
            "source_initial_heading": heading,
            "source_vx_command": metadata["vx_command"],
            "mass": metadata["mass_kg"],
            "gravity": 9.81,
            "leg_joint_names": metadata["leg_joint_names"],
            "wheel_joint_names": metadata["wheel_joint_names"],
            "target_jump_height": height,
            "target_takeoff_time": takeoff,
            "target_apex_time": apex,
            "target_landing_time": landing,
            "flight_start_time": takeoff,
            "apex_time": apex,
            "flight_end_time": landing,
            "sequence_duration": time[-1],
            "time_step": metadata["time_step"],
            "self_collisions_enabled": metadata["self_collisions_enabled"],
            "leg_effort_limit_nm": metadata["leg_effort_limit_nm"],
            "leg_velocity_limit_rad_s": metadata["leg_velocity_limit_rad_s"],
            "validation": {
                "success": True,
                "scope": metadata["independent_validation_scope"],
                "metrics": replay["metrics"],
                "batch_acceptance_count": metadata["batch_acceptance_count"],
                "batch_env_count": metadata["batch_env_count"],
                "robustness_certified": False,
            },
        },
        "time": time,
        "base_position": positions,
        "base_linear_velocity": velocities,
        "base_pitch": pitches,
        "base_pitch_rate": pitch_rates,
        "base_orientation": quaternions,
        "base_angular_velocity_world": angular_velocities,
        "joint_position": [s["joint_position"] for s in states],
        "joint_velocity": [s["joint_velocity"] for s in states],
        "wheel_position": [s["wheel_joint_position"] for s in states],
        "wheel_velocity": [s["wheel_joint_velocity"] for s in states],
    }
    # Fail before serialization rather than emitting non-standard NaN/Infinity.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cb, rb = args.candidate.read_bytes(), args.replay.read_bytes()
    payload = convert(
        json.loads(cb),
        json.loads(rb),
        candidate_sha=hashlib.sha256(cb).hexdigest(),
        replay_sha=hashlib.sha256(rb).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "samples": len(payload["time"]),
                "schedule": {
                    k: payload["metadata"][k]
                    for k in ("target_takeoff_time", "target_apex_time", "target_landing_time", "sequence_duration")
                },
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
