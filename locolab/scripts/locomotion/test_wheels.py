"""Script to test wheel direction by directly controlling joint velocities."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test wheel directions.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from locolab.source.locomotion.assets.robot import GO2W_CFG


def main():
    """Test wheel directions by directly setting joint velocity targets."""

    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn robot
    robot_cfg = GO2W_CFG.replace(prim_path="/World/Robot")
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn)

    # Create ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    # Create articulation before sim reset
    robot = Articulation(robot_cfg)

    # Play simulation - this initializes the articulation
    sim.reset()

    # Now articulation is ready
    robot.reset()

    # Get joint indices for wheels
    wheel_names = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
    wheel_indices = []
    for name in wheel_names:
        idx = robot.find_joints(name)[0][0]
        wheel_indices.append(idx)
    wheel_indices = torch.tensor(wheel_indices, device=sim.device)

    print("\n" + "=" * 60)
    print("WHEEL DIRECTION TEST - Direct Joint Control")
    print("=" * 60)
    print(f"Wheel joint indices: {wheel_indices.tolist()}")
    print(f"Wheel names: {wheel_names}")
    print("=" * 60 + "\n")

    # Test sequences: (name, [FL, FR, RL, RR] velocity in rad/s)
    tests = [
        ("All wheels STOP", [0.0, 0.0, 0.0, 0.0]),
        ("All wheels +10 rad/s", [10.0, 10.0, 10.0, 10.0]),
        ("All wheels -10 rad/s", [-10.0, -10.0, -10.0, -10.0]),
        ("Only FL +10", [10.0, 0.0, 0.0, 0.0]),
        ("Only FR +10", [0.0, 10.0, 0.0, 0.0]),
        ("Only RL +10", [0.0, 0.0, 10.0, 0.0]),
        ("Only RR +10", [0.0, 0.0, 0.0, 10.0]),
        ("Left +10, Right 0", [10.0, 0.0, 10.0, 0.0]),
        ("Left 0, Right +10", [0.0, 10.0, 0.0, 10.0]),
        ("Left +10, Right -10 (should rotate)", [10.0, -10.0, 10.0, -10.0]),
        ("Left -10, Right +10 (should rotate other way)", [-10.0, 10.0, -10.0, 10.0]),
    ]

    for test_name, wheel_vel_cmd in tests:
        print(f"\n>>> TEST: {test_name}")
        print(
            f"    Command: FL={wheel_vel_cmd[0]}, FR={wheel_vel_cmd[1]}, RL={wheel_vel_cmd[2]}, RR={wheel_vel_cmd[3]}"
        )

        # Reset robot position - UPRIGHT on ground
        # Quaternion for identity: [w, x, y, z] = [1, 0, 0, 0]
        robot.write_root_pose_to_sim(torch.tensor([[0, 0, 0.45, 1, 0, 0, 0]], device=sim.device))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))

        # Set default joint positions for legs
        default_pos = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))

        # Get leg joint indices (all joints except wheels)
        leg_indices = [i for i in range(robot.num_joints) if i not in wheel_indices.tolist()]
        leg_indices = torch.tensor(leg_indices, device=sim.device)
        leg_default_pos = robot.data.default_joint_pos[0, leg_indices]

        # Run for 200 steps (1 second at 0.005s dt)
        for step in range(300):
            # Set wheel velocity targets - shape must be (num_envs, len(joint_ids))
            wheel_vel_targets = torch.tensor([wheel_vel_cmd], device=sim.device, dtype=torch.float32)

            # Apply velocity targets to wheels (velocity control)
            robot.set_joint_velocity_target(wheel_vel_targets, joint_ids=wheel_indices)

            # Hold leg joints at default position (position control)
            robot.set_joint_position_target(leg_default_pos.unsqueeze(0), joint_ids=leg_indices)

            # Write data to sim
            robot.write_data_to_sim()

            # Step simulation
            sim.step()

            # Update robot state
            robot.update(sim.cfg.dt)

            # Print every 50 steps
            if step % 50 == 0:
                actual_wheel_vel = robot.data.joint_vel[0, wheel_indices]
                # Debug: print targets and torques
                vel_target = robot.data.joint_vel_target[0, wheel_indices]
                pos_target = robot.data.joint_pos_target[0, wheel_indices]
                applied_torque = robot.data.applied_torque[0, wheel_indices]

                wheel_pos = robot.data.joint_pos[0, wheel_indices]
                print(
                    f"    Step {step:3d}: "
                    f"wheel_pos=[{wheel_pos[0]:.1f}, {wheel_pos[1]:.1f}, "
                    f"{wheel_pos[2]:.1f}, {wheel_pos[3]:.1f}], "
                    f"wheel_vel=[{actual_wheel_vel[0]:.1f}, {actual_wheel_vel[1]:.1f}, "
                    f"{actual_wheel_vel[2]:.1f}, {actual_wheel_vel[3]:.1f}], "
                    f"pos_target=[{pos_target[0]:.1f}, {pos_target[1]:.1f}, "
                    f"{pos_target[2]:.1f}, {pos_target[3]:.1f}], "
                    f"vel_target=[{vel_target[0]:.1f}, {vel_target[1]:.1f}, "
                    f"{vel_target[2]:.1f}, {vel_target[3]:.1f}], "
                    f"torque=[{applied_torque[0]:.1f}, {applied_torque[1]:.1f}, "
                    f"{applied_torque[2]:.1f}, {applied_torque[3]:.1f}]"
                )

        input("    Press Enter to continue...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()
