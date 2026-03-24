"""
Launch file for Go2W real robot with ROS2 control.
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    # Robot description
    package_name = "go2w_description"
    package_path = get_package_share_directory(package_name)
    xacro_file = os.path.join(package_path, "xacro", "robot.xacro")
    robot_description = xacro.process_file(xacro_file).toxml()

    # Controller config
    controller_config = os.path.join(
        get_package_share_directory("robot_joint_controller"),
        "examples",
        "config",
        "robot_control_group.yaml",
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
    )

    # Controller manager
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            controller_config,
        ],
        output="screen",
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # Robot joint controller (group)
    robot_joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robot_joint_controller"],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation time",
            ),
            robot_state_publisher_node,
            controller_manager_node,
            joint_state_broadcaster_spawner,
            robot_joint_controller_spawner,
        ]
    )
