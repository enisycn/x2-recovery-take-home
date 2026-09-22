from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("x2_recovery_ros"))
    parameters = str(share / "config/recovery.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="isaac_ipc"),
            DeclareLaunchArgument("socket_path", default_value="/tmp/hrs_x2_recovery.sock"),
            DeclareLaunchArgument("policy_mode", default_value="checkpoint"),
            DeclareLaunchArgument("timeout_sec", default_value="10.0"),
            Node(
                package="x2_recovery_ros",
                executable="x2_recovery_node",
                name="x2_recovery",
                output="screen",
                parameters=[
                    parameters,
                    {
                        "backend": LaunchConfiguration("backend"),
                        "socket_path": LaunchConfiguration("socket_path"),
                        "policy_mode": LaunchConfiguration("policy_mode"),
                        "timeout_sec": LaunchConfiguration("timeout_sec"),
                    },
                ],
            ),
            Node(
                package="x2_recovery_ros",
                executable="x2_telemetry_node",
                name="x2_telemetry",
                output="screen",
                parameters=[parameters],
            ),
        ]
    )
