from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("x2_recovery_ros"))
    parameters = str(share / "config/recovery.yaml")
    return LaunchDescription(
        [
            Node(
                package="x2_recovery_ros",
                executable="x2_recovery_node",
                name="x2_recovery",
                output="screen",
                parameters=[parameters],
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

