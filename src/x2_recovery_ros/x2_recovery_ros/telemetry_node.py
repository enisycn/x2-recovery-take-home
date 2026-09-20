"""Compact telemetry consumer for the recovery demonstration."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class TelemetryNode(Node):
    def __init__(self) -> None:
        super().__init__("x2_telemetry")
        self.declare_parameter("joint_name", "left_knee_joint")
        self.declare_parameter("log_period_sec", 1.0)
        self._status = "UNKNOWN"
        self._last_joint: float | None = None

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, "/x2/recovery_status", self._on_status, status_qos
        )
        self.create_subscription(
            JointState, "/x2/joint_states", self._on_joint_state, 10
        )
        self.create_timer(float(self.get_parameter("log_period_sec").value), self._log)

    def _on_status(self, message: String) -> None:
        if message.data != self._status:
            self._status = message.data
            self.get_logger().info(f"status={self._status}")

    def _on_joint_state(self, message: JointState) -> None:
        selected = str(self.get_parameter("joint_name").value)
        try:
            index = message.name.index(selected)
        except ValueError:
            return
        if index < len(message.position):
            self._last_joint = float(message.position[index])

    def _log(self) -> None:
        selected = str(self.get_parameter("joint_name").value)
        position = "n/a" if self._last_joint is None else f"{self._last_joint:.4f} rad"
        self.get_logger().info(f"status={self._status} {selected}={position}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

