"""ROS 2 service and simulator bridge for one recovery attempt at a time."""

from __future__ import annotations

from pathlib import Path
import threading

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .session import AttemptGate, RecoverySession


class RecoveryNode(Node):
    def __init__(self) -> None:
        super().__init__("x2_recovery")
        share = Path(get_package_share_directory("x2_recovery_ros"))
        self.declare_parameter("backend", "reduced")
        self.declare_parameter("policy_mode", "checkpoint")
        self.declare_parameter("checkpoint", str(share / "artifacts/recovery_policy.npz"))
        self.declare_parameter("socket_path", "/tmp/hrs_x2_recovery.sock")
        self.declare_parameter("timeout_sec", 6.0)
        self.declare_parameter("seed", 101)
        self.declare_parameter("real_time", True)

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_publisher = self.create_publisher(
            String, "/x2/recovery_status", status_qos
        )
        self.joint_publisher = self.create_publisher(
            JointState, "/x2/joint_states", 10
        )
        self.service = self.create_service(
            Trigger, "/x2/start_recovery", self._handle_start
        )
        self._gate = AttemptGate()
        self._pending = False
        self._pending_lock = threading.Lock()
        self._attempt_index = 0
        self._worker: threading.Thread | None = None
        # Dispatching from a timer ensures the service callback returns its
        # acceptance response before simulator execution starts.
        self._dispatch_timer = self.create_timer(0.05, self._dispatch_pending)
        self._publish_status("IDLE")
        self.get_logger().info("X2 recovery ready")

    def _handle_start(self, request: Trigger.Request, response: Trigger.Response):
        del request
        if not self._gate.try_acquire():
            response.success = False
            response.message = "Recovery already running"
            return response
        with self._pending_lock:
            self._pending = True
        response.success = True
        response.message = "Recovery accepted"
        return response

    def _dispatch_pending(self) -> None:
        with self._pending_lock:
            if not self._pending:
                return
            self._pending = False
        self._attempt_index += 1
        self._publish_status("RUNNING")
        self._worker = threading.Thread(
            target=self._run_attempt,
            name=f"x2-recovery-{self._attempt_index}",
            daemon=True,
        )
        self._worker.start()

    def _run_attempt(self) -> None:
        try:
            session = RecoverySession(
                backend=str(self.get_parameter("backend").value),
                checkpoint=str(self.get_parameter("checkpoint").value),
                policy_mode=str(self.get_parameter("policy_mode").value),
                seed=int(self.get_parameter("seed").value) + self._attempt_index - 1,
                timeout_sec=float(self.get_parameter("timeout_sec").value),
                socket_path=str(self.get_parameter("socket_path").value),
                real_time=bool(self.get_parameter("real_time").value),
            )
            result = session.run(on_step=self._publish_joint_state)
            self._publish_status(result.status)
            if result.success:
                self.get_logger().info(f"Recovery succeeded in {result.steps} steps")
            else:
                self.get_logger().warning(
                    f"Recovery failed in {result.steps} steps: {result.failure_reason}"
                )
        except Exception as exc:
            self.get_logger().error(f"Recovery error: {exc}")
            self._publish_status("FAILED")
        finally:
            self._gate.release()

    def _publish_status(self, value: str) -> None:
        message = String()
        message.data = value
        self.status_publisher.publish(message)

    def _publish_joint_state(self, info: dict) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(info["joint_names"])
        message.position = [float(item) for item in info["joint_positions"]]
        self.joint_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecoveryNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
