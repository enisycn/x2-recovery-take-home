#!/usr/bin/env python3
"""Subscribe before triggering recovery, then test actual concurrent rejection."""
import json
import time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

rclpy.init()
node = rclpy.create_node('hrs_recovery_validation')
statuses, joints = [], []
qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 reliability=ReliabilityPolicy.RELIABLE)
node.create_subscription(String, '/x2/recovery_status', lambda msg: statuses.append(msg.data), qos)
node.create_subscription(JointState, '/x2/joint_states', lambda msg: joints.append({
    'names':list(msg.name), 'positions':list(msg.position),
    'stamp_sec':msg.header.stamp.sec, 'stamp_nanosec':msg.header.stamp.nanosec}), 10)
client = node.create_client(Trigger, '/x2/start_recovery')
try:
    assert client.wait_for_service(timeout_sec=15.), 'service unavailable'
    started = time.monotonic()
    first = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, first, timeout_sec=5.)
    assert first.done() and first.result().success, 'initial request not accepted'
    acceptance_s = time.monotonic() - started
    # Send the concurrent request immediately, without another CLI startup.
    second = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, second, timeout_sec=5.)
    assert second.done() and not second.result().success, 'concurrent request not rejected'
    deadline = time.monotonic() + 25.
    while time.monotonic() < deadline and not any(s in ('SUCCEEDED','FAILED') for s in statuses):
        rclpy.spin_once(node, timeout_sec=.1)
    assert 'SUCCEEDED' in statuses, f'physical recovery did not succeed: {statuses}'
    assert joints and len(joints[-1]['names'])==31 and len(joints[-1]['positions'])==31
    assert 'left_knee_joint' in joints[-1]['names']
    assert joints[-1]['stamp_sec'] > 0
    print(json.dumps({'backend':'Isaac Lab / PhysX', 'accepted':first.result().success,
        'acceptance_wall_s':acceptance_s, 'busy_rejected':not second.result().success,
        'statuses':statuses, 'joint_samples':len(joints), 'last_joint_state':joints[-1],
        'success':True},indent=2),flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
