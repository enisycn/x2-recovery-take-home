from pathlib import Path
import json
import socket
import threading

import pytest

from x2_recovery_ros.session import AttemptGate, RecoverySession


def checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts/recovery_policy.npz"


def test_attempt_gate_rejects_second_request_while_busy():
    gate = AttemptGate()
    assert gate.try_acquire()
    assert not gate.try_acquire()
    gate.release()
    assert gate.try_acquire()


def test_checkpoint_session_publishes_state_and_succeeds():
    samples = []
    result = RecoverySession(
        backend="reduced",
        checkpoint=checkpoint_path(),
        policy_mode="checkpoint",
        seed=101,
        timeout_sec=6.0,
        real_time=False,
    ).run(on_step=samples.append)
    assert result.success
    assert samples
    assert len(samples[-1]["joint_names"]) == len(samples[-1]["joint_positions"])


def test_zero_policy_reaches_failed_on_timeout():
    result = RecoverySession(
        backend="reduced",
        checkpoint=checkpoint_path(),
        policy_mode="zero",
        seed=1,
        timeout_sec=0.25,
        real_time=False,
    ).run()
    assert not result.success
    assert result.status == "FAILED"
    assert result.failure_reason == "timeout before stable two-foot stance"


def test_isaac_ipc_session_forwards_joint_state_and_result(tmp_path):
    socket_path = tmp_path / "isaac.sock"
    ready = threading.Event()

    # Some CI sandboxes block every socket family, including local AF_UNIX.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(socket_path))
    except PermissionError:
        pytest.skip("sandbox blocks Unix sockets")
    finally:
        probe.close()
    socket_path.unlink(missing_ok=True)

    def fake_server():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                request = json.loads(stream.readline())
                assert request["command"] == "start"
                for event in (
                    {"type": "step", "step": 1, "joint_names": ["joint"], "joint_positions": [0.2]},
                    {"type": "result", "success": True, "steps": 1, "failure_reason": ""},
                ):
                    stream.write((json.dumps(event) + "\n").encode())
                stream.flush()

    thread = threading.Thread(target=fake_server, daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0)
    samples = []
    result = RecoverySession(
        backend="isaac_ipc",
        checkpoint=checkpoint_path(),
        policy_mode="checkpoint",
        seed=101,
        timeout_sec=1.0,
        socket_path=str(socket_path),
        real_time=False,
    ).run(on_step=samples.append)
    thread.join(timeout=2.0)
    assert result.success
    assert samples[0]["joint_names"] == ["joint"]
