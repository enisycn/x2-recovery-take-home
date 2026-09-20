from pathlib import Path

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

