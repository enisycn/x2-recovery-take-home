"""Task-local pre-reset capture for evaluation and service diagnostics."""

from isaaclab.envs import ManagerBasedRLEnv


class X2RecoveryEnv(ManagerBasedRLEnv):
    terminal_observer = None
    terminal_snapshot = None

    def _reset_idx(self, env_ids):
        if self.terminal_observer is not None:
            # Call before scene/sensor/action/episode buffers are reset.
            self.terminal_snapshot = self.terminal_observer(self, env_ids)
        super()._reset_idx(env_ids)
