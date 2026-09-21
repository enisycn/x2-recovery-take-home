"""Task-local bounded relative joint-position action."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import (
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class BoundedRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply ``q_target = clip(q_current + beta * tanh(action), limits)``.

    HoST uses current-position increments, and FRASA uses an equivalent
    integrated desired-joint command.  Clamping the final target preserves the
    imported X2 limit margin while making a zero-initialized policy hold the
    current pose instead of commanding every asymmetric joint midpoint.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_position_targets = torch.zeros_like(self.processed_actions)

    @property
    def joint_position_targets(self) -> torch.Tensor:
        """Targets sampled once at the policy rate and held over decimation."""

        return self._joint_position_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        """Compute HoST's relative target once per policy decision.

        ``ActionManager.apply_action`` runs at every physics substep.  Computing
        a relative target there compounds the same increment ``decimation``
        times.  Snapshotting here implements the stated equation exactly:
        ``q_target(t) = q(t) + beta * a(t)``.
        """

        # A hard action clip made every raw value outside [-1, 1] physically
        # identical. PPO then drove 87% of deterministic actions into that
        # flat region. A smooth tanh bound preserves a useful gradient and the
        # action-magnitude reward can regularize the actual network output.
        self._raw_actions[:] = actions
        self._processed_actions = torch.tanh(self._raw_actions) * self._scale + self._offset
        current = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._joint_position_targets[:] = torch.clamp(
            current + self.processed_actions,
            min=limits[..., 0],
            max=limits[..., 1],
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._joint_position_targets,
            joint_ids=self._joint_ids,
        )


@configclass
class BoundedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for the bounded relative position action."""

    class_type: type[ActionTerm] = BoundedRelativeJointPositionAction
