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
    """Apply ``q_target = clip(q_current + beta * action, soft limits)``.

    HoST uses current-position increments, and FRASA uses an equivalent
    integrated desired-joint command.  Clamping the final target preserves the
    imported X2 limit margin while making a zero-initialized policy hold the
    current pose instead of commanding every asymmetric joint midpoint.
    """

    def apply_actions(self) -> None:
        targets = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        targets = torch.clamp(targets, min=limits[..., 0], max=limits[..., 1])
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class BoundedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for the bounded relative position action."""

    class_type: type[ActionTerm] = BoundedRelativeJointPositionAction

