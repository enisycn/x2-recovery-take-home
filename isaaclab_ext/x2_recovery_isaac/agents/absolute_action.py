"""Bounded absolute joint-position action used by HumanUP."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class BoundedAbsoluteJointPositionAction(JointPositionAction):
    """Apply ``q_target = clip(q_default + scale * action, soft_limits)``."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=limits[..., 0],
            max=limits[..., 1],
        )


@configclass
class BoundedAbsoluteJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = BoundedAbsoluteJointPositionAction
