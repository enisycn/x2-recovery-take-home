"""Eight symmetric recovery commands mapped into 31 bounded joint targets.

FRASA motivates reducing the sagittal search space. X2 keeps two additional
lateral balance commands. Limits here restrict commanded motion inside, never
expand, the official URDF ranges. This is an X2 adaptation, not a FRASA replica.
"""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# group, centre, half-range (radians). Left/right axes have the same URDF sign.
GROUPS = (
    (("left_hip_pitch_joint", "right_hip_pitch_joint"), -.9, 1.4),
    (("left_knee_joint", "right_knee_joint"), 1.15, 1.15),
    (("left_ankle_pitch_joint", "right_ankle_pitch_joint"), -.15, .70),
    (("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"), -.5, 2.0),
    (("left_elbow_joint", "right_elbow_joint"), -.8, .8),
    (("waist_pitch_joint",), 0., .30),
    (("left_ankle_roll_joint", "right_ankle_roll_joint"), 0., .15),
    (("left_hip_roll_joint", "right_hip_roll_joint"), 0., .15),
)


class SymmetricRecoveryAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, len(GROUPS)), device=env.device)
        self._targets = self._asset.data.default_joint_pos.torch.clone()
        self._joint_ids = list(range(self._asset.num_joints))
        self._map = torch.zeros((len(GROUPS), self._asset.num_joints), device=env.device)
        self._centre = self._targets[0].clone()
        for group, (names, centre, span) in enumerate(GROUPS):
            ids, resolved = self._asset.find_joints(list(names), preserve_order=True)
            if tuple(resolved) != names:
                raise RuntimeError(f'Unexpected joint mapping: {resolved}')
            self._centre[ids] = centre
            self._map[group, ids] = span

    @property
    def action_dim(self): return len(GROUPS)

    @property
    def raw_actions(self): return self._raw

    @property
    def processed_actions(self): return self._targets

    def process_actions(self, actions):
        self._raw[:] = actions
        targets = self._centre + torch.tanh(actions) @ self._map
        limits = self._asset.data.soft_joint_pos_limits.torch
        self._targets[:] = targets.clamp(min=limits[..., 0], max=limits[..., 1])

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._targets)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.
        self._targets[ids] = self._asset.data.default_joint_pos.torch[ids]


@configclass
class SymmetricRecoveryActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SymmetricRecoveryAction
    asset_name: str = 'robot'
