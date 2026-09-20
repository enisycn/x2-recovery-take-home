"""Robot names and limits taken from the official X2 Ultra v1.3.0 model."""

import numpy as np


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)

JOINT_LIMITS = np.asarray(
    [
        [-2.704, 2.556],
        [0.0, 2.4073],
        [-0.803, 0.453],
        [-2.704, 2.556],
        [0.0, 2.4073],
        [-0.803, 0.453],
    ],
    dtype=np.float64,
)

ACTION_NAMES = ("tuck", "arm_push", "leg_extend", "balance")

