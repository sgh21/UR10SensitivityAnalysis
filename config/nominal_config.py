"""Nominal robot constants used by the baseline.

Units are meters and radians.  The default table is a UR10-style Modified
Denavit-Hartenberg model and can be replaced with the plant-specific MD-H
table without touching the calibration algorithm.
"""

from __future__ import annotations

import numpy as np


NOMINAL_ROBOT = {
    "base_xyz": [0.0, 0.0, 0.0],
    "base_rpy": [0.0, 0.0, 0.0],
    "tool_xyz": [0.0, 0.0, 0.390],
    "tool_rpy": [0.0, 0.0, 0.0],
    "mdh": {
        # T(i-1, i) = Rx(alpha_i) * Tx(a_i) * Rz(theta_i) * Tz(d_i)
        "alpha": [0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0],
        "a": [0.0, 0.0, -0.6120, -0.5723, 0.0, 0.0],
        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
        "theta_offset": [0.0] * 6,
    },
    "joint_limits": [
        [-np.pi, np.pi],
        [-2.4, -0.4],
        [-2.6, 0.2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
    ],
}
