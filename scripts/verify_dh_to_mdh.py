"""Verify conversion from standard DH to the project's modified DH convention.

Input DH columns follow the source SystemConfig.py convention:
    [a, alpha, d, theta_offset]

Standard DH link transform:
    Rz(q + theta_offset) * Tz(d) * Tx(a) * Rx(alpha)

Project MDH link transform:
    Rx(alpha_m) * Tx(a_m) * Rz(q + theta_offset_m) * Tz(d_m)

The exact chain conversion is:
    alpha_m[0] = 0,       a_m[0] = 0
    alpha_m[i] = alpha[i-1], a_m[i] = a[i-1] for i > 0
    d_m[i] = d[i], theta_offset_m[i] = theta_offset[i]

The final standard-DH tail Tx(a6) * Rx(alpha6) must be appended after the
MDH chain, or absorbed into the tool transform.
"""

from __future__ import annotations

import argparse
import math

import numpy as np


UR10_DH_NOMINAL = np.array(
    [
        [0.0, math.pi / 2.0, 0.1273, 0.0],
        [-0.612, 0.0, 0.0, 0.0],
        [-0.5723, 0.0, 0.0, 0.0],
        [0.0, math.pi / 2.0, 0.163941, 0.0],
        [0.0, -math.pi / 2.0, 0.1157, 0.0],
        [0.0, 0.0, 0.0922, 0.0],
    ],
    dtype=float,
)

UR10_DH_DELTA = np.array(
    [
        [
            0.00151519014990144187,
            0.000174729661609918097,
            0.00061543913494158109,
            -5.85718756883034375e-05,
        ],
        [
            0.0114009672293980957,
            -0.00709088720918331188,
            -16.4530983452404946,
            0.191836362417154627,
        ],
        [
            0.234768924309791682,
            -0.00274154504728905393,
            -151.548455688576922,
            0.746561121634286784,
        ],
        [
            0.000045537526658417155,
            -0.000578855473748918214,
            168.00164179826703,
            -0.938381584982276307,
        ],
        [
            0.0000453539753039615049,
            0.000550870677181514168,
            -0.0000233111165712229784,
            -8.26046555372703106e-05,
        ],
        [
            0.0,
            0.0,
            0.0000630538353762755133,
            -3.45095783910384646e-05,
        ],
    ],
    dtype=float,
)

JOINT_LIMITS = np.array(
    [
        [-math.pi, math.pi],
        [-2.4, -0.4],
        [-2.6, 0.2],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
        [-math.pi, math.pi],
    ],
    dtype=float,
)

TOOL_XYZ_SDH = np.array([0.0, 0.0, 0.039], dtype=float)
TOOL_RPY_SDH = np.zeros(3, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    dh = UR10_DH_NOMINAL + UR10_DH_DELTA
    mdh, final_tail = standard_dh_to_project_mdh(dh)
    tool_mdh = final_tail @ make_transform(TOOL_XYZ_SDH, TOOL_RPY_SDH)

    report = verify_conversion(dh, mdh, final_tail, args.samples, args.seed)

    np.set_printoptions(precision=12, suppress=False)
    print("Standard DH initial table [a, alpha, d, theta_offset]:")
    print(dh)
    print()
    print("Project MDH arrays:")
    print("alpha =", mdh["alpha"].tolist())
    print("a =", mdh["a"].tolist())
    print("d =", mdh["d"].tolist())
    print("theta_offset =", mdh["theta_offset"].tolist())
    print()
    print("Final tail that must be appended after the MDH chain:")
    print("T_tail = Tx(a6) * Rx(alpha6)")
    print(final_tail)
    print()
    print("Equivalent MDH tool transform for TOOL_XYZ_SDH = [0, 0, 0.039]:")
    print(tool_mdh)
    print("tool_xyz_mdh =", tool_mdh[:3, 3].tolist())
    print()
    print("Verification report:")
    for key, value in report.items():
        print(f"{key}: {value:.16e}" if isinstance(value, float) else f"{key}: {value}")


def standard_dh_to_project_mdh(dh: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values = np.asarray(dh, dtype=float).reshape(6, 4)
    a = values[:, 0]
    alpha = values[:, 1]
    d = values[:, 2]
    theta_offset = values[:, 3]

    mdh = {
        "alpha": np.r_[0.0, alpha[:-1]],
        "a": np.r_[0.0, a[:-1]],
        "d": d.copy(),
        "theta_offset": theta_offset.copy(),
    }
    final_tail = trans_x(a[-1]) @ rot_x(alpha[-1])
    return mdh, final_tail


def verify_conversion(
    dh: np.ndarray,
    mdh: dict[str, np.ndarray],
    final_tail: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    lower = JOINT_LIMITS[:, 0]
    upper = JOINT_LIMITS[:, 1]

    max_matrix_abs = 0.0
    max_position_norm = 0.0
    max_rotation_angle = 0.0
    for _ in range(samples):
        q = rng.uniform(lower, upper)
        t_sdh = chain_standard_dh(dh, q)
        t_mdh = chain_project_mdh(mdh, q) @ final_tail
        diff = t_sdh - t_mdh
        max_matrix_abs = max(max_matrix_abs, float(np.max(np.abs(diff))))
        max_position_norm = max(
            max_position_norm, float(np.linalg.norm(t_sdh[:3, 3] - t_mdh[:3, 3]))
        )
        max_rotation_angle = max(
            max_rotation_angle, rotation_angle(t_sdh[:3, :3].T @ t_mdh[:3, :3])
        )

    return {
        "samples": int(samples),
        "seed": int(seed),
        "max_matrix_abs_error": max_matrix_abs,
        "max_position_error_m": max_position_norm,
        "max_rotation_error_rad": max_rotation_angle,
    }


def chain_standard_dh(dh: np.ndarray, q: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    for index in range(6):
        a, alpha, d, theta_offset = dh[index]
        transform = transform @ rot_z(q[index] + theta_offset) @ trans_z(d) @ trans_x(a) @ rot_x(alpha)
    return transform


def chain_project_mdh(mdh: dict[str, np.ndarray], q: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    for index in range(6):
        transform = (
            transform
            @ rot_x(mdh["alpha"][index])
            @ trans_x(mdh["a"][index])
            @ rot_z(q[index] + mdh["theta_offset"][index])
            @ trans_z(mdh["d"][index])
        )
    return transform


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return trans_x(xyz[0]) @ trans_y(xyz[1]) @ trans_z(xyz[2]) @ rot_x(rpy[0]) @ rot_y(rpy[1]) @ rot_z(rpy[2])


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def trans_x(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = float(distance)
    return transform


def trans_y(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[1, 3] = float(distance)
    return transform


def trans_z(distance: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[2, 3] = float(distance)
    return transform


def rotation_angle(rotation: np.ndarray) -> float:
    cos_angle = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, cos_angle))))


if __name__ == "__main__":
    main()
