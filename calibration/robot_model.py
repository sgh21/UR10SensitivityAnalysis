"""Forward model for the paper's multisource calibration baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.parameters import ErrorParameter, vector_to_components
from calibration.transforms import make_transform, modified_dh_transform, position_from_transform
from config.nominal_config import NOMINAL_ROBOT


@dataclass
class RobotParameters:
    """Nominal numerical parameters consumed by MD-H forward kinematics."""

    base_xyz: np.ndarray
    base_rpy: np.ndarray
    tool_xyz: np.ndarray
    tool_rpy: np.ndarray
    alpha: np.ndarray
    a: np.ndarray
    d: np.ndarray
    theta_offset: np.ndarray

    def copy(self) -> "RobotParameters":
        return RobotParameters(
            self.base_xyz.copy(),
            self.base_rpy.copy(),
            self.tool_xyz.copy(),
            self.tool_rpy.copy(),
            self.alpha.copy(),
            self.a.copy(),
            self.d.copy(),
            self.theta_offset.copy(),
        )


def load_nominal_robot(config: dict | None = None) -> RobotParameters:
    """Load nominal robot parameters from a dict matching ``NOMINAL_ROBOT``."""
    cfg = NOMINAL_ROBOT if config is None else config
    mdh = cfg["mdh"]
    return RobotParameters(
        base_xyz=np.asarray(cfg["base_xyz"], dtype=float).reshape(3),
        base_rpy=np.asarray(cfg["base_rpy"], dtype=float).reshape(3),
        tool_xyz=np.asarray(cfg["tool_xyz"], dtype=float).reshape(3),
        tool_rpy=np.asarray(cfg["tool_rpy"], dtype=float).reshape(3),
        alpha=np.asarray(mdh["alpha"], dtype=float).reshape(6),
        a=np.asarray(mdh["a"], dtype=float).reshape(6),
        d=np.asarray(mdh["d"], dtype=float).reshape(6),
        theta_offset=np.asarray(mdh["theta_offset"], dtype=float).reshape(6),
    )


class MultiSourceRobotModel:
    """Predict end-effector target positions from nominal plus error vector.

    The implementation follows the paper equations:
    kinematic MD-H errors, base/tool frame errors, reduction ratio error
    ``q * delta_rrd``, backlash ``direction * delta_backlash``, and load
    flexibility ``tau * delta_flex`` where ``tau = Jv.T @ F``.
    """

    def __init__(self, nominal: RobotParameters | None = None) -> None:
        self.nominal = load_nominal_robot() if nominal is None else nominal.copy()

    def position(
        self,
        joint_angles: np.ndarray,
        error_vector: np.ndarray,
        parameters: list[ErrorParameter],
        payload: float = 0.0,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one target xyz position."""
        return position_from_transform(
            self.transform(joint_angles, error_vector, parameters, payload, direction)
        )

    def batch_positions(
        self,
        joint_configs: np.ndarray,
        error_vector: np.ndarray,
        parameters: list[ErrorParameter],
        payloads: np.ndarray | float | None = None,
        directions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return N x 3 positions for N joint configurations."""
        configs = np.asarray(joint_configs, dtype=float)
        if configs.ndim != 2 or configs.shape[1] != 6:
            raise ValueError("joint_configs must be an N x 6 array.")
        payload_array = _normalize_payloads(payloads, len(configs))
        direction_array = _normalize_directions(directions, configs)
        return np.stack(
            [
                self.position(
                    configs[i], error_vector, parameters, payload_array[i], direction_array[i]
                )
                for i in range(len(configs))
            ],
            axis=0,
        )

    def transform(
        self,
        joint_angles: np.ndarray,
        error_vector: np.ndarray,
        parameters: list[ErrorParameter],
        payload: float = 0.0,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute full base-to-tool transform under the multisource errors."""
        comp = vector_to_components(error_vector, parameters)
        q = np.asarray(joint_angles, dtype=float).reshape(6)
        h = _infer_direction(q) if direction is None else np.asarray(direction, dtype=float).reshape(6)
        tau = self.joint_load_torque(q, comp, payload)

        q_eff = (
            q
            + self.nominal.theta_offset
            + comp["delta_theta"]
            + q * comp["rrd"]
            + h * comp["backlash"]
            + tau * comp["flex"]
        )

        transform = make_transform(
            self.nominal.base_xyz + comp["base_xyz"],
            self.nominal.base_rpy + comp["base_rpy"],
        )
        alpha = self.nominal.alpha + comp["delta_alpha"]
        a = self.nominal.a + comp["delta_a"]
        d = self.nominal.d + comp["delta_d"]
        for joint_index in range(6):
            transform = transform @ modified_dh_transform(
                alpha[joint_index], a[joint_index], q_eff[joint_index], d[joint_index]
            )
        return transform @ make_transform(
            self.nominal.tool_xyz + comp["tool_xyz"],
            self.nominal.tool_rpy + comp["tool_rpy"],
        )

    def joint_load_torque(
        self,
        joint_angles: np.ndarray,
        components: dict[str, np.ndarray],
        payload: float,
    ) -> np.ndarray:
        """Compute the virtual-work load torque ``tau = Jv.T @ [0,0,-mg]``."""
        mass = float(payload)
        if abs(mass) <= 1.0e-12:
            return np.zeros(6, dtype=float)
        q = np.asarray(joint_angles, dtype=float).reshape(6)
        base_position = self._rigid_position(q, components)
        jacobian = np.zeros((3, 6), dtype=float)
        eps = 1.0e-6
        for joint_index in range(6):
            q_step = q.copy()
            q_step[joint_index] += eps
            jacobian[:, joint_index] = (
                self._rigid_position(q_step, components) - base_position
            ) / eps
        return jacobian.T @ np.array([0.0, 0.0, -9.80665 * mass], dtype=float)

    def _rigid_position(self, joint_angles: np.ndarray, comp: dict[str, np.ndarray]) -> np.ndarray:
        """Rigid FK position used only for load-Jacobian finite differences."""
        transform = make_transform(
            self.nominal.base_xyz + comp["base_xyz"],
            self.nominal.base_rpy + comp["base_rpy"],
        )
        q = np.asarray(joint_angles, dtype=float).reshape(6)
        alpha = self.nominal.alpha + comp["delta_alpha"]
        a = self.nominal.a + comp["delta_a"]
        d = self.nominal.d + comp["delta_d"]
        q_eff = q + self.nominal.theta_offset + comp["delta_theta"] + q * comp["rrd"]
        for joint_index in range(6):
            transform = transform @ modified_dh_transform(
                alpha[joint_index], a[joint_index], q_eff[joint_index], d[joint_index]
            )
        return position_from_transform(
            transform
            @ make_transform(
                self.nominal.tool_xyz + comp["tool_xyz"],
                self.nominal.tool_rpy + comp["tool_rpy"],
            )
        )


def _normalize_payloads(payloads: np.ndarray | float | None, count: int) -> np.ndarray:
    if payloads is None:
        return np.zeros(count, dtype=float)
    if np.isscalar(payloads):
        return np.full(count, float(payloads), dtype=float)
    values = np.asarray(payloads, dtype=float).reshape(-1)
    if values.size != count:
        raise ValueError(f"Expected {count} payload values, got {values.size}.")
    return values


def _normalize_directions(directions: np.ndarray | None, configs: np.ndarray) -> np.ndarray:
    if directions is None:
        return np.stack([_infer_direction(q) for q in configs], axis=0)
    values = np.asarray(directions, dtype=float)
    if values.shape != configs.shape:
        raise ValueError(f"directions must have shape {configs.shape}, got {values.shape}.")
    values = np.sign(values)
    values[values == 0.0] = 1.0
    return values


def _infer_direction(q: np.ndarray) -> np.ndarray:
    direction = np.sign(np.asarray(q, dtype=float).reshape(6))
    direction[direction == 0.0] = 1.0
    return direction
