from enum import Enum

import numpy as np

from .quaternion import Quaternion, DualQuaternion


class Dimension(Enum):
    two = 2
    three = 3


class DH:
    """
    d: distance from the previous z axis to the next z axis
    theta: angle from the previous z axis to the next z axis
    a: distance from the previous x axis to the next x axis
    alpha: angle from the previous x axis to the next x axis
    follow wiki:
    https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    """

    def __init__(
            self,
            d: float,
            theta: float,
            a: float,
            alpha: float,
            calc_type: Dimension = Dimension.three,
    ) -> None:
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.dim = calc_type
        if self.dim == Dimension.two:
            self.d = 0
            self.alpha = 0

    @property
    def T(self) -> np.ndarray:
        return np.array(
            [
                [
                    np.cos(self.theta),
                    -np.sin(self.theta) * np.cos(self.alpha),
                    np.sin(self.theta) * np.sin(self.alpha),
                    self.a * np.cos(self.theta),
                ],
                [
                    np.sin(self.theta),
                    np.cos(self.theta) * np.cos(self.alpha),
                    -np.cos(self.theta) * np.sin(self.alpha),
                    self.a * np.sin(self.theta),
                ],
                [0, np.sin(self.alpha), np.cos(self.alpha), self.d],
                [0, 0, 0, 1],
            ]
        )

    def update_d(self, d) -> None:
        self.d = d

    def update_theta(self, theta) -> None:
        self.theta = theta

    def update_a(self, a) -> None:
        self.a = a

    def update_alpha(self, alpha) -> None:
        self.alpha = alpha

    def update(self, d, theta, a, alpha) -> None:
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha


class DualQuaternionReferenceFrame:
    """
    define a Reference Frame Y respect to frame X (parent) using dual quaternion
    Member:
        r: displacement, rYX_Y
        w: angular velocity
        v: linear velocity

        q: quaternion, a rotation form X to Y
        dq: dual quaternion, X displaces and rotates to Y
        wdq: dual quaternion of w, dual quaternion of wYX_Y
        rdq: dual quaternion of r, dual quaternion of rYX_Y
    """

    def __init__(
            self,
            displacement: np.ndarray,
            linear_velocity: np.ndarray,
            angular_velocity: np.ndarray,
            linear_acceleration: np.ndarray,
            angular_acceleration: np.ndarray,
            rotation: Quaternion,
    ) -> None:
        assert displacement.shape == (3,), "Displacement must be a 3D vector."
        assert linear_velocity.shape == (3,), "Linear velocity must be a 3D vector."
        assert angular_velocity.shape == (3,), "Angular velocity must be a 3D vector."
        self.r: Quaternion = self.add_as_pure_quaternion(displacement)
        self.v: Quaternion = self.add_as_pure_quaternion(linear_velocity)
        self.w: Quaternion = self.add_as_pure_quaternion(angular_velocity)
        self.a: Quaternion = self.add_as_pure_quaternion(linear_acceleration)
        self.alpha: Quaternion = self.add_as_pure_quaternion(angular_acceleration)
        self.q: Quaternion = rotation

        self.dq: DualQuaternion = DualQuaternion.from_quaternion_vector(
            self.q, self.r.vec
        )
        self.rdq: DualQuaternion = DualQuaternion.as_pure_real(self.r.vec)
        wdq_d = self.v + self.r.cross(self.w)
        self.wdq: DualQuaternion = DualQuaternion.from_real_dual(self.w, wdq_d)

    @staticmethod
    def add_as_pure_quaternion(vec: np.ndarray) -> Quaternion:
        assert vec.shape == (3,), "Vector must be a 3D vector."
        return Quaternion(np.hstack([0, vec]))

    def swap(self) -> "DualQuaternionReferenceFrame":
        new_r = self.q.rotate(self.r)
        new_v = self.q.rotate(self.v)
        new_w = self.q.rotate(self.w)
        new_a = self.q.rotate(self.a)
        new_alpha = self.q.rotate(self.alpha)

        new_wdq = self.dq.displace(self.wdq)
        new_rdq = self.dq.displace(self.rdq)
        new_dq = self.dq.conjugate()
        new_q = self.q.conjugate()

        return DualQuaternionReferenceFrame(
            new_r, new_v, new_w, new_a, new_alpha, new_q
        )

    def update(
            self,
            displacement: np.ndarray,
            linear_velocity: np.ndarray,
            angular_velocity: np.ndarray,
            linear_acceleration: np.ndarray,
            angular_acceleration: np.ndarray,
            rotation: Quaternion,
    ) -> None:
        assert displacement.shape == (3,), "Displacement must be a 3D vector."
        assert linear_velocity.shape == (3,), "Linear velocity must be a 3D vector."
        assert angular_velocity.shape == (3,), "Angular velocity must be a 3D vector."
        self.r: Quaternion = self.add_as_pure_quaternion(displacement)
        self.v: Quaternion = self.add_as_pure_quaternion(linear_velocity)
        self.w: Quaternion = self.add_as_pure_quaternion(angular_velocity)
        self.a: Quaternion = self.add_as_pure_quaternion(linear_acceleration)
        self.alpha: Quaternion = self.add_as_pure_quaternion(angular_acceleration)
        self.q: Quaternion = rotation

        self.dq: DualQuaternion = DualQuaternion.from_quaternion_vector(
            self.q, self.r.vec
        )
        self.rdq: DualQuaternion = DualQuaternion.as_pure_real(self.r.vec)
        wdq_d = self.v + self.r.cross(self.w)
        self.wdq: DualQuaternion = DualQuaternion.from_real_dual(self.w, wdq_d)
