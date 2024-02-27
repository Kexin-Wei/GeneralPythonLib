import warnings
import numpy as np
from ..utility.define_class import Dimension
from .quaternion import Quaternion, DualQuaternion


class DH:
    """
    d: distance from the previous z axis to the next z axis
    theta: angle from the previous z axis to the next z axis
    a: distance from the previous x axis to the next x axis
    alpha: angle from the previous x axis to the next x axis
    follow wiki: https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
    """

    def __init__(
        self,
        d: float,
        theta: float,
        a: float,
        alpha: float,
        calcType: Dimension = Dimension.three,
    ):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.dim = calcType
        if self.dim == Dimension.two:
            self.d = 0
            self.alpha = 0

    @property
    def T(self):
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

    def update_d(self, d):
        self.d = d

    def update_theta(self, theta):
        self.theta = theta

    def update_a(self, a):
        self.a = a

    def update_alpha(self, alpha):
        self.alpha = alpha

    def update(self, d, theta, a, alpha):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha


class Node:
    def __init__(
        self, node_name: str, parent_name: str, child_name: str = None
    ) -> None:
        """_
        Args:
            node_name (str): unique name in the chain, should be set as joint name
            parent_id (int):  parent node id
            child_id (int, optional): _description_. Defaults to None.
        """
        self.parent: set[str] = None
        self.child: set[str] = None
        self.node_name: str = node_name
        self.add_parent(parent_name)
        self.add_child(child_name)

    def add_parent(self, parent_name: str) -> None:
        assert (
            parent_name is not None
        ), f"Parent name cannot be None, node {self.node_name} add parent failed."
        if self.parent is None:
            self.parent = set([])
            self.parent.add(parent_name)
            return
        if parent_name in self.parent:
            warnings.warn(
                f"Parent name {parent_name} already exist in node {self.node_name}, ignored."
            )
            return
        self.parent.add(parent_name)

    def add_child(self, child_name: str) -> None:
        if child_name is None:
            return
        if self.child is None:
            self.child = set([])
            self.child.add(child_name)
            return
        if child_name in self.child:
            warnings.warn(
                f"Child name {child_name} already exist in node {self.node_name}, ignored."
            )
            return
        self.child.add(child_name)

    def remove_parent(self, parent_name: str):
        assert parent_name is not None, "Parent name cannot be None."
        if parent_name not in self.parent:
            warnings.warn(
                f"Parent name {parent_name} does not exist in node {self.node_name}, "
                f"remove parent failed."
            )
            return
        self.parent.remove(parent_name)

    def remove_child(self, child_name: str):
        if child_name is None:
            warnings.warn(
                f"Child name is None, node {self.node_name} remove child failed."
            )
            return
        if child_name not in self.child:
            warnings.warn(
                f"Child name {child_name} does not exist in node {self.node_name}, ignored."
            )
            return
        self.child.remove(child_name)


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

    def add_as_pure_quaternion(self, vec: np.ndarray) -> Quaternion:
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
    ):
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
