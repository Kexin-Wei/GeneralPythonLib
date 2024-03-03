import numpy as np
from typing import Union


class Quaternion:
    def __init__(self, v: np.ndarray) -> None:
        assert v.shape == (4,), "Quaternion must be a 4D vector."
        self.v = v

    @property
    def vec(self) -> np.ndarray:
        return self.v[1:]

    def __repr__(self) -> str:
        return f"Quaternion({self.v})"

    def __str__(self) -> str:
        return f"{self.v}"

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.v + other.v)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.v - other.v)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        if isinstance(other, Quaternion):
            q10, q20 = self.v[0], other.v[0]
            q1vec, q2vec = self.v[1:], other.v[1:]
            r0 = q10 * q20 - np.dot(q1vec, q2vec)
            rvec = q10 * q2vec + q20 * q1vec + np.cross(q1vec, q2vec)
            return Quaternion(np.hstack([r0, rvec]))
        elif isinstance(other, float):
            return Quaternion(self.v * other)
        else:
            assert False, (
                f"Quaternion can only be multiplied by another quaternion or a scalar. "
                f"Got {type(other)} instead."
            )

    def __truediv__(self, other: float) -> "Quaternion":
        assert other != 0, "Quaternion cannot be divided by 0."
        return Quaternion(self.v / other)

    def conjugate(self) -> "Quaternion":
        return Quaternion(np.hstack([self.v[0], -self.v[1:]]))

    def inverse(self) -> "Quaternion":
        return self.conjugate() / self.norm() ** 2

    def norm(self) -> float:
        norm = self * self.conjugate()
        return np.sqrt(norm.v[0])

    def unit(self) -> "Quaternion":
        return self / self.norm()

    def cross(self, other: "Quaternion") -> "Quaternion":
        q10, q20 = self.v[0], other.v[0]
        q1vec, q2vec = self.v[1:], other.v[1:]
        r_vec = q10 * q2vec + q20 * q1vec + np.cross(q1vec, q2vec)
        return Quaternion(np.hstack([0, r_vec]))

    def as_pure(self) -> "Quaternion":
        return Quaternion(np.hstack([0, self.v[1:]]))

    def rotate(self, rBA_X: "Quaternion") -> "Quaternion":
        """Rotate a quaternion by another quaternion
        rBA_Y = qYX.conj * rBA_X * qYX

        Args:
            rBA_X (Quaternion): waited to be rotated
        Returns:
            Quaternion: rBA_Y
        """
        return self.conjugate() * rBA_X * self

    @staticmethod
    def from_euler(theta: float, n: np.ndarray) -> "Quaternion":
        assert n.shape == (3,), "Axis must be a 3D vector."
        n = n / np.linalg.norm(n)
        return Quaternion(np.hstack([np.cos(theta / 2), np.sin(theta / 2) * n]))

    @staticmethod
    def as_pure(v: np.ndarray) -> "Quaternion":
        assert v.shape == (3,), "Vector must be a 3D vector."
        return Quaternion(np.hstack([0, v]))


class DualNumber:
    def __init__(self, v: np.ndarray) -> None:
        assert v.shape == (2,), "Dual number must be a 2D vector."
        self.v = v

    @property
    def real(self) -> float:
        return self.v[0]

    @property
    def dual(self) -> float:
        return self.v[1]

    def __repr__(self) -> str:
        return f"DualNumber({self.real}, {self.dual})"

    def __str__(self) -> str:
        return f"[{self.real}, {self.dual}]"

    def __add__(self, other: "DualNumber") -> "DualNumber":
        return DualNumber(self.v + other.v)

    def __sub__(self, other: "DualNumber") -> "DualNumber":
        return DualNumber(self.v - other.v)

    def __mul__(self, other: Union["DualNumber", float, int]) -> "DualNumber":
        if isinstance(other, DualNumber):
            r = self.real * other.real
            d = self.real * other.dual + self.dual * other.real
            return DualNumber(np.hstack([r, d]))
        elif isinstance(other, float) or isinstance(other, int):
            return DualNumber(self.v * other)
        else:
            assert False, (
                f"DualNumber can only be multiplied by another dual number or a scalar. "
                f"Got {type(other)} instead."
            )

    def inverse(self) -> "DualNumber":
        if self.real == 0:
            assert (
                False
            ), "Real part of dual number cannot be 0 when calculating its inverse."
        return DualNumber(np.hstack([1 / self.real, -self.dual / self.real**2]))

    def __pow__(self, p: int) -> "DualNumber":
        return DualNumber(
            np.hstack([self.real**p, p * self.real ** (p - 1) * self.dual])
        )

    def conjugate(self) -> "DualNumber":
        return DualNumber(np.hstack([self.real, -self.dual]))


class DualQuaternion:
    def __init__(self, v: np.ndarray) -> None:
        assert v.shape == (8,), "Dual quaternion must be a 8D vector."
        self.v = v

    @property
    def real(self) -> "Quaternion":
        return Quaternion(self.v[:4])

    @property
    def dual(self) -> "Quaternion":
        return Quaternion(self.v[4:])

    def __repr__(self) -> str:
        return f"DualQuaternion({self.real.v}, {self.dual.v})"

    def __str__(self) -> str:
        return f"{self.real.v}, {self.dual.v}"

    def __add__(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.v + other.v)

    def __sub__(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.v - other.v)

    def __truediv__(self, other: float) -> "DualQuaternion":
        assert other != 0, "DualQuaternion cannot be divided by 0."
        return DualQuaternion(self.v / other)

    def __mul__(self, other: "DualQuaternion") -> "DualQuaternion":
        if isinstance(other, DualQuaternion):
            r = self.real * other.real
            d = self.real * other.dual + self.dual * other.real
            return DualQuaternion(np.hstack([r.v, d.v]))
        elif isinstance(other, float) or isinstance(other, int):
            return DualQuaternion(self.v * other)
        else:
            assert False, (
                f"DualQuaternion can only be multiplied by another dual quaternion or a scalar. "
                f"Got {type(other)} instead."
            )

    def conjugate(self) -> "DualQuaternion":
        return DualQuaternion(
            np.hstack([self.real.conjugate().v, self.dual.conjugate().v])
        )

    def swap(self) -> "DualQuaternion":
        return DualQuaternion(np.hstack([self.dual.v, self.real.v]))

    def cross(self, other: "DualQuaternion") -> "DualQuaternion":
        r = self.real.cross(other.real)
        d = self.real.cross(other.dual) + self.dual.cross(other.real)
        return DualQuaternion(np.hstack([r.v, d.v]))

    def pure(self) -> "DualQuaternion":
        return DualQuaternion(
            np.hstack([self.real.as_pure().v, self.dual.as_pure().v])
        )

    def norm(self) -> DualNumber:
        norm = self * self.conjugate()
        return DualNumber(np.hstack([norm.real.v[0], norm.dual.v[0]]))

    def unit(self) -> "DualQuaternion":
        return self / self.norm()

    def inverse(self) -> "DualQuaternion":
        r = self.real.inverse()
        d = -r * self.dual * r
        return DualQuaternion(np.hstack([r.v, d.v]))

    @staticmethod
    def from_real_dual(real: Quaternion, dual: Quaternion) -> "DualQuaternion":
        return DualQuaternion(np.hstack([real.v, dual.v]))

    @staticmethod
    def as_pure_real(v: np.ndarray) -> "DualQuaternion":
        assert v.shape == (3,), "Vector must be a 3D vector."
        return DualQuaternion(np.hstack([0, v, np.zeros((4,))]))

    @staticmethod
    def from_quaternion_vector(
        qYX: Quaternion, tYX_X: np.ndarray
    ) -> "DualQuaternion":
        """build a dual quaternion from quaternion and translation vector
        Args:
            qYX (Quaternion): quaternion, or translation, ep. from reference frame D to B = qBD
            tYX_X (np.ndarray): translation vector, seen in the frame D = tBD_D
        """
        assert tYX_X.shape == (3,), "Vector must be a 3D vector."
        t = Quaternion(np.hstack([0, tYX_X]))
        d = t * qYX * 0.5
        return DualQuaternion(np.hstack([qYX.v, d.v]))

    @staticmethod
    def from_real_quaternion(q: Quaternion) -> "DualQuaternion":
        """build up a dual quaternion has zero dual part
        Args:
            q (Quaternion): rTI_I the quaternion of vector rTI in I frame
        """
        return DualQuaternion(np.hstack([q.v, np.zeros((4,))]))

    @staticmethod
    def from_euler_vector(
        theta: float, n: np.ndarray, v: np.ndarray
    ) -> "DualQuaternion":
        """build a dual quaternion from euler angle, rotate axism and vector
        Args:
            theta (float): rotate angle
            n (np.ndarray): the rotate axis unit vector
            v (np.ndarray): translation vector
        """
        q = Quaternion.from_euler(theta, n)
        return DualQuaternion.from_quaternion_vector(q, v)

    def displace(self, rYX_X: "DualQuaternion") -> "DualQuaternion":
        """displace a dual quaternion rYX_X by dqYX to get rYX_Y
        Args:
            rYX_X (DualQuaternion): dual quaternion of rYX_X

        Returns:
            DualQuaternion: rYX_Y
        """
        return self.conjugate() * rYX_X * self


if __name__ == "__main__":
    a = Quaternion(np.array([1, 2, 3, 4]))
    b = Quaternion(np.array([2, 1, 3, 3]))
    print(a + b)
    print(a - b)

    c = a * b
    print(c)

    print(a.conjugate())

    c = a.cross(b)
    print(c)

    print(a.norm() ** 2)
    print(a.unit())
    print(a.unit().conjugate() * a.unit())
    print(a.unit().norm())

    a = DualQuaternion(np.array([1, 2, 3, 4, 4, 3, 2, 1]))
    b = DualQuaternion(np.array([2, 1, 3, 3, 4, 1, 3, 1]))
    print(a + b)
    print(a * 10)
    print(a * b)
    print(a.conjugate())
    print(a.swap())

    print(a.cross(b))
    print(a.pure())
    print(a.norm())

    print(
        DualQuaternion.from_euler_vector(
            0, np.array([1, 0, 0]), np.array([1, 0, 0])
        )
    )
