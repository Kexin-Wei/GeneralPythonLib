import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from libs.utility.define_class import JointType, Dimension

LINK2D_PAIR = Tuple["Link2D", "Link2D"]


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
        self.m = 0  # matrix
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.dim = calcType
        if self.dim == Dimension.two:
            self.d = 0
            self.alpha = 0
        self.calculate_matrix()

    def update_d(self, d):
        self.d = d
        self.calculate_matrix()

    def update_theta(self, theta):
        self.theta = theta
        self.calculate_matrix()

    def update_a(self, a):
        self.a = a
        self.calculate_matrix()

    def update_alpha(self, alpha):
        self.alpha = alpha
        self.calculate_matrix()

    def update(self, d, theta, a, alpha):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.calculate_matrix()

    def calculate_matrix(self):
        self.m = np.array(
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


class Link2D(DH):
    """
    x: x position of the start point of the link
    y: y position of the start point of the link
    theta: angle of the link relative to the x axis
    l: length of the link
    label: label of the link of the start point, will display in the plot
    rad: if theta is in radian, set to True, otherwise False

    xn: x position of the end point of the link
    yn: y position of the end point of the link

    functions:
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float = 0,
        l: float = 1,
        label: str = "",
        rad: bool = False,
    ):
        super().__init__(0, theta, l, 0, Dimension.two)
        self.x = x
        self.y = y
        self.label = label
        if rad:
            self.new_theta(theta)
        else:
            self.new_theta(np.deg2rad(theta))

    @property
    def l(self):
        return self.a

    @property
    def theta(self):
        return self.theta

    @property
    def xn(self):
        return self.x + self.l * np.cos(self.theta)

    @property
    def yn(self):
        return self.y + self.l * np.sin(self.theta)

    def new_l(self, l: float):
        self.update_a(l)

    def new_theta(self, theta: float):
        self.update_theta(theta)

    def new_label(self, label: str):
        self.label = label

    def _plot_start_point(self, ax, color="black", markersize=5):
        ax.plot(self.x, self.y, "o", color=color, markersize=markersize)
        if self.label != "":
            ax.annotate(
                self.label,
                (self.x, self.y),
                color="black",
                fontsize=8,
                weight="heavy",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def _plot_end_point(self, ax, color="black"):
        ax.plot(self.end_x, self.end_y, "o", c=color, markersize=5)

    def _plot_link(self, ax, color="black"):
        ax.plot([self.x, self.end_x], [self.y, self.end_y], color=color)

    def plot(self, ax):
        self._plot_start_point(ax)
        self._plot_end_point(ax)
        self._plot_link(ax)


class Joint2D(Link2D):
    """all joints must have a link attached to it"""

    def __init__(
        self,
        j_type: JointType,
        x: float,
        y: float,
        theta: float = 0,
        l: float = 1,
        label: str = "",
        rad: bool = False,
    ):
        super().__init__(x, y, theta, l, label, rad)
        self.j_type = j_type

    def plot(self, ax, color="red"):
        self._plot_start_point(ax, color=color, markersize=10)
        self._plot_end_point(ax)
        self._plot_link(ax)

    def new_joint_value(self, j_value: float):
        if self.j_type == JointType.revolute:
            self.new_theta(j_value)
        elif self.j_type == JointType.prismatic:
            self.new_l(j_value)

    def new_pos(self, x, y):
        self.x = x
        self.y = y


class Robot2D:
    def __init__(self, joints: list[Joint2D], base: Tuple[float, float] = (0, 0)):
        self.joints = joints
        self._check_joints()

    def _check_joints(self):
        for i, joint in enumerate(self.joints):
            if i > 0:
                prev_joint = self.joints[i - 1]
                error_msg = (
                    f"The joints must be connected to each other. "
                    f"{joint.link.label} started at ({joint.link.x},{joint.link.y}), "
                    f"but the previous joint ended at ({prev_joint.link.end_x},{prev_joint.link.end_y})."
                )
                loc_error = (prev_joint.link.end_x - joint.link.x) ** 2 + (
                    prev_joint.link.end_y - joint.link.y
                ) ** 2
                assert loc_error < 1e-6, f"{error_msg}"

    @property
    def colors(self):
        n_joints = len(self.joints)
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=n_joints)
        colors = cmap(norm(range(n_joints)))
        return colors

    def _set_joint_pose(self, i_joint: int, pose: float):
        self.joints[i_joint].new_joint_value(pose)

    def plot(self, ax):
        for joint, c in zip(self.joints, self.colors):
            joint.plot(ax, color=c)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    link1 = Link2D(0, 0, 45, 1)
    link1.plot(ax)

    link2 = Link2D(1, 1, 90, 1)
    joint1 = Joint2D("revolute", link2, label="joint1")
    joint1.plot(ax)

    ax.set_aspect("equal")
    ax.grid()

    link3 = Link2D(0, 1, 90, 1)
    link4 = Link2D(0, 2, 45, 1)
    joint3 = Joint2D("revolute", link3, label="joint3")
    joint4 = Joint2D("revolute", link4, label="joint4")
    robot = Robot2D([joint3, joint4])
    robot.plot(ax)
    plt.show()
