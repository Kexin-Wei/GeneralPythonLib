import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

from typing import Tuple
from .reference_frame import DH, Dimension

LINK2D_PAIR = Tuple["Link2D", "Link2D"]


class JointType(Enum):
    revolute = "revolute"
    prismatic = "prismatic"


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
        name: str = "",
        rad: bool = False,
    ) -> None:
        super().__init__(0, theta, l, 0, Dimension.two)
        self.x = x
        self.y = y
        self.name = name
        if rad:
            self.new_theta(theta)
        else:
            self.new_theta(np.deg2rad(theta))

    @property
    def l(self) -> float:
        return self.a

    @property
    def xn(self) -> float:
        return self.x + self.l * np.cos(self.theta)

    @property
    def yn(self) -> float:
        return self.y + self.l * np.sin(self.theta)

    def new_l(self, l: float) -> None:
        self.update_a(l)

    def new_theta(self, theta: float) -> None:
        self.update_theta(theta)

    def new_label(self, label: str) -> None:
        self.name = label

    def _plot_start_point(self, ax: plt.Axes, color="black", markersize=5) -> plt.Axes:
        ax.plot(self.x, self.y, "o", color=color, markersize=markersize)
        if self.name != "":
            ax.annotate(
                self.name,
                (self.x, self.y),
                color="black",
                fontsize=8,
                weight="heavy",
                horizontalalignment="center",
                verticalalignment="center",
            )
        return ax

    def _plot_end_point(self, ax: plt.Axes, color="black") -> plt.Axes:
        ax.plot(self.xn, self.yn, "o", c=color, markersize=5)
        return ax

    def _plot_link(self, ax: plt.Axes, color="black") -> plt.Axes:
        ax.plot([self.x, self.xn], [self.y, self.yn], color=color)
        return ax

    def plot(self, ax: plt.Axes) -> plt.Axes:
        ax = self._plot_start_point(ax)
        ax = self._plot_end_point(ax)
        ax = self._plot_link(ax)
        return ax


class Joint2D(Link2D):
    """all joints must have a link attached to it"""

    def __init__(
        self,
        j_type: JointType,
        x: float,
        y: float,
        theta: float = 0,
        l: float = 1,
        name: str = "",
        j_range: tuple = (0, np.pi),
        rad: bool = False,
    ) -> None:
        super().__init__(x, y, theta, l, name, rad)
        self.j_type = j_type
        self.j_range = j_range

    def append_joint(
        self,
        j_type: JointType,
        theta: float,
        l: float,
        name: str = "",
        j_range: tuple = (0, np.pi),
    ) -> "Joint2D":
        return Joint2D(j_type, self.xn, self.yn, theta, l, name, j_range)

    def plot(self, ax: plt.Axes, color="red") -> plt.Axes:
        ax = self._plot_start_point(ax, color=color, markersize=25)
        ax = self._plot_end_point(ax)
        ax = self._plot_link(ax)
        return ax

    def new_joint_value(self, j_value: float) -> None:
        if self.j_type == JointType.revolute:
            self.new_theta(j_value)
        elif self.j_type == JointType.prismatic:
            self.new_l(j_value)

    def new_joint_range(self, j_range: tuple) -> None:
        self.j_range = j_range

    def new_pos(self, x, y) -> None:
        self.x = x
        self.y = y


# TODO: make LINK2D generalize to LINK3D
