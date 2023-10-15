import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from .reference_frame import DH
from ..utility.define_class import Dimension, JointType

LINK2D_PAIR = Tuple["Link2D", "Link2D"]


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
    ):
        super().__init__(0, theta, l, 0, Dimension.two)
        self.x = x
        self.y = y
        self.name = name
        if rad:
            self.new_theta(theta)
        else:
            self.new_theta(np.deg2rad(theta))

    @property
    def l(self):
        return self.a

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
        self.name = label

    def _plot_start_point(self, ax: plt.Axes, color="black", markersize=5):
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

    def _plot_end_point(self, ax: plt.Axes, color="black"):
        ax.plot(self.xn, self.yn, "o", c=color, markersize=5)

    def _plot_link(self, ax: plt.Axes, color="black"):
        ax.plot([self.x, self.xn], [self.y, self.yn], color=color)

    def plot(self, ax: plt.Axes):
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
        name: str = "",
        rad: bool = False,
    ):
        super().__init__(x, y, theta, l, name, rad)
        self.j_type = j_type

    def plot(self, ax: plt.Axes, color="red"):
        self._plot_start_point(ax, color=color, markersize=25)
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


# TODO: make LINK2D generalize to LINK3D
