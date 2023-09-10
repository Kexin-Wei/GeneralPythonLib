# numerical functions for robot kinematics
# using dh table
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Union
from dataclasses import dataclass
from ..utility.define_class import JointType, Dimension

LINK2D_PAIR = Tuple["Link2D", "Link2D"]


@dataclass
class Point:
    x: float
    y: float
    z: float = 0


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
        self.m = None  # matrix
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


# TODO: make LINK2D generalize to LINK3D
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
            self.parent = set().add(parent_name)
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
            self.child = set().add(child_name)
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


class KinematicChain:
    """define the relationship between joints
    by using DotInChain's parent and child to find the next joint
    and id mapped to join and link
    in forward kinematics or inverse kinematics
    """

    def __init__(self) -> None:
        self.nodes = {}
        self.base_name = "BASE"

    @property
    def node_names(self):
        return list(self.nodes.keys())

    def add_node(
        self, node_name: str, parent_name: str, child_name: str = None
    ) -> None:
        if len(self.nodes) == 0:
            if parent_name != self.base_name:
                warnings.warn(
                    f"Node name {node_name} is not base name, add node to base instead."
                )
            parent_name = self.base_name
        if node_name in self.node_names:
            warnings.warn(
                f"Node name {node_name} already exist, add node failed, edit existing node instead."
            )
            self.edit_node_connection(node_name, parent_name, child_name)
            return
        node = Node(node_name, parent_name, child_name)
        self.nodes[node_name] = node

    def _check_parent_name_exist(self, node_name: str) -> bool:
        assert node_name is not None, "Parent name cannot be None."
        return self._check_node_name_exist(node_name) or node_name == self.base_name

    def _check_node_name_exist(self, node_name: str) -> bool:
        assert node_name is not None, "Node name cannot be None."
        return node_name in self.node_names

    def get_node_by_name(self, node_name: str) -> Node:
        if self._check_node_name_exist(node_name):
            return self.nodes[node_name]
        warnings.warn(f"Node name does not exist, cannot get node by name {node_name}.")
        return None

    def add_parent_to_node(self, node_name: str, parent_name: str) -> None:
        assert self._check_node_name_exist(
            node_name
        ), f"Node name {node_name} does not exist."
        assert self._check_parent_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, "
            f"add parent to node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.add_parent(parent_name)

    def add_child_to_node(self, node_name: str, child_name: str) -> None:
        assert self._check_node_name_exist(
            node_name
        ), f"Node name {node_name} does not exist."
        assert self._check_node_name_exist(child_name), (
            f"Child name {child_name} does not exist, "
            f"add child to node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.add_child(child_name)

    def remove_parent_from_node(self, node_name: str, parent_name: str) -> None:
        assert self._check_node_name_exist(
            node_name
        ), f"Node name {node_name} does not exist."
        assert parent_name is not None, "Parent name cannot be None."
        assert self._check_node_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, "
            f"remove parent from node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.remove_parent(parent_name)

    def remove_child_from_node(self, node_name: str, child_name: str) -> None:
        assert self._check_node_name_exist(
            node_name
        ), f"Node name {node_name} does not exist."
        assert child_name is not None, "When deleting, child name cannot be None."
        assert self._check_node_name_exist(child_name), (
            f"Child name {child_name} does not exist, "
            f"remove child from node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.remove_child(child_name)

    def edit_node_connection(
        self, node_name: str, parent_name: str, child_name: str = None
    ) -> None:
        assert self._check_node_name_exist(
            node_name
        ), f"Node name {node_name} does not exist."
        assert self._check_parent_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, "
            f"edit node {node_name} failed."
        )
        assert child_name is None or self._check_node_name_exist(child_name), (
            f"Child name {child_name} does not exist, " f"edit node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.add_parent(parent_name)
        node.add_child(child_name)

    def get_node_child_to_end(self, node_name: str):
        if not self._check_node_name_exist(node_name):
            warnings.warn(f"Node name {node_name} does not exist.")
            return None

        node: Node = self.nodes[node_name]

        if node.child is None:
            return [node_name]
        # TODO: how to handle nest list
        if len(node.child) == 1:
            node_link = [node_name]
            child_link = self.get_node_child_to_end(list(node.child)[0])
            if isinstance(child_link[0], list):
                for l in child_link:
                    l.insert(0, node_name)
                return child_link
            assert isinstance(child_link[0], str)
            node_link.extend(child_link)
            return node_link

        if len(node.child) > 1:
            node_links = []
            for ch in node.child:
                node_links.append(self.get_node_child_to_end(ch))
            return node_links


class Robot2D(KinematicChain):
    def __init__(self) -> None:
        super().__init__()
        self.joint_name_map = {}

    @property
    def base(self):
        return self.base_name

    @property
    def joints(self):
        return list(self.joint_name_map.values())

    @property
    def colors(self):
        n_joints = len(self.joints)
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=n_joints)
        colors = cmap(norm(range(n_joints)))
        return colors

    def _check_joint_connection(self, joint_name: str, parent_name: str):
        if parent_name == self.base_name:
            return
        if parent_name not in self.joint_name_map.keys():
            assert False, f"Parent name {parent_name} does not exist."
        j = self.joint_name_map[joint_name]
        parent_j = self.joint_name_map[parent_name]
        loc_error = (parent_j.xn - j.x) ** 2 + (parent_j.yn - j.y) ** 2
        assert loc_error < 1e-6, (
            f"The joints must be connected to each other. "
            f"{j.name} started at ({j.x},{j.y}), "
            f"but the previous joint ended at ({parent_j.xn},{parent_j.yn})."
        )

    def add_joint(self, joint: Joint2D, parent) -> None:
        if isinstance(parent, str):
            parent_name = parent
        elif isinstance(parent, Joint2D):
            parent_name = parent.name
        else:
            assert False, f"Parent type {type(parent)} is not supported."
        self.add_node(joint.name, parent_name)
        self.joint_name_map[joint.name] = joint
        self._check_joint_connection(joint.name, parent_name)

    def add_joints(self, joints: list[Joint2D], joint_relation: dict = None):
        parent_name = self.base_name
        for i, j in enumerate(joints):
            if j.name in self.joint_name_map.keys():
                assert False, f"Joint name {j.name} already exist, add joint failed."
            if joint_relation is not None:
                parent_name = joint_relation[j.name]
            elif i > 0:
                parent_name = joints[i - 1].name
            self.add_joint(j, parent_name)

    def add_parallel_joint(
        self,
        joint_name: str,
        parent_name: str,
        add_joint_loc: Union[Point, list[float]],
        add_joint_type: JointType.revolute,
    ):
        assert self._check_node_name_exist(joint_name), (
            f"Node name {joint_name} does not exist, "
            f"add parallel joint to {parent_name} failed."
        )
        assert self._check_node_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, "
            f"add parent to node {joint_name} failed."
        )
        if isinstance(add_joint_loc, Point):
            x = add_joint_loc.x
            y = add_joint_loc.y
        elif isinstance(add_joint_loc, list):
            assert len(add_joint_loc) >= 2, (
                f"Add joint location {add_joint_loc} is not valid, "
                f"add parallel joint to {parent_name} failed."
            )
            x = add_joint_loc[0]
            y = add_joint_loc[1]
        new_knot = Joint2D(add_joint_type, x, y, name=f"{joint_name}_{parent_name}")
        self.add_joint(new_knot, joint_name)
        self.add_parent_to_node(new_knot.name, parent_name)

    def check_joint_parents_connection(self, joint_name: str):
        j = self.joint_name_map[joint_name]
        parent_names = self.nodes[joint_name].parent
        for parent_name in parent_names:
            self._check_joint_connection(joint_name, parent_name)

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.grid()
        for joint, c in zip(self.joints, self.colors):
            joint.plot(ax, color=c)
        plt.show()
        return ax
