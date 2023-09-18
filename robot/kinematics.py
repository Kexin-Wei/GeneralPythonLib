import warnings
import numpy as np
from ..utility.define_class import Dimension


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


class KinematicChain:
    """define the relationship between joints
    by using DotInChain's parent and child to find the next joint
    and id mapped to join and link
    in forward kinematics or inverse kinematics
    """

    def __init__(self) -> None:
        self.nodes = {}
        self.base_name = "BASE"
        self.base_node = Node(self.base_name, self.base_name, None)
        self.nodes[self.base_name] = self.base_node

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

        assert self._check_node_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, " "add node failed."
        )

        node = Node(node_name, parent_name, child_name)
        self.nodes[parent_name].add_child(node_name)
        self.nodes[node_name] = node

    def _check_node_name_exist(self, node_name: str) -> bool:
        assert node_name is not None, "Node name cannot be None."
        return node_name in self.node_names

    def get_node_by_name(self, node_name: str) -> Node:
        if self._check_node_name_exist(node_name):
            return self.nodes[node_name]
        warnings.warn(f"Node name does not exist, cannot get node by name {node_name}.")
        return None

    def add_parent_to_node(self, node_name: str, parent_name: str) -> None:
        if node_name == self.base_name:
            warnings.warn(
                f"Node name {node_name} is base name, cannot add parent to base."
            )
            return
        assert self._check_node_name_exist(node_name), (
            f"Node name {node_name} does not exist, " "add parent to node failed."
        )
        assert self._check_node_name_exist(parent_name), (
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
        if node_name == self.base_name:
            warnings.warn(
                f"Node name {node_name} is base name, cannot remove parent from base."
            )
            return
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
        assert self._check_node_name_exist(parent_name), (
            f"Parent name {parent_name} does not exist, "
            f"edit node {node_name} failed."
        )
        assert child_name is None or self._check_node_name_exist(child_name), (
            f"Child name {child_name} does not exist, " f"edit node {node_name} failed."
        )
        node: Node = self.nodes[node_name]
        node.add_parent(parent_name)
        node.add_child(child_name)

    def get_node_child_to_end(self, node_name: str) -> list[str]:
        if not self._check_node_name_exist(node_name):
            warnings.warn(f"Node name {node_name} does not exist.")
            return None

        node: Node = self.nodes[node_name]

        if node.child is None:
            return [node_name]

        node_links = []
        for ch in node.child:
            child_links = self.get_node_child_to_end(ch)
            assert len(child_links) > 0
            if isinstance(child_links[0], list):
                for l in child_links:
                    l.insert(0, node_name)
                    node_links.append(l)
            else:
                assert isinstance(child_links[0], str)
                child_links.insert(0, node_name)
                node_links.append(child_links)
        if len(node_links) == 1:
            return node_links[0]
        return node_links

    def get_structure(self):
        return self.get_node_child_to_end(self.base_name)
