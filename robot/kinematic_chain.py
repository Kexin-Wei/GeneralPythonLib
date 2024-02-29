from .reference_frame import Node
import warnings


class KinematicChain:
    """define the relationship between joints
    by using DotInChain's parent and child to find the next joint
    and id mapped to join and link
    in forward kinematics or inverse kinematics
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.base_name = "BASE"
        self.base_node = Node(self.base_name, self.base_name, None)
        self.nodes[self.base_name] = self.base_node

    @property
    def node_names(self) -> list[str]:
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

    def get_node_child_to_end(self, node_name: str) -> list:
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

    def get_structure(self) -> list:
        return self.get_node_child_to_end(self.base_name)
