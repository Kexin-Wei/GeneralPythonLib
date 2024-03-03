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
        self.base = Node(self.base_name, None, None)
        self.nodes[self.base_name] = self.base

    @property
    def node_names(self) -> list[str]:
        return list(self.nodes.keys())

    def add_node_to_parent(
        self,
        node: Node,
        parent: Node,
    ) -> None:
        # if no nodes, parent name must be base name
        if len(self.nodes) == 0:
            if parent.name != self.base_name:
                warnings.warn(
                    f"Node {node.name} is the first node, "
                    f"parent name should be {self.base_name}."
                    f"Add node to base now."
                )
                parent_name = self.base_name

        if node.name in self.node_names:
            warnings.warn(
                f"Node name {node.name} already exist, add node failed."
            )
            return

        if not self._check_node_name_exist(parent.name):
            warnings.warn(
                f"Parent name {parent.name} does not exist, "
                f"add node {node.name} failed."
            )
            return
        self.nodes[parent.name].add_child(node)
        self.nodes[node.name] = node

    def remove_node(self, node: Node):
        if not self._check_node_name_exist(node.name):
            warnings.warn(
                f"Node name {node.name} does not exist, remove node failed."
            )
            return
        if node.name == self.base_name:
            warnings.warn(
                f"Node name {node.name} is base name, cannot remove base."
            )
            return

        # remove child from node
        if len(node.child) > 0:
            for c in node.child:
                try:
                    child_node = self.nodes[c]
                except KeyError:
                    warnings.warn(
                        f"Child name {c} does not exist, "
                        f"ignore it and continue"
                    )
                    continue
                self.remove_node(child_node)

        # remove parent connection from node
        if len(node.parent) > 0:
            for p in node.parent:
                try:
                    parent_node = self.nodes[p]
                except KeyError:
                    warnings.warn(
                        f"Parent name {p} does not exist, "
                        f"ignore it and continue"
                    )
                    continue
                parent_node.remove_child(node)
        del self.nodes[node.name]

    def get_node(self, node_name: str) -> Node | None:
        if self._check_node_name_exist(node_name):
            return self.nodes[node_name]
        warnings.warn(
            f"Node name does not exist, cannot get node by name {node_name}."
        )
        return None

    def add_parent_connection(self, node: Node, parent: Node) -> None:
        if not self._check_node_and_parent_when_edit_node(
            node.name, parent.name
        ):
            return
        node.add_parent(parent)

    def add_child_connection(self, node: Node, child: Node) -> None:
        if not self._check_node_and_parent_when_edit_node(
            child.name, node.name
        ):
            return
        node.add_child(child)

    def remove_parent_connection(self, node: Node, parent: Node) -> None:
        if not self._check_node_and_parent_when_edit_node(
            node.name, parent.name
        ):
            return
        node.remove_parent(parent)

    def remove_child_connection(self, node: Node, child: Node) -> None:
        if not self._check_node_and_parent_when_edit_node(
            child.name, node.name
        ):
            return
        node.remove_child(child)

    def get_node_link_from_child_to_end(self, node: Node) -> None | list:
        if not self._check_node_name_exist(node.name):
            warnings.warn(f"Node name {node.name} does not exist.")
            return None

        if node.child is None:
            return [node.name]

        node_links = []
        for ch in node.child:
            child_node = self.get_node(ch)
            if child_node is None:
                warnings.warn(
                    f"Child name {ch} does not exist, "
                    f"continue search for each child"
                )
                continue
            child_links = self.get_node_link_from_child_to_end(child_node)
            if isinstance(child_links[0], list):
                for l in child_links:
                    l.insert(0, node.name)
                    node_links.append(l)
            elif isinstance(child_links[0], str):
                child_links.insert(0, node.name)
                node_links.append(child_links)

        # unpack list of list to list, if only one list in list
        if len(node_links) == 1:
            return node_links[0]
        return node_links

    def get_structure(self) -> list:
        return self.get_node_link_from_child_to_end(self.base)

    def _check_node_name_exist(self, node_name: str) -> bool:
        return node_name in self.node_names

    def _check_node_and_parent_when_edit_node(
        self, node_name: str, parent_name: str
    ) -> bool:
        if node_name == self.base_name:
            warnings.warn(
                f"Node name {node_name} is base name, "
                f"cannot add or remove parent from base."
            )
            return False
        if not self._check_node_name_exist(node_name):
            warnings.warn(
                f"Node name {node_name} does not exist, "
                f"add or remove parent from node failed."
            )
            return False
        if not self._check_node_name_exist(parent_name):
            warnings.warn(
                f"Parent name {parent_name} does not exist, "
                f"add or remove parent from node {node_name} failed."
            )
            return False
