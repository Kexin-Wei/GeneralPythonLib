# numerical functions for robot kinematics
# using dh table
import warnings
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union

from .joint import Joint2D, JointType, RadOrDeg
from .kinematic_chain import KinematicChain, Node
from .reference_frame import DH, Dimension


@dataclass
class Point:
    x: float
    y: float
    z: float = 0


class EndEffector2D(Node, DH):
    def __init__(
        self,
        theta: float = 0,
        radian: RadOrDeg = RadOrDeg.DEGREE,
        name: str = "end_effector",
    ) -> None:
        Node.__init__(self, name)
        DH.__init__(self, 0, theta, 0, 0, Dimension.two)
        if radian == RadOrDeg.DEGREE:
            self.update_theta(np.deg2rad(theta))

    def plot(
        self,
        ax: plt.Axes = None,
        scale: float = 5,
        color: str = "black",
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.grid()
        x, y, z = self.T[:3, 3]
        n_x = self.T[:3, 0]
        n_y = self.T[:3, 1]
        [dx1, dy1, dz1] = n_x * scale
        [dx2, dy2, dz2] = n_y * scale

        head_width = 0.2 * scale
        head_length = 0.2 * scale
        radius = 0.1 * scale
        ax.arrow(
            x,
            y,
            dx1,
            dy1,
            color="r",
            head_width=head_width,
            head_length=head_length,
        )
        ax.arrow(
            x,
            y,
            dx2,
            dy2,
            color="g",
            head_width=head_width,
            head_length=head_length,
        )
        cle = plt.Circle((x, y), radius, color=color)
        ax.add_patch(cle)
        return ax


class Robot2D(KinematicChain):
    def __init__(self) -> None:
        super().__init__()
        self.joint_name_map: dict[str, Joint2D] = {}
        self.parallel_joint_name_map: dict[str, Joint2D] = {}
        self.end_effector: EndEffector2D = None

    @classmethod
    def from_joint_list(cls, joint_list: list[Joint2D]) -> "Robot2D":
        robot = cls()
        robot.add_joints(joint_list)
        return robot

    @property
    def joints(self) -> list[Node]:
        return list(self.joint_name_map.values())

    @property
    def colors(self) -> np.ndarray:
        n_joints = len(self.joints)
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=n_joints)
        colors = cmap(norm(range(n_joints)))
        return colors

    @property
    def struct(self) -> list:
        return self.get_structure()

    @property
    def parallel_joints(self) -> list[Joint2D]:
        return list(self.parallel_joint_name_map.values())

    @property
    def parallel_colors(self) -> np.ndarray:
        n_joints = len(self.parallel_joints)
        cmap = plt.get_cmap("tab20")
        norm = plt.Normalize(vmin=0, vmax=n_joints)
        colors = cmap(norm(range(n_joints)))
        return colors

    @property
    def joint_range(self) -> list[tuple]:
        j_ranges = []
        for j in self.joints:
            j_ranges.append(j.j_range)
        return j_ranges

    def print_configuration_space(self) -> None:
        print("Configuration space in each joint:")
        for i, (j, j_range) in enumerate(zip(self.joints, self.joint_range)):
            if j.j_type == JointType.REVOLUTE:
                if j.rad == RadOrDeg.RADIAN:
                    print(f"Joint {i + 1} range: {j_range} in radian.")
                else:
                    print(f"Joint {i + 1} range: {j_range} in degree.")
            elif j.j_type == JointType.PRISMATIC:
                print(f"Joint {i + 1} range: {j_range} in meter.")
            else:
                print("mmmm")

    def get_joint_range_per_chain(self, chain: list[str]) -> list[tuple]:
        joint_range = []
        chain_no_base = chain[1:]
        for j_name in chain_no_base:
            if not self._check_node_name_exist(j_name):
                warnings.warn(
                    f"{j_name} does not exist, failed to get joint range."
                )
                continue
            joint_range.append(self.joint_name_map[j_name].j_range)
        return joint_range

    def _check_joint_connection(self, joint: Joint2D, parent: Joint2D) -> None:
        if parent.name == self.base_name:
            return
        if not self._check_node_name_exist(joint.name):
            warnings.warn(
                f"Joint name {joint.name} does not exist, "
                f"check joint connection failed."
            )
            return
        if not self._check_node_name_exist(parent.name):
            warnings.warn(
                f"Parent name {parent.name} does not exist, "
                f"check joint connection failed."
            )
            return
        loc_error = (parent.xn - joint.x) ** 2 + (parent.yn - joint.y) ** 2
        assert loc_error < 1e-6, (
            f"The joints must be connected to each other. "
            f"{joint.name} started at ({joint.x},{joint.y}), "
            f"but the previous joint ended at ({parent.xn},{parent.yn})."
        )

    def add_joint(self, joint: Joint2D, parent: Joint2D) -> None:
        self.add_node_to_parent(joint, parent)
        self.joint_name_map[joint.name] = joint
        self._check_joint_connection(joint, parent)

    def add_end_effector(
        self,
        parent: Joint2D,
        end_of_link: bool = True,
        same_orientation: bool = True,
        relation_matrix: np.ndarray = None,
    ) -> None:
        if not self._check_node_name_exist(parent.name):
            warnings.warn(
                f"Parent name {parent.name} does not exist, "
                f"add end effector failed."
            )
            return

        T = np.eye(4)
        if not end_of_link or not same_orientation:
            if relation_matrix is None or not isinstance(
                relation_matrix, np.ndarray
            ):
                warnings.warn(
                    f"Relation matrix is not valid"
                    f"add end effector failed, using same orientation and end of link."
                )
                end_of_link = True
                same_orientation = True
            elif relation_matrix.shape[0] > 3 or relation_matrix.shape[1] > 3:
                T = relation_matrix[:4, :4]
            elif (
                relation_matrix.shape[0] == 3 and relation_matrix.shape[1] == 3
            ):
                T[:2, :2] = relation_matrix[:2, :2]
                T[:2, 3] = relation_matrix[:2, 2]

        if end_of_link:
            T[0, 3] = parent.xn
            T[1, 3] = parent.yn
        if same_orientation:
            T[:2, :2] = parent.T[:2, :2]
        self.end_effector = EndEffector2D(name="end_effector")
        self.end_effector.update_T(T)
        self.add_node_to_parent(self.end_effector, parent)

    def add_joints(
        self, joints: list[Joint2D], joint_relation: dict = None
    ) -> None:
        for i, j in enumerate(joints):
            if j.name in self.joint_name_map.keys():
                warnings.warn(
                    f"Joint name {j.name} already exist, " f"add joint failed."
                )
                continue

            if joint_relation is not None:
                parent_name = joint_relation[j.name]
                if not self._check_node_name_exist(parent_name):
                    warnings.warn(
                        f"Parent name {parent_name} does not exist, "
                        f"add joint {j.name} failed."
                    )
                    continue
                parent = self.joint_name_map[parent_name]
            elif i > 0:
                parent = joints[i - 1]
            else:
                parent = self.base
            self.add_joint(j, parent)

    def add_parallel_joint_connection(
        self,
        joint: Joint2D,
        parent: Joint2D,
        add_joint_loc: Union[Point, list[float]],
        add_joint_type: JointType.REVOLUTE,
    ) -> None:

        if not self._check_node_name_exist(joint.name):
            warnings.warn(
                f"Node name {joint.name} does not exist, "
                f"add parallel joint to {parent.name} failed."
            )
        if not self._check_node_name_exist(parent.name):
            warnings.warn(
                f"Parent name {parent.name} does not exist, "
                f"add parent to node {joint.name} failed."
            )
        if isinstance(add_joint_loc, Point):
            x = add_joint_loc.x
            y = add_joint_loc.y
        else:  # isinstance(add_joint_loc, list):
            if not len(add_joint_loc) >= 2:
                warnings.warn(
                    f"Add joint location {add_joint_loc} is not valid, "
                    f"add parallel joint to {parent.name} failed."
                )
                return
            x = add_joint_loc[0]
            y = add_joint_loc[1]
        new_knot = Joint2D(
            add_joint_type, x, y, name=f"{joint.name}_{parent.name}"
        )
        self.add_joint(new_knot, joint)
        self.add_parent_connection(new_knot, parent)
        self.parallel_joint_name_map[new_knot.name] = new_knot

    @staticmethod
    def _plot_parallel_joint(
        ax: plt.Axes, joint: Joint2D, color="black"
    ) -> None:
        ax.plot(joint.x, joint.y, "h", color=color, markersize=25)
        if joint.name != "":
            ax.annotate(
                joint.name,
                (joint.x, joint.y),
                color="black",
                fontsize=8,
                weight="heavy",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def plot(self, ax: plt.Axes = None, show_fig: bool = True) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.grid()
        joint_ends = np.zeros((len(self.joints) * 2, 2))
        for i, (joint, c) in enumerate(zip(self.joints, self.colors)):
            if joint.name == self.base_name:
                continue
            if joint in self.parallel_joints:
                self._plot_parallel_joint(ax, joint, color=c)
            else:
                joint.plot(ax, color=c)
                joint_ends[i * 2] = [joint.x, joint.y]
                joint_ends[i * 2 + 1] = [joint.xn, joint.yn]
        prev_x_min, prev_x_max = ax.get_xlim()
        prev_y_min, prev_y_max = ax.get_ylim()
        rob_x_min, rob_x_max = joint_ends[:, 0].min(), joint_ends[:, 0].max()
        rob_y_min, rob_y_max = joint_ends[:, 1].min(), joint_ends[:, 1].max()
        x_range, y_range = rob_x_max - rob_x_min, rob_y_max - rob_y_min
        if x_range == 0:
            x_range = 5
        if y_range == 0:
            y_range = 5
        x_min = min(prev_x_min, rob_x_min)
        x_max = max(prev_x_max, rob_x_max)
        y_min = min(prev_y_min, rob_y_min)
        y_max = max(prev_y_max, rob_y_max)
        ax.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
        ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)
        if show_fig:
            plt.show()
        return ax

    def _chain_forward(self, chain: list[str]) -> np.ndarray:
        T = np.eye(4)
        for joint_name in chain:
            joint = self.joint_name_map[joint_name]
            T = T @ joint.T
        return T

    def forward(self) -> list[np.ndarray]:
        if not self.parallel_joints:
            ends = []
            for chain in self.struct:
                ends.append(self._chain_forward(chain))
            return ends
        else:
            pass  # TODO for parallel robot

    def inverse(self, end: np.ndarray):
        assert end.shape == (
            3,
        ), f"End point shape {end.shape} is not valid for inverse kinematics."
        if not self.parallel_joints:
            pass
        else:  # TODO for parallel robot
            pass

    @staticmethod
    def sample_j_range(j_range: list, n_samples: int) -> np.ndarray:
        j_range = np.array(j_range).swapaxes(0, 1)
        j_samples = np.linspace(j_range[0], j_range[1], n_samples)
        return j_samples

    def workspace_per_chain(
        self, chain: list[str], n_samples: int = 50
    ) -> np.ndarray:
        j_ranges = self.get_joint_range_per_chain(chain)
        j_samples = self.sample_j_range(j_ranges, n_samples)
        j_samples_stack = np.meshgrid(*j_samples.T)
        j_samples_stack = np.stack(j_samples_stack, axis=-1)
        j_samples_stack = j_samples_stack.reshape(-1, len(chain) - 1)
        ends = []
        chain_no_base = chain[1:]
        for j_v in j_samples_stack:
            for i, j_name in enumerate(chain_no_base):
                self.joint_name_map[j_name].new_joint_value(j_v[i])
            ends.append(self._chain_forward(chain_no_base))
        return np.array(ends)

    def workspace(
        self, n_samples: int = 50, end_effector_only: bool = True
    ) -> np.ndarray:
        if not end_effector_only:
            warnings.warn("Only end effector workspace is supported for now.")
            return None
        ends_by_chain = []
        for chain in self.struct:
            if self.end_effector.name in chain:
                ends_by_chain = self.workspace_per_chain(
                    chain, n_samples=n_samples
                )
        return np.concatenate(ends_by_chain, axis=0)

    def plot_workspace(self, ax: plt.Axes = None) -> plt.Axes:
        ax = self.plot(ax, show_fig=False)
        ends_by_chain = self.workspace()
        ax.plot(ends_by_chain[:, 0], ends_by_chain[:, 1], ".", color="black")
        plt.show()
        return ax
