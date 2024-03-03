# numerical functions for robot kinematics
# using dh table
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .joint import Joint2D, JointType, RadOrDeg
from .kinematic_chain import KinematicChain


@dataclass
class Point:
    x: float
    y: float
    z: float = 0


class Robot2D(KinematicChain):
    def __init__(self) -> None:
        super().__init__()
        self.joint_name_map: dict[str, Joint2D] = {}
        self.parallel_joint_name_map: dict[str, Joint2D] = {}

    @classmethod
    def from_joint_list(cls, joint_list: list[Joint2D]) -> "Robot2D":
        robot = cls()
        robot.add_joints(joint_list)
        return robot

    @property
    def base(self) -> str:
        return self.base_name

    @property
    def joints(self) -> list[Joint2D]:
        return list(self.joint_name_map.values())

    @property
    def colors(self) -> ndarray:
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
    def parallel_colors(self) -> ndarray:
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
                    print(f"Joint {i+1} range: {j_range} in radian.")
                else:
                    print(f"Joint {i+1} range: {j_range} in degree.")
            elif j.j_type == JointType.PRISMATIC:
                print(f"Joint {i+1} range: {j_range} in meter.")
            else:
                print("mmmm")

    def get_joint_range_per_chain(self, chain: list[str]) -> list[tuple]:
        joint_range = []
        chain_no_base = chain[1:]
        for j_name in chain_no_base:
            assert self._check_node_name_exist(
                j_name
            ), f"{j_name} does not exist, failed to get joint range."
            joint_range.append(self.joint_name_map[j_name].j_range)
        return joint_range

    def _check_joint_connection(
        self, joint_name: str, parent_name: str
    ) -> None:
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
        self.add_node_to_parent(joint, parent_name)
        self.joint_name_map[joint.name] = joint
        self._check_joint_connection(joint.name, parent_name)

    def add_joints(
        self, joints: list[Joint2D], joint_relation: dict = None
    ) -> None:
        parent_name = self.base_name
        for i, j in enumerate(joints):
            if j.name in self.joint_name_map.keys():
                assert (
                    False
                ), f"Joint name {j.name} already exist, add joint failed."
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
        add_joint_type: JointType.REVOLUTE,
    ) -> None:
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
        new_knot = Joint2D(
            add_joint_type, x, y, name=f"{joint_name}_{parent_name}"
        )
        self.add_joint(new_knot, joint_name)
        self.add_parent_connection(new_knot.name, parent_name)
        self.parallel_joint_name_map[new_knot.name] = new_knot

    def check_joint_parents_connection(self, joint_name: str) -> None:
        j = self.joint_name_map[joint_name]
        parent_names = self.nodes[joint_name].parent
        for parent_name in parent_names:
            self._check_joint_connection(joint_name, parent_name)

    def _plot_parallel_joint(
        self, ax: plt.Axes, joint: Joint2D, color="black"
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
            if joint in self.parallel_joints:
                self._plot_parallel_joint(ax, joint, color=c)
            else:
                joint.plot(ax, color=c)
                joint_ends[i * 2] = [joint.x, joint.y]
                joint_ends[i * 2 + 1] = [joint.xn, joint.yn]
        x_min, x_max = joint_ends[:, 0].min(), joint_ends[:, 0].max()
        y_min, y_max = joint_ends[:, 1].min(), joint_ends[:, 1].max()
        x_range, y_range = x_max - x_min, y_max - y_min
        if x_range == 0:
            x_range = 5
        ax.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
        if y_range == 0:
            y_range = 5
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

    def sample_j_range(self, j_range: list, n_samples: int) -> np.ndarray:
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
                ends.append(self.chain_forward(chain))
        return np.array(ends)

    def workspace(self, n_samples: int = 50) -> np.ndarray:
        ends_by_chain = []
        for chain in self.struct:
            ends_by_chain.append(
                self.workspace_per_chain(chain, n_samples=n_samples)
            )
        return np.concatenate(ends_by_chain, axis=0)

    def plot_workspace(self, ax: plt.Axes = None) -> plt.Axes:
        ax = self.plot(ax, show_fig=False)
        ends_by_chain = self.workspace()
        ax.plot(ends_by_chain[:, 0], ends_by_chain[:, 1], ".", color="black")
        plt.show()
        return ax
