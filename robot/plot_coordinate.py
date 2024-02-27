import matplotlib.pyplot as plt
import numpy as np


def plot_vec(
    origin: np.ndarray, vec: np.ndarray, ax: plt.Axes = None, color="k"
) -> plt.Axes:
    assert len(vec.shape) == 1, "Vector must be a 1D array."
    assert len(origin.shape) == 1, "Origin must be a 1D array."
    assert vec.shape[0] == origin.shape[0], (
        f"Vector and origin must have the same dimension. "
        f"Got {vec.shape[0]} and {origin.shape[0]} instead."
    )
    if ax is None:
        if vec.shape[0] == 3:
            ax = plt.figure().add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid()
    vec_len = np.linalg.norm(vec)
    if vec.shape[0] == 3:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
        )
    else:
        ax.quiver(
            origin[0],
            origin[1],
            vec[0],
            vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.05 * vec_len,
            headaxislength=5,
        )
    x_limit = ax.get_xlim()
    y_limit = ax.get_ylim()
    if x_limit[0] > origin[0]:
        x_limit = (origin[0], x_limit[1])
    if x_limit[1] < origin[0] + vec[0]:
        x_limit = (x_limit[0], origin[0] + vec[0])
    if y_limit[0] > origin[1]:
        y_limit = (origin[1], y_limit[1])
    if y_limit[1] < origin[1] + vec[1]:
        y_limit = (y_limit[0], origin[1] + vec[1])
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    if vec.shape[0] == 3:
        z_limit = ax.get_zlim()
        if z_limit[0] > origin[2]:
            z_limit = (origin[2], z_limit[1])
        if z_limit[1] < origin[2] + vec[2]:
            z_limit = (z_limit[0], origin[2] + vec[2])
        ax.set_zlim(z_limit)
    return ax


def plot_coord(
    origin: np.ndarray, xvec: np.ndarray, yvec: np.ndarray, ax: plt.Axes = None
) -> plt.Axes:
    assert len(xvec.shape) == 1, "X vector must be a 1D array."
    assert len(yvec.shape) == 1, "Z vector must be a 1D array."
    assert len(origin.shape) == 1, "Origin must be a 1D array."
    assert xvec.shape[0] == yvec.shape[0], (
        f"X vector and Z vector must have the same dimension. "
        f"Got {xvec.shape[0]} and {yvec.shape[0]} instead."
    )
    assert xvec.shape[0] == origin.shape[0], (
        f"X vector and origin must have the same dimension. "
        f"Got {xvec.shape[0]} and {origin.shape[0]} instead."
    )
    if ax is None:
        if xvec.shape[0] == 3:
            ax = plt.figure().add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.grid()
    ax = plot_vec(origin, xvec, ax, color="r")
    ax = plot_vec(origin, yvec, ax, color="g")
    if xvec.shape[0] == 3:
        zvec = np.cross(xvec, yvec)
        ax = plot_vec(origin, zvec, ax, color="b")
    return ax
