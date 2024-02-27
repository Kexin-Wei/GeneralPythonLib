import matplotlib.pyplot as plt
import numpy as np


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def set_axes_equal(ax: plt.Axes, limits=None):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    if limits is None:
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def plt_equal(ax, limits=None):
    ax.set_box_aspect((1, 1, 1))  # IMPORTANT - this is the new, key line
    set_axes_equal(ax, limits=limits)  # IMPORTANT - this is also required


def plt_show_equal(ax, block=False, limits=None):
    plt_equal(ax, limits=limits)
    plt.show(block=block)


def ax3d_handle(return_fig=False, **kwargs):
    if "ax" in kwargs:
        ax = kwargs["ax"]
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(projection="3d")
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")

    if return_fig:
        return ax, fig
    return ax


def create_axs(subplot_n, block=False, return_fig=False):
    r = int(np.floor(np.sqrt(subplot_n)))
    c = int(subplot_n / r)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axs = {}
    for i in range(subplot_n):
        axs[i] = fig.add_subplot(r, c, i + 1, projection="3d")
    if return_fig:
        return axs, fig
    return axs


def draw_mesh(mesh, ax):
    ax.cla()
    ax = mesh.plt_vtx(ax=ax)
    ax = mesh.plt_x(ax=ax)
    plt_equal(ax)
    return ax
