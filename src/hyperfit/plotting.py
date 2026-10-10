"""Figures for the notebook.

The colors follow a validated categorical palette. Series colors follow
a fixed slot order. Text and axes use neutral ink colors. Magnitude maps
use one blue hue from light to dark. Signed error maps use blue and red
with a gray midpoint.
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri
import networkx as nx
import numpy as np
import pandas as pd

from hyperfit import data, graph

SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
LINE_STYLES = ("-", "--", "-.", ":")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
DATA_MARKER = "#52514e"

_KIND_COLORS = {
    graph.OBSERVED: "#2a78d6",
    graph.PARAMETER: "#eb6834",
    graph.OPERATION: "#1baf7a",
    graph.CONSTANT: "#c3c2b7",
}
_KIND_NAMES = {
    graph.OBSERVED: "observed variable",
    graph.PARAMETER: "parameter (theta)",
    graph.OPERATION: "operation",
    graph.CONSTANT: "constant",
}
_SHARED_RING = "#c2185b"

SEQUENTIAL_BLUE = mpl.colors.LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING_BLUE_RED = mpl.colors.LinearSegmentedColormap.from_list(
    "diverging_blue_red",
    ["#184f95", "#86b6ef", "#f0efec", "#f19a99", "#b8322f"],
)

_PRETTY_NAMES = {
    "I1b": r"$\bar I_1$",
    "J": r"$J$",
    "l1": r"$\lambda_1$",
    "l2": r"$\lambda_2$",
    "l3": r"$\lambda_3$",
    "u": r"$u$",
    "I": r"$I$",
    "D1": r"$D_1$",
    "∇": r"$\nabla$",
    "FᵀF": r"$F^{T}F$",
    "∂/∂F": r"$\partial/\partial F$",
    "Div": r"$\mathrm{Div}$",
    "-2/3": r"$-\frac{2}{3}$",
}


def apply_style() -> None:
    """Sets the matplotlib defaults for all figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.dpi": 110,
        }
    )


def pretty_label(name: str) -> str:
    """Converts a variable or operation name into a mathtext label.

    Args:
        name: A name from a DAG node or a parameter name.

    Returns:
        A label for matplotlib text.
    """
    if name in _PRETTY_NAMES:
        return _PRETTY_NAMES[name]
    for prefix, symbol in (("mu", r"\mu"), ("alpha", r"\alpha")):
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            return rf"${symbol}_{{{name[len(prefix) :]}}}$"
    if name.startswith("C") and name[1:].isdigit():
        return rf"$C_{{{name[1:]}}}$"
    return name


def plot_dag(
    dag: graph.ExpressionDag,
    ax: plt.Axes,
    title: str,
    root_labels: dict[int, str] | None = None,
) -> None:
    """Draws a DAG from left to right.

    Args:
        dag: The DAG.
        ax: The axes to draw on.
        title: The axes title.
        root_labels: Optional text next to some nodes, by node index.
    """
    network = dag.to_networkx()
    positions = graph.layer_positions(dag)
    shared = set(dag.shared_nodes())
    colors = [_KIND_COLORS[dag.nodes[i].kind] for i in network.nodes]
    rings = [_SHARED_RING if i in shared else "#ffffff" for i in network.nodes]
    widths = [2.5 if i in shared else 1.0 for i in network.nodes]
    nx.draw_networkx_edges(
        network,
        positions,
        ax=ax,
        edge_color="#9a9993",
        arrows=True,
        arrowsize=10,
        width=0.9,
        node_size=900,
    )
    nx.draw_networkx_nodes(
        network,
        positions,
        ax=ax,
        node_color=colors,
        edgecolors=rings,
        linewidths=widths,
        node_size=900,
    )
    labels = {i: pretty_label(dag.nodes[i].label) for i in network.nodes}
    nx.draw_networkx_labels(network, positions, labels, ax=ax, font_size=9)
    for index, text in (root_labels or {}).items():
        x, y = positions[index]
        ax.annotate(
            text,
            (x, y),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": SURFACE,
                "ec": "none",
                "alpha": 0.9,
            },
        )
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor="#ffffff",
            label=_KIND_NAMES[kind],
        )
        for kind, color in _KIND_COLORS.items()
    ]
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="#ffffff",
            markeredgecolor=_SHARED_RING,
            markeredgewidth=2.0,
            label="shared node",
        )
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.margins(x=0.08, y=0.08)


def plot_adjacency(dag: graph.ExpressionDag, ax: plt.Axes, title: str) -> None:
    """Draws the adjacency matrix Z of a DAG.

    Args:
        dag: The DAG.
        ax: The axes to draw on.
        title: The axes title.
    """
    matrix = dag.adjacency_matrix()
    ax.imshow(
        matrix,
        cmap=mpl.colors.ListedColormap(["#f3f2ef", "#1c5cab"]),
        vmin=0,
        vmax=1,
    )
    labels = [pretty_label(node.label) for node in dag.nodes]
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_xlabel("j (input node)")
    ax.set_ylabel("i (node)")
    ax.set_title(title)
    ax.grid(False)


def plot_distance_matrix(table: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    """Draws a square matrix of tree edit distances with value labels.

    Args:
        table: A square table of distances.
        ax: The axes to draw on.
        title: The axes title.
    """
    values = table.to_numpy()
    image = ax.imshow(values, cmap=SEQUENTIAL_BLUE)
    ticks = np.arange(len(table))
    ax.set_xticks(ticks, table.columns, rotation=30, ha="right")
    ax.set_yticks(ticks, table.index)
    threshold = 0.6 * values.max()
    for (row, column), value in np.ndenumerate(values):
        color = "#ffffff" if value > threshold else TEXT_PRIMARY
        ax.text(column, row, str(value), ha="center", va="center", color=color)
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(image, ax=ax, label="tree edit distance", shrink=0.8)


def plot_test_data(
    test: data.HomogeneousTest,
    ax: plt.Axes,
    label: str,
    color: str = DATA_MARKER,
    marker: str = "o",
) -> None:
    """Draws the measured points of a homogeneous test.

    Dense tests show about 60 evenly spaced points, so the markers do not
    overlap.

    Args:
        test: The homogeneous test.
        ax: The axes to draw on.
        label: The legend label.
        color: The marker edge color.
        marker: The marker shape.
    """
    step = max(1, test.stretch.size // 60)
    ax.plot(
        test.stretch[::step],
        test.nominal_stress[::step],
        marker,
        color=color,
        markerfacecolor="none",
        label=label,
    )


def plot_model_curves(
    stretch: np.ndarray, curves: dict[str, np.ndarray], ax: plt.Axes
) -> None:
    """Draws model stress curves with fixed colors and line styles.

    Args:
        stretch: The stretch values of the curves.
        curves: The stress values of each curve, by legend label.
        ax: The axes to draw on.
    """
    for slot, (label, stress) in enumerate(curves.items()):
        ax.plot(
            stretch,
            stress,
            color=SERIES_COLORS[slot % len(SERIES_COLORS)],
            linestyle=LINE_STYLES[slot % len(LINE_STYLES)],
            label=label,
        )


def label_stress_axes(ax: plt.Axes, unit: str, title: str) -> None:
    """Adds the axis labels, the title and the legend of a stress plot.

    Args:
        ax: The axes.
        unit: The stress unit.
        title: The axes title.
    """
    ax.set_xlabel(r"stretch $\lambda$")
    ax.set_ylabel(f"nominal stress P [{unit}]")
    ax.set_title(title)
    ax.legend()


def plot_relative_residuals(
    stretch: np.ndarray,
    residuals: dict[str, np.ndarray],
    ax: plt.Axes,
    title: str,
) -> None:
    """Draws relative stress residuals against the stretch.

    Args:
        stretch: The stretch of each data point.
        residuals: The relative residual of each point, by legend label.
        ax: The axes to draw on.
        title: The axes title.
    """
    for slot, (label, values) in enumerate(residuals.items()):
        ax.plot(
            stretch,
            values,
            color=SERIES_COLORS[slot % len(SERIES_COLORS)],
            linestyle=LINE_STYLES[slot % len(LINE_STYLES)],
            label=label,
        )
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel(r"stretch $\lambda$")
    ax.set_ylabel("relative residual (model - data) / data")
    ax.set_title(title)
    ax.legend()


def plot_time_series(
    test: data.HomogeneousTest, axes: tuple[plt.Axes, plt.Axes]
) -> None:
    """Draws the stretch and the stress of a test against time.

    The two measures have different units, so they use two axes.

    Args:
        test: A homogeneous test with time data.
        axes: Two axes: stretch over time and stress over time.
    """
    axes[0].plot(test.time, test.stretch, color=SERIES_COLORS[0])
    axes[0].set_ylabel(r"stretch $\lambda$")
    axes[0].set_title(f"{test.material}: load history")
    axes[1].plot(test.time, test.nominal_stress, color=SERIES_COLORS[0])
    axes[1].set_ylabel("nominal stress P [MPa]")
    axes[1].set_xlabel("time t [s]")


def plot_cost_history(
    histories: dict[str, tuple[float, ...]], ax: plt.Axes
) -> None:
    """Draws the best CMA-ES cost after each generation.

    Args:
        histories: The cost history of each fit, by legend label.
        ax: The axes to draw on.
    """
    for slot, (label, history) in enumerate(histories.items()):
        ax.plot(
            np.arange(1, len(history) + 1),
            history,
            color=SERIES_COLORS[slot % len(SERIES_COLORS)],
            linestyle=LINE_STYLES[slot % len(LINE_STYLES)],
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel("best cost 0.5 ||r||²")
    ax.set_title("CMA-ES convergence")
    ax.legend()


def _triangulation(
    step: data.FieldStep, displacement: np.ndarray | None
) -> matplotlib.tri.Triangulation:
    """Builds a matplotlib triangulation of the mesh.

    Args:
        step: The field step.
        displacement: If given, the nodes move to the deformed position.

    Returns:
        The triangulation.
    """
    points = (
        step.coordinates
        if displacement is None
        else step.coordinates + displacement
    )
    return matplotlib.tri.Triangulation(
        points[:, 0], points[:, 1], step.connectivity
    )


def plot_nodal_field(
    step: data.FieldStep,
    values: np.ndarray,
    ax: plt.Axes,
    title: str,
    label: str,
    diverging: bool = False,
) -> None:
    """Draws a nodal scalar field on the deformed mesh.

    Args:
        step: The field step. The measured displacement deforms the mesh.
        values: One value per node.
        ax: The axes to draw on.
        title: The axes title.
        label: The color bar label.
        diverging: If True, use the blue-red map, centered at zero.
    """
    triangulation = _triangulation(step, step.displacement)
    if diverging:
        limit = float(np.max(np.abs(values))) or 1.0
        image = ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=DIVERGING_BLUE_RED,
            vmin=-limit,
            vmax=limit,
        )
    else:
        image = ax.tripcolor(
            triangulation, values, shading="gouraud", cmap=SEQUENTIAL_BLUE
        )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    plt.colorbar(image, ax=ax, label=label, shrink=0.85)


def plot_reactions(
    load_steps: list[int],
    measured: np.ndarray,
    predicted: dict[str, np.ndarray],
    ax: plt.Axes,
) -> None:
    """Draws the top reaction force against the load step.

    Args:
        load_steps: The load step numbers.
        measured: The measured reaction force of each load step.
        predicted: The predicted reaction force of each load step, by label.
        ax: The axes to draw on.
    """
    ax.plot(
        load_steps,
        measured,
        "o",
        color=DATA_MARKER,
        markerfacecolor="none",
        markersize=8,
        label="data",
    )
    for slot, (label, values) in enumerate(predicted.items()):
        ax.plot(
            load_steps,
            values,
            color=SERIES_COLORS[slot % len(SERIES_COLORS)],
            linestyle=LINE_STYLES[slot % len(LINE_STYLES)],
            marker="x",
            label=label,
        )
    ax.set_xlabel("load step (pseudo-time)")
    ax.set_ylabel("reaction force, top edge [-]")
    ax.set_title("Reaction force over the load history")
    ax.legend()


def plot_normalized_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], ax: plt.Axes
) -> None:
    """Draws uniaxial model curves divided by the initial shear modulus.

    Args:
        curves: (stretch, P / mu0) of each curve, by legend label.
        ax: The axes to draw on.
    """
    for slot, (label, (stretch, values)) in enumerate(curves.items()):
        ax.plot(
            stretch,
            values,
            color=SERIES_COLORS[slot % len(SERIES_COLORS)],
            linestyle=LINE_STYLES[slot % len(LINE_STYLES)],
            label=label,
        )
    ax.set_xlabel(r"stretch $\lambda$")
    ax.set_ylabel(r"$P / \mu_0$ [-]")
    ax.set_title("Uniaxial response shape, Yeoh N=3 fit of each source")
    ax.legend()


def save_figure(
    figure: plt.Figure, name: str, directory: pathlib.Path
) -> pathlib.Path:
    """Saves a figure as a PNG file.

    Args:
        figure: The figure.
        name: The file name without the extension.
        directory: The output directory. The function creates it if needed.

    Returns:
        The path of the PNG file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
