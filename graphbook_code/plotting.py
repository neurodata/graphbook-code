# -*- coding: utf-8 -*-
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Colormap

# from graspologic.plot.plot import _check_common_inputs, _process_graphs, _plot_groups
from graspologic.plot.plot import (
    _check_common_inputs,
    _process_graphs,
    make_axes_locatable,
    _plot_brackets,
    _sort_inds,
    _unique_like,
    _get_freqs,
)
from graspologic.utils import import_graph
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from typing import Union, Optional, Tuple
from matplotlib.axes import Axes
from scipy.sparse import csr_matrix
from graspologic.plot.plot import _check_common_inputs
from graspologic.preconditions import check_argument
from sklearn.utils import check_consistent_length
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke


cmaps = {"sequential": "Purples", "divergent": "RdBu_r", "qualitative": "tab10"}


def add_circle(x=0, y=0, radius=0.15, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    circle = Circle(
        (x, y),
        radius,
        clip_on=False,
        zorder=10,
        linewidth=1,
        edgecolor="black",
        facecolor=(0, 0, 0, 0.0125),
        path_effects=[withStroke(linewidth=5, foreground="w")],
    )
    ax.add_artist(circle)


def text(label, x, y, ax=None, *args, **kwargs):
    """
    Add text to a figure.
    """
    if ax is None:
        ax = plt.gca()
    left, width, bottom, height = 0.25, 0.5, 0.25, 0.5
    right = left + width
    top = bottom + height
    t = ax.text(
        x * (left + right),
        y * (bottom + top),
        label,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=32,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
        *args,
        **kwargs,
    )
    return t


class GraphColormap:
    """
    Default class for colormaps.
    """

    def __init__(self, color, discrete=True, k=None):
        """
        color corresponds to the name of the map type (sequential, divergent, quualitative).
        If discrete is true, discretizes the colormap. Must be true for qualitative colormap.
        """
        if color not in cmaps.keys():
            msg = "`color` option not a valid option."
            raise ValueError(msg)
        if (k is not None) and (not discrete):
            msg = "`k` only specified (optionally) for discrete colormaps."
            raise ValueError(msg)
        self.scale = color
        self.color = cmaps[color]
        self.discrete = discrete
        self.k = k
        kwargs = {}
        kwargs["as_cmap"] = not self.discrete
        if k is not None:
            kwargs["n_colors"] = self.k
        self.palette = sns.color_palette(self.color, **kwargs)


def networkplot(
    adjacency: Union[np.ndarray, csr_matrix],
    x: Union[np.ndarray, str],
    y: Union[np.ndarray, str],
    node_data: Optional[pd.DataFrame] = None,
    node_hue: Optional[Union[np.ndarray, str]] = None,
    palette: Optional[Union[str, list, dict]] = None,
    node_size: Optional[Union[np.ndarray, str]] = None,
    node_sizes: Optional[Union[list, dict, tuple]] = None,
    node_alpha: float = 0.8,
    edge_hue: str = "source",
    edge_linewidth: float = 0.2,
    edge_alpha: float = 0.2,
    title: str = "",
    context: str = "talk",
    font_scale: float = 1.0,
    figsize: Tuple[int, int] = (10, 10),
    ax: Optional[Axes] = None,
    legend: str = False,
    lckwargs: Tuple = {},
    skwargs: Tuple = {},
) -> Axes:
    # Alex Note: this is a better version of draw_layout_plot, soon
    # to be included in graspologic (PR open)
    r"""
    Plots 2D layout of input network. Allows for an adjacency matrix
    with ``x, y`` as 1D arrays that represent the coordinates of each
    node, or an adjacency matrix with ``node_data`` and ``x, y`` as
    keys. Note that the indices of the positions given are assumed to
    correspond with the adjacency matrix.
    Node colors are determined by ``node_hue`` and ``palette``, and if
    ``node_hue`` is None, all nodes will have the same default color
    used by :func:`seaborn.scatterplot`. If ``node_hue`` is given but
    ``palette`` is None, ``palette`` is set to 'Set1' and ``node_hue``
    will be treated as numeric variables. Edge colors are determined by
    its nodes, and ``edge_hue`` dictates whether the edges are colored
    based on its source or target nodes.
    Node sizes can also vary based on ``node_size`` and ``node_sizes``,
    and if ``node_size`` is None, all nodes will be of the same default
    size used by :func:`seaborn.scatterplot`. If ``node_size`` is given
    but ``node_sizes`` is None, ``node_size`` will be treated as numeric
    variables.
    Note that ``palette`` and ``node_sizes`` will not affect the output
    plot if ``node_hue`` and ``node_size`` are None, and ``node_hue`` and
    ``node_size`` must be the same types as ``x, y``.
    Parameters
    ----------
    adjacency: np.ndarray, csr_matrix
        Adjacency matrix of input network.
    x,y: np.ndarray, str
        Variables that specify the positions on the x and y axes. Either an
        array of x, y coordinates or a string that accesses a vector in
        ``node_data``. If ``x, y`` are arrays, they must be indexed the
        same way as the adjacency matrix of the input network.
    node_data: pd.DataFrame, optional, default: None
        Input data. When ``node_data`` is None, ``x, y`` must be np.ndarrays.
        When ``node_data`` is a dataframe, ``x, y`` must be strings. Must be
        indexed the same way as the adjacency matrix of the input network.
    node_hue: np.ndarray, str, optional, default: None
        Variable that produces nodes with different colors. Can be either
        categorical or numeric, and colors are mapped based on ``palette``.
        However if ``palette`` is None, ``node_hue`` is treated as numeric
        and 'Set1' is used as ``palette``.
    palette: str, list, dict, optional, default: None
        Method for choosing colors specified in ``node_hue``. Can be a string
        argument supported by :func:`seaborn.color_palette`, a list of colors,
        or a dictionary with ``node_hue`` variables as keys and colors as its
        values. Note that ``palette`` will not affect the plot if ``node_hue``
        is not given.
    node_size: np.ndarray, str, optional, default: None
        Variable that produces nodes with different sizes. Can be either categorical
        or numeric, and sizes are determined based on ``node_sizes``. If the
        argument ``node_sizes`` is None, ``node_size`` will be treated as
        numeric variables.
    node_sizes: list, dict, tuple, optional, default: None
        Method for choosing sizes specified in ``node_size``. Can be a list of
        sizes, a dictionary with ``node_size`` variables as keys and sizes as
        its values, or a tuple defining the minimum and maximum size values.
        Note that ``node_sizes`` will not affect the output plot if ``node_hue``
        is not given.
    node_alpha: float, default: 0.8
        Proportional opacity of the nodes.
    edge_hue: str, one of {source (default), target}
        Determines edge color based on its source or target node.
    edge_linewidth: float, default: 0.2
        Linewidth of the edges.
    edge_alpha: float, default: 0.2
        Proportional opacity of the edges.
    title: str
        Plot title.
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1.0
        Separate scaling factor to independently scale the size of the font
        elements.
    figsize : tuple of length 2, default: (10, 10)
        Size of the figure (width, height)
    ax: matplotlib.axes.Axes, optional, default: None
        Axes in which to draw the plot. Otherwise, will generate own axes.
    legend: False (default), or one of {brief, full, auto}
        How to draw the legend. If “brief”, numeric hue and size variables
        will be represented with a sample of evenly spaced values. If “full”,
        every group will get an entry in the legend. If “auto”, choose
        between brief or full representation based on number of levels. If
        False, no legend data is added and no legend is drawn.
    Returns
    -------
    ax : matplotlib axis object
        Output plot
    """

    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )

    index = range(adjacency.shape[0])
    if isinstance(x, np.ndarray):
        check_consistent_length(adjacency, x, y)
        check_argument(
            node_data is None, "If x and y are numpy arrays, meta_data must be None."
        )
        plot_df = pd.DataFrame(index=index)
        x_key = "x"
        y_key = "y"
        plot_df.loc[:, x_key] = x
        plot_df.loc[:, y_key] = y
        if node_hue is not None:
            check_argument(
                isinstance(node_hue, np.ndarray),
                "If x and y are numpy arrays, node_hue must be a list or a numpy array.",
            )
            check_consistent_length(x, node_hue)
            hue_key = "hue"
            plot_df.loc[:, hue_key] = node_hue
            if palette is None:
                palette = "Set1"
        else:
            hue_key = None
    elif isinstance(x, str):
        check_consistent_length(adjacency, node_data)
        check_argument(
            node_data is not None,
            "If x and y are strings, meta_data must be pandas DataFrame.",
        )
        plot_df = node_data.copy()
        x_key = x
        y_key = y
        if node_hue is not None:
            check_argument(
                isinstance(node_hue, str),
                "If x and y are strings, node_hue must also be a string.",
            )
            hue_key = node_hue
            if palette is None:
                palette = "Set1"
        else:
            hue_key = None
    else:
        raise TypeError("x and y must be numpy arrays or strings.")

    pre_inds, post_inds = adjacency.nonzero()
    pre = np.array(index)[pre_inds.astype(int)]
    post = np.array(index)[post_inds.astype(int)]
    rows = {"source": pre, "target": post}

    edgelist = pd.DataFrame(rows)
    pre_edgelist = edgelist.copy()
    post_edgelist = edgelist.copy()

    pre_edgelist["x"] = pre_edgelist["source"].map(plot_df[x_key])
    pre_edgelist["y"] = pre_edgelist["source"].map(plot_df[y_key])
    post_edgelist["x"] = post_edgelist["target"].map(plot_df[x_key])
    post_edgelist["y"] = post_edgelist["target"].map(plot_df[y_key])
    pre_coords = list(zip(pre_edgelist["x"], pre_edgelist["y"]))
    post_coords = list(zip(post_edgelist["x"], post_edgelist["y"]))
    coords = list(zip(pre_coords, post_coords))

    if node_hue is not None:
        if isinstance(palette, str):
            palette = sns.color_palette(
                palette, n_colors=len(plot_df[hue_key].unique())
            )
            plot_palette = dict(zip(plot_df[hue_key].unique(), palette))
        elif isinstance(palette, list):
            plot_palette = dict(zip(plot_df[hue_key].unique(), palette))
        elif isinstance(palette, dict):
            plot_palette = palette
        edgelist[hue_key] = edgelist[edge_hue].map(plot_df[hue_key])
        edge_colors = edgelist[hue_key].map(plot_palette)
    else:
        plot_palette = None
        edge_colors = None

    with sns.plotting_context(context=context, font_scale=font_scale):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(
            data=plot_df,
            x=x_key,
            y=y_key,
            hue=hue_key,
            palette=plot_palette,
            size=node_size,
            sizes=node_sizes,
            ax=ax,
            legend=legend,
            alpha=node_alpha,
            zorder=1,
            **skwargs,
        )
        plt.title(title)
        lc = LineCollection(
            segments=coords,
            alpha=edge_alpha,
            linewidths=edge_linewidth,
            colors=edge_colors,
            zorder=0,
            **lckwargs,
        )
        ax.add_collection(lc)
        ax.set(xticks=[], yticks=[])

    return ax


def draw_layout_plot(A, ax=None, pos=None, labels=None, node_color="qualitative"):
    if node_color not in cmaps.keys():
        raise ValueError(f"Your `node_color` must be in {list(cmaps.keys())}")

    G = nx.Graph(A)

    if ax is None:
        fig, ax = plt.subplots()
    if pos is None:
        pos = nx.spring_layout(G)
    else:
        pos = pos(G)

    options = {"edgecolors": "tab:gray", "node_size": 300}

    if labels is not None:
        n_unique = len(np.unique(labels))
        lab_dict = {}
        for j, lab in enumerate(np.unique(labels)):
            lab_dict[lab] = j
        cm = GraphColormap(node_color, discrete=True, k=n_unique)
        node_colors = [cm.palette[lab_dict[i]] for i in labels]
        commlist = [
            plt.Line2D((0, 1), (0, 0), color=cm.palette[col], marker="o", linestyle="")
            for lab, col in lab_dict.items()
        ]
        namelist = ["Community " + str(lab) for lab in lab_dict.keys()]
    else:
        cm = GraphColormap(node_color, discrete=True, k=1)
        node_colors = [cm.palette[0] for i in range(0, A.shape[0])]
    # draw
    nodes_plt = nx.draw_networkx_nodes(
        G, node_color=node_colors, pos=pos, ax=ax, **options
    )
    nx.draw_networkx_edges(G, alpha=0.5, pos=pos, width=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)

    if labels is not None:
        ax.legend(commlist, namelist)

    plt.tight_layout()


def draw_multiplot(
    A,
    pos=None,
    labels=None,
    xticklabels=False,
    yticklabels=False,
    node_color="qualitative",
    title=None,
):
    if node_color not in cmaps.keys():
        raise ValueError(f"Your `node_color` must be in {list(cmaps.keys())}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # heatmap
    hm = heatmap(
        A,
        ax=axs[0],
        cbar=False,
        color="sequential",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        inner_hier_labels=labels,
    )

    # layout plot
    draw_layout_plot(A, ax=axs[1], pos=None, labels=labels, node_color=node_color)
    if title is not None:
        plt.suptitle(title, fontsize=20, y=1.1)

    return axs


def plot_network(network, labels, color="sequential", *args, **kwargs):
    """
    Default plotting function for networks.
    """
    if color not in cmaps.keys():
        msg = "`color` option not a valid option."
        raise ValueError(msg)

    heatmap(network, labels, color=color, *args, **kwargs)


def draw_cartesian(xrange=(-2, 2), yrange=(-2, 2), ticks_frequency=5, ax=None):
    """
    Draw a cartesian coordinate axis.

    Parameters
    ----------
    xrange : tuple, optional
        xmin, xmax, by default (-2, 2)
    yrange : tuple, optional
        ymin, ymax, by default (-2, 2)
    ticks_frequency : int, optional
        interval for ticks, by default 5
    ax : [type], optional
        if None, makes new fig and ax, by default None

    Returns
    -------
    ax
        Cartesian coordinate axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    xmin, xmax = xrange
    ymin, ymax = yrange

    # Set identical scales for both axes
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect="equal")

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel("x", size=14, labelpad=-24, x=1.03)
    ax.set_ylabel("y", size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which="both", color="grey", linewidth=1, linestyle="-", alpha=0.05)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color="black", clip_on=False)
    ax.plot((1), (0), marker=">", transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker="^", transform=ax.get_xaxis_transform(), **arrow_fmt)

    return ax


def plot_latents(
    latent_positions,
    *,
    title=None,
    labels=None,
    ax=None,
    legend=True,
    fontdict=None,
    palette=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    if palette is None:
        palette = GraphColormap("qualitative").color
    if "s" not in kwargs:  # messy way to do this but w/e
        s = 10
    else:
        s = kwargs["s"]
        del kwargs["s"]

    plot = sns.scatterplot(
        x=latent_positions[:, 0],
        y=latent_positions[:, 1],
        hue=labels,
        s=s,
        ax=ax,
        palette=palette,
        color="k",
        **kwargs,
    )
    if title is not None:
        plot.set_title(title, wrap=True, fontdict=fontdict, loc="left")

    h, _ = plot.get_legend_handles_labels()
    if legend and h:
        ax.legend(title="Community")
    elif not legend and np.any(labels):
        ax.get_legend().remove()

    return plot


def add_legend(
    ax=None,
    legend_labels=["No Edge", "Edge"],
    colors=["white", "black"],
    bbox_to_anchor=(1.15, 0.5),
    **kwargs,
):
    fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    patches = []
    for c, l in zip(colors, legend_labels):
        patches.append(mpl.patches.Patch(facecolor=c, label=l, edgecolor="black"))

    fig.legend(
        patches,
        legend_labels,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        fontsize="x-large",
        loc="center right",
        bbox_to_anchor=bbox_to_anchor,
        **kwargs,
    )


def lined_heatmap(data, binary=True, lines_every_n=None, alpha=0.8, *args, **kwargs):
    if binary:
        ax = binary_heatmap(data, *args, **kwargs)
    else:
        ax = heatmap(data, *args, **kwargs)
    if lines_every_n is None:
        n = len(data) // 2
    else:
        n = lines_every_n
    ax.vlines(n, 0, n * 2, colors="black", lw=0.9, linestyle="dashed", alpha=alpha)
    ax.hlines(n, 0, n * 2, colors="black", lw=0.9, linestyle="dashed", alpha=alpha)
    return ax


def binary_heatmap(
    X,
    colors=["white", "black"],
    legend_labels=["No Edge", "Edge"],
    outline=True,
    legend=True,
    **kwargs,
):
    """
        Plots an unweighted graph as a black-and-white matrix with a binary colorbar.  Takes
    the same keyword arguments as ``plot.heatmap``.

        Parameters
        ----------
        X : nx.Graph or np.ndarray object
            Unweighted graph or numpy matrix to plot.

        colors : list-like or np.ndarray
            A list of exactly two colors to use for the heatmap.

        legend : bool, default = True
            If True, add a legend to the heatmap denoting which colors denote which
            ticklabels.

        legend_labels : list-like
            Binary labels to use in the legend. Not used if legend is False.

        outline: bool, default = False
            Whether to add an outline around the border of the heatmap.


        **kwargs : dict, optional
            All keyword arguments in ``plot.heatmap``.

    """
    if len(colors) != 2:
        raise ValueError("Colors must be length 2")
    if "center" in kwargs:
        raise ValueError("Center is not allowed for binary heatmaps.")
    if "cmap" in kwargs:
        raise ValueError(
            "cmap is not allowed in a binary heatmap. To change colors, use the `colors` parameter."
        )
    if not (isinstance(legend_labels, (list, tuple)) and len(legend_labels) == 2):
        raise ValueError("colorbar_ticklabels must be list-like and length 2.")

    # cbar doesn't make sense in the binary case, use legend instead
    kwargs["cbar"] = False
    if "cmap" not in kwargs:
        cmap = mpl.colors.ListedColormap(colors)
        kwargs["cmap"] = cmap
    ax = heatmap(X, center=None, **kwargs)
    if legend:
        no_edge_patch = mpl.patches.Patch(
            facecolor=colors[0], label=legend_labels[0], edgecolor="black"
        )
        edge_patch = mpl.patches.Patch(
            facecolor=colors[1], label=legend_labels[1], edgecolor="black"
        )
        ax.legend(
            [no_edge_patch, edge_patch],
            legend_labels,
            facecolor="white",
            edgecolor="black",
            framealpha=1,
            bbox_to_anchor=(1.25, 0.5),
            fontsize="x-large",
            loc="center right",
        )
    if outline:
        sns.despine(top=False, bottom=False, left=False, right=False)

    return ax


# included because I had to modify slightly
# for our purposes, and those modifications aren't in master
def heatmap(
    X,
    transform=None,
    figsize=(10, 10),
    title=None,
    context="talk",
    font_scale=1,
    xticklabels=False,
    yticklabels=False,
    color="sequential",
    vmin=None,
    vmax=None,
    center=None,
    cbar=True,
    inner_hier_labels=None,
    outer_hier_labels=None,
    hier_label_fontsize=30,
    ax=None,
    title_pad=None,
    sort_nodes=False,
    n_colors=None,
    **kwargs,
):
    r"""
    Plots a graph as a color-encoded matrix.

    Nodes can be grouped by providing ``inner_hier_labels`` or both
    ``inner_hier_labels`` and ``outer_hier_labels``. Nodes can also
    be sorted by the degree from largest to smallest degree nodes.
    The nodes will be sorted within each group if labels are also
    provided.

    Read more in the `Heatmap: Visualizing a Graph Tutorial
    <https://microsoft.github.io/graspologic/tutorials/plotting/heatmaps.html>`_

    Parameters
    ----------
    X : nx.Graph or np.ndarray object
        Graph or numpy matrix to plot

    transform : None, or string {'log', 'log10', 'zero-boost', 'simple-all', 'simple-nonzero'}

        - 'log'
            Plots the natural log of all nonzero numbers
        - 'log10'
            Plots the base 10 log of all nonzero numbers
        - 'zero-boost'
            Pass to ranks method. preserves the edge weight for all 0s, but ranks
            the other edges as if the ranks of all 0 edges has been assigned.
        - 'simple-all'
            Pass to ranks method. Assigns ranks to all non-zero edges, settling
            ties using the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{n^2 + 1}`
            where n is the number of nodes
        - 'simple-nonzero'
            Pass to ranks method. Same as simple-all, but ranks are scaled by
            :math:`\frac{rank(\text{non-zero edges})}{\text{# non-zero edges} + 1}`
        - 'binarize'
            Binarize input graph such that any edge weight greater than 0 becomes 1.

    figsize : tuple of integers, optional, default: (10, 10)
        Width, height in inches.

    title : str, optional, default: None
        Title of plot.

    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.

    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.

    xticklabels, yticklabels : bool or list, optional
        If list-like, plot these alternate labels as the ticklabels.

    color : str, colormap type. "sequential", "divergent", or "qualitative".

    vmin, vmax : floats, optional (default=None)
        Values to anchor the colormap, otherwise they are inferred from the data and
        other keyword arguments.

    center : float, default: 0
        The value at which to center the colormap

    cbar : bool, default: True
        Whether to draw a colorbar.

    inner_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes. If not None, will group the nodes
        according to these labels and plot the labels on the marginal

    outer_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes, ignored without ``inner_hier_labels``
        If not None, will plot these labels as the second level of a hierarchy on the
        marginals

    hier_label_fontsize : int
        Size (in points) of the text labels for the ``inner_hier_labels`` and
        ``outer_hier_labels``.

    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise will generate its own axes

    title_pad : int, float or None, optional (default=None)
        Custom padding to use for the distance of the title from the heatmap. Autoscales
        if None

    sort_nodes : boolean, optional (default=False)
        Whether or not to sort the nodes of the graph by the sum of edge weights
        (degree for an unweighted graph). If ``inner_hier_labels`` is passed and
        ``sort_nodes`` is True, will sort nodes this way within block.

    **kwargs : dict, optional
        additional plotting arguments passed to Seaborn's ``heatmap``
    """
    _check_common_inputs(
        figsize=figsize,
        title=title,
        context=context,
        font_scale=font_scale,
        hier_label_fontsize=hier_label_fontsize,
        title_pad=title_pad,
    )

    if color not in cmaps.keys():
        msg = "`color` option not a valid option."
        raise ValueError(msg)

    # Handle ticklabels
    if isinstance(xticklabels, list):
        if len(xticklabels) != X.shape[1]:
            msg = "xticklabels must have same length {}.".format(X.shape[1])
            raise ValueError(msg)

    elif not isinstance(xticklabels, (bool, int, list)):
        msg = "xticklabels must be a bool, int, or a list, not {}".format(
            type(xticklabels)
        )
        raise TypeError(msg)

    if isinstance(yticklabels, list):
        if len(yticklabels) != X.shape[0]:
            msg = "yticklabels must have same length {}.".format(X.shape[0])
            raise ValueError(msg)
    elif not isinstance(yticklabels, (bool, int, list)):
        msg = "yticklabels must be a bool, int, or a list, not {}".format(
            type(yticklabels)
        )
        raise TypeError(msg)
    # Handle cmap
    X = np.asarray(X)
    if n_colors is None:
        n_colors = 2 if len(np.unique(X)) == 2 else None
    if "cmap" not in kwargs:
        cmap = sns.color_palette(cmaps[color], n_colors=n_colors)
        kwargs["cmap"] = cmap
        if not isinstance(cmap, (str, list, Colormap)):
            msg = (
                "cmap must be a string, list of colors, or matplotlib.colors.Colormap,"
            )
            msg += " not {}.".format(type(cmap))
            raise TypeError(msg)
    else:
        cmap = kwargs["cmap"]

    # Handle center
    if center is not None:
        if not isinstance(center, (int, float)):
            msg = "center must be a integer or float, not {}.".format(type(center))
            raise TypeError(msg)

    # Handle cbar
    if not isinstance(cbar, bool):
        msg = "cbar must be a bool, not {}.".format(type(center))
        raise TypeError(msg)

    # Warning on labels
    if (inner_hier_labels is None) and (outer_hier_labels is not None):
        msg = "outer_hier_labels requires inner_hier_labels to be used."
        warnings.warn(msg)

    arr = import_graph(X)

    arr = _process_graphs(
        [arr], inner_hier_labels, outer_hier_labels, transform, sort_nodes
    )[0]

    # Global plotting settings
    CBAR_KWS = dict(shrink=0.7)  # norm=colors.Normalize(vmin=0, vmax=1))

    with sns.plotting_context(context, font_scale=font_scale):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        plot = sns.heatmap(
            arr,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws=CBAR_KWS,
            center=center,
            cbar=cbar,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        if title is not None:
            if title_pad is None:
                if inner_hier_labels is not None:
                    title_pad = 1.5 * font_scale + 1 * hier_label_fontsize + 30
                else:
                    title_pad = 1.5 * font_scale + 15
            plot.set_title(title, pad=title_pad)
        if inner_hier_labels is not None:
            if outer_hier_labels is not None:
                plot.set_yticklabels([])
                plot.set_xticklabels([])
                _plot_groups(
                    plot,
                    arr,
                    inner_hier_labels,
                    outer_hier_labels,
                    fontsize=hier_label_fontsize,
                )
            else:
                _plot_groups(plot, arr, inner_hier_labels, fontsize=hier_label_fontsize)
    return plot


def _plot_groups(ax, graph, inner_labels, outer_labels=None, fontsize=30):
    inner_labels = np.array(inner_labels)
    plot_outer = True
    if outer_labels is None:
        outer_labels = np.ones_like(inner_labels)
        plot_outer = False

    sorted_inds = _sort_inds(graph, inner_labels, outer_labels, False)
    inner_labels = inner_labels[sorted_inds]
    outer_labels = outer_labels[sorted_inds]

    inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
        inner_labels, outer_labels
    )
    inner_unique, _ = _unique_like(inner_labels)
    outer_unique, _ = _unique_like(outer_labels)

    n_verts = graph.shape[0]
    axline_kws = dict(linestyle="dashed", lw=0.9, alpha=0.3, zorder=3, color="grey")
    # draw lines
    for x in inner_freq_cumsum[1:-1]:
        ax.vlines(x, 0, n_verts + 1, **axline_kws)
        ax.hlines(x, 0, n_verts + 1, **axline_kws)

    # add specific lines for the borders of the plot
    pad = 0.001
    low = pad
    high = 1 - pad
    ax.plot((low, low), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (low, low), transform=ax.transAxes, **axline_kws)
    ax.plot((high, high), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (high, high), transform=ax.transAxes, **axline_kws)

    # generic curve that we will use for everything
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))

    divider = make_axes_locatable(ax)

    # inner curve generation
    inner_tick_loc = inner_freq.cumsum() - inner_freq / 2
    inner_tick_width = inner_freq / 2
    # outer curve generation
    outer_tick_loc = outer_freq.cumsum() - outer_freq / 2
    outer_tick_width = outer_freq / 2

    # top inner curves
    ax_x = divider.new_vertical(size="5%", pad=0.1, pack_start=False)
    ax.figure.add_axes(ax_x)
    _plot_brackets(
        ax_x,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "x",
        n_verts,
        fontsize,
    )
    # side inner curves
    if ax.get_yticklabels():
        yticklabel_fontsize = ax.get_yticklabels()[0].get_fontsize()
        ypad = yticklabel_fontsize * 0.03
    else:
        ypad = 0.0
    ax_y = divider.new_horizontal(size="5%", pad=ypad, pack_start=True)
    ax.figure.add_axes(ax_y)
    _plot_brackets(
        ax_y,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "y",
        n_verts,
        fontsize,
    )

    if plot_outer:
        # top outer curves
        pad_scalar = 0.35 / 30 * fontsize
        ax_x2 = divider.new_vertical(size="5%", pad=pad_scalar, pack_start=False)
        ax.figure.add_axes(ax_x2)
        _plot_brackets(
            ax_x2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "x",
            n_verts,
            fontsize,
        )
        # side outer curves
        ax_y2 = divider.new_horizontal(size="5%", pad=pad_scalar, pack_start=True)
        ax.figure.add_axes(ax_y2)
        _plot_brackets(
            ax_y2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "y",
            n_verts,
            fontsize,
        )
    return ax
