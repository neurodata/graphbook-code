# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import Colormap
from graspologic.plot.plot import _check_common_inputs, _process_graphs, _plot_groups
from graspologic.utils import import_graph
import warnings
import matplotlib.pyplot as plt


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
    cmap = mpl.colors.ListedColormap(colors)
    ax = heatmap(X, center=None, cmap=cmap, **kwargs)
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
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    center=0,
    cbar=True,
    inner_hier_labels=None,
    outer_hier_labels=None,
    hier_label_fontsize=30,
    ax=None,
    title_pad=None,
    sort_nodes=False,
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

    cmap : str, list of colors, or matplotlib.colors.Colormap, default: 'RdBu_r'
        Valid matplotlib color map.

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

    # Handle ticklabels
    if isinstance(xticklabels, list):
        if len(xticklabels) != X.shape[1]:
            msg = "xticklabels must have same length {}.".format(X.shape[1])
            raise ValueError(msg)

    elif not isinstance(xticklabels, (bool, int)):
        msg = "xticklabels must be a bool, int, or a list, not {}".format(
            type(xticklabels)
        )
        raise TypeError(msg)

    if isinstance(yticklabels, list):
        if len(yticklabels) != X.shape[0]:
            msg = "yticklabels must have same length {}.".format(X.shape[0])
            raise ValueError(msg)
    elif not isinstance(yticklabels, (bool, int)):
        msg = "yticklabels must be a bool, int, or a list, not {}".format(
            type(yticklabels)
        )
        raise TypeError(msg)
    # Handle cmap
    if not isinstance(cmap, (str, list, Colormap)):
        msg = "cmap must be a string, list of colors, or matplotlib.colors.Colormap,"
        msg += " not {}.".format(type(cmap))
        raise TypeError(msg)

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
            cmap=cmap,
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