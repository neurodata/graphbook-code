# -*- coding: utf-8 -*-
import seaborn as sns
from graspologic.plot import heatmap
import matplotlib as mpl


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
