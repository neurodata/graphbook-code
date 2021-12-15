# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, Collection, Optional

import numpy as np
from sklearn.utils import check_X_y

from graspologic.types import Dict, List, Tuple

from graspologic.cluster import GaussianCluster
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspologic.types import GraphRepresentation
from graspologic.utils import (
    augment_diagonal,
    cartesian_product,
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from graspologic.models import BaseGraphEstimator
from graspologic.models.sbm_estimators import _get_block_indices

def _check_common_inputs(
    n_components: Optional[int],
    min_comm: int,
    max_comm: int,
    cluster_kws: Dict[str, Any],
    embed_kws: Dict[str, Any],
) -> None:
    if not isinstance(n_components, int) and n_components is not None:
        raise TypeError("n_components must be an int or None")
    elif n_components is not None and n_components < 1:
        raise ValueError("n_components must be > 0")

    if not isinstance(min_comm, int):
        raise TypeError("min_comm must be an int")
    elif min_comm < 1:
        raise ValueError("min_comm must be > 0")

    if not isinstance(max_comm, int):
        raise TypeError("max_comm must be an int")
    elif max_comm < 1:
        raise ValueError("max_comm must be > 0")
    elif max_comm < min_comm:
        raise ValueError("max_comm must be >= min_comm")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")

    if not isinstance(embed_kws, dict):
        raise TypeError("embed_kws must be a dict")


class SBMEstimator(BaseGraphEstimator):
    r"""
    Stochastic Block Model
    The stochastic block model (SBM) represents each node as belonging to a block
    (or community). For a given potential edge between node :math:`i` and :math:`j`,
    the probability of an edge existing is specified by the block that nodes :math:`i`
    and :math:`j` belong to:
    :math:`P_{ij} = B_{\tau_i \tau_j}`
    where :math:`B \in \mathbb{[0, 1]}^{K x K}` and :math:`\tau` is an `n\_nodes`
    length vector specifying which block each node belongs to.
    Read more in the `Stochastic Block Model (SBM) Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/sbm.html>`_
    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt,
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are
        directed.
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in
        the graph where a node connects to itself.
    n_components : int, optional (default=None)
        Desired dimensionality of embedding for clustering to find communities.
        ``n_components`` must be ``< min(X.shape)``. If None, then optimal dimensions
        will be chosen by :func:`~graspologic.embed.select_dimension`.
    min_comm : int, optional (default=1)
        The minimum number of communities (blocks) to consider.
    max_comm : int, optional (default=10)
        The maximum number of communities (blocks) to consider (inclusive).
    cluster_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspologic.cluster.GaussianCluster`
    embed_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspologic.embed.AdjacencySpectralEmbed`
    Attributes
    ----------
    block_p_ : np.ndarray, shape (n_blocks, n_blocks)
        The block probability matrix :math:`B`, where the element :math:`B_{i, j}`
        represents the probability of an edge between block :math:`i` and block
        :math:`j`.
    p_mat_ : np.ndarray, shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model, from which graphs could be
        sampled.
    vertex_assignments_ : np.ndarray, shape (n_verts)
        A vector of integer labels corresponding to the predicted block that each node
        belongs to if ``y`` was not passed during the call to :func:`~graspologic.models.SBMEstimator.fit`.
    block_weights_ : np.ndarray, shape (n_blocks)
        Contains the proportion of nodes that belong to each block in the fit model.
    See also
    --------
    graspologic.models.DCSBMEstimator
    graspologic.simulations.sbm
    References
    ----------
    .. [1]  Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic
            blockmodels: First steps. Social networks, 5(2), 109-137.
    """

    block_p_: np.ndarray
    vertex_assignments_: np.ndarray

    def __init__(
        self,
        directed: bool = True,
        loops: bool = False,
        n_components: Optional[int] = None,
        min_comm: int = 1,
        max_comm: int = 10,
        cluster_kws: Dict[str, Any] = {},
        embed_kws: Dict[str, Any] = {},
    ):
        super().__init__(directed=directed, loops=loops)

        _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws)

        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = embed_kws

    def _estimate_assignments(self, graph: GraphRepresentation) -> None:
        """
        Do some kind of clustering algorithm to estimate communities
        There are many ways to do this, here is one
        """
        embed_graph = augment_diagonal(graph)
        latent = AdjacencySpectralEmbed(
            n_components=self.n_components, **self.embed_kws
        ).fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        vertex_assignments = gc.fit_predict(latent)  # type: ignore
        self.vertex_assignments_ = vertex_assignments

    def fit(
        self, graph: GraphRepresentation, y: Optional[Any] = None
    ) -> "SBMEstimator":
        """
        Fit the SBM to a graph, optionally with known block labels
        If y is `None`, the block assignments for each vertex will first be
        estimated.
        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit
        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph
        """
        graph = import_graph(graph)

        if not is_unweighted(graph):
            raise NotImplementedError(
                "Graph model is currently only implemented for unweighted graphs."
            )

        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_

            _, counts = np.unique(y, return_counts=True)
            self.block_weights_ = counts / graph.shape[0]
        else:
            check_X_y(graph, y)

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)

        if not self.loops:
            graph = remove_loops(graph)
        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        if not self.directed:
            block_p = symmetrize(block_p)
        self.block_p_ = block_p

        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        if not self.loops:
            p_mat = remove_loops(p_mat)
        self.p_mat_ = p_mat

        return self

    def _n_parameters(self) -> int:
        n_blocks: int = self.block_p_.shape[0]
        n_parameters = 0
        if self.directed:
            n_parameters += n_blocks ** 2
        else:
            n_parameters += int(n_blocks * (n_blocks + 1) / 2)
        if hasattr(self, "vertex_assignments_"):
            n_parameters += n_blocks - 1
        return n_parameters

    def _expand_labels(self, y, mode="abba"):
        if mode == "abba":
            return np.outer(1 - y, y) + np.outer(y, 1 - y) + 1
        elif mode == "abbd":
            return np.outer(1 - y, y) + np.outer(y, 1 - y) + 2 * np.outer(y, y) + 1
        elif mode == "abcd":
            return np.outer(y + 1, y + 2) - 1 - np.outer(y, y)
        elif mode == "abca":
            return np.outer(y + 1, y + 2) - 4 * np.outer(y, y) - 1
        return None

    def _fisher_exact_block_est(self, graph, labels, test_args):
        """
        A function for fisher exact block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        T = np.zeros((2, len(un_labs)))
        for idx, lab in enumerate(un_labs):
            T[0, idx] = (graph[labels == lab]).sum()
            T[1, idx] = len(graph[labels == lab]) - T[0, idx]
        return fisher_exact(
            T, workspace=T.sum() * 1.5, replicate=1000, simulate_pval=True, **test_args
        )

    def _chi2_block_est(self, graph, labels, test_args):
        """
        A function for fisher exact block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        T = np.zeros((2, len(un_labs)))
        for idx, lab in enumerate(un_labs):
            T[0, idx] = (graph[labels == lab]).sum()
            T[1, idx] = len(graph[labels == lab]) - T[0, idx]
        return chi2_contingency(T)[1]

    def _lrt_block_est(self, graph, labels, test_args):
        """
        A function for fisher exact block estimation for a 2-block SBM.
        """
        lrt_dat = DataFrame({"Edge": graph.flatten(), "Community": labels.flatten()})
        model_null = smf.glm(
            formula="Edge~1", data=lrt_dat, family=sm.families.Binomial()
        ).fit()
        model_alt = smf.glm(
            formula="Edge~Community", data=lrt_dat, family=sm.families.Binomial()
        ).fit()
        dof = model_null.df_resid - model_alt.df_resid
        lrs = 2 * (model_alt.llf - model_null.llf)
        return chi2.sf(lrs, df=dof)

    def _mgc_block_est(self, graph, labels, test_args):
        """
        A function for MGC block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        samples = [graph[labels == label] for label in un_labs]
        return KSample("MGC").test(*samples, **test_args)[1]

    def _dcorr_block_est(self, graph, labels, test_args):
        """
        A function for MGC block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        samples = [graph[labels == label] for label in un_labs]
        return KSample("Dcorr").test(*samples, **test_args)[1]

    def _kw_block_est(self, graph, labels, test_args):
        """
        AS function for Kruskal-Wallace block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        samples = [graph[labels == label] for label in un_labs]
        return kruskal(*samples)[1]

    def _anova_block_est(self, graph, labels, test_args):
        """
        A function for anova block estimation for a 2-block SBM.
        """
        un_labs = np.unique(labels)
        samples = [graph[labels == label] for label in un_labs]
        return f_oneway(*samples)[1]

    def estimate_block_structure(
        self,
        graph,
        y,
        candidates,
        test_method="mgc",
        test_args={},
        multitest_method="holm",
        alpha=0.05,
    ):
        """
        Estimate the block structure for 2-block SBMs.
        
        Parameters
        ----------
        graph: array_like or networkx.Graph
            Input graph to estimate a block structure for.
        y: array_like, length graph.shape[0]
            Categorical labels for the block assignments of the graph. Should have 
            2 unique entries.
        candidates: list of strings
            List of candidate models to sequentially test, in order, and will accept the candidate with
            the lowest Holm-Bonferroni corrected p-value.
            Should be a list of strings, where each entry is a 4-character string with acceptable
            entries "a", "b", "c", "d". The string :math:`x_{11}x_{12}x_{21}x_{22}` where :math:`x_{ij} \in \{"a", "b", "c", "d"\}`
            will test the candidate model :math:`[x_{11}, x_{21}; x_{12}, x_{22}]`, where entries that differ
            in the candidate string will correspond to a test of whether those entries differ in distribution.
            For example, the candidate models `["abcd", "abba"]` corresponds to testing 
            :math:`H_0: F_{11} \neq F_{12} \neq F_{21} \neq F_{22}` against :math:`H_1: F_{11} = F_{12} = F_{21} \neq F_{22}` and
            :math:`H_2: F_{11} = F_{22} \neq F_{21} = F_{12}`.
        test_method: string (default="fisher_exact")
            The method to use for estimating p-values associated with different block structures. Supported options are
            `"fisher_exact"` (Fisher Exact Test), `"chi2"` (Chi-squared test), and `"lrt"` (Likelihood Ratio Test)
            for unweighted graphs. Further, for both weighted and unweighted graphs, supported options are
            "mgc" (Multiscale Generalized Correlation), "kw" (Kruskal-Wallace Test), and "anova" (ANOVA).
        multitest_method: string (default="holm")
            The method used for correction for multiple hypotheses when determining an appropriate candidate
            model. Supported options are those from `statsmodels.stats.multitest.multitests()` in the
            `statsmodels` package. Default to `"holm"`, the Holm-Bonferroni step-down correction.
        alpha: float (default=.05)
            A probability, indicating the significance of the test. Defaults to :math:`\alpha=.05`.
        Returns
        -------
        p_val: the p-value associated with the test of the relevant candidate models.
        block_structure: A string indicating the optimal block structure with the lowest corrected p-value.
        """
        graph = import_graph(graph)

        if len(set(y)) != 2:
            raise ValueError("`y` vertex labels should have exactly 2 unique entries.")
        if test_method not in [
            "fisher_exact",
            "chi2",
            "lrt",
            "mgc",
            "kw",
            "anova",
            "dcorr",
        ]:
            raise ValueError("You have passed an unsupported method.")
        if (not is_unweighted(graph)) and (
            test_method in ["fisher_exact", "chi2", "lrt"]
        ):
            raise ValueError(
                "You have passed an unsupported method given a weighted graph."
            )
        for candidate in candidates:
            if len(candidate) != 4:
                raise ValueError(
                    "You have passed a candidate with too many characters."
                )
            if candidate not in ["abba", "abbd", "abcd", "abca"]:
                raise ValueError("You have passed an unsupported candidate model.")
        # run appropriate test
        if test_method == "fisher_exact":
            fn = self._fisher_exact_block_est
        elif test_method == "chi2":
            fn = self._chi2_block_est
        elif test_method == "lrt":
            fn = self._lrt_block_est
        elif test_method == "mgc":
            fn = self._mgc_block_est
        elif test_method == "dcorr":
            fn = self._dcorr_block_est
        elif test_method == "kw":
            fn = self._kw_block_est
        elif test_method == "anova":
            fn = self._anova_block_est
        # execute the statistical tests
        pvals = {}
        for candidate in candidates:
            can_label = self._expand_labels(y, candidate)
            # run test to obtain p-value
            pvals[candidate] = fn(*[graph, can_label], test_args)

        # run multitest method
        reject, cor_pvals, _, _ = multipletests(
            list(pvals.values()), alpha=alpha, method=multitest_method
        )
        idx_best = cor_pvals.argmin()
        if reject[idx_best]:
            return cor_pvals[idx_best], list(pvals.keys())[idx_best]
        else:
            return cor_pvals[idx_best], "aaaa"
