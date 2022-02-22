import numpy as np

from graspologic.utils import (
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from graspologic.simulations import sample_edges
from graspologic.models import BaseGraphEstimator
import warnings
from scipy.stats import mannwhitneyu
from scipy.stats import bernoulli
from typing import Any, Callable, Optional, Union
from typing import Dict, List, Set, Tuple

def ier(
    P: np.ndarray,
    rescale: bool = False,
    directed: bool = False,
    loops: bool = False,
    wt: Optional[Union[int, float, Callable]] = 1,
    wtargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    r"""
    Samples a random graph, given a probability matrix.
    :math:`P = (p_{ij})_{i,j = 1}^n` is the probability matrix, where each entry :math:`p_{ij} \in [0, 1]`
    indicates the probability of the :math:`(i, j)` edge between nodes :math:`i` and :math:`j` existing.
    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        A probability matrix, with entries between 0 and 1.
    rescale: boolean, optional (default=False)
        when ``rescale`` is True, will subtract the minimum value in
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Diagonal elements in P
        matrix are removed prior to rescaling (see above) which may affect behavior.
        Otherwise, edges are sampled in the diagonal.
    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument.
        This weight function will be randomly assigned for selected edges.
        If 1, graph produced is binary.
    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.
    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between
        vertices in a random graph based on their latent positions
    Examples
    --------
    >>> np.random.seed(1)
    Generate random latent positions using 2-dimensional Dirichlet distribution.
    >>> P = np.random.uniform(size=25).reshape((5, 5))
    Sample a binary RDPG using sampled latent positions.
    >>> rdpg(X, loops=False)
    array([[0., 1., 0., 0., 1.],
           [1., 0., 0., 1., 1.],
           [0., 0., 0., 1., 1.],
           [0., 1., 1., 0., 0.],
           [1., 1., 1., 0., 0.]])
    """

    A = sample_edges(P, directed=directed, loops=loops)

    # check weight function
    if (not np.issubdtype(type(wt), np.integer)) and (
        not np.issubdtype(type(wt), np.floating)
    ):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    if callable(wt):
        if wtargs is None:
            wtargs = dict()
        wts = wt(size=(np.count_nonzero(A)), **wtargs)
        A[A > 0] = wts
    else:
        A *= wt  # type: ignore
    return A