import numpy as np

from graspologic.utils import (
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
    cartprod,
)
from graspologic.models import BaseGraphEstimator
import warnings
from scipy.stats import mannwhitneyu
from scipy.stats import bernoulli

class SIEMEstimator(BaseGraphEstimator):
    """
    Stochastic Independent Edge Model
    
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
    Attributes
    ----------
    model: a dictionary of community names to a dictionary of edge indices and weights.
    K: the number of unique edge communities.
    n_vertices: the number of vertices in the graph.
    See also
    --------
    graspy.simulations.siem
    """

    def __init__(self, directed=True, loops=False):
        super().__init__(directed=directed, loops=loops)
        self.model = {}
        self.K = None
        self._has_been_fit = False

    def fit(self, graph, edge_comm):
        """
        Fits an SIEM to a graph.
        Parameters
        ----------
        graph : array_like [nxn] or networkx.Graph with n vertices
            Input graph to fit
        edge_comm : array_like [n x n]
            A matrix giving the community assignments for each edge within the adjacency matrix
            of `graph`.
        """
        graph = import_graph(graph)

        self.n_vertices = graph.shape[0]
        if not np.ndarray.all(np.isfinite(graph)):
            raise ValueError("`graph` has non-finite entries.")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("`graph` is not a square adjacency matrix.")
        if edge_comm.shape[0] != edge_comm.shape[1]:
            raise ValueError("`edge_comm` is not a square matrix.")
        if not graph.shape == edge_comm.shape:
            msg = """
            Your edge communities do not have the same number of vertices as the graph.
            Graph has %d vertices; edge community has %d vertices.
            """.format(
                graph.shape[0], edge_comm.shape[0]
            )
            raise ValueError(msg)

        siem = {
            x: {"edges": np.where(edge_comm == x), "weights": graph[edge_comm == x]}
            for x in np.unique(edge_comm)
        }
        self.model = siem
        self.K = len(self.model.keys())
        self.graph = graph
        if self._has_been_fit:
            warnings.warn("A model has already been fit. Overwriting previous model...")
        self._has_been_fit = True
        return

    def summarize(self, wts, wtargs):
        """
        Allows users to compute summary statistics for each edge community in the model.
        
        Parameters
        ----------
        wts: dictionary of callables
            A dictionary of summary statistics to compute for each edge community within the model.
            The keys should be the name of the summary statistic, and each entry should be a callable
            function accepting an unnamed argument for a vector or 1-d array as the first argument.
            Keys are names of the summary statistic, and values are the callable objects themselves.
        wtargs: dictionary of dictionaries
            A dictionary of dictionaries, where keys correspond to the names of summary statistics,
            and values are dictionaries of the trailing, named, parameters desired for the summary function. The
            keys of `wts` and `wtargs` should be identical.
        Returns
        -------
        summary: dictionary of summary statistics
            A dictionary where keys are edge community names, and values are a dictionary of summary statistics
            associated with each community.
        """
        # check that model has been fit
        if not self._has_been_fit:
            raise UnboundLocalError(
                "You must fit a model with `fit()` before summarizing the model."
            )
        # check keys for wt and wtargs are same
        if set(wts.keys()) != set(wtargs.keys()):
            raise ValueError("`wts` and `wtargs` should have the same key names.")
        # check wt for callables
        for key, wt in wts.items():
            if not callable(wt):
                raise TypeError("Each value of `wts` should be a callable object.")
        # check whether wtargs is a dictionary of dictionaries with first entry being None in sub-dicts
        for key, wtarg in wtargs.items():
            if not isinstance(wtarg, dict):
                raise TypeError(
                    "Each value of `wtargs` should be a sub-dictionary of class `dict`."
                )

        # equivalent of zipping the two dictionaries together
        wt_mod = {key: (wt, wtargs[key]) for key, wt in wts.items()}

        summary = {}
        for edge_comm in self.model.keys():
            summary[edge_comm] = {}
            for wt_name, (wt, wtarg) in wt_mod.items():
                # and store the summary statistic for this community
                summary[edge_comm][wt_name] = wt(
                    self.model[edge_comm]["weights"], **wtarg
                )
        return summary

    def compare(self, c1, c2, method=mannwhitneyu, methodargs=None):
        """
        A function for comparing two edge communities for a difference after a model has been fit.
        
        Parameters
        ----------
        c1: immutable
            A key in the model, from `self.model.keys()`, to be treated as the first entry
            to the comparison method.
        c2: immutable
            A key in the model, from `self.model.keys()`, to be treated as the second
            entry to the comparison method.
        method: callable
            A callable object to use for comparing the two objects. Should accept two unnamed
            leading vectors or 1-d arrays of edge weights.
        methodargs: dictionary
            A dictionary of named trailing arguments to be passed ot the comparison function of
            interest.
        """
        if not self._has_been_fit:
            raise UnboundLocalError(
                "You must fit a model with `fit()` before comparing communities in the model."
            )
        if not c1 in self.model.keys():
            raise ValueError("`c1` is not a key for the model.")
        if not c2 in self.model.keys():
            raise ValueError("`c2` is not a key for the model.")
        if not callable(method):
            raise TypeError("`method` should be a callable object.")
        if not isinstance(methodargs, dict):
            raise TypeError(
                "`methodargs` should be a dictionary of trailing arguments. Got type %s.".format(
                    type(methodargs)
                )
            )
        return method(
            self.model[c1]["weights"], self.model[c2]["weights"], **methodargs
        )


def siem(
    n,
    p,
    edge_comm,
    directed=False,
    loops=False,
    wt=None,
    wtargs=None,
    return_labels=False,
):
    """
    Samples a graph from the structured independent edge model (SIEM) 
    SIEM produces a graph with specified communities, in which each community can
    have different sizes and edge probabilities. 
    Read more in the :ref:`tutorials <simulations_tutorials>`
    Parameters
    ----------
    n: int
        Number of vertices
    p: float or list of floats of length K (k_communities)
        Probability of an edge existing within the corresponding communities.
        If a float, a probability, or a float greater than or equal to zero and less than or equal to 1.
        It is assumed that the probability is constant over all communities within the graph.
        If a list of floats of length K, each entry p[i] should be a float greater than or equal to zero
        and less than or equal to 1, where p[i] indicates the probability of an edge existing in the ith edge
        community.
    edge_comm: array-like shape (n, n)
        a square 2d numpy array or square numpy matrix of the edge community each edge is assigned to.
        All edges should be assigned a single community, taking values in the integers 1:K
        where K is the total number of unique communities. Note that edge_comm is expected to respect succeeding
        options passed in; particularly, directedness and loopiness. If loops is False, the entire diagonal of
        edge_comm should be 0.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    wt: object or list of K objects
        if Wt is an object, a weight function to use globally over
        the siem for assigning weights. If Wt is a list, a weight function for each of
        the edge communities to use for connection strengths Wt[i] corresponds to the weight function
        for edge community i. Default of None results in a binary graph
    wtargs: dictionary or array-like, shape
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].
    return_labels: boolean, optional (default=True)
        whether to return the community labels of each edge.
    Returns
    -------
    A: ndarray, shape (n, n)
        Sampled adjacency matrix
    labels: ndarray, shape (n, n)
        Square numpy array of labels for each of the edges. Returned if return_labels is True.
    """
    # check booleans
    if not isinstance(loops, bool):
        raise TypeError(
            "`loops` should be a boolean. You passed %s.".format(type(loops))
        )
    if not isinstance(directed, bool):
        raise TypeError(
            "`directed` should be a boolean. You passed %s.".format(type(directed))
        )
    # Check n
    if not isinstance(n, (int)):
        msg = "n must be a int, not {}.".format(type(n))
        raise TypeError(msg)
    # Check edge_comm
    if not isinstance(edge_comm, np.ndarray):
        msg = "edge_comm must be a square numpy array or matrix."
        raise TypeError(msg)
    try:
        if np.any(edge_comm != edge_comm.astype(int)):
            msg = "edge_comm must contain only natural numbers. Contains non-integers."
            raise ValueError(msg)
    except ValueError as err:
        err.message = (
            "edge_comm must contain only natural numbers. Contains non-numerics."
        )
        raise
    edge_comm = edge_comm.astype(int)
    K = edge_comm.max()  # number of communities
    if loops:
        if edge_comm.min() != 1:
            msg = "`edge_comm` should all be numbered sequentially from 1:K. The minimum is not 1."
            raise ValueError(msg)
        if len(np.unique(edge_comm)) != K:
            msg = "`edge_comm` should be numbered sequentially from 1:K. The sequence is not consecutive."
            raise ValueError(msg)
    elif not loops:
        if edge_comm[~np.eye(edge_comm.shape[0], dtype=bool)].min() != 1:
            msg = """Since your graph has no loops, all off-diagonal elements of`edge_comm`
            should have a minimum of 1. The minimum is not 1."""
            raise ValueError(msg)
        if np.any(np.diagonal(edge_comm) != 0):
            msg = """You requested a loopless graph, but assigned a diagonal element to a 
            non-zero community. All diagonal elements of `edge_comm` should be zero if
            `loops` is False."""
            raise ValueError(msg)
        if len(np.unique(edge_comm)) != K + 1:
            msg = """`edge_comm` should be numbered sequentially from 1:K for off-diagonals,
            and 0s on the diagonal. The sequence is not consecutive."""
            raise ValueError(msg)

    n = edge_comm.shape[0]
    if edge_comm.shape[0] != edge_comm.shape[1]:
        msg = "`edge_comm` should be square. `edge_comm` has dimensions [%d, %d]"
        raise ValueError(msg.format(edge_comm.shape[0], edge_comm.shape[1]))
    if len(edge_comm.shape) != 2:
        msg = "`edge_comm` should be a 2d array or a matrix, but `edge_comm` has %d dimensions."
        raise ValueError(msg.format(len(edge_comm.shape)))
    if (not directed) and np.any(edge_comm != edge_comm.T):
        msg = "You requested an undirected SIEM, but `edge_comm` is directed."

    # Check p
    if isinstance(p, float) or isinstance(p, int):
        p = p * np.ones(K)
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if len(p.shape) > 1:
            raise ValueError("p should be a float or a vector/list of length K.")
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p."
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)
        elif len(p) != K:
            msg = "# of Probabilities in `p` and # of Communities in `edge_comm` Don't match up."
            raise ValueError(msg)
    # Check wt and wtargs
    if (wt is not None) and (wtargs is None):
        raise TypeError(
            "wtargs should be a dictionary or a list of dictionaries. It is of type None."
        )
    if (wt is not None) and (wtargs is not None):
        if callable(wt):
            # extend the function to size of K
            wt = np.full(K, wt, dtype=object)
            if isinstance(wtargs, dict):
                wtargs = np.full(K, wtargs, dtype=object)
            else:
                for wtarg in wtargs:
                    if not isinstance(wtarg, dict):
                        raise TypeError(
                            "wtarg should be a dictionary or a list of dictionaries."
                        )
        elif isinstance(wt, list):
            if all(callable(x) for x in wt):
                # if not object, check dimensions
                if not isinstance(wtargs, list):
                    raise TypeError(
                        "Since wt is a list, wtargs should be a list of dictionaries."
                    )
                if len(wt) != K:
                    msg = "wt must have size K, not {}".format(len(wt))
                    raise ValueError(msg)
                if len(wtargs) != K:
                    msg = "wtargs must have size K, not {}".format(len(wtargs))
                    raise ValueError(msg)
            else:
                msg = "wt must contain all callable objects."
                raise TypeError(msg)
        else:
            msg = "wt must be a callable object or list of callable objects"
            raise TypeError(msg)

    # End Checks, begin simulation
    A = np.zeros((n, n))
    for i in range(1, K + 1):
        edge_comm_i = edge_comm == i
        A[np.where(edge_comm_i)] = bernoulli.rvs(p[i - 1], size=edge_comm_i.sum())

        if wt is not None:
            for k, l in zip(*np.where(edge_comm_i)):
                A[k, l] = A[k, l] * wt[i - 1](**wtargs[i - 1])
    # if not directed, just look at upper triangle and duplicate
    if not directed:
        A = symmetrize(A, method="triu")
    if return_labels:
        return (A, edge_comm)
    return A

