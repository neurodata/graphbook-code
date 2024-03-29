# -*- coding: utf-8 -*-

"""Top-level package for graphbook-code."""

__author__ = """Eric W. Bridgeford, Alex R. Loftus"""
__email__ = "ericwb95@gmail.com"
__version__ = "0.1.0"

__all__ = [
    "binary_heatmap",  # plotting
    "heatmap",
    "add_legend",
    "lined_heatmap",
    "plot_latents",
    "draw_layout_plot",
    "draw_multiplot",  # utils
    "text",
    "draw_cartesian",
    "networkplot",
    "SIEMEstimator",
    "siem",
    "add_circle",
    "ier",
    "plot_vector",
    "ohe_comm_vec",
    "generate_sbm_pmtx",
    "dcsbm",
    "lpm_from_sbm",
    "lpm_from_dcsbm",
    "generate_dcsbm_pmtx",
]

# star imports work only because we have __all__ defined
from .plotting import *
from .utils import *
from .siem import *
from .sbm import *
from .ier import *
