# -*- coding: utf-8 -*-

"""Top-level package for graphbook-code."""

__author__ = """Alex Loftus"""
__email__ = "aloftus2@jhu.edu"
__version__ = "0.1.0"

__all_ = [
    "binary_heatmap",  # plotting
    "heatmap",
    "add_legend",
    "lined_heatmap",
    "plot_latents",
    "draw_layout_plot",
    "draw_multiplot",  # utils
    "text",
]


from .plotting import *
from .utils import *
