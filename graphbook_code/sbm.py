# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np

def block_mtx_psd(B):
    """
    A function which indicates whether a block matrix
    B is positive semi-definite.
    """
    return np.all(np.linalg.eigvals(B) >= 0)

def ohe_comm_vec(z):
    """
    A function to generate the one-hot-encoded community
    assignment matrix from a community assignment vector.
    """
    # if the labels are zero-indexed, right-shift them
    if min(z) == 0:
        z = z + 1
    K = len(np.unique(z))
    n = len(z)
    C = np.zeros((n, K))
    for i, zi in enumerate(z):
        C[i, zi - 1] = 1
    return C


def generate_sbm_pmtx(z, B):
    """
    A function to generate the probability matrix for an SBM.
    """
    C = ohe_comm_vec(z)
    # probability matrix
    return C @ B @ C.T

def generate_dcsbm_pmtx(z, theta, B):
    """
    A function to generate the probability matrix for a DCSBM.
    """
    # uncorrected probability matrix
    Pp = generate_sbm_pmtx(z, B)
    theta = theta.reshape(-1)
    # apply the degree correction
    Theta = np.diag(theta)
    P = Theta @ Pp @ Theta.transpose()
    return P

def dcsbm(z, theta, B, directed=False, loops=False, return_prob=False):
    """
    A function to sample a DCSBM.
    """
    P = generate_dcsbm_pmtx(z, theta, B)
    robj = sample_edges(P, directed=directed, loops=loops)
    if return_prob:
        robj = (robj, P)
    return robj


def lpm_from_sbm(z, B):
    """
    A function to produce a latent position matrix from a
    community assignment vector and a block matrix.
    """
    if not block_mtx_psd(B):
        raise ValueError("Latent position matrices require PSD block matrices!")
    # one-hot encode the community assignment vector
    C = ohe_comm_vec(z)
    # compute square root matrix
    sqrtB = np.linalg.cholesky(B)
    # X = C*sqrt(B)
    return C @ sqrtB


def lpm_from_dcsbm(z, theta, B):
    """
    A function to produce a latent position matrix from a
    community assignment vector, a degree-correction vector,
    and a block matrix.
    """
    # X' = C*sqrt(B)
    Xp = lpm_from_sbm(z, B)
    # X = Theta*X' = Theta * C * sqrt(B)
    return np.diag(theta) @ Xp
