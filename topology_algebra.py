#!/usr/bin/env python3

import numpy as np


def augment(x):
    """
    Homogeneous coordinates:
        z -> [z, 1]

    Supports shape [d] or [n,d].
    """
    x = np.asarray(x)

    if x.ndim == 1:
        return np.concatenate([x, np.ones(1)])

    return np.concatenate(
        [x, np.ones((len(x), 1))],
        axis=1
    )


def compose_path(operators, path):
    """
    path='RAM' means:
        R acts first, then A, then M

    Column-vector convention:
        F_RAM = W_M W_A W_R

    Returned matrix works with row vectors via:
        x_aug @ F.T
    """
    d1 = next(iter(operators.values())).shape[0]

    F = np.eye(d1)

    for role in path:
        F = operators[role] @ F

    return F


def apply_operator(F, x):
    xa = augment(x)
    y = xa @ F.T

    # Last coordinate is homogeneous coordinate.
    return y[..., :-1]


def graph_operator(operators, paths, weights=None):
    """
    Linearized DAG/path-polynomial approximation:

        F_G = sum_p alpha_p F_p

    Useful later for fork/tree/router hypotheses.

    For a single path, use compose_path().
    """
    if weights is None:
        weights = np.ones(len(paths)) / len(paths)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    F = None

    for w, p in zip(weights, paths):
        P = compose_path(operators, p)

        if F is None:
            F = w * P
        else:
            F += w * P

    return F


def contrast_operator(F1, F2):
    """
    Topology contrast:
        Delta_G1,G2 = F_G1 - F_G2
    """
    return F1 - F2


def contrast_score(F1, F2, x):
    """
    Spectral/topology discrimination score:

        S(x) = || (F1-F2) [x;1] ||^2
    """
    K = contrast_operator(F1, F2)

    xa = augment(x)

    # Ignore homogeneous output coordinate.
    delta = xa @ K[:-1, :].T

    return np.sum(delta**2, axis=-1)
