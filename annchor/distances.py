# (C) Crown Copyright GCHQ
import Levenshtein as lev
import numpy as np
from numba import njit


@njit
def euclidean(x, y):
    """
    Euclidean distance.

    Parameters
    ----------
    x : float
    y : float
    """
    return np.linalg.norm(x - y)


def levenshtein(x, y):
    """
    Levenshtein distance.

    Parameters
    ----------
    x : float
    y : float
    """
    return lev.distance(x, y)
