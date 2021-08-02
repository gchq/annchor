import os
import numpy as np
from numba import njit
import scipy
import Levenshtein as lev


@njit
def euclidean(x, y):
    """
    Euclidean distance.
    """
    return np.linalg.norm(x - y)


def levenshtein(x, y):
    """
    Levenshtein distance.
    """
    return lev.distance(x, y)
