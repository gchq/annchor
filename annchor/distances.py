import os
import numpy as np
from numba import njit
import scipy
import Levenshtein as lev
from pynndescent.distances import kantorovich

package_directory = os.path.dirname(os.path.abspath(__file__))
cost_matrix = os.path.join(package_directory, "data", "wasserstein_matrix.npz")

M = np.load(cost_matrix)["arr_0"]


@njit
def wasserstein(x, y):
    """
    Custom wasserstein distance for the sklearn digits dataset.
    Hardcodes the cost matrix for simplicity.
    """
    return kantorovich(x, y, cost=M)


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


def cosine(x, y):
    """
    Cosine distance.
    """
    return scipy.spatial.distance.cosine(x, y)
