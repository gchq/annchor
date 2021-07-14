# numba jitted wasserstein (from pynndescent library)
# https://github.com/lmcinnes/pynndescent/blob/master/pynndescent/distances.py
import numpy as np
from pynndescent.optimal_transport import (
    allocate_graph_structures,
    initialize_graph_structures,
    initialize_supply,
    initialize_cost,
    network_simplex_core,
    total_cost,
    ProblemStatus,
)

_mock_identity = np.eye(2, dtype=np.float32)
_mock_ones = np.ones(2, dtype=np.float32)
_dummy_cost = np.zeros((2, 2), dtype=np.float64)

FLOAT32_EPS = np.finfo(np.float32).eps
FLOAT32_MAX = np.finfo(np.float32).max


from numba import njit


@njit(nogil=True)
def kantorovich(x, y, cost=_dummy_cost, max_iter=100000):

    row_mask = x != 0
    col_mask = y != 0

    a = x[row_mask].astype(np.float64)
    b = y[col_mask].astype(np.float64)

    a_sum = a.sum()
    b_sum = b.sum()

    # if not isclose(a_sum, b_sum):
    #     raise ValueError(
    #         "Kantorovich distance inputs must be valid probability distributions."
    #     )

    a /= a_sum
    b /= b_sum

    sub_cost = cost[row_mask, :][:, col_mask]

    node_arc_data, spanning_tree, graph = allocate_graph_structures(
        a.shape[0],
        b.shape[0],
        False,
    )
    initialize_supply(a, -b, graph, node_arc_data.supply)
    initialize_cost(sub_cost, graph, node_arc_data.cost)
    # initialize_cost(cost, graph, node_arc_data.cost)
    init_status = initialize_graph_structures(graph, node_arc_data, spanning_tree)
    if init_status == False:
        raise ValueError(
            "Kantorovich distance inputs must be valid probability distributions."
        )
    solve_status = network_simplex_core(
        node_arc_data,
        spanning_tree,
        graph,
        max_iter,
    )
    # if solve_status == ProblemStatus.MAX_ITER_REACHED:
    #     print("WARNING: RESULT MIGHT BE INACCURATE\nMax number of iteration reached!")
    if solve_status == ProblemStatus.INFEASIBLE:
        raise ValueError(
            "Optimal transport problem was INFEASIBLE. Please check " "inputs."
        )
    elif solve_status == ProblemStatus.UNBOUNDED:
        raise ValueError(
            "Optimal transport problem was UNBOUNDED. Please check " "inputs."
        )
    result = total_cost(node_arc_data.flow, node_arc_data.cost)

    return result


import os

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
