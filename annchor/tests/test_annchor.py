import numpy as np
from numba import njit
import Levenshtein as lev
from pynndescent.distances import kantorovich
import networkx as nkx

from annchor import Annchor, BruteForce, compare_neighbor_graphs
from annchor import Annchor, compare_neighbor_graphs
from annchor.datasets import load_digits, load_strings, load_graph_sp

# Set interation count for tests with optional iterations
niters = 1


def test_compare_neighbor_graphs(seed=42):
    np.random.seed(seed)
    neighbor_graph = load_digits()["neighbor_graph"]

    # Check self comparison is zero
    error = compare_neighbor_graphs(neighbor_graph, neighbor_graph, 30)
    assert error == 0

    # Check changing distances gives expected number of errors
    ixs, ds = neighbor_graph[0].copy(), neighbor_graph[1].copy()
    for i in range(ds.shape[0]):
        ds[i, np.random.randint(20, 100)] += np.random.random() + 0.01
    error = compare_neighbor_graphs(neighbor_graph, (ixs, ds), 100)
    assert error == ds.shape[0]

    # Check that these errors don't occur where we didn't inject them
    error = compare_neighbor_graphs(neighbor_graph, (ixs, ds), 20)
    assert error == 0


def test_digits(seed=42, niters=niters):

    # Set k-NN
    k = 25

    # Load digits
    data = load_digits()
    X = data["X"]
    M = data["cost_matrix"]
    neighbor_graph = data["neighbor_graph"]

    for it in range(niters):

        # Call ANNchor
        ann = Annchor(
            X,
            "wasserstein",
            func_kwargs={"cost_matrix": M},
            n_anchors=25,
            n_neighbors=k,
            n_samples=5000,
            p_work=0.16,
            random_seed=seed + it,
        )

        ann.fit()

        # Test accuracy
        error = compare_neighbor_graphs(neighbor_graph, ann.neighbor_graph, k)

        # 10 errors is relatively conservative in this parameter regime.
        # We should average much less.
        # 0-5 is typical, with a few outliers.
        assert error < 10


def test_strings(seed=42, niters=niters):

    # Set k-NN, metric
    k = 15

    strings_data = load_strings()
    X = strings_data["X"]
    neighbor_graph = strings_data["neighbor_graph"]

    for it in range(niters):

        # Call ANNchor
        ann = Annchor(
            X,
            "levenshtein",
            n_anchors=23,
            n_neighbors=k,
            random_seed=seed + it,
            n_samples=5000,
            p_work=0.12,
            niters=4,
        )

        ann.fit()

        # Test accuracy
        error = compare_neighbor_graphs(neighbor_graph, ann.neighbor_graph, k)

        # 15 errors is relatively conservative in this parameter regime.
        # We should average much less.
        # 0-5 is typical, with a few outliers
        assert error < 15


def test_graph_sp(seed=42, niters=niters):

    # Set k-NN, metric
    k = 15

    graph_sp_data = load_graph_sp()
    X = graph_sp_data["X"]
    neighbor_graph = graph_sp_data["neighbor_graph"]
    G = graph_sp_data["G"]

    def sp_dist(i, j):
        return nkx.dijkstra_path_length(G, i, j, weight="w")

    # Check some values of sp_dist
    assert np.isclose(sp_dist(0, 0), 0)
    assert np.isclose(sp_dist(2, 5), 0.1487023176704947)
    assert np.isclose(sp_dist(300, 701), 1.2342577780314983)

    for it in range(niters):

        # Call ANNchor
        ann = Annchor(
            X,
            sp_dist,
            n_anchors=20,
            n_neighbors=k,
            random_seed=seed + it,
            n_samples=5000,
            p_work=0.15,
            verbose=True,
        )

        ann.fit()

        # Test accuracy
        error = compare_neighbor_graphs(neighbor_graph, ann.neighbor_graph, k)

        # 10 errors is relatively conservative in this parameter regime.
        # We should average much less.
        # 0-5 is typical, with a few outliers
        assert error < 10


def test_bad_pwork():

    # Set k-NN, metric
    k = 15

    X = load_strings()["X"]

    ann = Annchor(X, "levenshtein", p_work=1.1)
    assert ann.p_work == 1.0

    ann = Annchor(X, "levenshtein", p_work=0.0)

    assert ann.p_work == 2 * ann.n_anchors * ann.nx / ann.N


def test_function_input():

    # Load digits
    data = load_digits()
    X = data["X"]
    M = data["cost_matrix"]
    nx = len(X)

    # not njit, no kwargs
    def wasserstein1(x, y):
        return kantorovich(x, y, cost=M)

    # not njit, kwargs
    def wasserstein2(x, y, cost=M):
        return kantorovich(x, y, cost=M)

    # njit, no kwargs
    @njit()
    def wasserstein3(x, y):
        return kantorovich(x, y, cost=M)

    # njit, kwargs
    @njit()
    def wasserstein4(x, y, cost=M):
        return kantorovich(x, y, cost=M)

    ann1 = Annchor(X, wasserstein1)
    ann2 = Annchor(X, wasserstein2, func_kwargs={"cost": M})
    ann3 = Annchor(X, wasserstein3)
    ann4 = Annchor(X, wasserstein4, func_kwargs={"cost": M})
    ann5 = Annchor(X, "wasserstein", func_kwargs={"cost_matrix": M})

    for it in range(10):
        i, j = np.random.randint(nx, size=2)
        assert np.isclose(ann1.f(X[i], X[j]), ann2.f(X[i], X[j]))
        assert np.isclose(ann2.f(X[i], X[j]), ann3.f(X[i], X[j]))
        assert np.isclose(ann3.f(X[i], X[j]), ann4.f(X[i], X[j]))
        assert np.isclose(ann4.f(X[i], X[j]), ann5.f(X[i], X[j]))

    bf1 = BruteForce(X, wasserstein1)
    bf2 = BruteForce(X, wasserstein2, func_kwargs={"cost": M})
    bf3 = BruteForce(X, wasserstein3)
    bf4 = BruteForce(X, wasserstein4, func_kwargs={"cost": M})
    bf5 = BruteForce(X, "wasserstein", func_kwargs={"cost_matrix": M})

    for it in range(10):
        i, j = np.random.randint(nx, size=2)
        assert np.isclose(bf1.f(X[i], X[j]), bf2.f(X[i], X[j]))
        assert np.isclose(bf2.f(X[i], X[j]), bf3.f(X[i], X[j]))
        assert np.isclose(bf3.f(X[i], X[j]), bf4.f(X[i], X[j]))
        assert np.isclose(bf4.f(X[i], X[j]), bf5.f(X[i], X[j]))


def test_brute_force():

    # Load digits
    data = load_digits()
    X = data["X"]
    M = data["cost_matrix"]
    neighbor_graph = data["neighbor_graph"]

    small_neighbor_graph = (
        np.array(
            [
                neighbor_graph[0][i][neighbor_graph[0][i] < 500][:10]
                for i in range(500)
            ]
        ),
        np.array(
            [
                neighbor_graph[1][i][neighbor_graph[0][i] < 500][:10]
                for i in range(500)
            ]
        ),
    )

    bruteforce = BruteForce(
        X[:500], "wasserstein", func_kwargs={"cost_matrix": M}
    )
    bruteforce.fit()

    error = compare_neighbor_graphs(
        small_neighbor_graph, bruteforce.neighbor_graph, 10
    )

    assert error == 0
