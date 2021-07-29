from annchor import Annchor, BruteForce, compare_neighbor_graphs
from annchor import Annchor, compare_neighbor_graphs
from annchor.datasets import load_digits, load_strings
from annchor.distances import wasserstein
import numpy as np
import Levenshtein as lev


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


def test_digits(seed=42):

    # Set k-NN
    k = 25

    # Load digits
    X = load_digits()["X"]
    y = load_digits()["y"]
    neighbor_graph = load_digits()["neighbor_graph"]

    for it in range(3):

        # Call ANNchor
        ann = Annchor(
            X,
            wasserstein,
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


def test_strings(seed=42):

    # Set k-NN, metric
    k = 15

    def levdist(a, b):
        return lev.distance(a, b)

    strings_data = load_strings()
    X = strings_data["X"]
    y = strings_data["y"]
    neighbor_graph = strings_data["neighbor_graph"]

    for it in range(3):

        # Call ANNchor
        ann = Annchor(
            X,
            levdist,
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


def test_brute_force():

    # Load digits
    X = load_digits()["X"]
    y = load_digits()["y"]
    neighbor_graph = load_digits()["neighbor_graph"]

    bruteforce = BruteForce(X, wasserstein)
    bruteforce.fit()

    error = compare_neighbor_graphs(
        neighbor_graph, bruteforce.neighbor_graph, 100
    )

    assert error == 0
