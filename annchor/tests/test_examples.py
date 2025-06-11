from collections import Counter

from numba import njit
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

from annchor import Annchor, BruteForce, compare_neighbor_graphs
from annchor.datasets import load_digits


def test_query_example():

    data = load_digits()
    X = data["X"]
    y = data["y"]
    M = data["cost_matrix"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ann = Annchor(
        X_train,
        "wasserstein",
        func_kwargs={"cost_matrix": M},
        n_anchors=25,
        n_neighbors=25,
        n_samples=5000,
        p_work=0.16,
    )

    ann.fit()

    Q = ann.query(X_test, p_work=0.2)

    errs = 0
    total = 0
    trials = 25
    for i in np.random.choice(np.arange(len(X_test)), trials):
        IJs = (
            np.vstack([np.arange(len(X_train)), i + np.zeros(len(X_train))])
            .astype(int)
            .T
        )
        ds = ann.get_exact_query_ijs(ann.f, X_train, X_test, IJs)
        errs += len(np.setdiff1d(np.argsort(ds)[:15], Q[0][i]))
        total += 15

    assert 1 - (errs / total) >= 0.99

    def get_most_common(arr):
        "Return the most common item from array arr"
        return Counter(arr).most_common(1)[0][0]

    y_pred = np.array(
        [get_most_common(y_train[Q[0][i]]) for i in range(len(X_test))]
    )

    assert np.sum(y_pred == y_test) / len(y_test) >= 0.95


def test_annchor_selective_subset():
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=5)  # First data set X,y
    U, v = make_moons(n_samples=1000, noise=0.1)  # Second data set U,v
    U = np.fliplr(U)

    # First data set
    knn = 15
    annX = Annchor(X, "euclidean", n_neighbors=knn, p_work=0.2)

    annX.fit()

    # Second data set
    knn = 15
    annU = Annchor(U, "euclidean", n_neighbors=knn, p_work=0.2)

    annU.fit()

    # First data set
    ssx = annX.annchor_selective_subset(y=y, alpha=0)
    assert len(ssx) == 90

    # Second data set
    ssu = annU.annchor_selective_subset(y=v, alpha=0)
    assert len(ssu) == 16


def test_custom_anchor_picker():
    # Specify some ANNchor params that we will keep constant
    n_anchors = 10
    p_work = 0.05

    # Create and visualise a test data set
    X, _, centers = make_blobs(
        centers=10, n_samples=1000, random_state=42, return_centers=True
    )

    # define our metric (just euclidean distance) and get exact k-NN graph
    @njit()
    def d(x, y):
        return np.linalg.norm(x - y)

    bruteforce = BruteForce(X, d)
    bruteforce.fit()

    # Run Annchor with default anchor picker (maxmin) and determine its accuracy
    annchor = Annchor(X, d, n_anchors=n_anchors, p_work=p_work)
    annchor.fit()
    n_errors = compare_neighbor_graphs(
        bruteforce.neighbor_graph, annchor.neighbor_graph, n_neighbors=15
    )
    assert n_errors == 0

    # Here we create an anchor picker class which uses a set of pre-chosen points in R2.

    class ExternalAnchorPicker:
        def __init__(self, A):
            # Initialise our anchor picker
            self.A = A  # Init our class with the list of pre-chosen points (A)
            self.is_anchor_safe = (
                False  # If your anchors do not come from the set X
            )
            # then is_anchor_safe should be False

        def get_anchors(self, ann: "Annchor"):
            # This is the main bulk of the class.
            # get_anchors should find the anchor points, and work out the distances to
            # each point in the data set.

            # define some shorthand variables
            nx = ann.nx
            na = ann.n_anchors

            # set random seed for reproducability
            np.random.seed(ann.random_seed)

            # D stores distances to anchor points
            # note: at this point D is shape (n_anchors, nx),
            #       but we transpose this after calculations.
            D = np.zeros((na, nx)) + np.infty

            v = lambda f: f

            # loop over our data set and calculate distance to anchor points
            for i in v(range(na)):

                D[i] = np.array([ann.f(x, self.A[i]) for x in ann.X])

            # Returns 3-tuple (A,D,n_evals)
            # A = array of indices of anchor points if they are in our data set, otherwise empty array
            # D = array of distances to anchor points
            # n_evals = number of calls to the metric
            return np.array([]), D.T, na * nx

    # Let's pick a ring of points surrounding our data and use those as the anchors
    theta = np.linspace(0, np.pi * 2, 11)[:-1]
    ring = np.vstack([15 * np.cos(theta), 15 * np.sin(theta)]).T

    ring_anchor_picker = ExternalAnchorPicker(ring)
    annchor_ring = Annchor(
        X,
        d,
        n_anchors=n_anchors,
        anchor_picker=ring_anchor_picker,
        p_work=p_work,
    )
    annchor_ring.fit()
    n_errors = compare_neighbor_graphs(
        bruteforce.neighbor_graph, annchor_ring.neighbor_graph, n_neighbors=15
    )
    assert n_errors == 0

    # Now let's try picking the centers of our data clusters and using those as the anchors
    center_anchor_picker = ExternalAnchorPicker(centers)
    annchor_center = Annchor(
        X,
        d,
        n_anchors=n_anchors,
        anchor_picker=center_anchor_picker,
        p_work=p_work,
    )
    annchor_center.fit()
    n_errors = compare_neighbor_graphs(
        bruteforce.neighbor_graph,
        annchor_center.neighbor_graph,
        n_neighbors=15,
    )
    assert n_errors == 1

    np.testing.assert_array_almost_equal(
        ring,
        np.array(
            [
                [1.50000000e01, 0.00000000e00],
                [1.21352549e01, 8.81677878e00],
                [4.63525492e00, 1.42658477e01],
                [-4.63525492e00, 1.42658477e01],
                [-1.21352549e01, 8.81677878e00],
                [-1.50000000e01, 1.83697020e-15],
                [-1.21352549e01, -8.81677878e00],
                [-4.63525492e00, -1.42658477e01],
                [4.63525492e00, -1.42658477e01],
                [1.21352549e01, -8.81677878e00],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_array_almost_equal(
        centers,
        np.array(
            [
                [-2.50919762, 9.01428613],
                [4.63987884, 1.97316968],
                [-6.87962719, -6.88010959],
                [-8.83832776, 7.32352292],
                [2.02230023, 4.16145156],
                [-9.58831011, 9.39819704],
                [6.64885282, -5.75321779],
                [-6.36350066, -6.3319098],
                [-3.91515514, 0.49512863],
                [-1.36109963, -4.1754172],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_array_equal(
        annchor.A, np.array([102, 674, 347, 586, 214, 963, 365, 348, 430, 429])
    )
