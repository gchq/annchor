import os
import numpy as np
import time

from numba import njit, prange, types
from numba.typed import Dict
from numba.core.registry import CPUDispatcher

from collections import Counter

from annchor.utils import *
from annchor.pickers import *
from annchor.samplers import *
from annchor.regressors import *
from annchor.error_predictors import *

from scipy.sparse import dok_matrix


class Annchor:

    """Annchor

    Quickly computes the approximate k-NN graph for slow metrics

    Parameters
    ----------
    X: np.array or list (required)
        The data set for which we want to find the k-NN graph.
    func: function, numba-jitted function (required) or string.
        The metric under which the k-NN graph should be evaluated.
        This can be a user supplied function or a string.
        Currently supported string arguments are
            * euclidean
            * cosine
            * levenshtein
            * wasserstein  (requires cost_matrix kwarg)
    func_kwargs: dict (optional, default None)
        Dictionary of keyword arguments for the metric
    n_anchors: int (optional, default 20)
        The number of anchor points. Increasing the number of anchors
        increases the k-NN graph accuracy at the expense of speed.
    n_neighbors: int (optional, default 15)
        The number of nearest neighbors to compute (i.e. the value of k
        for the k-NN graph).
    n_samples: int (optional, default 5000)
        The number of sample distances used to compute the error distribution
        (E = d-dhat).
    p_work: float (optional, default 0.1)
        The approximate percentage of brute force calculations which
        we are willing to make.
    anchor_picker: AnchorPicker (optional, default MaxMinAnchorPicker())
        The anchor picker class which specifies the anchor points.
    sampler: Sampler (optional, default SimpleStratifiedSampler())
        The sampler class which chooses the sample pairs.
    regression: Regression
        (optional, default SimpleStratifiedLinearRegression())
        The regression class which predicts distances from features.
    error_predictor: ErrorRegression
        (optional, default SimpleStratifiedErrorRegression())
        The error regression class which predicts error distributions.
    locality: int (optional, default 5)
        The number of anchor points to use in the permutation based k-NN
        (dhat) step.
    loc_thresh: int (optional, default 1)
        The minimum number of anchors in common for which we allow an item
        into NN_x.
    verbose: bool (optional, default False)
        Set verbose=True for more interim output.
    is_metric: bool (optional, default True)
        Set is_metric=False if the metric may violate the triangle inequality.
        With is_metric=True, predictions are clipped between upper/lower
        bounds, which may not be accurate if triangle inequality is violated.
    get_exact_ijs: function (optional, default None)
        An optional user supplied function for evaluating the metric on an
        array of indices. Useful if you wish to supply your own
        parallelisation.
        get_exact_ijs(f,X,IJ) should return
        np.array([f(X[i],X[j] for i,j in IJ]).
    backend: string (optional, default "loky")
        Specifies the joblib Parallel backend.
        Can be "loky" or "multiprocessing"
        In general "loky" seems to be more robust, but occasionally
        "multiprocessing" can be significantly faster


    """

    def __init__(
        self,
        X,
        func,
        func_kwargs=None,
        n_anchors=20,
        n_neighbors=15,
        n_samples=5000,
        p_work=0.1,
        anchor_picker=None,
        sampler=None,
        regression=None,
        error_predictor=None,
        random_seed=42,
        locality=5,
        loc_thresh=1,
        verbose=False,
        is_metric=True,
        get_exact_ijs=None,
        backend="loky",
        niters=2,
        lookahead=5,
    ):

        self.X = X
        self.nx = len(X)
        self.N = (self.nx * (self.nx - 1)) // 2

        self.f = get_function_from_input(func, func_kwargs)

        self.evals = 0

        self.n_anchors = n_anchors
        self.na = np.sum([self.nx - j for j in range(1, self.n_anchors + 1)])

        self.n_neighbors = n_neighbors
        self.p_work = p_work

        if self.p_work > 1:
            print("Warning: p_work should not exceed 1.  Setting it to 1.")
            self.p_work = 1.0
        if self.n_anchors * self.nx / self.N > self.p_work:
            print("Warning: Too many anchors for specified p_work.")
            self.p_work = 2 * self.n_anchors * self.nx / self.N
            print("Increasing p_work to %5.3f." % self.p_work)

        self.n_samples = n_samples

        if anchor_picker is None:
            anchor_picker = MaxMinAnchorPicker()
        if sampler is None:
            sampler = SimpleStratifiedSampler()
        if regression is None:
            regression = SimpleStratifiedLinearRegression()
        if error_predictor is None:
            error_predictor = SimpleStratifiedErrorRegression()
        self.anchor_picker = anchor_picker
        self.sampler = sampler
        self.regression = regression
        self.error_predictor = error_predictor

        self.random_seed = random_seed
        self.verbose = verbose
        self.locality = locality
        self.loc_thresh = loc_thresh
        self.is_metric = is_metric
        self.niters = niters
        self.lookahead = lookahead

        self.RefineApprox = None

        assert backend in ["loky", "multiprocessing"]
        if get_exact_ijs is None:
            self.get_exact_ijs = get_exact_ijs_(
                self.f, verbose=self.verbose, backend=backend
            )
        else:
            self.get_exact_ijs = get_exact_ijs

        test_parallelisation(self.get_exact_ijs,
                             self.f,
                             self.X,
                             self.nx,
                             backend,
                             s=20
                             )

    def get_anchors(self):

        """
        Gets the anchors and distances to anchors.
        Anchors are stored in self.A, distances in self.D.

        self.A: np.array, shape=(n_anchors,)
            Array of anchor indices.
        self.D: np.array, shape=(nx, n_anchors)
            Array of distances to anchor points.

        """

        self.A, self.D, evals = self.anchor_picker.get_anchors(self)

        self.evals += evals

    def get_locality(self):

        """
        Uses basic permutation/set method to find candidate nearest neighbours.

        Current Technique (Use something better in future):
            For each point i, find the set S_i of its nearest l=locality anchor
            points.
            For each other point j, calculate (S_i intersect S_j).
            Only consider pairs ij where |(S_i intersect S_j)|>=loc_thresh.

        self.check: Dict, keys=int64, val=int64[:]
            check[i] is the array of candidate nearest neighbour indices for
            index j.

        """
        start = time.time()

        na = self.n_anchors
        nx = self.nx

        # locality is number of nearest anchors to use in set
        # locality_thresh is number of elements in common required
        # to consider a pair of elements for nn candidacy
        locality = self.locality
        loc_thresh = self.loc_thresh
        sid = np.argsort(self.D, axis=1)[:, :locality]

        # Store candidate pairs in check
        # check[i] is a list of indices that are nn candidates for index i
        check = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )

        ix = np.arange(nx, dtype=np.int64)
        A = np.zeros((na, nx)).astype(int)
        for i in prange(sid.shape[0]):
            for j in sid[i]:
                A[j, i] = 1
        self.Amatrix = A
        for i in prange(nx):
            check[i] = ix[np.sum(A[sid[i], :], axis=0) >= loc_thresh]

        self.check = check

    def get_features(self):

        start = time.time()
        IJs = np.hstack([create_IJs(self.check, i) for i in range(self.nx)])
        IJs = np.fliplr(IJs.T)
        self.IJs = IJs
        n = IJs.shape[0]

        # IJs[:,0] should be sorted at this point
        # assert np.all(IJs[:,0]==np.sort(IJs[:,0]))
        #

        isort = np.arange(n).astype(np.int64)
        jsort = np.argsort(IJs[:, 1]).astype(np.int64)
        fi = IJs[:, 0]
        fj = IJs[jsort, 1]

        ixs = np.arange(n - 1)[(fi[1:] - fi[:-1]).astype(bool)] + 1
        ixs = np.insert(ixs, 0, 0)
        ixs = np.append(ixs, ixs[-1] + 1)
        jxs = np.arange(n - 1)[(fj[1:] - fj[:-1]).astype(bool)] + 1
        jxs = np.insert(jxs, 0, 0)
        jxs = np.append(jxs, ixs[-1] + 1)

        ufi = np.unique(fi)
        ufj = np.unique(fj)

        I = {i: np.array([]).astype(np.int64) for i in range(self.nx)}
        for i, j in enumerate(ufj):
            I[j] = jsort[jxs[i] : jxs[i + 1]]
        J = {i: np.array([]).astype(np.int64) for i in range(self.nx)}
        for i, j in enumerate(ufi):
            J[j] = isort[ixs[i] : ixs[i + 1]]

        self.I = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )
        for i in range(self.nx):
            self.I[i] = np.hstack([I[i], J[i]])

        if check_locality_size(self.I, self.nx, self.n_neighbors):
            raise Exception(
                "Error: Not enough candidates in pool for all indices.\n"
                + "Try again with lower loc_thresh."
            )

        dad = get_dad_ijs(IJs, self.D)
        bounds = get_bounds_njit_ijs(IJs, self.D)
        W = bounds[:, 1] - bounds[:, 0]
        anchors = np.zeros(shape=n)
        # anchors[(bounds[:, 1] - bounds[:, 0]) == 0] = 1
        for a in self.A:
            anchors[self.I[a]] = 1

        self.features = np.vstack([bounds.T, dad, anchors]).T

        self.feature_names = [
            "lower bound",
            "upper bound",
            "double anchor distance",
            "is anchor",
        ]

        i_is_anchor = self.feature_names.index("is anchor")
        self.not_computed_mask = self.features[:, i_is_anchor] < 1

    def get_sample(self):

        """
        Gets the sample of pairwise distances on which to train dhat/errors.

        self.G: np.array, shape=(n_samples,3)
            Array storing the sample distances and features (for future
            regression).
            G[i,:-1] are the features for sample pair i.
            G[i,-1] is the true distance for sample pair i.
        """

        (
            self.sample_ixs,
            self.n_samples,
            self.sample_bins,
        ) = self.sampler.sample(
            self.features,
            self.feature_names,
            self.n_samples,
            self.not_computed_mask,
            self.random_seed,
        )
        self.sample_features = self.features[self.sample_ixs]

        self.sample_ijs = sample_ijs = self.IJs[self.sample_ixs]

        self.sample_y = self.get_exact_ijs(self.f, self.X, self.sample_ijs)

        self.not_computed_mask[self.sample_ixs] = False
        self.evals += self.sample_y.shape[0]

    def fit_predict_regression(self):
        # fit

        self.regression.fit(
            self.sample_features,
            self.feature_names,
            self.sample_y,
            sample_bins=self.sample_bins,
        )

        # predict
        self.pred = self.regression.predict(self.features, self.feature_names)
        self.sample_predict = self.pred[self.sample_ixs]

        ilb = self.feature_names.index("lower bound")
        iub = self.feature_names.index("upper bound")
        self.pred = np.clip(
            self.pred, self.features[:, ilb], self.features[:, iub]
        )

        # if we don't satisfy the triangle inequality we should
        # put in the anchor point distances explicitly
        # (TODO: optimise this better)
        if not self.is_metric:
            for i, a in enumerate(self.A):
                ijs = self.IJs[self.I[a]]
                nas = np.sum(ijs * (ijs != a), axis=1)
                self.pred[self.I[a]] = self.D[nas, i]

        if self.RefineApprox is None:
            self.RefineApprox = self.pred.copy()
        else:
            self.RefineApprox[self.not_computed_mask] = self.pred[
                self.not_computed_mask
            ].copy()
        self.RefineApprox[self.sample_ixs] = self.sample_y

    def fit_predict_errors(self):

        self.error_predictor.fit(
            self.sample_features,
            self.feature_names,
            self.sample_y - self.sample_predict,
            sample_bins=self.sample_bins,
        )

        self.errors = self.error_predictor.predict(
            self.features, self.feature_names
        )

    def select_refine_candidate_pairs(self, w=0.5, it=0):

        nn = self.n_neighbors

        thresh = np.array(
            [
                np.partition(self.RefineApprox[self.I[i]], nn)[nn]
                for i in range(self.nx)
            ]
        )
        self.thresh = thresh

        if it == 0:
            self.RefineApprox = do_the_thing(
                self.nx,
                self.not_computed_mask,
                self.RefineApprox,
                self.I,
                3 * nn // 2,
            )

        p0 = (thresh[self.IJs[:, 0]] - self.RefineApprox)[
            self.not_computed_mask
        ]
        p1 = (thresh[self.IJs[:, 1]] - self.RefineApprox)[
            self.not_computed_mask
        ]
        p = np.max(np.vstack([p0, p1]), axis=0)

        errs = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],
        )
        for label in self.error_predictor.errs:
            errs[label] = self.error_predictor.errs[label]

        prob = get_probs(
            p,
            np.array(self.error_predictor.labels),
            self.errors[self.not_computed_mask],
            errs,
        )

        p_work = self.p_work
        # N = self.nx * (self.nx - 1) / 2
        n_refine = int((p_work * self.N - self.na - self.n_samples) * w) + 1

        n_refine = 0 if (n_refine < 0) else n_refine

        if n_refine >= prob.shape[0]:
            self.candidates = np.arange(prob.shape[0])
            self.next = np.arange(prob.shape[0])
        else:
            if n_refine * self.lookahead >= prob.shape[0]:
                large_part = np.arange(prob.shape[0])
            else:
                large_part = np.argpartition(-prob, n_refine * self.lookahead)[
                    : n_refine * self.lookahead
                ]

            argpart = np.argpartition(-prob[large_part], n_refine)
            self.candidates = large_part[argpart[:n_refine]]
            self.next = large_part[argpart[n_refine:]]

        self.nextback = np.arange(self.not_computed_mask.shape[0])[
            self.not_computed_mask
        ][self.next]

        mapback = np.arange(self.not_computed_mask.shape[0])[
            self.not_computed_mask
        ][self.candidates]

        IJs = self.IJs[mapback]

        exact = self.get_exact_ijs(self.f, self.X, IJs)
        self.evals += exact.shape[0]

        self.RefineApprox[mapback] = exact
        self.not_computed_mask[mapback] = False

    def update_anchor_points(self):
        dis = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )
        ds = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],
        )

        mapback = self.nextback
        for y in range(self.nx):
            mask = self.I[y][(~self.not_computed_mask[self.I[y]])]
            IJs = self.IJs[mask]
            dis[y] = IJs[IJs != y]
            iy = np.argsort(dis[y])
            dis[y] = dis[y][iy]
            ds[y] = self.RefineApprox[mask][iy]

        bounds = update_bounds(self.IJs[mapback], dis, ds)

        try:
            if not self.anchor_picker.is_anchor_safe:
                # If anchors are not in dataset, then we need to check if any
                # of the original anchors gave the best bounds
                bounds = np.vstack(
                    [
                        np.maximum(bounds[:, 0], self.features[mapback][:, 0]),
                        np.minimum(bounds[:, 1], self.features[mapback][:, 1]),
                    ]
                ).T
        except AttributeError:
            pass

        self.features[mapback, :2] = bounds

    def get_ann(self):

        # Get the nn-graph. Can probably optimise this more.

        ng = get_nn(
            self.nx,
            self.n_neighbors,
            self.RefineApprox,
            self.IJs,
            self.I,
            self.not_computed_mask,
        )

        self.neighbor_graph = (
            np.vstack([np.arange(self.nx), ng[0].T]).T,
            np.vstack([np.zeros(self.nx), ng[1].T]).T,
        )

    def fit(self):

        """
        Finds the approx nearest neighbour graph.
        """

        def timeit(item, origin, start):
            print(
                "%40s: %6.3f | %6.3f"
                % (item, time.time() - start, time.time() - origin)
            )
            return

        start = time.time()
        origin = time.time()
        if self.verbose:
            print("computing anchors...")
        self.get_anchors()
        if self.verbose:
            timeit("get_anchors", origin, start)

        start = time.time()
        if self.verbose:
            print("computing locality...")
        self.get_locality()
        if self.verbose:
            timeit("get_locality", origin, start)
        start = time.time()
        if self.verbose:
            print("computing features...")
        self.get_features()
        if self.verbose:
            timeit("get_features", origin, start)

        niters = self.niters
        for it in range(niters):

            start = time.time()
            if self.verbose:
                print("computing sample...")
            try:
                self.get_sample()
            except NothingToSample as err:
                if it == 0:
                    raise ValueError(
                        "Sampler raised NothingToSample on first iteration."
                    ) from err
                else:
                    print(
                        "Warning: main loop terminated early with nothing "
                        + "left to sample."
                    )
                    break
            finally:
                if self.verbose:
                    timeit("get_sample", origin, start)

            start = time.time()
            if self.verbose:
                print("fitting regression...")
            self.fit_predict_regression()
            if self.verbose:
                timeit("fit_predict_regression", origin, start)

            start = time.time()
            if self.verbose:
                print("fitting errors...")
            self.fit_predict_errors()
            if self.verbose:
                timeit("fit_predict_errors", origin, start)

            start = time.time()
            if self.verbose:
                print("selecting/refining candidate pairs (%d)" % it)
            self.select_refine_candidate_pairs(w=1 / niters, it=it)
            if self.verbose:
                timeit("select_refine_candidate_pairs", origin, start)

            if it < niters - 1:
                start = time.time()
                if self.verbose:
                    print("updating anchor points")
                self.update_anchor_points()
                if self.verbose:
                    timeit("update_anchor_points", origin, start)

        start = time.time()
        if self.verbose:
            print("generating neighbour graph")
        self.get_ann()
        if self.verbose:
            timeit("get_ann", origin, start)

    def to_sparse_matrix(self):
        """to_sparse_matrix
        Returns the K-NN graph as a dictionary of keys sparse distance matrix.
        """

        # Initialise sparse matrix
        D = dok_matrix((self.nx, self.nx), dtype=np.float64)

        for i, (js, ds) in enumerate(zip(*self.neighbor_graph)):
            # i is an index into our data
            # js is the list of indices that annchor has determined to be
            # nearest to i
            # ds is the list of distances corresponding to i,js
            for j, d in zip(js, ds):
                # symmetric, so update both pairs (i,j) and (j,i)
                D[i, j] = D[j, i] = d
        return D


class BruteForce:

    """BruteForce

    Computes the approximate k-NN graph by brute force

    Parameters
    ----------
    X: np.array or list (required)
        The data set for which we want to find the k-NN graph.
    func: function, numba-jitted function (required) or string.
        The metric under which the k-NN graph should be evaluated.
        This can be a user supplied function or a string.
        Currently supported string arguments are
            * euclidean
            * cosine
            * levenshtein
    func_kwargs: dict (optional, default None)
        Dictionary of keyword arguments for the metric
    get_exact_ijs: function (optional, default None)
        An optional user supplied function for evaluating the metric on an
        array of indices. Useful if you wish to supply your own
        parallelisation.
        get_exact_ijs(f,X,IJ) should return
        np.array([f(X[i],X[j] for i,j in IJ]).
    backend: string (optional, default "loky")
        Specifies the joblib Parallel backend.
        Can be "loky" or "multiprocessing"
        In general "loky" seems to be more robust, but occasionally
        "multiprocessing" can be significantly faster
    """

    def __init__(
        self,
        X,
        func,
        func_kwargs=None,
        verbose=False,
        get_exact_ijs=None,
        backend="loky",
    ):

        self.X = X
        self.nx = len(X)

        self.f = get_function_from_input(func, func_kwargs)

        self.verbose = verbose

        assert backend in ["loky", "multiprocessing"]
        if get_exact_ijs is None:
            self.get_exact_ijs = get_exact_ijs_(
                self.f, verbose=self.verbose, backend=backend
            )
        else:
            self.get_exact_ijs = get_exact_ijs

        test_parallelisation(self.get_exact_ijs,
                             self.f,
                             self.X,
                             self.nx,
                             backend,
                             s=20
                             )

    def fit(self):
        """get_neighbor_graph

        Gets the k-NN graph from the all pairs distance matrix

        """

        IJs = np.array(
            [(i, j) for i in range(self.nx - 1) for j in range(i + 1, self.nx)]
        )

        dists = self.get_exact_ijs(self.f, self.X, IJs)
        self.D = np.zeros(shape=(self.nx, self.nx))
        for ij, d in zip(IJs, dists):
            i, j = ij
            self.D[i, j] = self.D[j, i] = d
        self.neighbor_graph = (
            np.argsort(self.D, axis=1),
            np.sort(self.D, axis=1),
        )


def compare_neighbor_graphs(nng_1, nng_2, n_neighbors):

    """compare_neighbor_graphs

    Compares accuracy of k-NN graphs. The second graph is compared against the
    first.
    This measure of accuracy accounts for cases where the indices differ but
    the distances are equivalent.

    e.g. if nng_1[0][0]=[0, 1, 2, 3], nng_1[0][1]=[0, 1, 1, 2],

    and     nng_2[0][0]=[0, 1, 2, 4], nng_1[0][1]=[0, 1, 1, 2],

    There would be zero incorrect NN pairs, since both ix=3 and ix=4 are valid
    4th nearest neighbors.

    Parameters
    ----------
    nng_1: nearest neighbour graph (tuple of np.array)
        The first nearest neighbour graph, (indices, distances).
    nng_2: nearest neighbour graph (tuple of np.array)
        The second nearest neighbour graph, (indices, distances)..
    n_neighbors: int
        The number of nearest neighbors to consider

    Returns
    -------
    err: int
        The number of incorrect NN pairs.

    """

    nx = nng_1[0].shape[0]
    h = []
    for ix in range(nx):
        a = Counter(np.round(nng_1[1][ix][:n_neighbors], 3).astype(np.float32))
        b = Counter(np.round(nng_2[1][ix][:n_neighbors], 3).astype(np.float32))
        h.append(len(a - b))
    err = np.sum(h)

    return err
