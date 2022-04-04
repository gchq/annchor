import os
import numpy as np
import time

from numba import njit, prange, types, typeof
from numba.typed import Dict, List
from numba.core.registry import CPUDispatcher

from collections import Counter

from annchor.utils import *
from annchor.pickers import *
from annchor.samplers import *
from annchor.regressors import *
from annchor.error_predictors import *
from annchor.query_functions import *

from scipy.sparse import dok_matrix


class Annchor:

    """Annchor

    Quickly computes the approximate k-NN graph for slow metrics

    Parameters
    ----------
    X: np.array or list (required)
        The data set for which we want to find the k-NN graph.
    func: function, numba-jitted function or string (required)
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
    loc_min: int (optional, default 10*n_neighbors)
        The minimum number of points in NN_x
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
        loc_min=None,
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
        self.n_samples = n_samples

        if self.p_work > 1:
            print("Warning: p_work should not exceed 1.  Setting it to 1.")
            self.p_work = 1.0

        min_p_work = (2 * (self.na + self.n_samples) + 1) / self.N
        min_p_work = 1 if min_p_work > 1 else min_p_work

        if self.p_work < min_p_work:
            print("Warning: Too many anchors/samples for specified p_work.")
            print("Increasing p_work to %5.3f." % min_p_work)
            self.p_work = min_p_work
        if self.p_work > 0.75:
            print("Warning: High Value of p_work.")
            print(
                "Think about decreasing n_anchors or n_samples,"
                + " or using BruteForce."
            )

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
        self.loc_min = 10 * self.n_neighbors if loc_min is None else loc_min
        self.loc_min = np.clip(self.loc_min, 0, self.nx - 1)
        self.is_metric = is_metric
        self.niters = niters
        self.lookahead = lookahead

        self.RefineApprox = None

        assert backend in ["loky", "multiprocessing"]
        self.backend = backend

        if get_exact_ijs is None:
            self.get_exact_ijs = get_exact_ijs_(
                self.f, verbose=self.verbose, backend=backend
            )
        else:
            self.get_exact_ijs = get_exact_ijs

        test_parallelisation(
            self.get_exact_ijs, self.f, self.X, self.nx, backend, s=20
        )

        self.get_exact_query_ijs = None

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
        nn = self.n_neighbors
        nx = self.nx

        # locality is number of nearest anchors to use in set
        # locality_thresh is number of elements in common required
        # to consider a pair of elements for nn candidacy
        locality = self.locality
        # loc_thresh = self.loc_thresh
        self.sid = np.argsort(self.D, axis=1)[:, :locality]

        A = np.zeros((na, nx)).astype(int)
        for i in prange(self.sid.shape[0]):
            for j in self.sid[i]:
                A[j, i] = 1
        self.Amatrix = A

        # Store candidate pairs in check
        # check[i] is a list of indices that are nn candidates for index i

        self.check = get_check(
            self.Amatrix, self.sid, self.loc_thresh, self.loc_min, nx
        )

        self.I, self.IJs = get_IJs_from_check(self.check, nx)

        if check_locality_size(self.I, self.nx, self.n_neighbors):
            raise Exception(
                "Error: Not enough candidates in pool for all indices.\n"
                + "Try again with higher locality."
            )

    def get_features_IJ(self, IJs, I):
        n = IJs.shape[0]
        dad = get_dad_ijs(IJs, self.D)
        bounds = get_bounds_njit_ijs(IJs, self.D)
        # W = bounds[:, 1] - bounds[:, 0]

        anchors = np.zeros(shape=n)
        # anchors[(bounds[:, 1] - bounds[:, 0]) == 0] = 1
        for a in self.A:
            anchors[I[a]] = 1

        features = np.vstack([bounds.T, dad, anchors]).T

        feature_names = [
            "lower bound",
            "upper bound",
            "double anchor distance",
            "is anchor",
        ]

        i_is_anchor = feature_names.index("is anchor")
        not_computed_mask = features[:, i_is_anchor] < 1

        return feature_names, features, not_computed_mask

    def get_features(self):

        (
            self.feature_names,
            self.features,
            self.not_computed_mask,
        ) = self.get_features_IJ(self.IJs, self.I)

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

    def update_anchor_points(self, timeout=10, chunk_size=200000):
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

        n = mapback.shape[0]
        r = np.ceil(n / chunk_size).astype(int)
        k = len(mapback[::r])

        start = time.time()
        for i in range(r):
            _mapback = mapback[i * k : (i + 1) * k]
            bounds = update_bounds(self.IJs[_mapback], dis, ds)

            bounds = np.vstack(
                [
                    np.maximum(bounds[:, 0], self.features[_mapback][:, 0]),
                    np.minimum(bounds[:, 1], self.features[_mapback][:, 1]),
                ]
            ).T

            self.features[_mapback, :2] = bounds
            if time.time() - start > timeout:
                break

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
        eps = np.nextafter(0, 1, dtype=np.float64)
        for i, (js, ds) in enumerate(zip(*self.neighbor_graph)):
            # i is an index into our data
            # js is the list of indices that annchor has determined to be
            # nearest to i
            # ds is the list of distances corresponding to i,js
            for j, d in zip(js, ds):
                # symmetric, so update both pairs (i,j) and (j,i)
                D[i, j] = D[j, i] = d + eps
        return D

    def query(self, Q, nn=15, p_work=0.3, get_exact_query_ijs=None):

        nq = len(Q)
        na = self.n_anchors * nq
        nbf = nq * self.nx
        limit = ((nq * nn * 3) // 2 - 1 + na) / nbf
        if p_work < limit:
            print("Warning: p_work too low")
            print("Increasing p_work to %5.3f" % limit)
            p_work = limit

        return query_(
            self,
            Q,
            nn=nn,
            p_work=p_work,
            get_exact_query_ijs=get_exact_query_ijs,
        )

    def get_nearest_enemies(self, y, nn=3, loc_min=100):
        """get_nearest_enemies
        Returns the nearest enemy graph
        """
        nx = self.nx
        lens = "len(y)=%d, len(X)=%d" % (len(y), nx)
        dim_err = "Label dimension mismatch: " + lens
        assert len(y) == nx, dim_err
        labels, counts = np.unique(y, return_counts=True)
        count_err = "At least one label occurs fewer times " + (
            "than specified nn=%d" % nn
        )
        assert np.all(counts >= nn), count_err

        def f(arr, i):
            return arr[y != y[i]]

        check = get_check(
            self.Amatrix, self.sid, self.loc_thresh, loc_min, nx, f=f
        )

        for i in range(nx):
            z = np.zeros(nx)
            z[check[i]] = 1
            z[self.check[i]] -= 1
            check[i] = np.nonzero(z > 0)[0]
        I, IJs = get_IJs_from_check(check, nx)

        (feature_names, features, not_computed_mask) = self.get_features_IJ(
            IJs, I
        )
        pred = self.regression.predict(features, feature_names)
        ilb = feature_names.index("lower bound")
        iub = feature_names.index("upper bound")
        pred = np.clip(pred, features[:, ilb], features[:, iub])
        nijs = len(self.IJs)
        for i in prange(nx):
            self.I[i] = np.hstack((self.I[i], I[i] + nijs))
        self.IJs = np.vstack((self.IJs, IJs))
        self.not_computed_mask = np.hstack(
            (self.not_computed_mask, not_computed_mask)
        )
        self.RefineApprox = np.hstack((self.RefineApprox, pred))
        self.features = np.vstack((self.features, features))

        RA = self.RefineApprox
        I = self.I
        not_computed_mask = self.not_computed_mask
        IJs = self.IJs
        get_exact_ijs = self.get_exact_ijs

        ngi = np.zeros(shape=(nx, nn), dtype=np.int64)
        ngd = np.zeros(shape=(nx, nn))

        ixs = {}
        fis = {}
        lixs = 0
        for i in range(nx):
            f = IJs[I[i]]
            mask = f[:, 0] == i
            fis[i] = fi = f[:, 1] * mask + f[:, 0] * (1 - mask)
            label_mask = y[fi] != y[i]
            asort = np.argsort(RA[I[i]][label_mask])
            ncm = not_computed_mask[I[i]][label_mask][asort][:50]
            ixs[i] = I[i][label_mask][asort][:50][ncm]
            lixs += len(ixs[i])
        if lixs > 0:
            ixs = np.hstack([ixs[i] for i in range(nx) if len(ixs[i]) > 0])
            d = get_exact_ijs(self.f, self.X, self.IJs[ixs])
            RA[ixs] = d
            not_computed_mask[ixs] = False

        for i in prange(nx):
            Ii = I[np.int64(i)]
            d = RA[Ii]
            mx = np.max(d)
            d[not_computed_mask[Ii]] += mx

            fi = fis[i]

            d[y[fi] == y[i]] += mx
            t = np.partition(d, nn - 1)[nn]
            mask = d <= t
            iy = Ii[mask][np.argsort(d[mask])][:nn]

            ngd[i, :] = RA[iy]
            ngi[i, :] = fi[mask][np.argsort(d[mask])][:nn]

        self.nearest_enemy_graph = (ngi, ngd)

    def annchor_selective_subset(self, y, dne=None, alpha=0):

        if dne is None:
            try:
                dne = self.nearest_enemy_graph[1][:, 0]
            except AttributeError:
                self.get_nearest_enemies(y)
                dne = self.nearest_enemy_graph[1][:, 0]

        zero_dist_enemies = np.argwhere(dne == 0)
        if len(zero_dist_enemies) > 0:
            error = (
                "Error: The following indices are distance zero from a point "
                + " with a different label:\n"
            )
            for i in zero_dist_enemies:
                error += "\t %d\n" % i
            raise Exception(error)

        alpha_dne = dne / (1 + alpha)

        ix = np.arange(len(self.X))

        ngi, ngd = self.neighbor_graph

        ebuffer = np.array(
            [
                np.searchsorted(_ngd, _dne - 1e-6)
                for _ngd, _dne in zip(ngd, alpha_dne)
            ]
        )
        buffer = [_ngi[:eb] for _ngi, eb in zip(ngi, ebuffer)]
        rss = ix[ebuffer == 1]

        present = np.isin(ngi, rss)

        amaxpresent = np.argmax(present, axis=1)
        anypresent = np.any(present, axis=1)

        rssbuffer = amaxpresent + ebuffer * (~anypresent)
        done = np.array(rssbuffer < ebuffer)

        rest = ix[~done]

        while len(rest) > 0:
            stack = np.hstack([buffer[t] for t in ix[~done]])
            a, b = np.unique(stack, return_counts=True)
            nxt = a[np.argmax(b)]

            rss = np.append(rss, nxt)

            mid = time.time()

            # present = np.isin(ngi, rss[-1])
            # return present,ngi,rss,ebuffer,done

            present = np.isin(ngi[~done], rss[-1])

            amaxpresent = np.argmax(present, axis=1)
            anypresent = np.any(present, axis=1)

            rssbuffer = amaxpresent + ebuffer[~done] * (~anypresent)
            done[~done] += rssbuffer < ebuffer[~done]
            rest = ix[~done]

        dists = self.RefineApprox.copy()
        iub = self.feature_names.index("upper bound")
        dists[self.not_computed_mask] = self.features[
            self.not_computed_mask, iub
        ]

        def get_full_ngi_ngd(i):
            isort = np.argsort(dists[self.I[i]])
            ngi = np.sum(self.IJs[self.I[i][isort]], axis=1) - i
            ngd = dists[self.I[i]][isort]
            return np.insert(ngi, 0, i).astype(int), np.insert(ngd, 0, 0)

        res = [get_full_ngi_ngd(i) for i in tq(range(self.nx))]
        ngi = [r[0] for r in res]
        ngd = [r[1] for r in res]
        ebuffer = np.array(
            [
                np.searchsorted(_ngd, _dne - 1e-6)
                for _ngd, _dne in zip(ngd, alpha_dne)
            ]
        )
        buffer = [_ngi[:eb] for _ngi, eb in zip(ngi, ebuffer)]
        ssarr = np.array(
            [
                np.isin(rss, buffer[i], assume_unique=True)
                for i in tq(range(self.nx))
            ]
        )
        a = np.zeros(len(ssarr))
        j = 0
        for i in tq(range(len(rss))):
            del_ssarr = np.delete(ssarr, i - j, axis=1)
            m = np.min(np_sum(del_ssarr, axis=1))
            if m != 0:
                ssarr = del_ssarr
                j += 1
                a[i] = 1

        #
        return np.delete(rss, np.arange(len(ssarr))[a.astype(bool)])

    def alpha_rss(self, y, dne=None, alpha=0):

        if dne is None:
            try:
                dne = self.nearest_enemy_graph[1][:, 0]
            except AttributeError:
                self.get_nearest_enemies(y)
                dne = self.nearest_enemy_graph[1][:, 0]

        ix = np.argsort(dne)
        rss = [ix[0]]

        alpha_dne = dne / (1 + alpha)
        self.rssDs = {}
        for i in ix:
            ds = self.get_exact_ijs(
                self.f, self.X, np.array([[i, r] for r in rss])
            )
            self.rssDs[i] = ds
            dnnR = np.min(ds)
            dne_alpha = alpha_dne[i]
            if (dnnR > dne_alpha) or np.isclose(dnnR, dne_alpha):
                rss.append(i)
        return np.array(rss)


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

        test_parallelisation(
            self.get_exact_ijs, self.f, self.X, self.nx, backend, s=20
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
