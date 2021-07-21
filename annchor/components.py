from abc import ABC, abstractmethod

import numpy as np


from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import os
from annchor.utils import *
from numba import njit, prange, types
from numba.typed import Dict

from tqdm.auto import tqdm as tq

CPU_COUNT = os.cpu_count()

########################################################
# Annchor Pickers


class MaxMinAnchorPicker:
    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = np.zeros(na).astype(int)
        ix = np.random.randint(nx)

        if ann.verbose:
            v = lambda f: tq(f)
        else:
            v = lambda f: f

        for i in v(range(na)):
            A[i] = ix
            IJs = np.array([[ix, j] for j in range(ann.nx)])
            D[i] = ann.get_exact_ijs(ann.f, ann.X, IJs)
            ix = np.argmax(np_min(D, 0))

        return A, D.T, na * nx


class ExternalAnchorPicker:
    def __init__(self, A):
        self.A = A
        self.is_anchor_safe = False

    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        if ann.verbose:
            v = lambda f: tq(f)
        else:
            v = lambda f: f

        for i in v(range(na)):
            D[i] = np.array([ann.f(x, self.A[i]) for x in ann.X])

        return np.array([]), D.T, na * nx


class SelectedAnchorPicker:
    def __init__(self, A):
        self.A = A

    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = self.A

        IJ = np.array([[i, j] for i in A for j in range(nx)])
        D = ann.get_exact_ijs(ann.f, ann.X, IJ)
        D = D.reshape(na, nx)

        return A, D.T, na * nx

class RandomAnchorPicker:
    def get_anchors(self, ann: "Annchor"):
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)

        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = np.random.choice(np.arange(nx),na,replace=False)

        IJ = np.array([[i, j] for i in A for j in range(nx)])
        D = ann.get_exact_ijs(ann.f, ann.X, IJ)
        D = D.reshape(na, nx)

        return A, D.T, na * nx

########################################################
# Samplers


class NothingToSample(Exception):
    pass


class Sampler(ABC):
    """
    Abstract base class for Samplers.
    They need to implement two methods, get_partition and sample_partition:
     * get_partition(sample_feature, n_samples) should return a pair (sample_bins, new_n_samples)
     * sample_partition should return the sample indices

    This base class implements sample_partition in a simple way, so descendants may
    either implement get_partition only, or optionally override sample_partition as well.
    """

    def __init__(self, partition_feature_name, n_partitions):
        self.partition_feature_name = partition_feature_name
        self.n_partitions = n_partitions
        self.loop_num = 0

    @abstractmethod
    def get_partition(self, sample_feature, new_samples):
        pass

    def sample_partition(
        self, indices, n_samples, sample_feature, sample_bins, random_seed
    ):
        bin_size = n_samples // self.n_partitions
        remainder = n_samples % self.n_partitions

        samples = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )
        samples = loop_partitions(
            samples,
            indices,
            sample_feature,
            sample_bins,
            self.n_partitions,
            bin_size,
            remainder,
            random_seed,
            self.loop_num,
        )
        self.loop_num += 1

        for i in range(self.n_partitions):
            if len(samples[i]) < 2:
                raise Exception("Some sampler bins contain too few samples")

        sample_ixs = np.hstack([samples[i] for i in range(self.n_partitions)])

        return sample_ixs

    def sample(
        self, features, feature_names, n_samples, not_computed_mask, random_seed
    ):
        if not not_computed_mask.any():
            raise NothingToSample()

        i_feature = feature_names.index(self.partition_feature_name)
        sample_feature = features[not_computed_mask][:, i_feature]
        indices = np.arange(not_computed_mask.shape[0])[not_computed_mask]

        sample_bins, new_n_samples = self.get_partition(sample_feature, n_samples)
        if new_n_samples != n_samples:
            print(
                "Warning: n_samples has changed from %d to %d."
                % (n_samples, new_n_samples)
            )
        n_samples = new_n_samples

        if n_samples == 0:
            raise NothingToSample()

        sample_ixs = self.sample_partition(
            indices, n_samples, sample_feature, sample_bins, random_seed
        )

        if n_samples != sample_ixs.shape[0]:
            print("Warning: Some bins contained fewer samples than requested")

        return sample_ixs, sample_ixs.shape[0], sample_bins


class SimpleStratifiedSampler(Sampler):
    def __init__(self, partition_feature_name="double anchor distance", n_partitions=7):
        super().__init__(partition_feature_name, n_partitions)

    def get_partition(self, sample_feature, n_samples):
        n = sample_feature.shape[0]
        iq1 = int(n / 100)
        iq3 = int(99 * n / 100)

        if (iq1 * self.n_partitions) < n_samples:
            iq1 = int(n / 10)
            iq3 = int(9 * n / 10)

        if (iq1 * self.n_partitions) < n_samples:
            n_samples = iq1 * self.n_partitions
            print(
                "Warning: n_samples too large for data set size. Reducing n_samples to %d."
                % n_samples
            )

        q1 = np.partition(sample_feature, iq1)[iq1]
        q3 = np.partition(sample_feature, iq3)[iq3]

        sample_bins = np.linspace(q1, q3, self.n_partitions - 1)
        sample_bins = np.hstack([-np.infty, sample_bins, np.infty])
        return sample_bins, n_samples


class ClusterSampler(Sampler):
    def __init__(self, partition_feature_name="double anchor distance", n_partitions=5):
        super().__init__(partition_feature_name, n_partitions)

    def get_partition(self, sample_feature, n_samples):
        n = sample_feature.shape[0]

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.n_partitions)
        labels = kmeans.fit_predict(sample_feature.reshape(-1, 1))

        partitions = np.array(
            [
                [
                    np.min(sample_feature[labels == i]),
                    np.max(sample_feature[labels == i]),
                ]
                for i in range(self.n_partitions)
            ]
        )

        partitions = np.sort(partitions.flatten())
        sample_bins = partitions[1:-1:2]
        sample_bins = np.hstack([-np.infty, sample_bins, np.infty])
        return sample_bins, n_samples


class SamplingError(Exception):
    def __init__(self, message):
        super().__init__(message)


########################################################
# Error Predictors


class SimpleStratifiedLinearRegression:
    def __init__(
        self,
        reg_feature_names=["lower bound", "upper bound", "double anchor distance"],
        partition_feature_name="double anchor distance",
        n_partitions=7,
    ):
        # regression = LinearRegression,
        # regression_kwargs={}):

        self.n_partitions = n_partitions
        self.LRs = [LinearRegression() for n in range(self.n_partitions)]
        self.partition_feature_name = partition_feature_name
        self.reg_feature_names = reg_feature_names

        return

    def fit(self, sample_features, feature_names, sample_y, sample_bins=None):

        i_partition_feature = feature_names.index(self.partition_feature_name)
        i_features = [
            i for i, name in enumerate(feature_names) if name in self.reg_feature_names
        ]

        F = sample_features[:, i_partition_feature]

        if sample_bins is None:
            n = F.shape[0]
            iq1 = int(n / 100)
            iq3 = int(99 * n / 100)
            q1 = np.partition(F, iq1)[iq1]
            q3 = np.partition(F, iq3)[iq3]

            sample_bins = np.linspace(q1, q3, self.n_partitions - 1)
            self.sample_bins = np.hstack([-np.infty, sample_bins, np.infty])
        else:
            self.n_partitions = sample_bins.shape[0] - 1
            self.sample_bins = sample_bins

        for nbin in range(self.n_partitions):
            mask = (F > self.sample_bins[nbin]) * (F <= self.sample_bins[nbin + 1])
            self.LRs[nbin].fit(sample_features[mask][:, i_features], sample_y[mask])

    def predict(self, features, feature_names):

        i_partition_feature = feature_names.index(self.partition_feature_name)
        i_features = [
            i for i, name in enumerate(feature_names) if name in self.reg_feature_names
        ]

        X = features[:, i_features]
        y = np.zeros(X.shape[0])
        F = features[:, i_partition_feature]

        def predict_bin(nbin):
            mask = (F > self.sample_bins[nbin]) * (F <= self.sample_bins[nbin + 1])
            return self.LRs[nbin].predict(X[mask])

        preds = Parallel(n_jobs=CPU_COUNT)(
            delayed(predict_bin)(nbin) for nbin in range(self.n_partitions)
        )

        for nbin, pred in enumerate(preds):
            mask = (F > self.sample_bins[nbin]) * (F <= self.sample_bins[nbin + 1])
            y[mask] = pred
            # self.LRs[nbin].predict(X[mask])
        return y


class SimpleStratifiedErrorRegression:
    def __init__(self, partition_feature_name="double anchor distance", n_partitions=7):
        self.n_partitions = n_partitions
        self.partition_feature_name = partition_feature_name
        self.labels = range(n_partitions)

    def fit(self, sample_features, feature_names, sample_error, sample_bins=None):

        i_feature = feature_names.index(self.partition_feature_name)

        sample_feature = sample_features[:, i_feature]

        if sample_bins is None:
            n = sample_feature.shape[0]
            iq1 = int(n / 100)
            iq3 = int(99 * n / 100)
            q1 = np.partition(sample_feature, iq1)[iq1]
            q3 = np.partition(sample_feature, iq3)[iq3]

            sample_bins = np.linspace(q1, q3, self.n_partitions - 1)
            self.partition_bins = np.hstack([-np.infty, sample_bins, np.infty])
        else:
            self.n_partitions = sample_bins.shape[0] - 1
            self.partition_bins = sample_bins

        self.errs = {}  # np.zeros(shape=self.n_partitions)
        for nbin in range(self.n_partitions):

            mask = (sample_feature >= self.partition_bins[nbin]) * (
                sample_feature <= self.partition_bins[nbin + 1]
            )
            err = np.sort(sample_error[mask])
            self.errs[nbin] = err

    def predict(self, features, feature_names):
        labels = np.empty(shape=features.shape[0]).astype(int)
        i_feature = feature_names.index(self.partition_feature_name)
        feature = features[:, i_feature]

        for nbin in range(self.n_partitions):

            mask = (feature >= self.partition_bins[nbin]) * (
                feature <= self.partition_bins[nbin + 1]
            )
            labels[mask] = nbin
        return labels

    def update_errors(self, errors, partitions):

        for i in range(self.n_partitions):

            mask = partitions == i

            new_errors = np.sort(
                np.hstack([self.errs[i], errors[mask][np.abs(errors[mask]) > 0.000001]])
            )
            self.errs[i] = new_errors
