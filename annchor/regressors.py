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


class SimpleStratifiedLinearRegression:
    def __init__(
        self,
        reg_feature_names=[
            "lower bound",
            "upper bound",
            "double anchor distance",
        ],
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
            i
            for i, name in enumerate(feature_names)
            if name in self.reg_feature_names
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
            mask = (F > self.sample_bins[nbin]) * (
                F <= self.sample_bins[nbin + 1]
            )
            self.LRs[nbin].fit(
                sample_features[mask][:, i_features], sample_y[mask]
            )

    def predict(self, features, feature_names):

        i_partition_feature = feature_names.index(self.partition_feature_name)
        i_features = [
            i
            for i, name in enumerate(feature_names)
            if name in self.reg_feature_names
        ]

        X = features[:, i_features]
        y = np.zeros(X.shape[0])
        F = features[:, i_partition_feature]

        def predict_bin(nbin):
            mask = (F > self.sample_bins[nbin]) * (
                F <= self.sample_bins[nbin + 1]
            )
            return self.LRs[nbin].predict(X[mask])

        preds = Parallel(n_jobs=CPU_COUNT)(
            delayed(predict_bin)(nbin) for nbin in range(self.n_partitions)
        )

        for nbin, pred in enumerate(preds):
            mask = (F > self.sample_bins[nbin]) * (
                F <= self.sample_bins[nbin + 1]
            )
            y[mask] = pred
            # self.LRs[nbin].predict(X[mask])
        return y
