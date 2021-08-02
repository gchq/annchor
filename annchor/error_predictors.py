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


class SimpleStratifiedErrorRegression:
    def __init__(
        self, partition_feature_name="double anchor distance", n_partitions=7
    ):
        self.n_partitions = n_partitions
        self.partition_feature_name = partition_feature_name
        self.labels = range(n_partitions)

    def fit(
        self, sample_features, feature_names, sample_error, sample_bins=None
    ):

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
                np.hstack(
                    [
                        self.errs[i],
                        errors[mask][np.abs(errors[mask]) > 0.000001],
                    ]
                )
            )
            self.errs[i] = new_errors
