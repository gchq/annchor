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


class NothingToSample(Exception):
    pass


class Sampler(ABC):
    """
    Abstract base class for Samplers.
    They need to implement two methods, get_partition and sample_partition:
     * get_partition(sample_feature, n_samples) should return a
     pair (sample_bins, new_n_samples)
     * sample_partition should return the sample indices

    This base class implements sample_partition in a simple way, so descendants
    may either implement get_partition only, or optionally override
    sample_partition as well.
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
        self,
        features,
        feature_names,
        n_samples,
        not_computed_mask,
        random_seed,
    ):
        if not not_computed_mask.any():
            raise NothingToSample()

        i_feature = feature_names.index(self.partition_feature_name)
        sample_feature = features[not_computed_mask][:, i_feature]
        indices = np.arange(not_computed_mask.shape[0])[not_computed_mask]

        sample_bins, new_n_samples = self.get_partition(
            sample_feature, n_samples
        )
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
    def __init__(
        self, partition_feature_name="double anchor distance", n_partitions=7
    ):
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
                "Warning: n_samples too large for data set size.\n"
                + "Reducing n_samples to %d." % n_samples
            )

        q1 = np.partition(sample_feature, iq1)[iq1]
        q3 = np.partition(sample_feature, iq3)[iq3]

        sample_bins = np.linspace(q1, q3, self.n_partitions - 1)
        sample_bins = np.hstack([-np.infty, sample_bins, np.infty])
        return sample_bins, n_samples


class ClusterSampler(Sampler):
    def __init__(
        self, partition_feature_name="double anchor distance", n_partitions=5
    ):
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
