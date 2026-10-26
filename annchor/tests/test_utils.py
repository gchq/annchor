# (C) Crown Copyright GCHQ
import numpy as np
from joblib import Parallel as JoblibParallel

import annchor.utils as utils


def test_python_distance_parallelisation_has_no_fixed_timeout(monkeypatch):
    parallel_kwargs = []

    def recording_parallel(*args, **kwargs):
        parallel_kwargs.append(kwargs)
        return JoblibParallel(n_jobs=1)

    monkeypatch.setattr(utils, "Parallel", recording_parallel)

    def distance(x, y):
        return abs(x - y)

    X = np.array([0.0, 2.0, 5.0])
    Z = np.array([1.0, 4.0])

    get_exact = utils.get_exact_ijs_(distance)
    result = get_exact(distance, X, np.array([[0, 1], [1, 2]]))
    assert np.array_equal(result, np.array([2.0, 3.0]))

    get_exact_query = utils.get_exact_query_ijs_(distance)
    query_result = get_exact_query(distance, X, Z, np.array([[0, 0], [2, 1]]))
    assert np.array_equal(query_result, np.array([1.0, 1.0]))

    assert len(parallel_kwargs) == 2
    assert all("timeout" not in kwargs for kwargs in parallel_kwargs)
