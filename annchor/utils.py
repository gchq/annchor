import numpy as np

from numba import njit, prange, types
from numba.typed import Dict
from numba.core.registry import CPUDispatcher

from joblib import Parallel, delayed

import os
from tqdm.auto import tqdm as tq

from annchor.distances import euclidean, levenshtein
from pynndescent.distances import kantorovich
from scipy.spatial.distance import cosine

from multiprocessing.context import TimeoutError

CPU_COUNT = os.cpu_count()


@njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Numba support for numpy funcs with axis option.
    From github user joelrich: (https://github.com/numba/numba/issues/1269).
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


@njit
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)


@njit
def np_argmin(array, axis):
    return np_apply_along_axis(np.argmin, axis, array)


def get_function_from_input(func, func_kwargs):

    if isinstance(func, str):
        allowed_strings = {
            "euclidean": euclidean,
            "cosine": cosine,
            "levenshtein": levenshtein,
            "wasserstein": None,
        }
        assert (
            func in allowed_strings
        ), "Error: The string must be one of {}".format(allowed_strings)

        if func == "wasserstein":
            assert (
                "cost_matrix" in func_kwargs
            ), "Error: wassetstein metric requires cost_function kwarg"

            M = func_kwargs["cost_matrix"]

            @njit()
            def wasserstein(x, y):
                return kantorovich(x, y, cost=M)

            f = wasserstein
        else:
            f = allowed_strings[func]
    else:
        if func_kwargs is None:
            f = func
        else:

            # Handle numba func with kwargs
            if isinstance(func, CPUDispatcher):
                list_kwargs = tuple(func_kwargs.values())

                @njit()
                def f(x, y):
                    return func(x, y, *list_kwargs)

            else:

                def f(x, y):
                    return func(x, y, **func_kwargs)

    return f


def get_exact_ijs_(f, parallel=True, verbose=False, backend="loky"):
    """
    Takes the metric f and returns the function get_exact_ijs(f,X,IJ), which
    calculates the distances between pairs given in array IJ.

    Parameters
    ----------
    f: function
        The metric. Takes two points from the data set and calculates their
        distance.


    Outputs
    -------
    get_exact_ijs: function
        get_exact_ijs(f,X,IJ) is the function that returns distances between
        pairs given in array IJ,
        i.e. np.array([f(X[i],X[j]) for i,j in IJ]).
    """
    if not parallel:

        def get_exact(f, X, IJ):
            def _f(ij):
                i, j = ij
                return f(X[i], X[j])

            fIJ = np.array([_f(ij) for ij in tq(IJ)])

            return fIJ

        return get_exact

    if isinstance(f, CPUDispatcher):

        @njit(parallel=True)
        def get_exact(f, X, IJ):
            exact = np.zeros(len(IJ))
            for ix in prange(len(IJ)):
                i, j = IJ[ix]
                exact[ix] = f(X[i], X[j])
            return exact

    else:
        if verbose:

            def get_exact(f, X, IJ):

                fIJ = np.array(
                    Parallel(n_jobs=CPU_COUNT, backend=backend, timeout=30)(
                        delayed(f)(X[i], X[j]) for i, j in tq(IJ, leave=False)
                    )
                )

                return fIJ

        else:

            def get_exact(f, X, IJ):

                fIJ = np.array(
                    Parallel(n_jobs=CPU_COUNT, backend=backend, timeout=30)(
                        delayed(f)(X[i], X[j]) for i, j in IJ
                    )
                )

                return fIJ

    return get_exact


def test_parallelisation(get_exact_ijs, f, X, nx, backend, s=20):
    try:
        get_exact_ijs(
            f, X, np.random.randint(nx, size=(s, 2))
        )
    except TimeoutError:
        print("TimeoutError: Parallelisation failed.")
        if isinstance(f, CPUDispatcher):
            print(
                "Currently using numba parallelisation, try"
                + " specifying custom parallelistation with"
                + " get_exact_ijs keyword argument."
            )
        elif backend == "loky":
            print(
                "Current backend is 'loky', try backend='multiprocessing',"
                + " or specifying custom parallelistation with"
                + " get_exact_ijs keyword argument."
            )
        elif backend == "multiprocessing":
            print(
                "Current backend is 'multiprocessing', try backend='loky',"
                + " or specifying custom parallelistation with"
                + " get_exact_ijs keyword argument."
            )
        raise TimeoutError()


@njit(parallel=True, fastmath=True)
def get_bounds_njit_ijs(IJs, D):
    """
    Calculates the triangle inequality bounds for pair (i,j).

    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    D: np.array, shape=(nx,na)
        Array of distances to anchor points


    Outputs
    -------
    bounds: np.array, shape=(?,2)
        Array of lower and upper bounds for pairs in IJs.
    """

    n = IJs.shape[0]
    bounds = np.zeros(shape=(n, 2))
    for k in prange(n):
        i = IJs[k][0]
        j = IJs[k][1]
        bounds[k, 0] = np.max(np.abs(D[i] - D[j]))
        bounds[k, 1] = np.min(D[i] + D[j])

    return bounds


@njit()
def get_bounds_alt(disi, disj, dsi, dsj):
    """
    Convoluted function to get upper/lower bounds quickly.
    """
    ub, lb = np.infty, 0
    j0 = 0
    for i in range(disi.shape[0]):
        for j in range(j0, disj.shape[0]):
            if disi[i] <= disj[j]:
                j0 = j
                if disi[i] == disj[j]:
                    a = dsi[i] + dsj[j]
                    b = np.abs(dsi[i] - dsj[j])
                    ub = np.minimum(ub, a)
                    lb = np.maximum(lb, b)
                    j0 += 1
                break

    return lb, ub


@njit(parallel=True)
def update_bounds(IJs, dis, ds):
    """
    Updates the bounds on distances i,j in IJ.

    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    dis: numba.typed.Dict
        Dict of computed pair indices (sorted by dist)
    ds: numba.typed.Dict
        Dict of computed pair distances (sorted by dist)

    Outputs
    -------
    bounds: np.array, shape=(?,2)
        Array of upper/lower bounds for pairs in IJs.
    """

    bounds = np.empty(shape=(IJs.shape))
    for it in prange(IJs.shape[0]):
        i, j = IJs[it]
        a = get_bounds_alt(dis[i], dis[j], ds[i], ds[j])
        bounds[it] = a

    return bounds


@njit(fastmath=True)
def get_dad_ijs(IJs, D):
    """
    Calculates the double anchor distance for pair (i,j).

    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    D: np.array, shape=(nx,na)
        Array of distances to anchor points


    Outputs
    -------
    dad: np.array, shape=(?,)
        Array of double anchor distances for pairs in IJs.
    """
    n = IJs.shape[0]
    dad = np.zeros(shape=(n))
    cA = np_argmin(D, 1)
    for k in range(n):
        i, j = IJs[k]
        dad[k] = D[i, int(cA[j])] + D[j, int(cA[i])]

    return dad / 2


@njit(parallel=True)
def get_nn(nx, nn, RA, IJs, I, not_computed_mask):
    """
    Calculates the nearest neighbor graph.

    Parameters
    ----------
    nx: int
        Number of points in the data set.
    nn: int
        Number of nearest neighbors.
    RA: np.array
        Array of refine approximate distances
    IJs: np.array
        Array of pairs i,j corresponding to the approx distances
    I: dict
        Dictionary mapping indices of the data set to indices in IJs/RA.


    Outputs
    -------
    ngi: np.array, shape=(nx,nn)
        neighbor graph indices.
        ngi[i][j] is the index of the jth closest point to index i.

    ngd: np.array, shape=(nx,nn)
        neighbor graph distances.
        ngd[i][j] is the distance of the jth closest point to index i.

    """
    ngi = np.zeros(shape=(nx, nn - 1), dtype=np.int64)
    ngd = np.zeros(shape=(nx, nn - 1))
    for i in prange(nx):
        Ii = I[np.int64(i)]
        d = RA[Ii]
        mx = np.max(d)
        d[not_computed_mask[Ii]] += mx
        t = np.partition(d, nn - 1)[nn - 1]
        mask = d <= t
        iy = Ii[mask][np.argsort(d[mask])][: nn - 1]

        ngd[i, :] = RA[iy]

        f = IJs[iy]
        mask = f[:, 0] == i
        ngi[i, :] = f[:, 1] * mask + f[:, 0] * (1 - mask)
    return ngi, ngd


@njit
def create_IJs(check, i):
    mask = check[i] > i
    ones = (np.ones(check[i][mask].shape) * i).astype(np.int64)
    IJs = np.vstack((check[i][mask], ones))
    return IJs


@njit
def sample_partition(
    indices, sample_feature, sample_bins, nbin, bin_size, remainder
):
    mask = (sample_feature >= sample_bins[nbin]) * (
        sample_feature < sample_bins[nbin + 1]
    )
    n_mask = np.sum(mask)

    ixmask = indices[mask]
    if ixmask.shape[0] < (bin_size + (nbin < remainder)):
        return ixmask
    return np.random.choice(
        ixmask, size=(bin_size + (nbin < remainder)), replace=False
    )


@njit
def loop_partitions(
    samples,
    indices,
    sample_feature,
    sample_bins,
    nbins,
    bin_size,
    remainder,
    random_seed,
    loop_num,
):
    np.random.seed(random_seed + loop_num)
    for nbin in range(nbins):

        samples[np.int64(nbin)] = sample_partition(
            indices, sample_feature, sample_bins, nbin, bin_size, remainder
        )
    return samples


@njit(parallel=True)
def get_probs(p, labels, errors_ncm, errs):
    prob = np.empty(shape=p.shape)
    for nlabel in prange(labels.shape[0]):
        label = labels[nlabel]
        mask = errors_ncm == label
        prob[mask] = np.searchsorted(errs[label], p[mask])
        prob[mask] /= len(errs[label])
    return prob


@njit(parallel=True)
def check_locality_size(I, nx, nn):
    a = np.zeros(nx)
    for i in range(nx):
        a[i] = (I[i].shape[0]) < nn
    return np.any(a)


@njit()
def argpartition(a, k):
    dxs = np.partition(a, k)[k]
    return np.arange(len(a))[a < dxs]


@njit()
def do_the_thing(nx, ncm, RA, I, nmin):

    l = np.arange(ncm.shape[0])

    for i in range(nx):
        _i = np.int64(i)
        mask = ncm[I[np.int64(_i)]]
        n_computed = np.sum(~mask)
        n_todo = nmin - n_computed
        if n_todo > 0:
            ixs = argpartition(RA[I[_i]][mask], n_todo)  # [:n_todo]
            mapback = l[I[_i]][mask][ixs]
            RA[mapback] = -1

    return RA
