import numpy as np

from numba import njit, prange, types
from numba.typed import Dict
from numba.core.registry import CPUDispatcher

from annchor.utils import *


def get_query_anchor_dists(ann, Q):
    nq = len(Q)
    na = ann.n_anchors
    IJs = np.array([[i, j] for j in range(nq) for i in range(na)])
    D = ann.get_exact_query_ijs(ann.f, ann.X[ann.A], Q, IJs)
    return D.reshape((nq, na))


def get_query_locality(ann, QD):
    na = ann.n_anchors
    nq = len(QD)
    ix = np.arange(ann.nx, dtype=np.int64)
    # locality is number of nearest anchors to use in set
    # locality_thresh is number of elements in common required
    # to consider a pair of elements for nn candidacy
    locality = ann.locality
    loc_thresh = ann.loc_thresh
    sid = np.argsort(QD, axis=1)[:, :locality]

    check = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )

    for i in prange(nq):
        check[i] = ix[np.sum(ann.Amatrix[sid[i], :], axis=0) >= ann.loc_thresh]
    #
    return check


def get_query_features(ann, QD, check):
    nq = len(QD)
    IJs = np.hstack(
        [
            np.stack([check[i], np.ones(check[i].shape) * i]).astype(int)
            for i in range(nq)
        ]
    ).T

    QI = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )

    nijs = np.arange(len(IJs))

    csum = np.cumsum(np.insert([len(check[key]) for key in check], 0, 0))

    for i in range(nq):
        QI[i] = nijs[csum[i] : csum[i + 1]]

    check = get_query_locality(ann, QD)
    dad = get_query_dad_ijs(IJs, ann.D, QD)
    bounds = get_query_bounds_njit_ijs(IJs, ann.D, QD)
    anchors = np.isin(IJs[:, 0], ann.A)
    Qfeatures = np.vstack([bounds.T, dad, anchors]).T
    Q_not_computed_mask = Qfeatures[:, 3] < 1
    return IJs, QI, Qfeatures, Q_not_computed_mask


@njit(fastmath=True)
def get_query_dad_ijs(IJs, D, QD):
    """
    Calculates the double anchor distance for query pair (i,j).


    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    D: np.array, shape=(nx,na)
        Array of distances to anchor points
    QD: np.array, shape=(nx,na)
        Array of query distances to anchor points

    Outputs
    -------
    dad: np.array, shape=(?,)
        Array of double anchor distances for pairs in IJs.
    """
    n = IJs.shape[0]
    dad = np.zeros(shape=(n))
    cA = np_argmin(D, 1)
    cQA = np_argmin(QD, 1)

    for k in range(n):
        i, j = IJs[k]
        dad[k] = D[i, int(cQA[j])] + QD[j, int(cA[i])]

    return dad / 2


@njit(parallel=True, fastmath=True)
def get_query_bounds_njit_ijs(IJs, D, QD):
    """
    Calculates the triangle inequality bounds for query pair (i,j).

    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    D: np.array, shape=(nx,na)
        Array of distances to anchor points
    QD: np.array, shape=(nx,na)
        Array of query distances to anchor points

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
        bounds[k, 0] = np.max(np.abs(D[i] - QD[j]))
        bounds[k, 1] = np.min(D[i] + QD[j])

    return bounds


def select_refine_candidate_query_pairs(
    ann, IJs, Q, QI, QRA, Qncm, Qerrors, p_work, nn
):

    # nn = ann.n_neighbors
    nq = len(Q)

    thresh = np.array([np.partition(QRA[QI[i]], nn)[nn] for i in range(nq)])

    QRA = guarantee_nmin(
        nq,
        Qncm,
        QRA,
        QI,
        3 * nn // 2,
    )
    # p0 = (thresh[IJs[:, 0]] - QRA)[Qncm]
    p = (thresh[IJs[:, 1]] - QRA)[Qncm]
    # p = np.max(np.vstack([p0, p1]), axis=0)

    errs = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:],
    )
    for label in ann.error_predictor.errs:
        errs[label] = ann.error_predictor.errs[label]
    prob = get_probs(
        p,
        np.array(ann.error_predictor.labels),
        Qerrors[Qncm],
        errs,
    )

    # n_refine = (4*nn*nq)//2 #fix this

    nbf = nq * ann.nx
    na = ann.n_anchors * nq

    n_refine = int((p_work * nbf - na)) + 1

    candidates = np.argpartition(-prob, n_refine)[:n_refine]

    mapback = np.arange(Qncm.shape[0])[Qncm][candidates]

    exact = ann.get_exact_query_ijs(ann.f, ann.X, Q, IJs[mapback])
    QRA[mapback] = exact
    Qncm[mapback] = False

    return QRA, Qncm


def query_(ann, Q, nn=15, p_work=0.3, get_exact_query_ijs=None):

    if get_exact_query_ijs is None:
        if ann.get_exact_query_ijs is None:
            ann.get_exact_query_ijs = get_exact_query_ijs_(
                ann.f, verbose=ann.verbose, backend=ann.backend
            )
    else:
        ann.get_exact_query_ijs = get_exact_query_ijs

    nq = len(Q)
    QD = get_query_anchor_dists(ann, Q)
    check = get_query_locality(ann, QD)
    IJs, QI, Qfeatures, Q_not_computed_mask = get_query_features(
        ann, QD, check
    )
    Qpred = ann.regression.predict(Qfeatures, ann.feature_names)
    ilb = ann.feature_names.index("lower bound")
    iub = ann.feature_names.index("upper bound")
    Qpred = np.clip(Qpred, Qfeatures[:, ilb], Qfeatures[:, iub])
    Qerrors = ann.error_predictor.predict(Qfeatures, ann.feature_names[:-1])
    QRA = Qpred.copy()

    QRA, Q_not_computed_mask = select_refine_candidate_query_pairs(
        ann, IJs, Q, QI, QRA, Q_not_computed_mask, Qerrors, p_work, nn
    )

    ngi, ngd = get_nn(nq, nn + 1, QRA, IJs, QI, Q_not_computed_mask)

    return ngi, ngd


###############


def legacy_query_(ann, Z, get_exact_query_ijs=None, k=5, alpha=1.4, beta=1.4):

    if get_exact_query_ijs is None:
        if ann.get_exact_query_ijs is None:
            ann.get_exact_query_ijs = get_exact_query_ijs_(
                ann.f, verbose=ann.verbose, backend=ann.backend
            )
    else:
        ann.get_exact_query_ijs = get_exact_query_ijs

    DQP = ann.D
    DP = ann.D[ann.A]

    As, Ds, lMs, nevals = query_dm(
        Z,
        ann.X[ann.A],
        DP,
        ann.f,
        ann.get_exact_query_ijs,
        k=k,
        alpha=alpha,
        init=0,
    )

    DD = np.array(
        [np.linalg.norm(DQP[:, As[i]] - Ds[i], axis=1) for i in range(len(Z))]
    )
    isort = np.argsort(DD)

    def collect(i, R, Q, DD, isort, f, get_exact_query_ijs, k):
        ix = np.searchsorted(DD[i, isort[i]] / DD[i, isort[i][k]], beta)
        nni, ndi = isort[i][:ix], get_exact_query_ijs(
            f, R, Q, np.array([[i, j] for j in isort[i][:ix]])
        )
        dsort = np.argsort(ndi)
        return nni[dsort][:k], ndi[dsort][:k]  # ,ix

    res = [
        collect(i, Z, ann.X, DD, isort, ann.f, ann.get_exact_query_ijs, k)
        for i in range(len(Z))
    ]
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def query_dm(Q, P, DP, f, query_parallel, k=0, alpha=1.2, init=0):

    As = {}
    Ds = {}
    lMs = {}
    nevals = 0

    nq = len(Q)
    mp = len(P)
    ix = np.arange(nq)

    a = np.zeros((nq, 1)).astype(int) + init
    ijs = np.array([[i, _a] for i, _a in zip(ix, a[:, -1])])
    da = query_parallel(f, Q, P, ijs)
    nevals += len(ijs)

    dm = da[:, np.newaxis]
    di = a[:, -1].copy()

    M = (da[:, np.newaxis] - DP[a[:, -1], :])[:, :, np.newaxis]
    lM = np.linalg.norm(M, axis=2)
    a1 = np.argmin(lM, axis=1)

    mask = np.any(a1[:, np.newaxis] == a, axis=1)  # a1==a[:,-1]

    # print(ix[mask],dm[mask],di[mask])
    for i, _a, _d, _lM in zip(ix[mask], a[mask], dm[mask], lM[mask]):
        As[i] = _a
        Ds[i] = _d
        lMs[i] = _lM

    for loop in range(mp):
        ix = ix[~mask]
        if len(ix) == 0:
            break
        a = np.concatenate([a[~mask], a1[~mask, np.newaxis]], axis=1)

        M = M[~mask]

        dm = dm[~mask]
        ijs = np.array([[i, _a] for i, _a in zip(ix, a[:, -1])])
        da = query_parallel(f, Q, P, ijs)
        nevals += len(ijs)

        dm = np.concatenate([dm, da[:, np.newaxis]], axis=1)
        M1 = (da[:, np.newaxis] - DP[a[:, -1], :])[:, :, np.newaxis]
        M = np.concatenate([M, M1], axis=2)
        lM = np.linalg.norm(M, axis=2)
        a1 = np.argmin(lM, axis=1)

        mask = np.any(a1[:, np.newaxis] == a, axis=1)  # a1==a[:,-1]

        for i, _a, _d, _lM in zip(ix[mask], a[mask], dm[mask], lM[mask]):
            As[i] = _a
            Ds[i] = _d
            lMs[i] = _lM
    itodo = {
        i: np.arange(mp)[lMs[i] < (lMs[i][np.argsort(lMs[i])[k]] * alpha)]
        for i in range(nq)
    }
    itodo = {
        i: itodo[i][np.isin(itodo[i], As[i], assume_unique=True, invert=True)]
        for i in range(nq)
    }
    litodo = np.cumsum([0] + [len(itodo[i]) for i in range(nq)])
    todo = np.array([[i, j] for i in range(nq) for j in itodo[i]])
    dtodo = query_parallel(f, Q, P, todo)
    nevals += len(todo)

    dtodo = {i: dtodo[litodo[i] : litodo[i + 1]] for i in range(nq)}
    for i in range(nq):
        Ds[i] = np.hstack([Ds[i], dtodo[i]])
        isort = np.argsort(Ds[i])
        As[i] = np.hstack([As[i], itodo[i]])[isort]
        Ds[i] = Ds[i][isort]

    return As, Ds, lMs, nevals


#######
