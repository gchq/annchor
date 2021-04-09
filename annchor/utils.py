import numpy as np
from numba import njit, prange, types
from numba.typed import Dict

from joblib import Parallel, delayed

import os


from sklearn.linear_model import LinearRegression


CPU_COUNT = os.cpu_count()

@njit
def np_apply_along_axis(func1d, axis, arr):
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

def get_dists_(f,low_cpu):

    if "numba" in str(type(f)):
        
        if low_cpu:
            def get_dists(ix,f,X,nx):
                return np.array([f(X[j],X[ix]) for j in prange(nx)])

        else:
            @njit(parallel=True)
            def get_dists(ix,f,X,nx):
                d = np.empty(shape=(nx))
                for j in prange(nx):
                    d[j] = f(X[j],X[ix])
                return d
        
    else:
        def get_dists(ix,f,X,nx):
            
            def _f(j):  
                return f(X[j],X[ix])

            d = np.array(Parallel(n_jobs=CPU_COUNT)(delayed(_f)(j) for j in range(nx)))
            #d = np.empty(shape=(nx))
            #for j in prange(nx):
            #    d[j] = f(X[j],X[ix])
            return d
        
    return get_dists


def get_RA_(f):
    
    if "numba" in str(type(f)):

        @njit(parallel=True)
        def get_RA(f, X, IJ, RefineApprox):
            for ix in prange(len(IJ)):
                i, j = IJ[ix]
                RefineApprox[i, j] = RefineApprox[j, i] = f(X[i], X[j])
            return RefineApprox

    else:

        def get_RA(f, X, IJ, RefineApprox):
            
            def _f(ij):
                i,j=ij
                return f(X[i],X[j])

            fIJ = Parallel(n_jobs=CPU_COUNT)(delayed(_f)(ij) for ij in IJ)
            
            for ij,f in zip(IJ,fIJ):
                i, j = ij
                RefineApprox[i, j] = RefineApprox[j, i] = f
                
            #for ix in prange(len(IJ)):
            #    i, j = IJ[ix]
            #    RefineApprox[i, j] = RefineApprox[j, i] = f(X[i], X[j])
            return RefineApprox

    return get_RA

def get_exact_ijs_(f):
    
    if "numba" in str(type(f)):

        @njit(parallel=True)
        def get_exact(f, X, IJ):
            exact = np.zeros(len(IJ))
            for ix in prange(len(IJ)):
                i, j = IJ[ix]
                exact[ix] = f(X[i], X[j])
            return exact

    else:

        def get_exact(f, X, IJ):
            
            def _f(ij):
                i,j=ij
                return f(X[i],X[j])
            
            fIJ = np.array(Parallel(n_jobs=CPU_COUNT)(delayed(_f)(ij) for ij in IJ))
            
            #exact = np.zeros(len(IJ))
            #for ix in prange(len(IJ)):
            #    i, j = IJ[ix]
            #    exact[ix] = f(X[i], X[j])
            return fIJ

    return get_exact



@njit(parallel=True,fastmath=True)
def get_approx_njit(D,nx):
    Approx = np.zeros(shape=(nx, nx, 2))
    for i in prange(nx-1):
        for j in prange(i+1,nx):
            lb = np.max(np.abs(D[i] - D[j]))
            ub = np.min(D[i] + D[j])


            Approx[i,j,0] = lb
            Approx[j,i,0] = lb
            Approx[i, j, 1] = ub
            Approx[j,i,1] = ub

    return Approx


@njit(parallel=True,fastmath=True)
def get_approx_njit_W(D,nx):
    Approx = np.zeros(shape=(nx, nx, 2))
    for i in prange(nx-1):
        for j in prange(i+1,nx):
            lb = np.max(np.abs(D[i] - D[j]))
            ub = np.min(D[i] + D[j])


            Approx[i,j,0] = lb
            Approx[j,i,0] = lb
            Approx[i, j, 1] = ub
            Approx[j,i,1] = ub

    return np.argmax(np_sum(Approx[:,:,1]-Approx[:,:,0],0))

@njit(parallel=True,fastmath=True)
def get_approx_njit_ijs(ijs,D):
    n = ijs.shape[0]
    Approx = np.zeros(shape=(n,2))
    for k in prange(n):
        i = ijs[k][0]
        j = ijs[k][1]
        Approx[k,0] = np.max(np.abs(D[i] - D[j]))
        Approx[k,1] = np.min(D[i] + D[j])

    return Approx


@njit(fastmath=True)
def get_approx_njit_ij(i,j,D):
    approx=np.zeros(2)
    approx[0]=np.max(np.abs(D[i] - D[j]))
    approx[1]=np.min(D[i] + D[j])
    return approx


@njit(fastmath=True)
def lrf(i,j,D,coef,intercept):
    approx = get_approx_njit_ij(i,j,D)
    d= np.sum(approx*coef)+intercept
    b1 = (d<approx[0])
    b2 = (d>approx[1])


    return approx[0],approx[1],b1*approx[0] + b2*approx[1] + (1-b1)*(1-b2)*d

@njit()
def f(i,d_hat_adj,d_hat,k,IXs,LRApprox,thresh,check,ix_dict):
    # check are ijs to be checked
    # d_hat_adj array of vals corresponding to 
    dsort = np.argsort((d_hat_adj[ix_dict[i]]))  # argsorts d_hat_adj
    nn = check[i][dsort]                         # sorts check by d_hat_adj
    a0 = IXs.shape[1]   
    a1 = nn.shape[0]
    IXs[i,:a1] = nn[:a0]                         # top a0 indices go into IXs
    LRApprox[i,:a1] = d_hat_adj[ix_dict[i]][dsort][:a0] # top a0/a1 distances into LRApprox
    thresh[i] = np.sort(d_hat[ix_dict[i]])[k]

@njit(parallel=True)
def get_LRApprox_lomem(d_hat_adj,d_hat,k,IXs,LRApprox,thresh,check,ix_dict):
    n = LRApprox.shape[0]
    for i in prange(n):
        f(i,d_hat_adj,d_hat,k,IXs,LRApprox,thresh,check,ix_dict)
        

def lexsort_based(data):
    #From stackoverflow: https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array/31097277
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

@njit(parallel=True)
def search_evaluated_ijs(k,nx,IJs,exact):
    nns = np.empty(shape=(nx,k-1),dtype=np.int32)
    dists = np.empty(shape=(nx,k-1))


    for ix in prange(nx):
        mask0 = IJs[:,0]==ix
        mask1 = IJs[:,1]==ix

        exact_ix = np.hstack((exact[mask0],exact[mask1]))
        isort = np.argsort(exact_ix)[:(k-1)]
        nns[ix] = np.vstack((IJs[mask0],np.fliplr(IJs[mask1]))).T[1][isort]
        dists[ix] = exact_ix[isort]
    return nns,dists

@njit
def np_argmin(array, axis):
    return np_apply_along_axis(np.argmin, axis, array)


@njit(fastmath=True)
def get_F(nx,D):
    F = np.zeros(shape=(nx,nx))
    cA = np_argmin(D,1)
    for i in range(nx):
        for j in range(nx):
            F[i,j] = D[i,int(cA[j])]
    F = F+F.T
    return F/2


