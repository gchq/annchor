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

@njit
def np_argmin(array, axis):
    return np_apply_along_axis(np.argmin, axis, array)



def get_dists_(f,low_cpu):
    '''
    Takes the metric f and returns the function get_dists(ix,f,X,nx), which
    calculates the distance from all data points to X[ix]. 
    Useful for getting distances to anchor points.
    
    Parameters
    ----------
    f: function
        The metric. Takes two points from the data set and calculates their distance.
    low_cpu: bool
        Flag which shows whether the user wants low_cpu mode (some numba functions slow on few cores).
        
    Outputs
    -------
    get_dists: function
        get_dists(ix,f,X,nx) is the function that returns an array of distances to data point X[ix],
        i.e. np.array([f(X[ix],y) for y in X]).
    '''

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
            return d
        
    return get_dists


def get_exact_ijs_(f):
    '''
    Takes the metric f and returns the function get_exact_ijs(f,X,IJ), which
    calculates the distances between pairs given in array IJ.
    
    Parameters
    ----------
    f: function
        The metric. Takes two points from the data set and calculates their distance.

        
    Outputs
    -------
    get_exact_ijs: function
        get_exact_ijs(f,X,IJ) is the function that returns distances between pairs given in array IJ,
        i.e. np.array([f(X[i],X[j]) for i,j in IJ]).
    '''
    
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
            
            return fIJ

    return get_exact


@njit(parallel=True,fastmath=True)
def get_bounds_njit_ijs(IJs,D):
    '''
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
    '''
    
    n = IJs.shape[0]
    bounds = np.zeros(shape=(n,2))
    for k in prange(n):
        i = IJs[k][0]
        j = IJs[k][1]
        bounds[k,0] = np.max(np.abs(D[i] - D[j]))
        bounds[k,1] = np.min(D[i] + D[j])

    return bounds


@njit(fastmath=True)
def get_dad_ijs(IJs,D):
    '''
    Calculates the double anchor distance for pair (i,j).
    
    Parameters
    ----------
    IJs: np.array, shape=(?,2)
        Array of i,j pairs
    D: np.array, shape=(nx,na)
        Array of distances to anchor points

        
    Outputs
    -------
    dad: np.array, shape=(?,2)
        Array of double anchor distances for pairs in IJs.
    '''
    n=IJs.shape[0]
    dad = np.zeros(shape=(n))
    cA = np_argmin(D,1)
    for k in range(n):
        i,j = IJs[k]
        dad[k] = D[i,int(cA[j])]+D[j,int(cA[i])]
    
    return dad/2


@njit(parallel=True)
def get_nn(nx,nn,RA,IJs,I):
    '''
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

    '''
    ngi = np.zeros(shape = (nx,nn),dtype=np.int64)
    ngd = np.zeros(shape = (nx,nn))
    for i in (prange(nx)):
        Ii = I[np.int64(i)]
        d = RA[Ii]
        t = np.partition(d,nn)[nn] 
        mask = d<t
        iy = Ii[mask][np.argsort(d[mask])]

        ngd[i,:] = RA[iy]

        f = IJs[iy]
        mask = f[:,0]==i
        ngi[i,:] = f[:,1]*mask+f[:,0]*(1-mask)
    return ngi,ngd

@njit
def create_IJs(check,i):
    mask = check[i]>i
    ones = (np.ones(check[i][mask].shape)*i).astype(np.int64)
    IJs = np.vstack((check[i][mask],
                     ones
                    ))
    return IJs
   
@njit
def sample_partition(
        indices,
        sample_feature,
        sample_bins,
        nbin,
        bin_size,
        remainder):
    mask = ((sample_feature>=sample_bins[nbin]) *
            (sample_feature<=sample_bins[nbin+1]))
    n_mask = np.sum(mask)

    return np.random.choice(indices[mask],
                           size=(bin_size+(nbin<remainder)),
                            replace=False)
@njit(parallel=True)
def loop_partitions(
        samples,
        indices,
        sample_feature,
        sample_bins,
        nbins,
        bin_size,
        remainder):

    for nbin in prange(nbins):

        samples[np.int64(nbin)] = sample_partition(
                            indices,
                            sample_feature,
                            sample_bins,
                            nbin,
                            bin_size,
                            remainder)
    return samples

@njit(parallel=True)
def get_probs(p,labels,errors_ncm,errs):
    prob = np.empty(shape=p.shape)
    for nlabel in prange(labels.shape[0]):
        label=labels[nlabel]
        mask = errors_ncm==label
        prob[mask] = np.searchsorted(errs[label],
                                     p[mask])
        prob[mask]/=len(errs[label])
    return prob