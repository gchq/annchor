import os
import numpy as np
import time

from numba import njit, prange, types
from numba.typed import Dict
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor as KNNR


from collections import Counter


from annchor.utils import *
from annchor.components import *

class Annchor:      
        
    """Annchor
    
    Quickly computes the approximate k-NN graph for slow metrics
    
    Parameters
    ----------
    X: np.array or list (required)
        The data set for which we want to find the k-NN graph.
    func: function or numba-jitted function (required)
        The metric under which the k-NN graph should be evaluated.  
    n_anchors: int (optional, default 20)
        The number of anchor points. Increasing the number of anchors 
        increases the k-NN graph accuracy at the expense of speed.
    n_neighbors: int (optional, default 15)
        The number of nearest neighbors to compute (i.e. the value of k
        for the k-NN graph).
    n_samples: int (optional, default 5000) 
        The number of sample distances used to compute the error distribution
        (E = dhat-d).
    p_work: float (optional, default 0.1)
        The approximate percentage of brute force calculations which
        we are willing to make.
    sampler: Sampler (optional, default SimpleStratifiedSampler())
        The sampler class which chooses the sample pairs.
    regression: Regression (optional, default SimpleStratifiedDistanceRegression())
        The regression class which predicts distances from features.
    error_predictor: ErrorRegression (optional, default SimpleStratifiedErrorRegression())
        The error regression class which predicts error distributions.
    threshold: int (optional, default None)
        The neighbor threshold at which we prune the candidate nearest
        neighbour set.
    locality: int (optional, default 5)
        The number of anchor points to use in the permutation based k-NN
        (dhat) step.
    loc_thresh: int (optional, default 2)
        The minimum number of anchors in common for which we allow an item
        into NN_x.
    max_checks: int (optional, default nx/5)
        The max number of candidates per index.
    verbose: bool (optional, default False)
        Set verbose=True for more interim output.


    """
      
    def __init__(
        self,
        X,
        func,
        n_anchors=20,
        n_neighbors=15,
        n_samples=5000, 
        p_work=0.1,
        anchor_picker=MaxMinAnchorPicker(),
        sampler=SimpleStratifiedSampler(),
        regression=SimpleStratifiedDistanceRegression(),
        error_predictor=SimpleStratifiedErrorRegression(),
        threshold=None,
        random_seed=42,
        locality=5,
        loc_thresh=2,
        max_checks=None,
        verbose=False,
        low_cpu=False,
        ww=0.5
    ):  

        self.X = X
        self.nx = X.shape[0]
        self.N = (self.nx*(self.nx-1))//2

        self.f = func
        self.evals = 0
        
        self.n_anchors = n_anchors
        self.n_neighbors = n_neighbors
        self.p_work=p_work
        
        if self.n_anchors*self.nx/self.N>self.p_work:
            print('Warning: Too many anchors for specified p_work.')
            self.p_work = 2*self.n_anchors*self.nx/self.N
            print('Increasing p_work to %5.3f.' % self.p_work)
        
        self.n_samples = n_samples
        self.anchor_picker = anchor_picker
        self.sampler = sampler
        self.regression = regression
        self.error_predictor = error_predictor
        self.threshold = self.n_neighbors if (threshold==None) else threshold
        self.random_seed = random_seed
        self.verbose=verbose
        self.low_cpu=low_cpu
        self.locality = locality
        self.loc_thresh = loc_thresh
        self.max_checks = self.nx//5 if (max_checks==None) else max_checks
        
        self.Q = []
        self.LRApprox = None
        self.RefineApprox = None
        self.LRDict = {i: {i: 0} for i in range(self.nx)}
        self.IJ = {}
        self.IJs = []
        self.nn = np.zeros((self.nx, self.n_neighbors))

        self.get_RA = get_RA_(self.f)
        self.get_exact_ijs = get_exact_ijs_(self.f)
        self.ww=ww
        
        
    def get_anchors(self):
        

        '''
        Gets the anchors and distances to anchors. 
        Anchors are stored in self.A, distances in self.D.

        self.A: np.array, shape=(n_anchors,)
            Array of anchor indices.
        self.D: np.array, shape=(nx, n_anchors)
            Array of distances to anchor points.

        '''

        self.anchor_picker.init_ann(self)
        
        self.A,self.D,evals = self.anchor_picker.get_anchors()
                  
        self.evals+=evals
  
       
    def get_features(self):
        
        '''
        Gets the features required for predicting approximate distances.
        
        self.Bounds: np.array, shape=(nx,nx,2)
            Array storing triangle inequality bounds. 
            Bounds[i,j,0] is the lower bound on distance ij.
            Bounds[i,j,1] is the upper bound on distance ij.
            
        self.W: np.array, shape=(nx,nx)
            Array of widths, i.e. difference between upper and lower bounds.
            diff[i,j] is the width of the bounds on distance ij.
            
        self.F: np.array, shape=(nx,nx)
            Array of very approximate double distances. 
            F[i,j] is d(x_i,a_j)+d(x_j,a_i), 
            where a_k is the closest anchor to x_k.
            
            
        self.features: np.array, shape=(nx*(nx-1)/2,?)
            Array of features on which to sample, predict, etc.
            Indices correspond to the upper right diagonal of all-pairs matrices.
            
        '''
    
        
        # Get upper/lower bounds
        # upper bound -> Bounds[:,:,1]
        # lower bound -> Bounds[:,:,0]
        self.Bounds = get_approx_njit(self.D,self.nx)
        
        # Width of bounds
        self.W = self.Bounds[:,:,1]-self.Bounds[:,:,0]
        
        #
        self.F = get_F(self.nx,self.D)
                
        
        anchors = np.zeros(self.nx)
        anchors[self.A[self.A<self.nx]]=1
        IA,JA = np.meshgrid(anchors,anchors)
        
        
        II,JJ = np.meshgrid(np.arange(self.nx),np.arange(self.nx))
        self.features = np.vstack([II[II>JJ].T,
                                   JJ[II>JJ].T,
                                   self.Bounds[II>JJ,:].T,
                                   self.F[II>JJ].T,
                                   (IA+JA)[II>JJ]
                                  ]).T

        self.feature_names = ['i',
                              'j',
                              'lower bound',
                              'upper bound',
                              'double anchor distance',
                              'is anchor'
                             ]
        
        i_is_anchor = self.feature_names.index('is anchor')
        self.not_computed_mask = self.features[:,i_is_anchor]<1
        
    def get_sample(self):
        
        '''
        Gets the sample of pairwise distances on which to train dhat/errors.
        
        self.G: np.array, shape=(n_samples,3)
            Array storing the sample distances and features (for future regression).
            G[i,:-1] are the features for sample pair i.
            G[i,-1] is the true distance for sample pair i.
        '''

        
        self.sample_ixs = self.sampler.sample(self.features,
                                              self.feature_names,
                                              self.n_samples,
                                             self.not_computed_mask)
        self.sample_features = self.features[self.sample_ixs]
                  
        ixs = [i for i,name in enumerate(self.feature_names) if name in ['i','j']]
        self.sample_ijs = sample_ijs = self.sample_features[:,ixs].astype(int)       

        if self.low_cpu:
            # If you have few CPU cores, calling self.get_exact_ijs is slow,
            # better to just iterate without numba assistance
            self.sample_y = np.array([self.f(self.X[ij[0]],
                                             self.X[ij[1]]
                                            ) 
                                      for ij in self.sample_ijs
                                     ])
   
        else:

            self.sample_y = self.get_exact_ijs(self.f,
                                               self.X,
                                               self.sample_ijs)
        
        self.not_computed_mask[self.sample_ixs]=False
        self.evals+=self.sample_y.shape[0]
        
        
    def fit_predict_regression(self):
        #fit

        self.regression.fit(self.sample_features,
                            self.feature_names,
                            self.sample_y)
        
        #predict
        self.pred = self.regression.predict(self.features,
                                            self.feature_names)
        self.sample_predict = self.pred[self.sample_ixs]
        
        self.pred[self.sample_ixs]=self.sample_y
        
        self.LRApprox = np.zeros(shape=(self.nx,self.nx))
            
        II,JJ = np.meshgrid(np.arange(self.nx),np.arange(self.nx))
        self.LRApprox[II>JJ] = self.pred
        self.LRApprox += self.LRApprox.T
        
        self.LRApprox = np.clip(
            self.LRApprox, self.Bounds[:, :, 0], self.Bounds[:, :, 1]
        )
        
        # Instantiate the nx*nx all-pairs refined distance matrix
        # This will be updated later with refinements
        if not np.any(self.RefineApprox):
            self.RefineApprox = self.LRApprox.copy()
        else:
            c = self.RefineApprox[II>JJ][~self.not_computed_mask]
            nc = self.pred[self.not_computed_mask]
            e = np.zeros(shape=self.N)
            e[self.not_computed_mask] = nc
            e[~self.not_computed_mask] = c
            e[self.sample_ixs]=self.sample_y
            self.RefineApprox = np.zeros(shape=(self.nx,self.nx))
            self.RefineApprox[II>JJ] = e
            self.RefineApprox += self.RefineApprox.T
            print(self.RefineApprox[20,126])
            
        
    def fit_predict_errors(self):
        
        self.error_predictor.fit(self.sample_features,
                                 self.feature_names,
                                 self.sample_y-self.sample_predict
                                )
                
        self.errors = self.error_predictor.predict(self.features,
                                                   self.feature_names)
        #self.errors = np.zeros(shape=(self.nx,self.nx))
        #II,JJ = np.meshgrid(np.arange(self.nx),np.arange(self.nx))
        #self.errors[II>JJ] = error_predictor.predict(self.features[:,4])
        #self.errors += self.errors.T
        
    def select_candidate_pairs(self,w=0.5):

        II,JJ = np.meshgrid(np.arange(self.nx),np.arange(self.nx))

        rasort  = np.sort(self.RefineApprox,axis=1)[:,self.threshold]



        p0 = (rasort - self.RefineApprox )[II>JJ][self.not_computed_mask]
        p1 = (rasort - self.RefineApprox ).T[II>JJ][self.not_computed_mask]
        p = np.max(np.vstack([p0,p1]),axis=0)
        prob = np.empty(shape=p.shape)
        for label in self.error_predictor.labels:
            mask = self.errors[self.not_computed_mask]==label
            prob[mask] = np.searchsorted(self.error_predictor.errs[label],
                                         p[mask])
            prob[mask]/=len(self.error_predictor.errs[label])



        ixs = [i for i,name in enumerate(self.feature_names) if name in ['i','j']]

        p_work = self.p_work
        N = self.nx*(self.nx-1)/2
        max_refine = int((p_work*N-self.n_anchors*self.nx-self.n_samples)*w)
        
        max_refine = 0 if (max_refine<0) else max_refine
            
        
        self.candidates = np.argsort(-prob)[:max_refine]

        
        self.IJs = self.features[self.not_computed_mask][self.candidates][:,ixs].astype(int)
        self.IJs = list(map(tuple,self.IJs))
           
        self._IJS = self.features[self.not_computed_mask][self.candidates][:,ixs].astype(int)
        mapback = np.arange(self.not_computed_mask.shape[0])[self.not_computed_mask][self.candidates]
        self.not_computed_mask[mapback]=False
    
    def refine_candidate_pairs(self):
        '''
        Iterate over the pairs in self.IJ and evaluate them exactly.
        
        self.IJs: list of tuples
            The list of candidate pairs which we will refine, i.e. calculate the
            exact distance.
            
        self.RefineApprox: np.array, shape=(nx,nx)
            Array to hold the refined distances. 
            Initially a copy of LRApprox, RefineApprox will be updated to hold 
            true distances for pairs deemed close enough to potentially be nearest
            neighbours.
        '''
        if len(self.IJs)>0:
            self.RefineApprox = self.get_RA(self.f, self.X, np.array(self.IJs),self.RefineApprox)

        self.evals +=len(self.IJs)


            
    def get_ann(self):
        
        # Get the nn-graph. Can probably optimise this more.

        nnixs = np.argsort(self.RefineApprox, axis=1)[:, : self.n_neighbors]
        I = np.meshgrid(np.arange(self.nx), np.arange(self.nx))[1][
            :, : self.n_neighbors
        ]
        
        self.neighbor_graph = (nnixs, self.RefineApprox[I, nnixs])
        return nnixs, self.RefineApprox[I, nnixs]

    
    def fit(self):
        """fit

            Calls the ANNchor algorithm for the parameters supplied to the class.
            
            Once called, the neighbor graph is available in Annchor.neighbor_graph.

        """
        start = time.time()

        if self.verbose: print('computing anchors...')
        self.get_anchors()
        if self.verbose: print('get_anchors:', time.time()-start)

        if self.verbose: print('computing features...')
        self.get_features()
        if self.verbose: print('get_features:', time.time()-start)

        if self.verbose: print('computing sample...')
        self.get_sample()
        if self.verbose: print('get_sample:', time.time()-start)

        if self.verbose: print('fitting regression...')
        self.fit_predict_regression()
        if self.verbose: print('fit_predict_regression:', time.time()-start)

        if self.verbose: print('fitting errors...')
        self.fit_predict_errors()
        if self.verbose: print('fit_predict_errors:', time.time()-start)
            
        if self.verbose: print('selecting candidate pairs (1)')
        self.select_candidate_pairs()
        if self.verbose: print('select_candidate_pairs:', time.time()-start)

        if self.verbose: print('refining candidate pairs (1)')
        self.refine_candidate_pairs()
        if self.verbose: print('refine_candidate_pairs:', time.time()-start)
            
        if self.verbose: print('selecting candidate pairs (2)')
        self.select_candidate_pairs()
        if self.verbose: print('select_candidate_pairs:', time.time()-start)

        if self.verbose: print('refining candidate pairs (2)')
        self.refine_candidate_pairs()
        if self.verbose: print('refine_candidate_pairs:', time.time()-start)
            
        if self.verbose: print('generating neighbour graph')
        self.get_ann()
        if self.verbose: print('get_ann:', time.time()-start)


### Low mem
#      The low mem routines try the algorithm without storing things in nx*nx matrices.
#      Necessary when nx gets big (>7000 or so).

    def get_sample_lm(self):
        
        '''
        Get the sample pairs and their bounds and true distances.
        
        self.G: np.array, shape=(n_samples,3)
            Array storing the sample distances and bounds (for future regression).
            G[i,0] is the lower bound for sample pair i.
            G[i,1] is the upper bound for sample pair i.
            G[i,2] is the true distance for sample pair i.
        '''


        start = time.time()
        np.random.seed(self.random_seed)
        n = self.n_samples
        self.G = np.zeros((n, 3))


        non_anchors = np.array([ix for ix in set(np.arange(self.nx)).difference(set(self.A))])
        IJs = np.array([np.random.choice(non_anchors, replace=False, size=2) for k in range(n)])
        approx = get_approx_njit_ijs(IJs,self.D)
        self.G[:,:2] = approx
        self.G[:,2] = self.get_exact_ijs(self.f, self.X, IJs)
        #for k in range(n):
        #    i, j = np.random.choice(non_anchors, replace=False, size=2)
        #    G[k][:2] = [ab.Approx[i, j, 0], ab.Approx[i, j, 1]]
        #    d = ab.f(ab.X[i], ab.X[j])
        #    G[k][2] = d
        #    ab.Q.append((i, j, d))

        self.evals += n
        #print('get_sample: %5.3f' % (time.time()-start))


    def get_lr_partitions_lm(self):
        
        ''' 
        Finds the error distributions.
        Low memory partitions based on Width alone for now
            
        self.LR: sklearn.linear_model.LinearRegression
            Linear Regression model fitted by samples in G.
            
        self.dts: List of floats
            Partition edges for width space.
            E.g. partition 0 holds widths between dts[0] and dts[1],
                 partition 1 holds widths between dts[1] and dts[2].
                 etc
                 
        self.hs: List of arrays
            List containing the distribution of errors for different partitions.
            hs[i] is the sorted errors for sample pairs belonging to partition i.
            
   
        '''
        start = time.time()


        # Fit linear regression according to samples stored in G
        self.LR = self.regression
        self.LR.fit(self.G[:, :2], self.G[:, 2])

        # Calculate the widths (diff)
        diff = self.G[:,1]-self.G[:,0]
        ixs = np.argsort(diff)
        ng = len(ixs)

        splits=self.partitions
        
        # Partition based on linear splitting of the distribution of widths
        self.dts = [0]+[diff[ixs[i*ng//splits]] for i in range(1,splits)]+[np.infty]

        # Calculate the error distributions in each partition
        self.hs = []
        for ix in [ ixs[i*ng//splits:(i+1)*ng//splits] for i in range(splits)]:
        #ix = np.argsort(G[:, 2])[: n // 3]
            r = np.clip(self.LR.predict(self.G[ix, :2]),self.G[ix, 0],self.G[ix, 1]) - self.G[ix, 2]
        #ab.r1, ab.r2 = np.mean(r), np.std(r)
            self.hs.append(np.sort(r))
        #print('get_lr_partitions: %5.3f' % (time.time()-start))


    def get_locality_lm(self):
        
        '''
        Uses basic permutation/set method to find candidate nearest neighbours.
        
        Current Technique (Use something better in future):
            For each point i, find the set S_i of its nearest l=locality anchor points.
            For each other point j, calculate (S_i intersect S_j).
            Only consider pairs ij where |(S_i intersect S_j)|>=loc_thresh.
            
        self.check: Dict, keys=int64, val=int64[:]
            check[i] is the array of candidate nearest neighbour indices for index j.
        
        '''
        start = time.time()

        na = self.n_anchors
        nx = self.nx

        # locality is number of nearest anchors to use in set
        # locality_thresh is number of elements in common required to consider a pair of elements for nn candidacy
        locality = self.locality
        loc_thresh=self.loc_thresh
        sid = np.argsort(self.D,axis=1)[:,:locality]

        # Store candidate pairs in check
        # check[i] is a list of indices that are nn candidates for index i
        check = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )

        ix = np.arange(nx)
        A = np.zeros((na,nx)).astype(int)
        for i in prange(sid.shape[0]):
            for j in sid[i]:
                A[j,i]=1
        for i in prange(nx):
            check[i] = ix[np.sum(A[sid[i],:],axis=0)>=loc_thresh]  

        self.check = check
        #print('get_locality: %5.3f' % (time.time()-start))
            


    def get_LRApprox_lo_mem(self):
        
        # In the low mem version, the approximate distances are stored in a long list
        
        '''
        Get the approximate distances
        
        self.IXs: np.array, shape=(nx,max_checks)
            Array of candidate nn-pair indices. 
            IXs[i,j] is the index of the jth closest element to i according to the approximation.
            A value of IXs[i,j]=-1 denotes that there were fewer than j elements in check[i].
            
        self.LRApprox: np.array, shape=(nx,max_checks)
            Array of approx distances. 
            LRApprox[i,j] is the approx distance between index i and index IXs[i,j].
            
        self.thresh: np.array, shape=(nx,)
            Array of thresholds (i.e. distance of the k-th approx nearest neighbour.

        '''
        self.get_sample_lm()


        self.get_lr_partitions_lm() 

        self.get_locality_lm()

        s = time.time()
        IJs = []
        ix_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],
        )
        k=0
        for i in (range(self.nx)):
            q = self.check[i]
            IJs.append(np.vstack([(i+np.zeros(q.shape)).astype(int),q]))
            ix_dict[i] = (np.arange(k,len(q)+k))
            k+=len(q)
        IJs = np.hstack(IJs).T

        #print('get IJs and indices %5.3f' % (time.time()-s) )

        Approx = get_approx_njit_ijs(IJs,self.D)
        #print('get approx IJs %5.3f' % (time.time()-s) )


        diff = Approx[:,1]-Approx[:,0]

        coef = self.LR.coef_
        intercept = self.LR.intercept_

        d_hat = np.sum(Approx*coef,axis=1)+intercept
        #print('do lin reg %5.3f' % (time.time()-s) )

        d_hat_adj=d_hat.copy()

        #adjust d by partition #thistakeslongest
        for i,h in enumerate(self.hs):
            nh = h.shape[0]
            adj = h[int(self.min_prob*nh)]
            mask1 = d_hat_adj>self.dts[i]
            mask2 = d_hat_adj<self.dts[i+1] 
            d_hat_adj[mask1*mask2]+=adj

        #print('adjust by partition %5.3f' % (time.time()-s) )



        b1 = (d_hat<Approx[:,0])
        b2 = (d_hat>Approx[:,1])
        d_hat = b1*Approx[:,0]+b2*Approx[:,1]+(1-b1)*(1-b2)*d_hat

        #print('clip by bounds %5.3f' % (time.time()-s) )

        #from numba import prange,njit
        self.IXs = np.zeros(shape=(self.nx,self.max_checks),dtype=int)-1
        self.LRApprox = np.zeros(shape=(self.nx,self.max_checks))+9999
        self.thresh = np.empty(shape=(self.nx))


        get_LRApprox_lomem(d_hat_adj,
                            d_hat,
                            self.n_neighbors,
                            self.IXs,
                            self.LRApprox,
                            self.thresh,
                            self.check,
                            ix_dict)
        #print('get_LRApprox_lomem %5.3f' % (time.time()-s) )


    def get_candidate_nns_lo_mem(self):
        
        '''
            Refines the approximation by calculating true distances for
            all pairs deemed close enough to be nearest neighbours.
        '''
        IYs = []

        for i in (range(self.nx)):
            q = self.IXs[i][self.LRApprox[i]<self.thresh[i]]
            IYs.append(np.vstack([(i+np.zeros(q.shape)).astype(int),q]))

        IYs = np.hstack(IYs)

        mask = IYs[0]>IYs[1]
        IJs = np.hstack([np.flipud(IYs[:,mask]),(IYs[:,~mask])])
        IJs = IJs[:,IJs[0]!=IJs[1]]
        IJs = lexsort_based(IJs.T)

        exact = self.get_exact_ijs(self.f,self.X,IJs)
        self.evals += IJs.shape[0]
        nns,dists = search_evaluated_ijs(self.n_neighbors,self.nx,IJs,exact)

        nns = np.hstack([np.reshape(np.arange(self.nx),(-1,1)),nns])
        dists = np.hstack([np.reshape(np.zeros(self.nx),(-1,1)),dists])

        self.neighbor_graph = nns,dists
            
            
    def fit_lo_mem(self):
        
        '''
        Finds the approx nearest neighbour graph in the low memory regime.
        '''
        start = time.time()
        if self.verbose: print('computing anchors...')
        self.get_anchors()
        if self.verbose: print('get_anchors:', time.time()-start)
            
        if self.verbose: print('getting LRApprox...')
        self.get_LRApprox_lo_mem()
        if self.verbose: print('get_LRApprox_lo_mem:', time.time()-start)

        if self.verbose: print('evaluating cnns...')
        self.get_candidate_nns_lo_mem()
        if self.verbose: print('get_candidate_nns_lo_mem:', time.time()-start)


    
    # Plotting & Diagnostics

    def plot_anchor_distance_decay(self, ax):

        max_min_decay = [
            np.max(np.min(self.D[:, :i], axis=1)) for i in range(1, self.n_anchors + 1)
        ]
        median_decay = [
            np.median(np.min(self.D[:, :i], axis=1))
            for i in range(1, self.n_anchors + 1)
        ]
        ax.plot(max_min_decay, "-o")
        ax.plot(median_decay, "-x")
        return

    def plot_LR_approx(self, ax):
        ix = np.argsort(self.G[:, 2])
        ax.plot(self.G[ix])
        ax.plot(self.LR.predict(self.G[:, :2])[ix])
        return

    
    
    

class BruteForce:      
        
    """BruteForce
    
    Computes the approximate k-NN graph by brute force
    
    Parameters
    ----------
    X: np.array or list (required)
        The data set for which we want to find the k-NN graph.
    func: function or numba-jitted function (required)
        The metric under which the k-NN graph should be evaluated.  

    """
       
    def __init__(
        self,
        X,
        func
    ):    
        
        self.X = X
        self.nx = X.shape[0]
        self.f = func
        
        

    def get_neighbor_graph(self):
        """get_neighbor_graph
    
            Gets the k-NN graph from the all pairs distance matrix

        """
        if "numba" in str(type(self.f)):
            
            
            @njit(parallel=True)
            def get_exact(f, X, IJ):
                nx=X.shape[0]
                exact = np.zeros(shape=(nx,nx))
                for ix in prange(len(IJ)):
                    i, j = IJ[ix]
                    exact[i,j] = exact[j,i] = f(X[i], X[j])
                return exact
            IJs = np.array([(i,j) for i in range(self.nx-1) for j in range(i+1,self.nx)])
            self.D = get_exact(self.f,self.X,IJs)
            self.neighbor_graph = np.argsort(self.D,axis=1),np.sort(self.D,axis=1)
            
        else:
            
            CPU_COUNT = os.cpu_count()

            def get_exact(f, X, IJ):

                def _f(ij):
                    i,j=ij
                    return f(X[i],X[j])

                return np.array(Parallel(n_jobs=CPU_COUNT)(delayed(_f)(ij) for ij in IJ))

            IJs = [(i,j) for i in range(self.nx-1) for j in range(i+1,self.nx)]
            dists = get_exact(self.f,self.X,IJs)
            self.D = np.zeros(shape=(self.nx,self.nx))
            for ij,d in zip(IJs,dists):
                i,j = ij
                self.D[i,j] = self.D[j,i] = d
            self.neighbor_graph = np.argsort(self.D,axis=1),np.sort(self.D,axis=1)


def compare_neighbor_graphs(nng_1, nng_2, n_neighbors):

    """compare_neighbor_graphs
    
    Compares accuracy of k-NN graphs. The second graph is compared against the first.
    This measure of accuracy accounts for cases where the indices differ but the distances
    are equivalent. 
    
    e.g. if nng_1[0][0]=[0, 1, 2, 3], nng_1[0][1]=[0, 1, 1, 2],
    
    and     nng_2[0][0]=[0, 1, 2, 4], nng_1[0][1]=[0, 1, 1, 2],
    
    There would be zero incorrect NN pairs, since both ix=3 and ix=4 are valid 4th nearest neighbors.
    
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
        a = Counter( np.round(nng_1[1][ix][:n_neighbors],3).astype(np.float32) )
        b = Counter( np.round(nng_2[1][ix][:n_neighbors],3).astype(np.float32) )
        h.append(len(a-b))
    err = np.sum(h)

    return err

        
        
    
