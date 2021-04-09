import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import os
from annchor.utils import *


CPU_COUNT = os.cpu_count()


class MaxMinAnchorPicker:
    
    def __init__(self):
        pass
    
    def init_ann(self,ann):
        self.ann = ann
        
    def get_anchors(self):
        ann= self.ann
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

        get_dists = get_dists_(ann.f,ann.low_cpu)
        
        for i in range(na):
            A[i] = ix  
            D[i] = get_dists(ix,ann.f,ann.X,nx)
            ix = np.argmax(np_min(D, 0))
            
        return A,D.T,na*nx

class RandomAnchorPicker:
    
    def __init__(self):
        pass
    
    def init_ann(self,ann):
        self.ann = ann
        
    def get_anchors(self):
        ann= self.ann
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)
        
        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = np.random.choice(ann.nx,size=na,replace=False)

        get_dists = get_dists_(ann.f,ann.low_cpu)
        
        for i in range(na):
            D[i] = get_dists(A[i],ann.f,ann.X,nx)
            
        return A,D.T,na*nx

class SelectedAnchorPicker:
    
    def __init__(self,A):
        self.A=A
        pass
    
    def init_ann(self,ann):
        self.ann = ann
        
    def get_anchors(self):
        ann= self.ann
        nx = ann.nx
        na = ann.n_anchors
        np.random.seed(ann.random_seed)
        
        # D stores distances to anchor points
        # note: at this point D is shape (n_anchors, nx),
        #       but we transpose this after calculations.
        D = np.zeros((na, nx)) + np.infty

        # A stores anchor indices
        A = self.A

        get_dists = get_dists_(ann.f,ann.low_cpu)
        
        for i in range(na):
            D[i] = get_dists(A[i],ann.f,ann.X,nx)
            
        return A,D.T,na*nx      
    

class MaxBoundAnchorPicker:
    
    def __init__(self):
        pass
    
    def init_ann(self,ann):
        self.ann = ann
        
    def get_anchors(self):
        ann= self.ann
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

        get_dists = get_dists_(ann.f,ann.low_cpu)
        
        for i in range(na):
            A[i] = ix  
            D[i] = get_dists(ix,ann.f,ann.X,nx)
            ix = get_approx_njit_W(D.T[:,:i+1],ann.nx)
            #ix = np.argmax(npsum(bounds[:,:,1]-bounds[:,:,0],axis=0))
            
        return A,D.T,na*nx

class SimpleStratifiedSampler:
    
    def __init__(self,
                 partition_feature_name='double anchor distance',
                 n_partitions=7):

        self.partition_feature_name = partition_feature_name
        self.n_partitions = n_partitions
        
    def sample(self,features,feature_names,n_samples,not_computed_mask):
        
        #i_na = feature_names.index('is anchor')
        i_feature = feature_names.index(self.partition_feature_name)
        #no_anchor_mask = features[:,i_na]==0
        sample_feature = features[not_computed_mask][:,i_feature]

        indices = np.arange(not_computed_mask.shape[0])[not_computed_mask]
        
        sorted_sample_feature = np.sort(sample_feature)
        q1 = sorted_sample_feature[int(sample_feature.shape[0]/100)]
        q3 = sorted_sample_feature[int(99*sample_feature.shape[0]/100)]


        sample_bins = np.linspace(q1,q3,self.n_partitions-1)
        sample_bins=np.hstack([-np.infty,sample_bins,np.infty])

        bin_size = n_samples//self.n_partitions
        remainder = n_samples%self.n_partitions


        
        samples = {}
        for nbin in range(self.n_partitions):

            mask = ((sample_feature>=sample_bins[nbin]) *
                    (sample_feature<=sample_bins[nbin+1]))
            n_mask = np.sum(mask)
            if n_mask==0:
                raise SamplingError(
                                'No samples in bin (%5.3f,%5.3f). Reduce n_partitions.'
                                % (sample_bins[nbin],sample_bins[nbin+1]))

            samples[nbin] = np.random.choice(indices[mask],
                                             size=(bin_size+(nbin<remainder)),
                                             replace=False)

        sample_ixs = np.hstack([samples[i] for i in range(self.n_partitions)])

       
        return sample_ixs
        
    

class SamplingError(Exception):
    def __init__(self, message):
        super().__init__(message)

#    def __str__(self):
#        return f'{self.salary} -> {self.message}'


class SimpleStratifiedDistanceRegression:
    def __init__(self,
                 reg_feature_names=['lower bound','upper bound',
                                      'double anchor distance'],
                 partition_feature_name='double anchor distance',
                 n_partitions=7,
                 regression = LinearRegression,
                 regression_kwargs={}):
        
        self.n_partitions = n_partitions
        self.LRs = [regression(**regression_kwargs) for n in range(self.n_partitions)]
        self.partition_feature_name = partition_feature_name
        self.reg_feature_names =  reg_feature_names


        return
    
    def fit(self,sample_features,feature_names,sample_y):
    
        i_partition_feature = feature_names.index(self.partition_feature_name)
        i_features = [i for i,name in enumerate(feature_names) 
                      if name in self.reg_feature_names]

        
        F = sample_features[:,i_partition_feature]
        sorted_sample_feature = np.sort(F)
        q1 = sorted_sample_feature[int(F.shape[0]/100)]
        q3 = sorted_sample_feature[int(99*F.shape[0]/100)]


        sample_bins = np.linspace(q1,q3,self.n_partitions-1)
        self.sample_bins=np.hstack([-np.infty,sample_bins,np.infty])    
        

        F = sample_features[:,i_partition_feature]
        for nbin in range(self.n_partitions):
            mask = ((F>self.sample_bins[nbin]) * (F<=self.sample_bins[nbin+1]))           
            self.LRs[nbin].fit(sample_features[mask][:,i_features],sample_y[mask])
        

    def predict(self,features,feature_names):
        
        i_partition_feature = feature_names.index(self.partition_feature_name)
        i_features = [i for i,name in enumerate(feature_names) 
                      if name in self.reg_feature_names]
        
        X = features[:,i_features]
        y = np.zeros(X.shape[0])
        F = features[:,i_partition_feature]

        def predict_bin(nbin):
            mask = ((F>self.sample_bins[nbin]) * (F<=self.sample_bins[nbin+1]))
            return self.LRs[nbin].predict(X[mask])
        preds = Parallel(n_jobs=CPU_COUNT)(
                    delayed(predict_bin)(nbin) for nbin in range(self.n_partitions)
                )
        
        
        for nbin,pred in enumerate(preds):
            mask = ((F>self.sample_bins[nbin]) * (F<=self.sample_bins[nbin+1]))
            y[mask]=pred
            #self.LRs[nbin].predict(X[mask])
        return y
    
class SimpleStratifiedErrorRegression:
    def __init__(self,
                 partition_feature_name='double anchor distance',
                 n_partitions=7
                ):
        self.n_partitions=n_partitions
        self.partition_feature_name = partition_feature_name
        self.labels = range(n_partitions)
        
    def fit(self,sample_features,feature_names,sample_error):
        
        i_feature = feature_names.index(self.partition_feature_name)


        sample_feature = sample_features[:,i_feature]
        sorted_sample_feature = np.sort(sample_feature)
        q1 = sorted_sample_feature[int(sample_feature.shape[0]/100)]
        q3 = sorted_sample_feature[int(99*sample_feature.shape[0]/100)]


        sample_bins = np.linspace(q1,q3,self.n_partitions-1)
        self.partition_bins=np.hstack([-np.infty,sample_bins,np.infty])


        self.errs = {}#np.zeros(shape=self.n_partitions)
        for nbin in range(self.n_partitions):
            
            mask = ((sample_feature>=self.partition_bins[nbin]) *
                    (sample_feature<=self.partition_bins[nbin+1]))
            err = np.sort(sample_error[mask] )
            self.errs[nbin]=err

    def predict(self,features,feature_names):
        labels = np.empty(shape=features.shape[0]).astype(int)
        i_feature = feature_names.index(self.partition_feature_name)
        feature = features[:,i_feature]

        for nbin in range(self.n_partitions):

            mask = ((feature>=self.partition_bins[nbin]) *
                    (feature<=self.partition_bins[nbin+1]))
            labels[mask]=nbin
        return labels


