import numpy as np
import time
import Levenshtein as lev
import os
from annchor.datasets import load_strings




k=15

def levdist(a,b):
    return lev.distance(a,b)

strings_data = load_strings()
X = strings_data['X']
y = strings_data['y']
neighbor_graph = strings_data['neighbor_graph']

nx = X.shape[0]

from annchor import Annchor
from annchor import BruteForce
from annchor import compare_neighbor_graphs
import nmslib





def call_bruteforce(seed,params):
    start_time = time.time()

    bruteforce = BruteForce(X,levdist)
    bruteforce.get_neighbor_graph()

    t = time.time()-start_time

    error = compare_neighbor_graphs(neighbor_graph,
                                    bruteforce.neighbor_graph,
                                    k)

    return t, error

def call_hnsw(seed,params):
    start_time = time.time()

    # specify some parameters
    CPU_COUNT = os.cpu_count()

    index_time_params = params
    
    index_time_params['indexThreadQty']=CPU_COUNT

    
    #{'M': 20,
    #'efConstruction': 100}

    # create the index
    index = nmslib.init(method='hnsw',
                        space='leven',
                        dtype=nmslib.DistType.INT,
                        data_type=nmslib.DataType.OBJECT_AS_STRING)

    index.addDataPointBatch(data=list(X))
    index.createIndex(index_time_params,print_progress=True)

    # query the index
    res = index.knnQueryBatch(list(X), k=k, num_threads=CPU_COUNT)
    hnsw_neighbor_graph = [np.array([x[0]for x in res]),np.array([x[1]for x in res])]

    t = time.time()-start_time


    # Test accuracy
    error = compare_neighbor_graphs(neighbor_graph,
                                    hnsw_neighbor_graph,
                                    k)
    return t, error


def call_annchor(seed,params):
    start_time = time.time()

    # Call ANNchor
    ann = Annchor(X,
                  levdist,
                  n_neighbors=k, 
                  random_seed=seed,
                  n_samples=3000,
                  partitions=3,
                  **params
                 )
                 # n_anchors=25
                 # min_prob=0.02,
                 # random_seed=0,
                 # n_samples=1000,
                 # partitions=1)

    ann.fit()

    t = time.time()-start_time

    # Test accuracy
    error = compare_neighbor_graphs(neighbor_graph,
                                    ann.neighbor_graph,
                                    k)

    return t, error


runs = [(call_hnsw,{'efConstruction':50,'M':15,'post':0}),
        (call_hnsw,{'efConstruction':50,'M':15,'post':2}),
        (call_hnsw,{'efConstruction':100,'M':20,'post':0}),
        (call_hnsw,{'efConstruction':100,'M':20,'post':2}),
        (call_annchor,{'n_anchors':15,'min_prob':0.2}),
        (call_annchor,{'n_anchors':20,'min_prob':0.15}),
        (call_annchor,{'n_anchors':20,'min_prob':0.1}),
        (call_annchor,{'n_anchors':20,'min_prob':0.02}),
        (call_bruteforce, {})
         ]
seed=0



for it in range(20):
    print('Commencing iteration %d' % it)
    with open('runs/benchmark_levenshtein.txt','a') as f:
        for run in runs:
            call,params = run
            t,error = call(seed,params)

            seed+=1
            
            name = str(call.__name__).split('_')[1]
            out = '%-15s|%-8.2f|%-8d|%-60s' % (name,t,error,str(params))
            print(out)
            f.write(out+'\n')




