import numpy as np
import time
from annchor.datasets import load_digits as load_data




k=15

X = load_data()['X']
y = load_data()['y']
neighbor_graph = load_data()['neighbor_graph']

nx = X.shape[0]


from annchor.distances import wasserstein
from annchor import Annchor
from annchor import BruteForce
from annchor import compare_neighbor_graphs
from pynndescent import NNDescent



def call_bruteforce(seed,params):
    start_time = time.time()

    bruteforce = BruteForce(X,wasserstein)
    bruteforce.get_neighbor_graph()

    t = time.time()-start_time

    error = compare_neighbor_graphs(neighbor_graph,
                                    bruteforce.neighbor_graph,
                                    k)

    return t, error

def call_pynn(seed,params):
    start_time = time.time()

    # Call nearest neighbour descent
    nndescent = NNDescent(X,
                          n_neighbors=k,
                          metric=wasserstein,
                          random_state=seed,
                          **params)

    t = time.time()-start_time


    # Test accuracy
    error = compare_neighbor_graphs(neighbor_graph,
                                    nndescent.neighbor_graph,
                                    k)
    return t, error


def call_annchor(seed,params):
    start_time = time.time()

    # Call ANNchor
    ann = Annchor(X,
                  wasserstein,
                  n_neighbors=k, 
                  random_seed=seed,
                  n_samples=1000,
                  partitions=1,
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





runs = [(call_pynn,{'n_trees':11,'n_iters':15,'max_candidates':None}),
        (call_pynn,{'n_trees':7,'n_iters':11,'max_candidates':None}),
        (call_pynn,{'n_trees':7,'n_iters':11,'max_candidates':5}),
        (call_pynn,{'n_trees':5,'n_iters':5,'max_candidates':5}),
        (call_annchor,{'n_anchors':25,'min_prob':0.5}),
        (call_annchor,{'n_anchors':25,'min_prob':0.2}),
        (call_annchor,{'n_anchors':25,'min_prob':0.1}),
        (call_annchor,{'n_anchors':25,'min_prob':0.01}),
        (call_bruteforce, {})
         ]
seed=0



for it in range(20):
    print('Commencing iteration %d' % it)
    with open('runs/benchmark_wasserstein_%d.txt' % k,'a') as f:
        for run in runs:
            call,params = run
            t,error = call(seed,params)

            seed+=1
            
            name = str(call.__name__).split('_')[1]
            out = '%-15s|%-8.2f|%-8d|%-60s' % (name,t,error,str(params))
            print(out)
            f.write(out+'\n')


