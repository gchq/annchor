Basic Parameters
================

``n_anchors``
~~~~~~~~~~~~~
The number of anchor points (waypoints) we wish to use. The default value is 20, and should increase with the complexity of the underlying metric space.

Picking an optimal value of ``n_anchors`` is more of an art than a science at this stage. For simple metric spaces we can get away with an extremely low number of anchors; for example ``n_anchors=5`` does pretty well for Euclidean in 2D! For most use cases we suggest between 20 and 30 anchor points. Remember that adding more anchor points will use more metric evaluations - there is a trade off to be made between picking more anchors, and evaluating more potential nearest neighbors. Given a fixed value of ``p_work`` (more on this later) the number of calls to the metric is also fixed at ``n_evals = n_anchors*nx + n_samples + n_refine``, thus increasing ``n_anchors`` reduces ``n_samples`` and vice versa.

``n_neighbors``
~~~~~~~~~~~~~~~
Simply the number of nearest neighbors we are looking to find (i.e. the *k* in *k*\-NN).

``n_samples``
~~~~~~~~~~~~~
ANNchor works by trying to approximate the metric given a variety of features (e.g. lower/upper bound). Under the hood this uses a standard machine learning approach, which naturally requires training. The parameter ``n_samples`` specifies how many pairwise distances we evaluate in order to create a training set on which to learn the metric. If this parameter is too small then we risk not accurately learning the underlying metric. Too big and we end up wasting valuable calls to the metric on non-nearest neighbor pairs.

``p_work``
~~~~~~~~~~
At its heart, ANNchor wants to make as few calls to an expensive metric as possible. Of course, the more calls to the metric we make, the more accurate we will be (at the expense of run time). The ``p_work`` parameter allows us to control this trade off.
In the brute force case we would make ``N=nx*(nx-1)//2`` calls to the metric, and ANNchor will make at most ``p_work*N`` calls to the metric, i.e. ``p_work`` is the percentage of brute force work we are willing to do. 

``locality`` and ``loc_thresh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These parameters specify how we cut down the number of pairwise distances that may be considered throughout the algorithm. We use a simple waypointing mechanism: For each point find the set of its ``locality`` nearest anchors, and only consider pairs which share at least ``loc_thresh`` anchors between their sets.

``verbose``
~~~~~~~~~~~
Simple boolean flag determines whether or not we print diagnostics/progress messages.

Advanced Parameters
===================
For most use cases the parameters described in the previous section will suffice. However, there are some other parameters available which allow greater control over the ANNchor algorithm.

``anchor_picker``
~~~~~~~~~~~~~~~~~
The ``anchor_picker`` parameter allows you to supply you own anchor picker class. This class determines how anchors are sampled from the data. By default we use a max-min algorithm that picks anchors sequentially, aiming to pick the next anchor to be the point with the largest distance to an existing anchor.


``sampler``
~~~~~~~~~~~
The ``sampler`` parameter allows you to supply your own sampler class. This tells ANNchor how to decide which pairs of points to pick for the sampling (training) phase.

``regression``
~~~~~~~~~~~~~~
We can specify the machine learning algorithm to train on the features of the sample points. By default this is a stratified linear regression, but could in principle be any ml algorithm. Remember that this needs to be much faster than the slow metric in order to succeed.

``error_predictor``
~~~~~~~~~~~~~~~~~~~
The ``error_predictor`` class tries to predict the error associated with the regression, which is useful for assigning probabilities to predictions.

``is_metric``
~~~~~~~~~~~~~
This boolean argument should be set to false when our metric is not a true metric (in particular whether it violates the triangle inequality). This can sometimes be the case for approximate metrics. If the metric approximate, ANNchor needs to take better care of the upper and lower triangle inequality bounds since they may no longer represent reality.

``get_exact_ijs``
~~~~~~~~~~~~~~~~~
This parameter allows the user to specify their own parallelisation for computing large numbers of pairwise distances. 

