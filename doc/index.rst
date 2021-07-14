.. ANNchor documentation master file, created by
   sphinx-quickstart on Tue Jan 12 12:27:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ANNchor: Fast Approximate *k*\-NN Graph Construction for Slow Metrics
=====================================================================

ANNchor is an algorithm for the fast construction of approximate *k*\-nearest neighbour (*k*\-NN) graphs in a metric space where the metric is slow to compute.
Given a metric space (*X*, *d*), we use anchor points (i.e. a small subset of *X*) and the triangle inequality to construct an approximation to *d* that is fast to compute.
The fast metric is used to quickly identify candidate nearest neighbours, which are then checked by the slow metric. 
In this way, ANNchor aims to use significantly fewer calls to the slow metric than current state-of-the-art techniques for *k*\-NN graph construction (e.g. Nearest Neighbour Descent, HNSW). 
This leads to significant decreases in algorithm run-time when calls to the slow metric are the bottleneck of the computational process.


**Installation**

Install from source (clone the annchor repo from github):

.. code:: bash

    pip install -e ./annchor


.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide
   parameters




.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. toctree::
   :caption: API Reference:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
