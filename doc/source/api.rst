ANNchor API Guide
=================

ANNchor contains two classes, the ANNchor base class :class:`~annchor.Annchor`, and a
brute force class :class:`~annchor.BruteForce`.

Annchor
-------

.. autoclass:: annchor.Annchor
   :members:

.. autoclass:: annchor.BruteForce
   :members:

Utility Functions
----------------------

The annchor package also includes some utility functions, e.g. finding 
neighbor graph accuracy.

.. autofunction:: annchor.compare_neighbor_graphs


Data sets
---------

ANNchor comes with various data sets for benchmarking.

.. automodule:: annchor.datasets
   :members:

