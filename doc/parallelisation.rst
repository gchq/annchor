Parallelisation
===============

The main bottleneck that Annchor encounters is the computation of many calls to a given metric.
Fortunately, these calculations are embarrassingly parallel, which makes our life a little easier.

However, parallelisation is a tricky beast. There's rarely a one-size-fits-all approach to selecting the best parallelisation.
There are many factors at play, e.g. the software/hardware that is available to you, or the details of your particular metric.

Annchor tries its best to select an appropriate parallelisation for you (e.g. joblib, numba), but we realise
that we will never understand a specific use case better than you, the user. The user is familiar with their metric,
and their computer architecture, and thus is best placed to decide the parallelisation.

Thus, you can supply your own parallelisation to Annchor. This is done through the ``get_exact_ijs`` keyword.
Specifically you can provide a function as described below:

.. code:: python3

    def my_custom_parallelisation(f, X, IJs):
      """
      Parameters
      ----------
      f: your metric (function f(X[i],X[j]) = r for some real r>=0)
      X: your data set
      IJs: a numpy array of index pairs (indices to pairs of items in X, to be evaluated by f)
      """

      # Custom parallelisation code here.
      # should return equivalent to
      result = np.array([f(X[i], X[j]) for i,j in IJs])

      return result

It is worth checking that your custom parallelisation works as expected prior to running Annchor.
You should run something like the following as a quick sanity check:

.. code:: python3

    f = # your function here
    X = # your data here

    nx = len(X)
    s = 20 # number of items on which to test parallelisation
    IJs = np.random.randint(nx, size=(s, 2))

    serial = np.array([f(X[i], X[j]) for i,j in IJs])

    parallel = my_custom_parallelisation(f, X, IJs)

    assert np.isclose(serial, parallel)

Note that annchor will run a quick test to ensure the parallelisation works, but will not currently check that it returns the expected result as run in serial.

We recommend that the user takes some time to experiment and determine the best parallelisation for their metric/architecture combination, especially if
the expected runtime is high.
