<img align="left" src="https://github.com/gchq/annchor/raw/main/doc/images/logo.svg" width="300">

# ANNchor
A python library implementing ANNchor:<br>
*k*-nearest neighbour graph construction for slow metrics.

## User Guide
For user guide and documentation, go to ```/doc/_build/index.html```

<br></br>

## What is ANNchor?
ANNchor is a python library which constructs approximate *k*-nearest neighbour graphs for slow metrics. 
The *k*-NN graph is an extremely useful data structure that appears in a wide variety of applications, for example: clustering, dimensionality reduction, visualisation and exploratory data analysis (EDA). However, if we want to use a slow metric, these *k*-NN graphs can take an exceptionally long time to compute.
Typical slow metrics include the Wasserstein metric (Earth Mover's distance) applied to images, and Levenshtein (Edit) distance on long strings, where the time taken to compute these distances is significantly longer than a typical Euclidean distance.

ANNchor uses Machine Learning methods to infer true distances between points in a data set from a variety of features derived from anchor points (aka landmarks/waypoints). In practice, this means that ANNchor does not make as many calls to the underlying metric as other state of the art *k*-NN graph generation techniques. This translates to quicker run times, especially when the metric is slow.

Results from ANNchor can easily be combined with other popular libraries in the Data Science community. In the docs we give examples of how to use ANNchor in an EDA pipeline alongside [UMAP](https://github.com/lmcinnes/umap) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan).

## Installation
Clone this repo and install with pip:
```bash
pip install -e annchor/
```

## Basic Usage

```python
import numpy as np
import annchor

X =          #your data, list/np.array of items
distance =   #your distance function, distance(X[i],X[j]) = d

ann = annchor.Annchor(X,
                      distance,
                      n_anchors=15,
                      n_neighbors=15,
                      p_work=0.1)
ann.fit()

print(ann.neighbor_graph)

```

## Examples
Examples can be found in the Examples subfolder.
Extra python packages will be required to run the examples.
These packages can be installed via:
```bash
pip install -r annchor/Examples/requirements.txt
```

