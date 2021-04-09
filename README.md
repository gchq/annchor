<img align="left" src="https://github.com/gchq/annchor/raw/main/doc/images/logo.svg" width="300">

# ANNchor
A python library implementing ANNchor:<br>
*k*-nearest neighbour graph construction for slow metrics.

## User Guide
For user guide and documentation, go to ```/doc/_build/index.html```

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
