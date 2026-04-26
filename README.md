## SNN: Fast and exact fixed-radius neighbor search

[![!pypi](https://img.shields.io/pypi/v/snnpy?color=white)](https://pypi.org/project/snnpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/532659733.svg)](https://zenodo.org/doi/10.5281/zenodo.10275013)


SNN is a fast and exact fixed-radius nearest neighbor search algorithm [1]. It uses the first principal component of the data to prune the search space and speeds up Euclidean distance computations using high-level BLAS routines. SNN is implemented in native Python. On many problems, SNN is faster than KDtree and Balltree in the scikit-learn package. 

### Reproducibility

To reproduce the experiments from the paper [1], see the instructions and code in the `exp` subfolder.

### Python installation

The native Python implementation of SNN can be installed by:

```sh
pip install snnpy
```

If you build from source and need a custom OpenBLAS location, you can pass paths via environment variables:

```sh
# Linux/macOS
OPENBLAS_DIR=/path/to/OpenBLAS pip install .
# or explicitly:
OPENBLAS_INCLUDE_DIR=/path/to/OpenBLAS/include OPENBLAS_LIB_DIR=/path/to/OpenBLAS/lib pip install .
# Optional: set library basename (default on Linux/macOS is "blas")
OPENBLAS_LIB_NAME=openblas OPENBLAS_DIR=/path/to/OpenBLAS pip install .
```

```powershell
# Windows (PowerShell)
$env:OPENBLAS_DIR='D:\path\to\OpenBLAS'
pip install .
# or explicitly:
$env:OPENBLAS_INCLUDE_DIR='D:\path\to\OpenBLAS\include'
$env:OPENBLAS_LIB_DIR='D:\path\to\OpenBLAS\lib'
# Optional: override library basename (default on Windows is "libopenblas")
$env:OPENBLAS_LIB_NAME='libopenblas'
pip install .
```

### Usage

```python
import numpy as np
from snnpy import *
from time import time

n_samples = 100000
n_dim =  100
radius = 3.5
rng = np.random.RandomState(0)
X = rng.random_sample((n_samples, n_dim))  

# build SNN model
st = time()
snn_model = build_snn_model(X)  
print("SNN index time:", time()-st)
# will be faster if return_dist is False, then no distance information come out

# query neighbors of X[0]
st = time()
ind,dist = snn_model.query_radius(X[0], radius, return_distance=True)
# If remove the returning of the associated distance, use: ind, dist = snn_model.query_radius(X[0], radius, return_distance=False)
sort_ind = np.argsort(dist)
print("SNN query time:", time()-st)

# print total number and top five indices
print("number of neighbors:", len(ind))
print("indices of closest five:", ", ".join([str(i) for i in ind[sort_ind][:5]]))

# EXAMPLE OUTPUT
# SNN index time: 0.2224433422088623
# SNN query time: 0.009207725524902344
# number of neighbors: 550
# indices of closest five: 0, 27279, 69983, 65906, 97095
```

For the compiled `pybind11` backend (`snnpy.snnomp`), advanced radius query methods are available:

* `query_radius_advanced(...)`
* `query_radius_batch_advanced(...)`

They support:
* `metric`: `"euclidean"`, `"sqeuclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cosine"` (similar to common `scikit-learn` metrics).
* `groups`: integer array of shape `(n_samples,)`.
* `max_per_group`: cap on neighbors returned from each group.
* `return_distance`: optionally return distances together with indices.
* `p`: Minkowski power (used when `metric="minkowski"`).
* `fallback_to_nearest_if_empty`: if `True`, returns the nearest point when no neighbors are found inside radius (default `False` keeps current behavior).

The same fallback flag is also available in standard methods:
* `query_radius(..., fallback_to_nearest_if_empty=False)`
* `query_radius_batch(..., fallback_to_nearest_if_empty=False)`

KNN methods are also available in the compiled backend:
* `query_knn(new_data, k, return_distance=False, groups=None, max_per_group=-1)`
* `query_knn_batch(new_data, k, return_distance=False, groups=None, max_per_group=-1)`
  
KNN methods also support metric selection like advanced radius methods via:
* `metric`: `"euclidean"`, `"sqeuclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cosine"`
* `p`: Minkowski power (used when `metric="minkowski"`).

For KNN, if `groups` and positive `max_per_group` are provided, the method returns up to `k` nearest neighbors while limiting each group count by `max_per_group`.

For `query_radius_advanced` / `query_radius_batch_advanced`, if `groups=None`, `max_per_group<=0`, `metric="euclidean"` and `return_distance=False`, the implementation uses the faster non-group radius path (no per-group limiting logic).
Batch compiled methods return Python lists of NumPy arrays (not Python int lists), which is much more memory-efficient for large outputs.

Compare this to sklearn's KDTree:

```python
from sklearn.neighbors import KDTree
st = time()
tree = KDTree(X)    
print("KDTree index time:", time()-st)
st = time()
ind2 = tree.query_radius(X[0].reshape(1, -1), radius)
print("KDTree query time:", time()-st)
print("number of neighbors:", len(ind2[0]))

# KDTree index time: 7.597502946853638
# KDTree query time: 0.08962678909301758
# number of neighbors: 550
```

### License
All the content in this repository is licensed under the MIT License. 


## Reference

```
Chen X, Güttel S. 2024. Fast and exact fixed-radius neighbor search based on sorting. PeerJ Computer Science 10:e1929 https://doi.org/10.7717/peerj-cs.1929
```
