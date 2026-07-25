"""Shared type aliases for transparentml."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Public-facing input types
FeatureMatrix = ArrayLike
"""Input accepted for a 2D array as shape (n_samples, n_features).
Can be a numpy array, nested list, pandas Dataframe."""

TargetVector = ArrayLike
"""Input accepted for a 1D array of shape (n_samples,).
Can be a numpy array, list, or pandas Series."""

ClassLabels = NDArray[np.int64]
"""1D array of shape (n_samples,) containing predicted class labels."""

# Internal types
FeatureMatrixArray = NDArray[np.float64]
TargetVectorArray = NDArray[np.float64]
WeightVector = NDArray[np.float64]