from __future__ import annotations

__all__ = ["Scalar", "Array"]

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray as Array

# Todo: rename scalar to float, scalars to scalar
Scalar = Union[float, np.floating[Any], Array[np.floating[Any]]]
Scalars = Union[np.floating[Any], Array[np.floating[Any]]]
