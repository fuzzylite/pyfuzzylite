from __future__ import annotations

__all__ = [
    "Array",
    "Float",
    "Scalar",
    "ScalarArray",
]

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray as Array

Float = Union[float, np.floating[Any]]
Scalar = Union[float, np.floating[Any], Array[np.floating[Any]]]
ScalarArray = Array[np.floating[Any]]
