from __future__ import annotations

__all__ = [
    "Array",
    "Float",
    "Scalar",
    "ScalarArray",
    "array",
    "float_type",
    "to_float",
    "scalar",
]

from collections.abc import Sequence
from typing import Any, Union, overload

import numpy as np
from numpy.typing import NDArray as Array

Float = Union[float, np.floating[Any]]
Scalar = Union[float, np.floating[Any], Array[np.floating[Any]]]
ScalarArray = Array[np.floating[Any]]

float_type = np.float64


def to_float(x: Any, /) -> float:
    """Convert the value into a floating point defined by the library
    @param x is the value to convert.
    """
    return float_type(x)  # type: ignore


@overload
def scalar(x: Sequence[Any] | Array[Any], /) -> ScalarArray:
    ...


@overload
def scalar(x: Any, /) -> Scalar:
    ...


def scalar(x: Sequence[Any] | Array[Any] | Any, /) -> ScalarArray | Scalar:
    """Convert the values into a floating point value  defined by the library
    @param x is the value to convert.
    """
    return np.asarray(x, dtype=float_type)


def array(x: Any, /, **kwargs: Any) -> Array[Any]:
    """Convert the value into a floating point defined by the library
    @param x is the value to convert.
    """
    return np.asarray(x, **kwargs)
