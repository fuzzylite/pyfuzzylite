"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = ["Array", "Scalar", "ScalarArray", "Self"]

import sys
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray as Array

Scalar = Union[float, np.floating[Any], Array[np.floating[Any]]]
ScalarArray = Array[np.floating[Any]]

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
