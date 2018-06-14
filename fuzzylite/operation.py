"""
 pyfuzzylite (TM), a fuzzy logic control library in Python.
 Copyright (C) 2010-2017 FuzzyLite Limited. All rights reserved.
 Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>

 This file is part of pyfuzzylite.

 pyfuzzylite is free software: you can redistribute it and/or modify it under
 the terms of the FuzzyLite License included with the software.

 You should have received a copy of the FuzzyLite License along with
 pyfuzzylite. If not, see <http://www.fuzzylite.com/license/>.

 pyfuzzylite is a trademark of FuzzyLite Limited
 fuzzylite is a registered trademark of FuzzyLite Limited.
"""

import math
from typing import Callable, Optional, Text, Union


class Operation(object):
    """
    Operation
    """

    @staticmethod
    def eq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b))

    @staticmethod
    def neq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b))

    @staticmethod
    def gt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b)
                    ) and a > b

    @staticmethod
    def ge(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b)
                or a > b)

    @staticmethod
    def le(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b)
                or a < b)

    @staticmethod
    def lt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b)
                    ) and a < b

    @staticmethod
    def logical_and(a: float, b: float) -> bool:
        return Operation.eq(a, 1.0) and Operation.eq(b, 1.0)

    @staticmethod
    def logical_or(a: float, b: float) -> bool:
        return Operation.eq(a, 1.0) or Operation.eq(b, 1.0)

    @staticmethod
    def valid_name(name: str) -> str:
        result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
        return result if result else "unnamed"

    @staticmethod
    def str(x: Union[float, object], decimals: Optional[int] = None) -> Text:
        if not decimals:
            from . import lib
            decimals = lib.decimals
        if isinstance(x, float):
            return f"{x:.{decimals}f}"
        return str(x)

    @staticmethod
    def scale(x: float, from_minimum: float, from_maximum: float, to_minimum: float,
              to_maximum: float) -> float:
        return ((to_maximum - to_minimum) / (from_maximum - from_minimum) * (x - from_minimum)
                + to_minimum)

    @staticmethod
    def bound(x: float, minimum: float, maximum: float) -> float:
        if x > maximum:
            return maximum
        if x < minimum:
            return minimum
        return x

    @staticmethod
    def arity_of(method: Callable) -> int:
        import inspect
        signature = inspect.signature(method)
        required_parameters = [parameter for parameter in signature.parameters.values()
                               if parameter.default == inspect.Parameter.empty]
        return len(required_parameters)

    @staticmethod
    def pi() -> float:
        return math.pi


Op = Operation
