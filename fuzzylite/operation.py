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

import inspect
from typing import Callable

import fuzzylite


# TODO convert to module or static object

class Operation(object):
    '''
    Operation
    '''

    @staticmethod
    def eq(a: float, b: float, absolute_tolerance: float = None):
        return (a == b
                or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                 if not absolute_tolerance else absolute_tolerance)
                or (a != a and b != b))

    @staticmethod
    def neq(a: float, b: float, absolute_tolerance: float = None):
        return not (a == b
                    or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                     if not absolute_tolerance else absolute_tolerance)
                    or (a != a and b != b))

    @staticmethod
    def gt(a: float, b: float, absolute_tolerance: float = None):
        return not (a == b
                    or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                     if not absolute_tolerance else absolute_tolerance)
                    or (a != a and b != b)
                    ) and a > b

    @staticmethod
    def ge(a: float, b: float, absolute_tolerance: float = None):
        return (a == b
                or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                 if not absolute_tolerance else absolute_tolerance)
                or (a != a and b != b)
                or a > b)

    @staticmethod
    def le(a: float, b: float, absolute_tolerance: float = None):
        return (a == b
                or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                 if not absolute_tolerance else absolute_tolerance)
                or (a != a and b != b)
                or a < b)

    @staticmethod
    def lt(a: float, b: float, absolute_tolerance: float = None):
        return not (a == b
                    or abs(a - b) < (fuzzylite.library().absolute_tolerance
                                     if not absolute_tolerance else absolute_tolerance)
                    or (a != a and b != b)
                    ) and a < b

    @staticmethod
    def logical_and(a: float, b: float):
        return 1.0 if Operation.eq(a, 1.0) and Operation.eq(b, 1.0) else 0.0

    @staticmethod
    def logical_or(a: float, b: float):
        return 1.0 if Operation.eq(a, 1.0) or Operation.eq(b, 1.0) else 0.0

    @staticmethod
    def valid_name(name: str) -> str:
        result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
        return result if result else "unnamed"

    @staticmethod
    def str(x, decimals: float = None) -> str:
        if isinstance(x, float):
            return ("{:.%sf}" % (decimals if decimals else fuzzylite.library().decimals)).format(x)
        return str(x)

    @staticmethod
    def scale(x: float, from_minimum: float, from_maximum: float, to_minimum: float,
              to_maximum: float) -> float:
        return (to_maximum - to_minimum) / (from_maximum - from_minimum) * (
                x - from_minimum) + to_minimum

    @staticmethod
    def bound(x: float, minimum: float, maximum: float):
        if x > maximum:
            return maximum
        if x < minimum:
            return minimum
        return x

    @staticmethod
    def arity_of(method: Callable):
        signature = inspect.signature(method)
        required_parameters = [parameter for parameter in signature.parameters.values()
                               if parameter.default == inspect.Parameter.empty]
        return len(required_parameters)
