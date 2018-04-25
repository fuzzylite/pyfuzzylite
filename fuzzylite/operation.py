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

import fuzzylite as fl


class Operation(object):
    '''
    Operation
    '''

    @staticmethod
    def valid_name(name: str) -> str:
        result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
        return result if result else "unnamed"

    @staticmethod
    def str(x) -> str:
        if isinstance(x, float):
            return ("{:.%sf}" % fl.DECIMALS).format(x)
        return str(x)

    @staticmethod
    def scale(x: float, from_minimum: float, from_maximum: float, to_minimum: float, to_maximum: float) -> float:
        return (to_maximum - to_minimum) / (from_maximum - from_minimum) * (x - from_minimum) + to_minimum

    @staticmethod
    def bound(x: float, minimum: float, maximum: float):
        if x > maximum: return maximum
        if x < minimum: return minimum
        return x

    @staticmethod
    def arity_of(self, method: Callable):
        if not method:
            raise ValueError("expected a method or function, but found none")
        return len([parameter for parameter in inspect.signature(self.method).parameters.values()
                    if parameter.default == inspect.Parameter.empty])
