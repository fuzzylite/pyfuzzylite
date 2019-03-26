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

__all__ = ["Hedge", "Any", "Extremely", "Not", "Seldom", "Somewhat", "Very", "HedgeLambda",
           "HedgeFunction"]

import math
import typing
from typing import Callable

if typing.TYPE_CHECKING:
    from .term import Function


class Hedge:

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def hedge(self, x: float) -> float:
        raise NotImplementedError()


class Any(Hedge):

    def hedge(self, x: float) -> float:
        return 1.0


class Extremely(Hedge):

    def hedge(self, x: float) -> float:
        return 2.0 * x * x if x <= 0.5 else (1.0 - 2.0 * (1.0 - x) * (1.0 - x))


class Not(Hedge):

    def hedge(self, x: float) -> float:
        return 1.0 - x


class Seldom(Hedge):

    def hedge(self, x: float) -> float:
        return math.sqrt(0.5 * x) if x <= 0.5 else (1.0 - math.sqrt(0.5 * (1.0 - x)))


class Somewhat(Hedge):

    def hedge(self, x: float) -> float:
        return math.sqrt(x)


class Very(Hedge):

    def hedge(self, x: float) -> float:
        return x * x


class HedgeLambda(Hedge):

    def __init__(self, name: str, function: Callable[[float], float]) -> None:
        self._name = name
        self.function = function

    @property
    def name(self) -> str:
        return self._name

    def hedge(self, x: float) -> float:
        return self.function(x)


class HedgeFunction(Hedge):

    def __init__(self, function: 'Function') -> None:
        self.function = function

    @property
    def name(self) -> str:
        return self.function.name

    def hedge(self, x: float) -> float:
        return self.function.membership(x)
