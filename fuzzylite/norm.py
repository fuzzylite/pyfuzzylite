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

import typing

if typing.TYPE_CHECKING:
    from fuzzylite.engine import Engine


class Norm(object):
    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def compute(self, a: float, b: float) -> float:
        raise NotImplementedError()


class TNorm(Norm):
    def compute(self, a: float, b: float) -> float:
        raise NotImplementedError()


class AlgebraicProduct(TNorm):
    def compute(self, a: float, b: float) -> float:
        return a * b


class BoundedDifference(TNorm):
    def compute(self, a: float, b: float) -> float:
        return max(0.0, a + b - 1.0)


class DrasticProduct(TNorm):
    # todo: op.eq(1.0, max(a,b)) ?
    def compute(self, a: float, b: float) -> float:
        return min(a, b) if max(a, b) == 1.0 else 0.0


class EinsteinProduct(TNorm):
    def compute(self, a: float, b: float) -> float:
        return (a * b) / (2.0 - (a + b - a * b))


class HamacherProduct(TNorm):
    # todo: op.neq(a+b,0.0) ?
    def compute(self, a: float, b: float) -> float:
        return (a * b) / (a + b - a * b) if a + b != 0.0 else 0.0


class Minimum(TNorm):
    def compute(self, a: float, b: float) -> float:
        return min(a, b)


class NilpotentMinimum(TNorm):
    def compute(self, a: float, b: float) -> float:
        return min(a, b) if a + b > 1.0 else 0.0


class TNormFunction(TNorm):
    __slots__ = ("f",)

    def __init__(self, formula: str, engine: typing.Optional['Engine'] = None) -> None:
        from fuzzylite.term import Function
        self.f = Function(self.class_name, formula, engine)
        self.f.load()

    def compute(self, a: float, b: float) -> float:
        self.f.variables['a'] = a
        self.f.variables['b'] = b
        return self.f.evaluate()


class SNorm(Norm):
    def compute(self, a: float, b: float) -> float:
        raise NotImplementedError()


class AlgebraicSum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return a + b - (a * b)


class BoundedSum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return min(1.0, a + b)


class DrasticSum(SNorm):
    # todo: op.eq(a+b,0.0) ?
    def compute(self, a: float, b: float) -> float:
        return max(a, b) if min(a, b) == 0.0 else 1.0


class EinsteinSum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return (a + b) / (1.0 + a * b)


class HamacherSum(SNorm):
    # todo: op.neq(a+b,0.0) ?
    def compute(self, a: float, b: float) -> float:
        return (a + b - 2.0 * a * b) / (1.0 - a * b) if a * b != 1.0 else 1.0


class Maximum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return max(a, b)


class NilpotentMaximum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return max(a, b) if a + b < 1.0 else 1.0


class NormalizedSum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return (a + b) / max(1.0, a + b)


class UnboundedSum(SNorm):
    def compute(self, a: float, b: float) -> float:
        return a + b


class SNormFunction(SNorm):
    __slots__ = ("f",)

    def __init__(self, formula: str, engine: typing.Optional['Engine'] = None) -> None:
        from fuzzylite.term import Function
        self.f = Function(self.class_name, formula, engine)
        self.f.load()

    def compute(self, a: float, b: float) -> float:
        self.f.variables['a'] = a
        self.f.variables['b'] = b
        return self.f.evaluate()
