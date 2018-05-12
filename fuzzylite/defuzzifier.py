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

from enum import Enum, auto
from math import isfinite, nan

from .operation import Operation as Op
from .term import Aggregated, Constant, Function, Linear, Term


class Defuzzifier(object):
    @property
    def class_name(self):
        return self.__class__.__name__

    def configure(self, parameters: str):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()


class IntegralDefuzzifier(Defuzzifier):
    __slots__ = ["resolution"]

    default_resolution = 100

    def __init__(self):
        self.resolution = IntegralDefuzzifier.default_resolution

    def __str__(self):
        return f"{self.class_name} {self.parameters()}"

    def configure(self, parameters: str):
        if parameters:
            self.resolution = int(parameters)

    def parameters(self):
        return Op.str(self.resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()


class Bisector(IntegralDefuzzifier):
    def __init__(self):
        super().__init__()

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        counter = resolution
        left = right = 0
        x_left, x_right = (minimum, maximum)
        left_area = right_area = 0.0

        # TODO: Improve?
        while counter > 0:
            counter = counter - 1
            if left_area <= right_area:
                x_left = minimum + (left + 0.5) * dx
                left_area += term.membership(x_left)
                left += 1
            else:
                x_right = maximum - (right + 0.5) * dx
                right_area += term.membership(x_right)
                right += 1

        # Inverse weighted average to compensate
        return (left_area * x_right + right_area * x_left) / (left_area + right_area)


class Centroid(IntegralDefuzzifier):
    def __init__(self):
        super().__init__()

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        area = x_centroid = float(0.0)
        for i in range(0, resolution):
            x = minimum + (i + 0.5) * dx
            y = term.membership(x)
            x_centroid += y * x
            area += y
        return x_centroid / area


# TODO: Implement
class LargestOfMaximum(IntegralDefuzzifier):
    pass


# TODO: Implement
class MeanOfMaximum(IntegralDefuzzifier):
    pass


# TODO: Implement
class SmallestOfMaximum(IntegralDefuzzifier):
    pass


class WeightedDefuzzifier(Defuzzifier):
    __slots__ = ["type"]

    class Type(Enum):
        Automatic = auto(),
        TakagiSugeno = auto(),
        Tsukamoto = auto()

    def __init__(self):
        self.type = WeightedDefuzzifier.Type.Automatic

    def __str__(self):
        return f"{self.class_name} {self.parameters()}"

    def configure(self, parameters: str):
        if parameters:
            self.type = WeightedDefuzzifier.Type[parameters]

    def parameters(self):
        return self.type.name if self.type else ""

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()

    def infer_type(self, term: Term):
        if isinstance(term, (Constant, Linear, Function)):
            return WeightedDefuzzifier.Type.TakagiSugeno
        return WeightedDefuzzifier.Type.Tsukamoto


class WeightedAverage(WeightedDefuzzifier):
    def __init__(self):
        super().__init__()

    def defuzzify(self, fuzzy_output: Aggregated, *_):
        if not fuzzy_output.terms:
            return nan
        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output.terms[0])

        weighted_sum = weights = 0.0
        if this_type == WeightedDefuzzifier.Type.TakagiSugeno:
            # Provides Takagi-Sugeno and Inverse Tsukamoto of Functions
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.membership(w)
                weighted_sum += w * z
                weights += w
        else:
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.tsukamoto(w, fuzzy_output.minimum, fuzzy_output.maximum)
                weighted_sum += w * z
                weights += w

        return weighted_sum / weights


class WeightedSum(WeightedDefuzzifier):
    def __init__(self):
        super().__init__()

    def defuzzify(self, fuzzy_output: Aggregated, *_):
        if not fuzzy_output.terms:
            return nan
        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output.terms[0])

        weighted_sum = 0.0
        if this_type == WeightedDefuzzifier.Type.TakagiSugeno:
            # Provides Takagi-Sugeno and Inverse Tsukamoto of Functions
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.membership(w)
                weighted_sum += w * z
        else:
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.tsukamoto(w, fuzzy_output.minimum, fuzzy_output.maximum)
                weighted_sum += w * z

        return weighted_sum
