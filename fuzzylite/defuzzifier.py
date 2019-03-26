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

__all__ = ["Defuzzifier", "IntegralDefuzzifier", "Bisector", "Centroid", "LargestOfMaximum",
           "MeanOfMaximum", "SmallestOfMaximum", "WeightedDefuzzifier", "WeightedAverage",
           "WeightedSum"]

import enum
import math
from math import nan
from typing import Optional, Union

from .operation import Op
from .term import Aggregated, Constant, Function, Linear, Term


class Defuzzifier:

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def configure(self, parameters: str) -> None:
        raise NotImplementedError()

    def parameters(self) -> str:
        raise NotImplementedError()

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()


class IntegralDefuzzifier(Defuzzifier):
    default_resolution = 100

    def __init__(self, resolution: Optional[int] = None) -> None:
        self.resolution = resolution if resolution else IntegralDefuzzifier.default_resolution

    def __str__(self) -> str:
        return f"{self.class_name} {self.parameters()}"

    def configure(self, parameters: str) -> None:
        if parameters:
            self.resolution = int(parameters)

    def parameters(self) -> str:
        return Op.str(self.resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()


class Bisector(IntegralDefuzzifier):

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__(resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not math.isfinite(minimum + maximum):
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

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__(resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not math.isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        area = x_centroid = 0.0
        for i in range(0, resolution):
            x = minimum + (i + 0.5) * dx
            y = term.membership(x)
            x_centroid += y * x
            area += y
        return x_centroid / area


class LargestOfMaximum(IntegralDefuzzifier):

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__(resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not math.isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        y_max = -math.inf
        x_largest = maximum
        for i in range(0, resolution):
            x = minimum + (i + 0.5) * dx
            y = term.membership(x)
            if Op.ge(y, y_max):
                y_max = y
                x_largest = x
        return x_largest


class MeanOfMaximum(IntegralDefuzzifier):

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__(resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not math.isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        y_max = -math.inf
        x_smallest = minimum
        x_largest = maximum
        find_x_largest = False
        for i in range(0, resolution):
            x = minimum + (i + 0.5) * dx
            y = term.membership(x)
            if Op.gt(y, y_max):
                y_max = y
                x_smallest = x
                x_largest = x
                find_x_largest = True
            elif find_x_largest and Op.eq(y, y_max):
                x_largest = x
            elif Op.lt(y, y_max):
                find_x_largest = False
        return (x_largest + x_smallest) / 2.0


class SmallestOfMaximum(IntegralDefuzzifier):

    def __init__(self, resolution: Optional[int] = None) -> None:
        super().__init__(resolution)

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        if not math.isfinite(minimum + maximum):
            return nan
        resolution = self.resolution
        dx = (maximum - minimum) / resolution
        y_max = -math.inf
        x_smallest = minimum
        for i in range(0, resolution):
            x = minimum + (i + 0.5) * dx
            y = term.membership(x)
            if Op.gt(y, y_max):
                y_max = y
                x_smallest = x
        return x_smallest


class WeightedDefuzzifier(Defuzzifier):
    @enum.unique
    class Type(enum.Enum):
        Automatic, TakagiSugeno, Tsukamoto = range(3)

    def __init__(self, type: Optional[Union[str, 'WeightedDefuzzifier.Type']] = None) -> None:
        if type is None:
            type = WeightedDefuzzifier.Type.Automatic
        elif isinstance(type, str):
            type = WeightedDefuzzifier.Type[type]
        self.type = type

    def __str__(self) -> str:
        return f"{self.class_name} {self.parameters()}"

    def configure(self, parameters: str) -> None:
        if parameters:
            self.type = WeightedDefuzzifier.Type[parameters]

    def parameters(self) -> str:
        return self.type.name if self.type else ""

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> float:
        raise NotImplementedError()

    def infer_type(self, term: Term) -> 'WeightedDefuzzifier.Type':
        if isinstance(term, (Constant, Linear, Function)):
            return WeightedDefuzzifier.Type.TakagiSugeno
        return WeightedDefuzzifier.Type.Tsukamoto


class WeightedAverage(WeightedDefuzzifier):

    def __init__(self, type: Optional[Union[str, 'WeightedDefuzzifier.Type']] = None) -> None:
        super().__init__(type)

    def defuzzify(self, fuzzy_output: Term,
                  unused_minimum: float = nan, unused_maximum: float = nan) -> float:
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(f"expected an Aggregated term, but found {type(fuzzy_output)}")

        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        if not fuzzy_output.terms:
            return nan

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

    def __init__(self, type: Optional[Union[str, 'WeightedDefuzzifier.Type']] = None) -> None:
        super().__init__(type)

    def defuzzify(self, fuzzy_output: Term,
                  unused_minimum: float = nan, unused_maximum: float = nan) -> float:
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(f"expected an Aggregated term, but found {type(fuzzy_output)}")

        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        if not fuzzy_output.terms:
            return nan

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
