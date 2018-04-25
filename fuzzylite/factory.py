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

from .activation import *
from .defuzzifier import *
from .hedge import *
from .norm import *
from .term import *


class ConstructionFactory(object):
    __slots__ = ["constructors"]

    def __init__(self):
        self.constructors: Dict[str, function] = {}

    @property
    def class_name(self):
        return self.__class__.__name__

    def construct(self, key: str) -> object:
        if key in self.constructors:
            if self.constructors[key]:
                return self.constructors[key]()
        raise ValueError(f"constructor of '{key}' not found in {self.class_name}")


class CloningFactory(object):
    __slots__ = ["objects"]

    def __init__(self):
        self.objects: Dict[str, object] = {}

    @property
    def class_name(self):
        return self.__class__.__name__

    def clone(self, key: str):
        if key in self.objects:
            if self.objects[key]:
                return self.objects[key].clone()
            return None
        raise ValueError(f"object with key '{key}' not found in {self.class_name}")


class ActivationFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for activation in [First, General, Highest, Last, Lowest, Proportional, Threshold]:
            self.constructors[activation().class_name] = activation


class DefuzzifierFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for defuzzifier in [Bisector, Centroid, LargestOfMaximum, MeanOfMaximum, SmallestOfMaximum,
                            WeightedAverage, WeightedSum]:
            self.constructors[defuzzifier().class_name] = defuzzifier

    # # TODO: Implement?
    # def construct(self, key: str, parameter: Union[int, str]):
    #     raise NotImplementedError()


# TODO: implement
class FunctionFactory(CloningFactory):
    def __init__(self):
        super().__init__()


class HedgeFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for hedge in [Any, Extremely, Not, Seldom, Somewhat, Very]:
            self.constructors[hedge().name] = hedge


class SNormFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for snorm in [AlgebraicSum, BoundedSum, DrasticSum, EinsteinSum, HamacherSum,
                      Maximum, NilpotentMaximum, NormalizedSum, UnboundedSum]:
            self.constructors[snorm().class_name] = snorm


class TNormFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for tnorm in [AlgebraicProduct, BoundedDifference, DrasticProduct, EinsteinProduct,
                      HamacherProduct, Minimum, NilpotentMinimum]:
            self.constructors[tnorm().class_name] = tnorm


class TermFactory(ConstructionFactory):
    def __init__(self):
        super().__init__()
        self.constructors[""] = type(None)

        for term in [Bell, Binary, Concave, Constant, Cosine, Discrete,
                     Function, Gaussian, GaussianProduct, Linear, PiShape, Ramp,
                     Rectangle, Sigmoid, SigmoidDifference, SigmoidProduct,
                     Spike, SShape, Trapezoid, Triangle, ZShape]:
            self.constructors[term().class_name] = term
