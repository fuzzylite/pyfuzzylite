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

import copy
import unittest

from fuzzylite import *
from tests.assert_component import ComponentAssert


class FactoryAssert(ComponentAssert):
    def has_class_name(self, name: str):
        self.test.assertEqual(self.actual.class_name, name)
        return self

    def constructs_exactly(self, name_type: Dict[str, type]):
        self.test.assertDictEqual(name_type, self.actual.constructors)
        for name, clazz in name_type.items():
            self.test.assertEqual(type(self.actual.construct(name)), clazz)
        return self

    def clones_exactly(self, name_instance: Dict[str, object]):
        self.test.assertSetEqual(set(self.actual.objects.keys()), set(name_instance.keys()))
        for name, instance in name_instance.items():
            self.test.assertEqual(str(self.actual.clone(name)), str(instance))
        return self


class TestFactory(unittest.TestCase):
    def test_construction_factory(self):
        assert_that = FactoryAssert(self, ConstructionFactory())
        assert_that.has_class_name("ConstructionFactory").constructs_exactly({})

        class Example(object):
            def __str__(self):
                return "instance of Example"

        assert_that.actual.constructors["example"] = Example

        assert_that.constructs_exactly({"example": Example})

        self.assertEqual(str(assert_that.actual.construct("example")), "instance of Example")

    def test_activation_factory(self):
        FactoryAssert(self, ActivationFactory()) \
            .has_class_name("ActivationFactory") \
            .constructs_exactly({"": type(None), "First": First, "General": General, "Highest": Highest, "Last": Last,
                                 "Lowest": Lowest, "Proportional": Proportional, "Threshold": Threshold})

    def test_defuzzifier_factory(self):
        FactoryAssert(self, DefuzzifierFactory()) \
            .has_class_name("DefuzzifierFactory") \
            .constructs_exactly({"": type(None),
                                 "Bisector": Bisector, "Centroid": Centroid, "LargestOfMaximum": LargestOfMaximum,
                                 "MeanOfMaximum": MeanOfMaximum, "SmallestOfMaximum": SmallestOfMaximum,
                                 "WeightedAverage": WeightedAverage, "WeightedSum": WeightedSum
                                 })

    def test_hedge_factory(self):
        FactoryAssert(self, HedgeFactory()) \
            .has_class_name("HedgeFactory") \
            .constructs_exactly({"": type(None),
                                 "any": Any, "extremely": Extremely, "not": Not, "seldom": Seldom,
                                 "somewhat": Somewhat, "very": Very
                                 })

    def test_snorm_factory(self):
        FactoryAssert(self, SNormFactory()) \
            .has_class_name("SNormFactory") \
            .constructs_exactly({"": type(None),
                                 "AlgebraicSum": AlgebraicSum, "BoundedSum": BoundedSum, "DrasticSum": DrasticSum,
                                 "EinsteinSum": EinsteinSum, "HamacherSum": HamacherSum, "Maximum": Maximum,
                                 "NilpotentMaximum": NilpotentMaximum, "NormalizedSum": NormalizedSum,
                                 "UnboundedSum": UnboundedSum
                                 })

    def test_tnorm_factory(self):
        FactoryAssert(self, TNormFactory()) \
            .has_class_name("TNormFactory") \
            .constructs_exactly({"": type(None),
                                 "AlgebraicProduct": AlgebraicProduct, "BoundedDifference": BoundedDifference,
                                 "DrasticProduct": DrasticProduct, "EinsteinProduct": EinsteinProduct,
                                 "HamacherProduct": HamacherProduct, "Minimum": Minimum,
                                 "NilpotentMinimum": NilpotentMinimum
                                 })

    def test_term_factory(self):
        FactoryAssert(self, TermFactory()) \
            .has_class_name("TermFactory") \
            .constructs_exactly({"": type(None),
                                 "Bell": Bell, "Binary": Binary, "Concave": Concave, "Constant": Constant,
                                 "Cosine": Cosine, "Discrete": Discrete, "Function": Function, "Gaussian": Gaussian,
                                 "GaussianProduct": GaussianProduct, "Linear": Linear, "PiShape": PiShape, "Ramp": Ramp,
                                 "Rectangle": Rectangle, "Sigmoid": Sigmoid, "SigmoidDifference": SigmoidDifference,
                                 "SigmoidProduct": SigmoidProduct, "Spike": Spike, "SShape": SShape,
                                 "Trapezoid": Trapezoid, "Triangle": Triangle, "ZShape": ZShape
                                 })

    def test_cloning_factory(self):
        assert_that = FactoryAssert(self, CloningFactory())
        assert_that.has_class_name("CloningFactory").clones_exactly({})

        class Example(object):
            def __init__(self, property):
                self.property = property

            def __str__(self):
                return f"Example({str(self.property)})"

            def clone(self):
                return copy.copy(self)

        assert_that.actual.objects["example"] = Example("Clone of Example")
        assert_that.clones_exactly({"example": Example("Clone of Example")})
        self.assertEqual(assert_that.actual.clone("example").property, "Clone of Example")

    @unittest.skip("Testing of Function")
    def test_function_factory(self):
        pass



if __name__ == '__main__':
    unittest.main()
