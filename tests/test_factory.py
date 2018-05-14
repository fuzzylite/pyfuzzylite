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

import unittest
from typing import *

from fuzzylite import *
from tests.assert_component import BaseAssert, ComponentAssert


class FactoryAssert(ComponentAssert):
    def has_class_name(self, name: str):
        self.test.assertEqual(self.actual.class_name, name)
        return self

    def constructs_exactly(self, name_type: Dict[str, type]):
        self.test.assertDictEqual(name_type, self.actual.constructors)
        for name, clazz in name_type.items():
            self.test.assertEqual(type(self.actual.construct(name)), clazz)
        return self

    def copies_exactly(self, name_instance: Dict[str, object]):
        self.test.assertSetEqual(set(self.actual.objects.keys()), set(name_instance.keys()))
        for name, instance in name_instance.items():
            self.test.assertEqual(str(self.actual.copy(name)), str(instance))
            self.test.assertNotEqual(repr(self.actual.copy(name)), repr(instance))
        return self


class FunctionFactoryAssert(BaseAssert):
    def contains_exactly(self, elements: Set[str], element_type: Function.Element.Type = None):
        if element_type == Function.Element.Type.Operator:
            self.test.assertSetEqual(self.actual.operators(), elements)
        elif element_type == Function.Element.Type.Function:
            self.test.assertSetEqual(self.actual.functions(), elements)
        else:
            self.test.assertSetEqual(set(self.actual.objects.keys()), elements)

            # self.test.assertEqual(key, element.method.__name__)
        return self

    def operation_is(self, operation_value: Dict[Tuple[str, Tuple[float]], float]):
        for operation, expected_value in operation_value.items():
            name = operation[0]
            args = operation[1]
            element = self.actual.objects[name]
            value = element.method(*args)
            message = f"{name}({', '.join([Op.str(x) for x in args])})"
            self.test.assertAlmostEqual(expected_value, value, places=15,
                                        msg=f"{message} == {Op.str(value)}, "
                                            f"but expected {message} == {Op.str(expected_value)}")
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
            .constructs_exactly(
            {"": type(None), "First": First, "General": General, "Highest": Highest, "Last": Last,
             "Lowest": Lowest, "Proportional": Proportional, "Threshold": Threshold})

    def test_defuzzifier_factory(self):
        FactoryAssert(self, DefuzzifierFactory()) \
            .has_class_name("DefuzzifierFactory") \
            .constructs_exactly({"": type(None),
                                 "Bisector": Bisector, "Centroid": Centroid,
                                 "LargestOfMaximum": LargestOfMaximum,
                                 "MeanOfMaximum": MeanOfMaximum,
                                 "SmallestOfMaximum": SmallestOfMaximum,
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
                                 "AlgebraicSum": AlgebraicSum, "BoundedSum": BoundedSum,
                                 "DrasticSum": DrasticSum,
                                 "EinsteinSum": EinsteinSum, "HamacherSum": HamacherSum,
                                 "Maximum": Maximum,
                                 "NilpotentMaximum": NilpotentMaximum,
                                 "NormalizedSum": NormalizedSum,
                                 "UnboundedSum": UnboundedSum
                                 })

    def test_tnorm_factory(self):
        FactoryAssert(self, TNormFactory()) \
            .has_class_name("TNormFactory") \
            .constructs_exactly({"": type(None),
                                 "AlgebraicProduct": AlgebraicProduct,
                                 "BoundedDifference": BoundedDifference,
                                 "DrasticProduct": DrasticProduct,
                                 "EinsteinProduct": EinsteinProduct,
                                 "HamacherProduct": HamacherProduct, "Minimum": Minimum,
                                 "NilpotentMinimum": NilpotentMinimum
                                 })

    def test_term_factory(self):
        FactoryAssert(self, TermFactory()) \
            .has_class_name("TermFactory") \
            .constructs_exactly({"": type(None),
                                 "Bell": Bell, "Binary": Binary, "Concave": Concave,
                                 "Constant": Constant,
                                 "Cosine": Cosine, "Discrete": Discrete, "Function": Function,
                                 "Gaussian": Gaussian,
                                 "GaussianProduct": GaussianProduct, "Linear": Linear,
                                 "PiShape": PiShape, "Ramp": Ramp,
                                 "Rectangle": Rectangle, "Sigmoid": Sigmoid,
                                 "SigmoidDifference": SigmoidDifference,
                                 "SigmoidProduct": SigmoidProduct, "Spike": Spike, "SShape": SShape,
                                 "Trapezoid": Trapezoid, "Triangle": Triangle, "ZShape": ZShape
                                 })

    def test_copy_factory(self):
        assert_that = FactoryAssert(self, CopyFactory())
        assert_that.has_class_name("CopyFactory").copies_exactly({})

        class Example(object):
            def __init__(self, value):
                self.property = value

            def __str__(self):
                return f"Example({str(self.property)})"

            def clone(self):
                return copy.copy(self)

        assert_that.actual.objects["example"] = Example("Clone of Example")
        assert_that.copies_exactly({"example": Example("Clone of Example")})
        self.assertEqual(assert_that.actual.copy("example").property, "Clone of Example")


class TestFunctionFactory(unittest.TestCase):
    def test_factory_matches_keys_and_names(self):
        for key, element in FunctionFactory().objects.items():
            self.assertEqual(key, element.name)
            # if it is a function, the name should be contained in
            # in the methods name
            if element.element_type == Function.Element.Type.Function:
                self.assertIn(key, element.method.__name__)

    def test_arity(self):
        acceptable: Dict[Type[Exception], Set[str]] = {
            ZeroDivisionError: {"%", "/", "fmod", "^"},
            ValueError: {"acos", "acosh", "asin", "atanh", "fmod",
                         "log", "log10", "log1p", "sqrt", "pow"}
        }

        errors = []

        def evaluate(function_element: Function.Element, parameters: List[float]):
            try:
                function_element.method(*parameters)
            except Exception as ex:
                if not (type(ex) in acceptable
                        and function_element.name in acceptable[type(ex)]):
                    errors.append(ex)
                    print(f"{function_element.name}:"
                          f"{function_element.method.__name__}({parameters}): {ex.__class__}")

        from random import Random
        random = Random()

        for element in FunctionFactory().objects.values():
            if element.arity == 0:
                evaluate(element, [])
            else:
                for i in range(1000):
                    a = Op.scale(random.randint(0, 100), 0, 100, -10, 10)
                    b = Op.scale(random.randint(0, 100), 0, 100, -10, 10)
                    if element.arity == 1:
                        evaluate(element, [a])
                        evaluate(element, [b])
                    else:
                        evaluate(element, [a, a])
                        evaluate(element, [a, b])
                        evaluate(element, [b, a])
                        evaluate(element, [b, b])

        self.assertListEqual([], errors)

    def test_factory_contains_exactly(self):
        FunctionFactoryAssert(self, FunctionFactory()) \
            .contains_exactly(
            {'!', '~', '^', '*', '/', '%', '+', '-', 'and', 'or'},
            Function.Element.Type.Operator) \
            .contains_exactly(
            {'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil',
             'cos', 'cosh', 'eq', 'exp', 'fabs', 'floor', 'fmod', 'ge', 'gt', 'le',
             'log', 'log10', 'log1p', 'lt', 'max', 'min', 'neq', 'pow', 'round', 'sin',
             'sinh', 'sqrt', 'tan', 'tanh'},
            Function.Element.Type.Function)

    def test_function_operators(self):
        FunctionFactoryAssert(self, FunctionFactory()) \
            .operation_is(
            {
                ("!", (0,)): 1, ("!", (1,)): 0,
                ("~", (1,)): -1, ("~", (-2,)): 2, ("~", (0,)): 0,
                ("^", (3, 3)): 27, ("^", (9, 0.5)): 3,
                ("*", (-2, 3)): -6, ("*", (3, -2)): -6,
                ("/", (6, 3)): 2, ("/", (3, 6)): 0.5,
                ("%", (6, 3)): 0, ("%", (3, 6)): 3, ("%", (3.5, 6)): 3.5, ("%", (6, 3.5)): 2.5,
                ("+", (2, 3)): 5, ("+", (2, -3)): -1,
                ("-", (2, 3)): -1, ("-", (2, -3)): 5,
                ("and", (1, 0)): 0, ("and", (1, 1)): 1,
                ("or", (1, 0)): 1, ("or", (0, 0)): 0,
            })

    @unittest.skip("Until Function is ready")
    def test_function_precedence(self):
        pass


if __name__ == '__main__':
    unittest.main()
