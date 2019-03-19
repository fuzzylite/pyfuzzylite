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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

import fuzzylite as fl
from tests.assert_component import BaseAssert


class FactoryAssert(BaseAssert[Union[fl.ConstructionFactory[Any], fl.CloningFactory[Any]]]):

    def has_class_name(self, name: str) -> 'FactoryAssert':
        self.test.assertEqual(self.actual.class_name, name)
        return self

    def contains(self, name: Union[str, Iterable[str]], contains: bool = True) -> 'FactoryAssert':
        if isinstance(name, str):
            name = [name]
        for string in name:
            if contains:
                self.test.assertIn(string, self.actual,
                                   f"'{string}' is not in factory {self.actual.class_name}")
            else:
                self.test.assertNotIn(string, self.actual,
                                      f"'{string}' is in factory {self.actual.class_name}")
        return self

    def constructs_exactly(self, name_type: Dict[str, type]) -> 'FactoryAssert':
        if not isinstance(self.actual, fl.ConstructionFactory):
            raise ValueError(f"expected an instance of {fl.ConstructionFactory}, "
                             f"but got {self.actual}")
        self.test.assertDictEqual(name_type, self.actual.constructors)
        for name, clazz in name_type.items():
            self.test.assertEqual(type(self.actual.construct(name)), clazz)
        return self

    def copies_exactly(self, name_instance: Dict[str, object]) -> 'FactoryAssert':
        if not isinstance(self.actual, fl.CloningFactory):
            raise ValueError(f"expected an instance of {fl.CloningFactory}, "
                             f"but got {self.actual}")
        self.test.assertSetEqual(set(self.actual.objects.keys()), set(name_instance.keys()))
        for name, instance in name_instance.items():
            self.test.assertEqual(str(self.actual.copy(name)), str(instance))
            self.test.assertNotEqual(repr(self.actual.copy(name)), repr(instance))
        return self


class FunctionFactoryAssert(BaseAssert[fl.FunctionFactory]):

    def contains_exactly(self, elements: Set[str],
                         element_type: Optional[
                             fl.Function.Element.Type] = None) -> 'FunctionFactoryAssert':
        if element_type == fl.Function.Element.Type.Operator:
            self.test.assertSetEqual(set(self.actual.operators().keys()), elements)
        elif element_type == fl.Function.Element.Type.Function:
            self.test.assertSetEqual(set(self.actual.functions().keys()), elements)
        else:
            self.test.assertSetEqual(set(self.actual.objects.keys()), elements)

            # self.test.assertEqual(key, element.method.__name__)
        return self

    def operation_is(self, operation_value: Dict[Tuple[str, Sequence[float]], float]) \
            -> 'FunctionFactoryAssert':
        for operation, expected_value in operation_value.items():
            name = operation[0]
            args = operation[1]
            element = self.actual.objects[name]
            value = element.method(*args)
            message = f"expected {name}({', '.join([fl.Op.str(x) for x in args])}) to result in " \
                      f"{fl.Op.str(expected_value)}, but got {fl.Op.str(value)}"
            self.test.assertAlmostEqual(expected_value, value, places=15, msg=message)
        return self


class TestFactory(unittest.TestCase):

    def test_construction_factory(self) -> None:
        actual: fl.ConstructionFactory[Any] = fl.ConstructionFactory()
        assert_that = FactoryAssert(self, actual)
        assert_that.has_class_name("ConstructionFactory").constructs_exactly({})

        class Example:

            def __str__(self) -> str:
                return "instance of Example"

        actual.constructors["example"] = Example

        assert_that.contains("example")
        assert_that.constructs_exactly({"example": Example})

        self.assertEqual(str(actual.construct("example")), "instance of Example")

    def test_activation_factory(self) -> None:
        FactoryAssert(self, fl.ActivationFactory()) \
            .has_class_name("ActivationFactory") \
            .contains(["First", "Last", "Threshold"]) \
            .contains(["Second", "Third"], False) \
            .constructs_exactly({"First": fl.First, "General": fl.General, "Highest": fl.Highest,
                                 "Last": fl.Last, "Lowest": fl.Lowest,
                                 "Proportional": fl.Proportional,
                                 "Threshold": fl.Threshold})

    def test_defuzzifier_factory(self) -> None:
        FactoryAssert(self, fl.DefuzzifierFactory()) \
            .has_class_name("DefuzzifierFactory") \
            .contains(["Bisector", "MeanOfMaximum", "WeightedSum"]) \
            .contains(["Something", "Else"], False) \
            .constructs_exactly({"Bisector": fl.Bisector, "Centroid": fl.Centroid,
                                 "LargestOfMaximum": fl.LargestOfMaximum,
                                 "MeanOfMaximum": fl.MeanOfMaximum,
                                 "SmallestOfMaximum": fl.SmallestOfMaximum,
                                 "WeightedAverage": fl.WeightedAverage,
                                 "WeightedSum": fl.WeightedSum
                                 })

    def test_hedge_factory(self) -> None:
        FactoryAssert(self, fl.HedgeFactory()) \
            .has_class_name("HedgeFactory") \
            .contains(["any", "seldom", "very"]) \
            .contains(["very much", "often"], False) \
            .constructs_exactly({"any": fl.Any, "extremely": fl.Extremely, "not": fl.Not,
                                 "seldom": fl.Seldom, "somewhat": fl.Somewhat, "very": fl.Very
                                 })

    def test_snorm_factory(self) -> None:
        FactoryAssert(self, fl.SNormFactory()) \
            .has_class_name("SNormFactory") \
            .contains(["AlgebraicSum", "EinsteinSum", "UnboundedSum"]) \
            .contains(["AlgebraicProduct", "EinsteinProduct", "UnboundedProduct"], False) \
            .constructs_exactly({"AlgebraicSum": fl.AlgebraicSum, "BoundedSum": fl.BoundedSum,
                                 "DrasticSum": fl.DrasticSum,
                                 "EinsteinSum": fl.EinsteinSum, "HamacherSum": fl.HamacherSum,
                                 "Maximum": fl.Maximum,
                                 "NilpotentMaximum": fl.NilpotentMaximum,
                                 "NormalizedSum": fl.NormalizedSum,
                                 "UnboundedSum": fl.UnboundedSum
                                 })

    def test_tnorm_factory(self) -> None:
        FactoryAssert(self, fl.TNormFactory()) \
            .has_class_name("TNormFactory") \
            .contains(["AlgebraicProduct", "EinsteinProduct", "NilpotentMinimum"]) \
            .contains(["AlgebraicSum", "EinsteinSum", "UnboundedSum"], False) \
            .constructs_exactly({"AlgebraicProduct": fl.AlgebraicProduct,
                                 "BoundedDifference": fl.BoundedDifference,
                                 "DrasticProduct": fl.DrasticProduct,
                                 "EinsteinProduct": fl.EinsteinProduct,
                                 "HamacherProduct": fl.HamacherProduct, "Minimum": fl.Minimum,
                                 "NilpotentMinimum": fl.NilpotentMinimum
                                 })

    def test_term_factory(self) -> None:
        FactoryAssert(self, fl.TermFactory()) \
            .has_class_name("TermFactory") \
            .contains(["Bell", "Gaussian", "ZShape"]) \
            .contains(["Star", "Cube", "Sphere"], False) \
            .constructs_exactly({"Bell": fl.Bell, "Binary": fl.Binary, "Concave": fl.Concave,
                                 "Constant": fl.Constant,
                                 "Cosine": fl.Cosine, "Discrete": fl.Discrete,
                                 "Function": fl.Function,
                                 "Gaussian": fl.Gaussian,
                                 "GaussianProduct": fl.GaussianProduct, "Linear": fl.Linear,
                                 "PiShape": fl.PiShape, "Ramp": fl.Ramp,
                                 "Rectangle": fl.Rectangle, "Sigmoid": fl.Sigmoid,
                                 "SigmoidDifference": fl.SigmoidDifference,
                                 "SigmoidProduct": fl.SigmoidProduct, "Spike": fl.Spike,
                                 "SShape": fl.SShape,
                                 "Trapezoid": fl.Trapezoid, "Triangle": fl.Triangle,
                                 "ZShape": fl.ZShape
                                 })

    def test_copy_factory(self) -> None:
        actual: fl.CloningFactory[Any] = fl.CloningFactory()
        assert_that = FactoryAssert(self, actual)
        assert_that.has_class_name("CloningFactory").copies_exactly({})

        class Example(object):

            def __init__(self, value: str) -> None:
                self.property = value

            def __str__(self) -> str:
                return f"Example({str(self.property)})"

        actual.objects["example"] = Example("Clone of Example")

        assert_that.copies_exactly({"example": Example("Clone of Example")})

        self.assertEqual(actual.copy("example").property, "Clone of Example")


class TestFunctionFactory(unittest.TestCase):

    def test_factory_precedence(self) -> None:
        precedence_expected = {0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 50,
                               6: 40, 7: 30, 8: 20, 9: 10, 10: 0}
        factory = fl.FunctionFactory()
        for p, e in precedence_expected.items():
            self.assertEqual(e, factory._precedence(p))

    def test_factory_matches_keys_and_names(self) -> None:
        for key, element in fl.FunctionFactory().objects.items():
            self.assertEqual(key, element.name)
            # if it is a function, the name should be contained in
            # in the methods name
            if element.type == fl.Function.Element.Type.Function:
                self.assertIn(key, element.method.__name__)

    def test_arity(self) -> None:
        acceptable: Dict[Type[Exception], Set[str]] = {
            ZeroDivisionError: {"%", "/", "fmod", "^", "**"},
            ValueError: {"acos", "acosh", "asin", "atanh", "fmod",
                         "log", "log10", "log1p", "sqrt", "pow"}
        }

        errors = []

        def evaluate(function_element: fl.Function.Element, parameters: List[float]) -> None:
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

        for element in fl.FunctionFactory().objects.values():
            if element.arity == 0:
                evaluate(element, [])
            else:
                for i in range(1000):
                    a = fl.Op.scale(random.randint(0, 100), 0, 100, -10, 10)
                    b = fl.Op.scale(random.randint(0, 100), 0, 100, -10, 10)
                    if element.arity == 1:
                        evaluate(element, [a])
                        evaluate(element, [b])
                    else:
                        evaluate(element, [a, a])
                        evaluate(element, [a, b])
                        evaluate(element, [b, a])
                        evaluate(element, [b, b])

        self.assertListEqual([], errors)

    def test_factory_contains_exactly(self) -> None:
        FunctionFactoryAssert(self, fl.FunctionFactory()) \
            .contains_exactly(
            {'!', '~', '^', '**', '*', '/', '%', '+', '-', '.+', '.-', 'and', 'or'},
            fl.Function.Element.Type.Operator) \
            .contains_exactly(
            {'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil',
             'cos', 'cosh', 'eq', 'exp', 'fabs', 'floor', 'fmod', 'ge', 'gt', 'le',
             'log', 'log10', 'log1p', 'lt', 'max', 'min', 'neq', 'pi', 'pow', 'round', 'sin',
             'sinh', 'sqrt', 'tan', 'tanh'},
            fl.Function.Element.Type.Function)

    def test_function_operators(self) -> None:
        FunctionFactoryAssert(self, fl.FunctionFactory()) \
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

    @unittest.skip("Until fl.Function is ready")
    def test_function_precedence(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
