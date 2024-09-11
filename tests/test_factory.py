"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import itertools
import unittest
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Union,
)

import numpy as np

import fuzzylite as fl
from fuzzylite.types import Self
from tests.assert_component import BaseAssert


class FactoryAssert(BaseAssert[Union[fl.ConstructionFactory[Any], fl.CloningFactory[Any]]]):
    """Factory assert."""

    def has_class_name(self, name: str) -> FactoryAssert:
        """Asserts the factory has the expected class name."""
        self.test.assertEqual(fl.Op.class_name(self.actual), name)
        return self

    def contains(self, name: str | Iterable[str], contains: bool = True) -> FactoryAssert:
        """Asserts whether the factory contains specific class names."""
        if isinstance(name, str):
            name = [name]
        for string in name:
            if contains:
                self.test.assertIn(
                    string,
                    self.actual,
                    f"'{string}' is not in factory {fl.Op.class_name(self.actual)}",
                )
            else:
                self.test.assertNotIn(
                    string,
                    self.actual,
                    f"'{string}' is in factory {fl.Op.class_name(self.actual)}",
                )
        return self

    def constructs_exactly(self, name_type: dict[str, type]) -> FactoryAssert:
        """Asserts the factory constructs expected types from the names."""
        if not isinstance(self.actual, fl.ConstructionFactory):
            raise ValueError(
                f"expected an instance of {fl.ConstructionFactory}, but got {self.actual}"
            )
        self.test.assertDictEqual(name_type, self.actual.constructors)
        for name, clazz in name_type.items():
            self.test.assertEqual(type(self.actual.construct(name)), clazz)
        return self

    def copies_exactly(self, name_instance: dict[str, object]) -> FactoryAssert:
        """Assert the factory clones the objects from the class names."""
        if not isinstance(self.actual, fl.CloningFactory):
            raise ValueError(f"expected an instance of {fl.CloningFactory}, but got {self.actual}")
        self.test.assertSetEqual(set(self.actual.objects.keys()), set(name_instance.keys()))
        for name, instance in name_instance.items():
            self.test.assertEqual(str(self.actual.copy(name)), str(instance))
            self.test.assertNotEqual(repr(self.actual.copy(name)), repr(instance))
        return self


class FunctionFactoryAssert(BaseAssert[fl.FunctionFactory]):
    """Function Factory assert."""

    def contains_exactly(
        self,
        elements: set[str],
        element_type: fl.Function.Element.Type | None = None,
    ) -> Self:
        """Assert the factory contains only the expected elements."""
        if element_type == fl.Function.Element.Type.Operator:
            self.test.assertSetEqual(set(self.actual.operators().keys()), elements)
        elif element_type == fl.Function.Element.Type.Function:
            self.test.assertSetEqual(set(self.actual.functions().keys()), elements)
        else:
            self.test.assertSetEqual(set(self.actual.objects.keys()), elements)
        return self

    def operation_is(self, operation_value: dict[tuple[str, Sequence[float]], float]) -> Self:
        """Assert the operation on the sequence of values results in the expected value."""
        for operation, expected in operation_value.items():
            name = operation[0]
            args = operation[1]
            element = self.actual.objects[name]
            obtained = element.method(*args)
            message = (
                f"expected {name}({', '.join([fl.Op.str(x) for x in args])}) to result in "
                f"{fl.Op.str(expected)}, but got {fl.Op.str(obtained)}"
            )
            np.testing.assert_allclose(
                obtained,
                expected,
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
                err_msg=message,
            )
        return self

    def precedence_is_the_same(self, *operators: str) -> Self:
        """Assert the precedence of the operators is the same as the expected."""
        elements = self.actual.operators()
        precedence = {operator: elements[operator].precedence for operator in operators}
        same_precedence = set(precedence.values())
        self.test.assertEqual(
            len(same_precedence), 1, msg=f"precedence is not the same: {precedence}"
        )
        return self

    def precedence_is_higher(self, a: str, b: str) -> Self:
        """Assert the precedence of the operators is different."""
        elements = self.actual.operators()
        self.test.assertGreater(
            elements[a].precedence,
            elements[b].precedence,
            msg=f"expected precedence of {a} ({elements[a].precedence}) > {b} ({elements[b].precedence}), "
            f"but got {elements[a].precedence} <= {elements[b].precedence}",
        )
        return self


class TestFactory(unittest.TestCase):
    """Test factories."""

    def test_construction_factory(self) -> None:
        """Test the construction factory on an arbitrary class."""
        FactoryAssert(self, fl.ConstructionFactory()).has_class_name(
            "ConstructionFactory"
        ).constructs_exactly({})

        class Example:
            def __str__(self) -> str:
                return "instance of Example"

        FactoryAssert(
            self, fl.ConstructionFactory(constructors={"example": Example})
        ).constructs_exactly({"example": Example})

        self.assertEqual(
            str(fl.ConstructionFactory(constructors={"example": Example}).construct("example")),
            "instance of Example",
        )

    def test_iter_len_getitem_setitem(self) -> None:
        """Test iter, len, getitem, and setitem on construction factories."""
        self.assertEqual(0, len(fl.ConstructionFactory()))
        self.assertEqual(23, len(fl.TermFactory()))

        factory = fl.DefuzzifierFactory()
        self.assertEqual(factory["Centroid"], fl.Centroid)

        factory["CoG"] = fl.Centroid
        self.assertEqual(factory["CoG"], fl.Centroid)

        iterator = iter(factory)
        for defuzzifier in factory:
            self.assertEqual(defuzzifier, next(iterator))

        defuzzifiers = [
            "Bisector",
            "Centroid",
            "LargestOfMaximum",
            "MeanOfMaximum",
            "SmallestOfMaximum",
            "WeightedAverage",
            "WeightedSum",
            "CoG",
        ]
        for index, defuzzifier in enumerate(factory):
            self.assertEqual(defuzzifiers[index], defuzzifier)

        self.assertEqual([d for d in factory], defuzzifiers)

    def test_activation_factory(self) -> None:
        """Test the activation factory."""
        FactoryAssert(self, fl.ActivationFactory()).has_class_name("ActivationFactory").contains(
            ["First", "Last", "Threshold"]
        ).contains(["Second", "Third"], False).constructs_exactly(
            {
                "First": fl.First,
                "General": fl.General,
                "Highest": fl.Highest,
                "Last": fl.Last,
                "Lowest": fl.Lowest,
                "Proportional": fl.Proportional,
                "Threshold": fl.Threshold,
            }
        )

    def test_defuzzifier_factory(self) -> None:
        """Test the defuzzifier factory."""
        FactoryAssert(self, fl.DefuzzifierFactory()).has_class_name("DefuzzifierFactory").contains(
            ["Bisector", "MeanOfMaximum", "WeightedSum"]
        ).contains(["Something", "Else"], False).constructs_exactly(
            {
                "Bisector": fl.Bisector,
                "Centroid": fl.Centroid,
                "LargestOfMaximum": fl.LargestOfMaximum,
                "MeanOfMaximum": fl.MeanOfMaximum,
                "SmallestOfMaximum": fl.SmallestOfMaximum,
                "WeightedAverage": fl.WeightedAverage,
                "WeightedSum": fl.WeightedSum,
            }
        )

    def test_hedge_factory(self) -> None:
        """Test the hedge factory."""
        FactoryAssert(self, fl.HedgeFactory()).has_class_name("HedgeFactory").contains(
            ["any", "seldom", "very"]
        ).contains(["very much", "often"], False).constructs_exactly(
            {
                "any": fl.Any,
                "extremely": fl.Extremely,
                "not": fl.Not,
                "seldom": fl.Seldom,
                "somewhat": fl.Somewhat,
                "very": fl.Very,
            }
        )

    def test_snorm_factory(self) -> None:
        """Test the S-Norm factory."""
        FactoryAssert(self, fl.SNormFactory()).has_class_name("SNormFactory").contains(
            ["AlgebraicSum", "EinsteinSum", "UnboundedSum"]
        ).contains(
            ["AlgebraicProduct", "EinsteinProduct", "UnboundedProduct"], False
        ).constructs_exactly(
            {
                "AlgebraicSum": fl.AlgebraicSum,
                "BoundedSum": fl.BoundedSum,
                "DrasticSum": fl.DrasticSum,
                "EinsteinSum": fl.EinsteinSum,
                "HamacherSum": fl.HamacherSum,
                "Maximum": fl.Maximum,
                "NilpotentMaximum": fl.NilpotentMaximum,
                "NormalizedSum": fl.NormalizedSum,
                "UnboundedSum": fl.UnboundedSum,
            }
        )

    def test_tnorm_factory(self) -> None:
        """Test the T-Norm factory."""
        FactoryAssert(self, fl.TNormFactory()).has_class_name("TNormFactory").contains(
            ["AlgebraicProduct", "EinsteinProduct", "NilpotentMinimum"]
        ).contains(["AlgebraicSum", "EinsteinSum", "UnboundedSum"], False).constructs_exactly(
            {
                "AlgebraicProduct": fl.AlgebraicProduct,
                "BoundedDifference": fl.BoundedDifference,
                "DrasticProduct": fl.DrasticProduct,
                "EinsteinProduct": fl.EinsteinProduct,
                "HamacherProduct": fl.HamacherProduct,
                "Minimum": fl.Minimum,
                "NilpotentMinimum": fl.NilpotentMinimum,
            }
        )

    def test_term_factory(self) -> None:
        """Test the term factory."""
        FactoryAssert(self, fl.TermFactory()).has_class_name("TermFactory").contains(
            ["Bell", "Gaussian", "ZShape"]
        ).contains(["Star", "Cube", "Sphere"], False).constructs_exactly(
            {
                "Arc": fl.Arc,
                "Bell": fl.Bell,
                "Binary": fl.Binary,
                "Concave": fl.Concave,
                "Constant": fl.Constant,
                "Cosine": fl.Cosine,
                "Discrete": fl.Discrete,
                "Function": fl.Function,
                "Gaussian": fl.Gaussian,
                "GaussianProduct": fl.GaussianProduct,
                "Linear": fl.Linear,
                "PiShape": fl.PiShape,
                "Ramp": fl.Ramp,
                "Rectangle": fl.Rectangle,
                "SemiEllipse": fl.SemiEllipse,
                "Sigmoid": fl.Sigmoid,
                "SigmoidDifference": fl.SigmoidDifference,
                "SigmoidProduct": fl.SigmoidProduct,
                "Spike": fl.Spike,
                "SShape": fl.SShape,
                "Trapezoid": fl.Trapezoid,
                "Triangle": fl.Triangle,
                "ZShape": fl.ZShape,
            }
        )

    def test_cloning_factory(self) -> None:
        """Test the cloning factory."""
        FactoryAssert(self, fl.CloningFactory()).has_class_name("CloningFactory").copies_exactly({})

        class Example:
            def __init__(self, value: str) -> None:
                self.property = value

            def __str__(self) -> str:
                return f"Example({str(self.property)})"

        FactoryAssert(
            self, fl.CloningFactory(objects={"example": Example("Clone of Example")})
        ).copies_exactly({"example": Example("Clone of Example")})


class TestFunctionFactory(unittest.TestCase):
    """Test the function factory."""

    def test_factory_precedence(self) -> None:
        """Test the precedence values are internally inversed."""
        precedence_expected = {
            0: 100,
            1: 90,
            2: 80,
            3: 70,
            4: 60,
            5: 50,
            6: 40,
            7: 30,
            8: 20,
            9: 10,
            10: 0,
        }
        factory = fl.FunctionFactory()
        for p, e in precedence_expected.items():
            self.assertEqual(e, factory._precedence(p))

    def test_len_iter_getitem_setitem(self) -> None:
        """Test iter, len, getitem, and setitem on cloning factories."""
        self.assertEqual(0, len(fl.CloningFactory()))
        self.assertEqual(47, len(fl.FunctionFactory()))

        factory = fl.FunctionFactory()
        self.assertEqual(factory["sin"], factory.objects["sin"])

        factory["sine"] = factory["sin"]
        self.assertEqual(factory["sine"], factory.objects["sin"])

        iterator = iter(factory)
        for function in factory:
            self.assertEqual(function, next(iterator))

        functions = list(factory.operators().keys()) + list(factory.functions().keys())
        for index, function in enumerate(factory):
            self.assertEqual(functions[index], function)

        self.assertEqual([f for f in factory], functions)

    def test_factory_matches_keys_and_names(self) -> None:
        """Test the registration names of functions match the function names."""
        exceptions = {
            "acos": "arccos",
            "asin": "arcsin",
            "atan": "arctan",
            "atan2": "arctan2",
            "acosh": "arccosh",
            "asinh": "arcsinh",
            "atanh": "arctanh",
            "pi": "<lambda>",
        }
        for key, element in fl.FunctionFactory().objects.items():
            self.assertEqual(key, element.name)
            # if it is a function, the name should be contained in
            # in the methods name
            if element.type == fl.Function.Element.Type.Function:
                if key in exceptions:
                    self.assertIn(exceptions[key], element.method.__name__)
                else:
                    self.assertIn(key, element.method.__name__)

    def test_arity(self) -> None:
        """Tests correct arity of functions by calling functions without raising exceptions."""
        values = [-np.inf, -10, -5, -1, -0.5, 0, np.nan, 0.5, 1, 5, 10, np.inf]

        for element in fl.FunctionFactory().objects.values():
            if element.arity == 0:
                element.method()
            else:
                for a, b in itertools.combinations(values, 2):
                    if element.arity == 1:
                        element.method(a)
                        element.method(b)
                    else:
                        element.method(a, a)
                        element.method(a, b)
                        element.method(b, a)
                        element.method(b, b)

    def test_factory_contains_exactly(self) -> None:
        """Test the factory contains all the operators and functions."""
        expected_operators = "! % * ** + - .+ .- / ^ and or ~"
        expected_functions = (
            "abs acos acosh asin asinh atan atan2 atanh ceil cos "
            "cosh eq exp fabs floor fmod ge gt le log "
            "log10 log1p lt max min neq pi pow round sin "
            "sinh sqrt tan tanh"
        )
        FunctionFactoryAssert(self, fl.FunctionFactory()).contains_exactly(
            set(expected_operators.split()),
            fl.Function.Element.Type.Operator,
        ).contains_exactly(
            set(expected_functions.split()),
            fl.Function.Element.Type.Function,
        )

    def test_function_operators(self) -> None:
        """Tests the operators yield correct resutls."""
        FunctionFactoryAssert(self, fl.FunctionFactory()).operation_is(
            {
                ("!", (0,)): 1,
                ("!", (1,)): 0,
                ("~", (1,)): -1,
                ("~", (-2,)): 2,
                ("~", (0,)): 0,
                ("^", (3, 3)): 27,
                ("^", (9, 0.5)): 3,
                ("*", (-2, 3)): -6,
                ("*", (3, -2)): -6,
                ("/", (6, 3)): 2,
                ("/", (3, 6)): 0.5,
                ("%", (6, 3)): 0,
                ("%", (3, 6)): 3,
                ("%", (3.5, 6)): 3.5,
                ("%", (6, 3.5)): 2.5,
                ("+", (2, 3)): 5,
                ("+", (2, -3)): -1,
                ("-", (2, 3)): -1,
                ("-", (2, -3)): 5,
                ("and", (1, 0)): 0,
                ("and", (1, 1)): 1,
                ("or", (1, 0)): 1,
                ("or", (0, 0)): 0,
            }
        )

    def test_function_precedence(self) -> None:
        """Tests the precedence of operators."""
        (
            FunctionFactoryAssert(self, fl.FunctionFactory())
            .precedence_is_the_same("!", "~")
            .precedence_is_the_same("^", "**", ".-", ".+")
            .precedence_is_the_same("*", "/", "%")
            .precedence_is_the_same("+", "-")
        )
        (
            FunctionFactoryAssert(self, fl.FunctionFactory())
            .precedence_is_higher("!", ".+")
            .precedence_is_higher("^", "%")
            .precedence_is_higher("*", "-")
            .precedence_is_higher("+", "and")
            .precedence_is_higher("and", "or")
        )

        np.testing.assert_allclose(
            fl.Function("f", "(10 + 5) * 2 - 3 / 4 ** 2", load=True).evaluate(),
            29.8125,
        )


if __name__ == "__main__":
    unittest.main()
