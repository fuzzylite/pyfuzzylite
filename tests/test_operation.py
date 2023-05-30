"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited
fuzzylite is a registered trademark of FuzzyLite Limited.
"""
from __future__ import annotations

import math
import unittest
from typing import Callable

import numpy.testing as npt

import fuzzylite as fl


class AssertOperator:
    """Assert operator."""

    def __init__(self, commutative: bool = True, scalable: bool = True) -> None:
        """Construct assert operator."""
        self.commutative = commutative
        self.scalable = scalable

    def assert_operator(
        self,
        operator: Callable[[fl.Scalar, fl.Scalar], fl.Scalar],
        a: fl.Scalar,
        b: fl.Scalar,
        expected: fl.Scalar,
    ) -> AssertOperator:
        """Assert operator."""
        obtained = operator(a, b)
        npt.assert_equal(obtained, expected)
        return self

    def assert_that(
        self,
        operator: Callable[[fl.Scalar, fl.Scalar], fl.Scalar],
        operands_result: dict[tuple[fl.Scalar, fl.Scalar], fl.Scalar],
    ) -> AssertOperator:
        """Assert that."""
        for (a, b), expected in operands_result.items():
            obtained = operator(a, b)
            npt.assert_equal(
                obtained,
                expected,
                err_msg=f"{str(operator.__qualname__)}(a={a}, b={b}) = {obtained}, but expected {expected}",
            )
            if self.commutative:
                obtained = operator(b, a)
                npt.assert_equal(
                    obtained,
                    expected,
                    err_msg=f"{str(operator.__qualname__)}(b={b}, a={a}) = {obtained}, but expected {expected}",
                )

        if self.scalable:
            values = operands_result.keys()
            a = fl.array([a for a, _ in values])
            b = fl.array([b for _, b in values])
            expected = fl.array([z for z in operands_result.values()])
            obtained = operator(a, b)
            npt.assert_equal(
                obtained,
                expected,
                err_msg=f"{str(operator.__qualname__)}(a={a}, b={b}) = {obtained}, but expected {expected}",
            )
            if self.commutative:
                obtained = operator(b, a)
                npt.assert_equal(
                    obtained,
                    expected,
                    err_msg=f"{str(operator.__qualname__)}(b={b}, a={a}) = {obtained}, but expected {expected}",
                )
        return self


# TODO: Complete tests.
class TestOperation(unittest.TestCase):
    """Test operation."""

    def test_operator_eq(self) -> None:
        """Test operator eq."""
        AssertOperator(commutative=True, scalable=True).assert_that(
            fl.Op.eq,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (fl.inf, fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 0.0,
                (-0.0, -0.0): 1.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_operator_neq(self) -> None:
        """Test operator neq."""
        AssertOperator(commutative=True, scalable=True).assert_that(
            fl.Op.neq,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 1.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 1.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_gt(self) -> None:
        """Test operator gt."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.gt,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 1.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_lt(self) -> None:
        """Test operator lt."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.lt,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 0.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_ge(self) -> None:
        """Test operator ge."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.ge,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 1.0,
                (-0.0, 0.0): 1.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_operator_le(self) -> None:
        """Test operator le."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.le,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (-fl.inf, fl.inf): 1.0,
                (-fl.inf, -fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 0.0,
                (-0.0, 0.0): 1.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_valid_identifier(self) -> None:
        """Test what a valid identifier is."""
        self.assertEqual(fl.Op.as_identifier("  xx  "), "xx")  # trims
        self.assertEqual(
            fl.Op.as_identifier("   ~!@#$%^&*()+{}[]:;\"'<>?/,   "), "unnamed"
        )
        self.assertEqual(fl.Op.as_identifier("abc123_.ABC"), "abc123_.ABC")
        self.assertEqual(fl.Op.as_identifier("      "), "unnamed")

    def test_str(self) -> None:
        """Test string operation uses global decimals."""
        fl.lib.decimals = 3
        self.assertEqual(fl.Op.str(0.3), "0.300")
        self.assertEqual(fl.Op.str(-0.3), "-0.300")
        self.assertEqual(fl.Op.str(3), "3")
        self.assertEqual(fl.Op.str(3.0001), "3.000")

        self.assertEqual(fl.Op.str(math.inf), "inf")
        self.assertEqual(fl.Op.str(-math.inf), "-inf")
        self.assertEqual(fl.Op.str(math.nan), "nan")

        fl.lib.decimals = 5
        self.assertEqual(fl.Op.str(0.3), "0.30000")

        fl.lib.decimals = 0
        self.assertEqual(fl.Op.str(0.3), "0")

        fl.lib.decimals = 3

    def test_scale(self) -> None:
        """Test linear interpolation."""
        self.assertEqual(fl.Op.scale(0, 0, 1, -10, 10), -10.0)
        self.assertEqual(fl.Op.scale(0.5, 0, 1, -10, 10), 0.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, -10, 10), 10)

        self.assertEqual(fl.Op.scale(0, 0, 1, 0, 10), 0.0)
        self.assertEqual(fl.Op.scale(0.5, 0, 1, 0, 10), 5.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, 0, 10), 10)

        self.assertEqual(fl.Op.scale(-1, 0, 1, 0, 10), -10.0)
        self.assertEqual(fl.Op.scale(2, 0, 1, 0, 10), 20)

        self.assertEqual(math.isnan(fl.Op.scale(math.nan, 0, 1, 0, 10)), True)
        self.assertEqual(fl.Op.scale(math.inf, 0, 1, 0, 10), math.inf)
        self.assertEqual(fl.Op.scale(-math.inf, 0, 1, 0, 10), -math.inf)

    @unittest.skip("Revisit describe() method")
    def test_describe(self) -> None:
        """Test describe."""
        # TODO: Revisit describe method
        self.assertEqual(
            "OutputVariable[{"
            "'default_value': 'nan', 'defuzzifier': 'None', "
            "'fuzzy': 'term: x Aggregated []', "
            "'lock_previous': 'False', 'previous_value': 'nan'"
            "}]",
            fl.Op.describe(fl.OutputVariable("x", "an x", terms=[fl.Triangle("t")])),
        )
        self.assertEqual(
            "InputVariable[{}]",
            fl.Op.describe(fl.InputVariable("x", "an x", terms=[fl.Triangle("t")])),
        )


if __name__ == "__main__":
    unittest.main()
