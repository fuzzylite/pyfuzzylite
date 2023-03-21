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
import unittest
from typing import Dict, Tuple

import fuzzylite as fl
from tests.assert_component import BaseAssert


class NormAssert(BaseAssert[fl.Norm]):
    """Assert norms."""

    def is_t_norm(self) -> "NormAssert":
        """Assert it is a T-Norm."""
        self.test.assertIsInstance(self.actual, fl.TNorm)
        return self

    def is_s_norm(self) -> "NormAssert":
        """Assert it is an S-Norm."""
        self.test.assertIsInstance(self.actual, fl.SNorm)
        return self

    def evaluates(
        self, abz: Dict[Tuple[float, float], float], commutative: bool = True
    ) -> "NormAssert":
        """Assert the norm produces the expected values."""
        for ab, z in abz.items():
            self.test.assertEqual(z, self.actual.compute(*ab), f"in ({ab})")
            if commutative:
                self.test.assertEqual(
                    z,
                    self.actual.compute(*reversed(ab)),
                    f"when ({tuple(reversed(ab))})",
                )
        return self


class TestTNorm(unittest.TestCase):
    """Test T-Norms."""

    def test_algebraic_product(self) -> None:
        """Test the algebraic product."""
        NormAssert(self, fl.AlgebraicProduct()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.125,
                (0.50, 0.50): 0.250,
                (0.50, 0.75): 0.375,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_bounded_difference(self) -> None:
        """Test the bounded difference."""
        NormAssert(self, fl.BoundedDifference()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.0,
                (0.50, 0.50): 0.0,
                (0.50, 0.75): 0.25,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_drastic_product(self) -> None:
        """Test the drastic product."""
        NormAssert(self, fl.DrasticProduct()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.00,
                (0.50, 0.50): 0.00,
                (0.50, 0.75): 0.00,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_einstein_product(self) -> None:
        """Test the einstein product."""
        NormAssert(self, fl.EinsteinProduct()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.09090909090909091,
                (0.50, 0.50): 0.20,
                (0.50, 0.75): 0.3333333333333333,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_hamacher_product(self) -> None:
        """Test the hamacher product."""
        NormAssert(self, fl.HamacherProduct()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.2,
                (0.50, 0.50): 0.3333333333333333,
                (0.50, 0.75): 0.42857142857142855,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_minimum(self) -> None:
        """Test the minimum."""
        NormAssert(self, fl.Minimum()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.25,
                (0.50, 0.50): 0.50,
                (0.50, 0.75): 0.50,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )

    def test_nilpotent_minimum(self) -> None:
        """Test the nilpotent minimum."""
        NormAssert(self, fl.NilpotentMinimum()).is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.00,
                (0.00, 0.50): 0.00,
                (0.00, 0.75): 0.00,
                (0.00, 1.00): 0.00,
                (0.50, 0.25): 0.0,
                (0.50, 0.50): 0.00,
                (0.50, 0.75): 0.50,
                (1.00, 0.00): 0.00,
                (1.00, 0.25): 0.25,
                (1.00, 0.50): 0.50,
                (1.00, 0.75): 0.75,
                (1.00, 1.00): 1.00,
            }
        )


class TestSNorm(unittest.TestCase):
    """Test the S-Norms."""

    def test_algebraic_sum(self) -> None:
        """Test the algebraic sum."""
        NormAssert(self, fl.AlgebraicSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.625,
                (0.50, 0.50): 0.75,
                (0.50, 0.75): 0.875,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_bounded_sum(self) -> None:
        """Test the bounded sum."""
        NormAssert(self, fl.BoundedSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.75,
                (0.50, 0.50): 1.00,
                (0.50, 0.75): 1.00,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_drastic_sum(self) -> None:
        """Test the drastic sum."""
        NormAssert(self, fl.DrasticSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 1.00,
                (0.50, 0.50): 1.00,
                (0.50, 0.75): 1.00,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_einstein_sum(self) -> None:
        """Test the einstein sum."""
        NormAssert(self, fl.EinsteinSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.6666666666666666,
                (0.50, 0.50): 0.80,
                (0.50, 0.75): 0.9090909090909091,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_hamacher_sum(self) -> None:
        """Test the hamacher sum."""
        NormAssert(self, fl.HamacherSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.5714285714285714,
                (0.50, 0.50): 0.6666666666666666,
                (0.50, 0.75): 0.80,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_maximum(self) -> None:
        """Test the maximum."""
        NormAssert(self, fl.Maximum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.50,
                (0.50, 0.50): 0.50,
                (0.50, 0.75): 0.75,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_nilpotent_maximum(self) -> None:
        """Test the nilpotent maximum."""
        NormAssert(self, fl.NilpotentMaximum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.50,
                (0.50, 0.50): 1.00,
                (0.50, 0.75): 1.00,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_normalized_sum(self) -> None:
        """Test the normalised sum."""
        NormAssert(self, fl.NormalizedSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.75,
                (0.50, 0.50): 1.00,
                (0.50, 0.75): 1.00,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_unbounded_sum(self) -> None:
        """Test the unbounded sum."""
        NormAssert(self, fl.UnboundedSum()).is_s_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.75,
                (0.50, 0.50): 1.00,
                (0.50, 0.75): 1.25,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.25,
                (1.00, 0.50): 1.50,
                (1.00, 0.75): 1.75,
                (1.00, 1.00): 2.00,
            }
        )


class TestNormFunctions(unittest.TestCase):
    """Test the norm functions."""

    def test_norm_function(self) -> None:
        """Test the norm function."""
        NormAssert(
            self, fl.NormFunction(fl.Function.create("AlgebraicSum", "a + b - (a * b)"))
        ).exports_fll("NormFunction").is_s_norm().is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.625,
                (0.50, 0.50): 0.75,
                (0.50, 0.75): 0.875,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )

    def test_norm_lambda(self) -> None:
        """Test the norm lambda."""
        NormAssert(self, fl.NormLambda(lambda a, b: a + b - (a * b))).exports_fll(
            "NormLambda"
        ).is_s_norm().is_t_norm().evaluates(
            {
                (0.00, 0.00): 0.00,
                (0.00, 0.25): 0.25,
                (0.00, 0.50): 0.50,
                (0.00, 0.75): 0.75,
                (0.00, 1.00): 1.00,
                (0.50, 0.25): 0.625,
                (0.50, 0.50): 0.75,
                (0.50, 0.75): 0.875,
                (1.00, 0.00): 1.00,
                (1.00, 0.25): 1.00,
                (1.00, 0.50): 1.00,
                (1.00, 0.75): 1.00,
                (1.00, 1.00): 1.00,
            }
        )


if __name__ == "__main__":
    unittest.main()
