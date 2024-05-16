"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

import fuzzylite as fl
from fuzzylite import inf, nan
from tests.assert_component import BaseAssert


class NormAssert(BaseAssert[fl.Norm]):
    """Assert norms."""

    def is_t_norm(self) -> NormAssert:
        """Assert it is a T-Norm."""
        self.test.assertIsInstance(self.actual, fl.TNorm)
        return self

    def is_s_norm(self) -> NormAssert:
        """Assert it is an S-Norm."""
        self.test.assertIsInstance(self.actual, fl.SNorm)
        return self

    def evaluates(
        self,
        abz: dict[tuple[float, float], float],
        commutative: bool = True,
        associative: bool = True,
    ) -> NormAssert:
        """Assert the norm produces the expected values."""
        for ab, z in abz.items():
            npt.assert_allclose(z, self.actual.compute(*ab), equal_nan=True, err_msg=f"in ({ab})")
            # self.test.assertEqual(z, self.actual.compute(*ab), f"in ({ab})")
            if commutative:
                npt.assert_allclose(
                    z,
                    self.actual.compute(*reversed(ab)),
                    equal_nan=True,
                    err_msg=f"when ({tuple(reversed(ab))})",
                )
                # self.test.assertEqual(
                #     z,
                #     self.actual.compute(*reversed(ab)),
                #     f"when ({tuple(reversed(ab))})",
                # )
            if associative:
                a: fl.Scalar = ab[0]
                b: fl.Scalar = ab[1]
                c: fl.Scalar = sum(ab) / 2
                abc = self.actual.compute(self.actual.compute(a, b), c)
                bca = self.actual.compute(self.actual.compute(b, c), a)
                cab = self.actual.compute(self.actual.compute(c, a), b)
                np.testing.assert_allclose(abc, bca, atol=fl.settings.atol, rtol=fl.settings.rtol)
                np.testing.assert_allclose(bca, cab, atol=fl.settings.atol, rtol=fl.settings.rtol)
        # Test as numpy array
        a = fl.array([x[0] for x in abz])
        b = fl.array([x[1] for x in abz])
        expected = fl.array([z for z in abz.values()])
        obtained = self.actual.compute(a, b)
        np.testing.assert_equal(expected, obtained)
        if commutative:
            obtained = self.actual.compute(b, a)
            np.testing.assert_equal(expected, obtained)
        if associative:
            c = (a + b) / 2
            abc = self.actual.compute(self.actual.compute(a, b), c)
            bca = self.actual.compute(self.actual.compute(b, c), a)
            cab = self.actual.compute(self.actual.compute(c, a), b)
            np.testing.assert_allclose(abc, bca, rtol=fl.settings.rtol, atol=fl.settings.atol)
            np.testing.assert_allclose(bca, cab, rtol=fl.settings.rtol, atol=fl.settings.atol)
        return self


class TestTNorm(unittest.TestCase):
    """Test T-Norms."""

    def test_algebraic_product(self) -> None:
        """Test the algebraic product."""
        NormAssert(self, fl.AlgebraicProduct()).is_t_norm().repr_is(
            "fl.AlgebraicProduct()"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): inf,
                (inf, -inf): -inf,
                (-inf, -inf): inf,
            }
        )

    def test_bounded_difference(self) -> None:
        """Test the bounded difference."""
        NormAssert(self, fl.BoundedDifference()).is_t_norm().repr_is(
            "fl.BoundedDifference()"
        ).exports_fll("BoundedDifference").evaluates(
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
                (nan, nan): nan,
                (inf, inf): inf,
                (inf, -inf): nan,
                (-inf, -inf): 0,
            }
        )

    def test_drastic_product(self) -> None:
        """Test the drastic product."""
        NormAssert(self, fl.DrasticProduct()).is_t_norm().repr_is(
            "fl.DrasticProduct()"
        ).exports_fll("DrasticProduct").evaluates(
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
                (nan, nan): 0.0,
                (inf, inf): 0.0,
                (inf, -inf): 0.0,
                (-inf, -inf): 0.0,
            }
        )

    def test_einstein_product(self) -> None:
        """Test the einstein product."""
        NormAssert(self, fl.EinsteinProduct()).is_t_norm().repr_is(
            "fl.EinsteinProduct()"
        ).exports_fll("EinsteinProduct").evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): nan,
            }
        )

    def test_hamacher_product(self) -> None:
        """Test the hamacher product."""
        NormAssert(self, fl.HamacherProduct()).is_t_norm().repr_is(
            "fl.HamacherProduct()"
        ).exports_fll("HamacherProduct").evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): nan,
            }
        )

    def test_minimum(self) -> None:
        """Test the minimum."""
        NormAssert(self, fl.Minimum()).is_t_norm().repr_is("fl.Minimum()").exports_fll(
            "Minimum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): inf,
                (inf, -inf): -inf,
                (-inf, -inf): -inf,
            }
        )

    def test_nilpotent_minimum(self) -> None:
        """Test the nilpotent minimum."""
        NormAssert(self, fl.NilpotentMinimum()).is_t_norm().repr_is(
            "fl.NilpotentMinimum()"
        ).exports_fll("NilpotentMinimum").evaluates(
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
                (nan, nan): 0.0,
                (inf, inf): inf,
                (inf, -inf): 0.0,
                (-inf, -inf): 0.0,
            }
        )


class TestSNorm(unittest.TestCase):
    """Test the S-Norms."""

    def test_algebraic_sum(self) -> None:
        """Test the algebraic sum."""
        NormAssert(self, fl.AlgebraicSum()).is_s_norm().repr_is("fl.AlgebraicSum()").exports_fll(
            "AlgebraicSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )

    def test_bounded_sum(self) -> None:
        """Test the bounded sum."""
        NormAssert(self, fl.BoundedSum()).is_s_norm().repr_is("fl.BoundedSum()").exports_fll(
            "BoundedSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): 1.0,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )

    def test_drastic_sum(self) -> None:
        """Test the drastic sum."""
        NormAssert(self, fl.DrasticSum()).is_s_norm().repr_is("fl.DrasticSum()").exports_fll(
            "DrasticSum"
        ).evaluates(
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
                (nan, nan): 1.0,
                (inf, inf): 1.0,
                (inf, -inf): 1.0,
                (-inf, -inf): 1.0,
            }
        )

    def test_einstein_sum(self) -> None:
        """Test the einstein sum."""
        NormAssert(self, fl.EinsteinSum()).is_s_norm().repr_is("fl.EinsteinSum()").exports_fll(
            "EinsteinSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): nan,
            }
        )

    def test_hamacher_sum(self) -> None:
        """Test the hamacher sum."""
        NormAssert(self, fl.HamacherSum()).is_s_norm().repr_is("fl.HamacherSum()").exports_fll(
            "HamacherSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): nan,
            }
        )

    def test_maximum(self) -> None:
        """Test the maximum."""
        NormAssert(self, fl.Maximum()).is_s_norm().repr_is("fl.Maximum()").exports_fll(
            "Maximum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): inf,
                (inf, -inf): inf,
                (-inf, -inf): -inf,
            }
        )

    def test_nilpotent_maximum(self) -> None:
        """Test the nilpotent maximum."""
        NormAssert(self, fl.NilpotentMaximum()).is_s_norm().repr_is(
            "fl.NilpotentMaximum()"
        ).exports_fll("NilpotentMaximum").evaluates(
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
                (nan, nan): 1.0,
                (inf, inf): 1.0,
                (inf, -inf): 1.0,
                (-inf, -inf): -inf,
            }
        )

    def test_normalized_sum(self) -> None:
        """Test the normalised sum."""
        NormAssert(self, fl.NormalizedSum()).is_s_norm().repr_is("fl.NormalizedSum()").exports_fll(
            "NormalizedSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )

    def test_unbounded_sum(self) -> None:
        """Test the unbounded sum."""
        NormAssert(self, fl.UnboundedSum()).is_s_norm().repr_is("fl.UnboundedSum()").exports_fll(
            "UnboundedSum"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): inf,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )


class TestNormFunctions(unittest.TestCase):
    """Test the norm functions."""

    def test_norm_function(self) -> None:
        """Test the norm function."""
        NormAssert(
            self, fl.NormFunction(fl.Function.create("AlgebraicSum", "a + b - (a * b)"))
        ).exports_fll("NormFunction").is_s_norm().is_t_norm().repr_is(
            "fl.NormFunction(fl.Function('AlgebraicSum', 'a + b - (a * b)'))"
        ).evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )

    def test_norm_lambda(self) -> None:
        """Test the norm lambda."""
        NormAssert(self, fl.NormLambda(lambda a, b: a + b - (a * b))).exports_fll(
            "NormLambda"
        ).is_s_norm().is_t_norm().repr_is("fl.NormLambda(lambda a, b: ...)").evaluates(
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
                (nan, nan): nan,
                (inf, inf): nan,
                (-inf, inf): nan,
                (inf, -inf): nan,
                (-inf, -inf): -inf,
            }
        )


if __name__ == "__main__":
    unittest.main()
