"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import copy
import operator
import re
import unittest
from collections.abc import Sequence
from typing import Callable, NoReturn

import numpy as np

import fuzzylite as fl
import fuzzylite.library
from fuzzylite import Scalar, inf, nan
from fuzzylite.types import Self
from tests.assert_component import BaseAssert


class TermAssert(BaseAssert[fl.Term]):
    """Term assert."""

    def has_name(self, name: str, height: float = 1.0) -> Self:
        """Assert the term has the expected name and height."""
        self.test.assertEqual(self.actual.name, name)
        self.test.assertEqual(self.actual.height, height)
        return self

    def takes_parameters(self, parameters: int) -> Self:
        """Assert the term takes the number of parameters for configuration."""
        with self.test.assertRaises(ValueError) as error:
            self.actual.__class__().configure("")
        self.test.assertIn(
            str(error.exception),
            {
                f"expected {parameters} parameters, but got 0: ''",
                f"expected {parameters} parameters (or {parameters + 1} including height), but got 0: ''",
            },
        )

        too_many_parameters = " ".join(str(x + 1) for x in range(parameters + 2))
        with self.test.assertRaises(ValueError) as error:
            self.actual.__class__().configure(too_many_parameters)
        self.test.assertIn(
            str(error.exception),
            {
                f"expected {parameters} parameters, but got {parameters + 2}: '{too_many_parameters}'",
                f"expected {parameters} parameters (or {parameters + 1} including height), but got {parameters + 2}: '{too_many_parameters}'",
            },
        )

        return self

    def is_monotonic(self, monotonic: bool = True) -> Self:
        """Assert the term is monotonic."""
        self.test.assertEqual(monotonic, self.actual.is_monotonic())
        return self

    def is_not_monotonic(self) -> Self:
        """Assert the term is not monotonic."""
        self.test.assertEqual(False, self.actual.is_monotonic())
        return self

    def configured_as(self, parameters: str) -> Self:
        """Configure the term with the parameters."""
        self.actual.configure(parameters)
        return self

    def has_memberships(
        self, x_mf: dict[float, float], heights: Sequence[float] | None = None
    ) -> Self:
        """Assert the term's membership function produces $f(x{_keys}) = mf_{values}$."""
        inputs = fl.scalar(list(x_mf.keys()))
        if heights is None:
            heights = [0.0, 0.25, 0.5, 0.75, 1.0]
        for height in heights:
            self.actual.height = height
            outputs = height * fl.scalar(list(x_mf.values()))
            self.has_membership(dict(zip(inputs, outputs)))
        return self

    def has_membership(self, x_mf: dict[float, float]) -> Self:
        """Assert the term's membership function produces $f(x{_keys}) = mf_{values}$."""
        inputs = fl.scalar(list(x_mf.keys()))
        expected = fl.scalar(list(x_mf.values()))
        if isinstance(self.actual, fl.Linear):
            self.test.assertIsNotNone(self.actual.engine)
            # membership function of linear terms do not depend on parameter inputs,
            # instead it depends on the input values of the variables.
            # here we extend each input value to match the length of the inputs
            # so the obtained membership has the same shape as the expected values
            for input_variable in self.actual.engine.input_variables:  # type: ignore
                input_variable.value = np.full_like(inputs, np.atleast_1d(input_variable.value)[0])

        obtained = self.actual.membership(inputs)
        np.testing.assert_allclose(
            expected,
            obtained,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )
        return self

    def membership_fails(self, x: float, exception: type[Exception], message: str) -> Self:
        """Assert the membership function raises the exception when evaluating $f(x)$."""
        with self.test.assertRaises(exception) as error:
            self.actual.membership(x)
        self.test.assertEqual(str(error.exception), message, msg=f"when x={x:.3f}")
        return self

    def has_tsukamotos(self, x_mf: dict[float, float]) -> Self:
        """Assert the term computes all Tsukamoto values correctly."""
        self.test.assertEqual(True, self.actual.is_monotonic())
        expected = fl.scalar([mf for mf in x_mf.values()])
        obtained = self.actual.tsukamoto(fl.scalar([x for x in x_mf]))
        np.testing.assert_allclose(
            expected,
            obtained,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )
        return self

    def apply(
        self,
        func: Callable[..., None],
        args: Sequence[str] = (),
        **keywords: dict[str, object],
    ) -> Self:
        """Applies function on the term with the arguments and keywords as parameters."""
        func(self.actual, *args, **keywords)
        return self


class TestTerm(unittest.TestCase):
    """Test terms."""

    def setUp(self) -> None:
        """Display the entire diff in tests."""
        self.maxDiff = None

    def test_term(self) -> None:
        """Test the base term."""

        class BaseTerm(fl.Term):
            """Base term for testing."""

            def membership(self, x: Scalar) -> Scalar:
                """Returns nan for testing."""
                return np.full_like(x, nan)

        (
            TermAssert(self, BaseTerm("x", 0.5))
            .has_name("x", height=0.5)
            .repr_is("BaseTerm('x', 0.5)", with_alias="*", validate=False)
            .repr_is(
                "tests.test_term.BaseTerm('x', 0.5)",
                with_alias="",
                validate=False,
            )
            .exports_fll("term: x BaseTerm 0.500")
            .is_not_monotonic()
        )

        discrete_base = BaseTerm().discretize(-1, 1, 10, midpoints=False)
        xy = {
            -1.0: nan,
            -0.8: nan,
            -0.6: nan,
            -0.4: nan,
            -0.2: nan,
            0.0: nan,
            0.2: nan,
            0.4: nan,
            0.6: nan,
            0.8: nan,
            1.0: nan,
        }
        np.testing.assert_allclose(discrete_base.x(), fl.array([x for x in xy]))
        np.testing.assert_allclose(discrete_base.y(), fl.array([y for y in xy.values()]))

        discrete_base = BaseTerm().discretize(-1, 1, 10, midpoints=True)
        xy = {
            -0.9: nan,
            -0.7: nan,
            -0.5: nan,
            -0.3: nan,
            -0.1: nan,
            0.1: nan,
            0.3: nan,
            0.5: nan,
            0.7: nan,
            0.9: nan,
        }
        np.testing.assert_allclose(discrete_base.x(), fl.array([x for x in xy]))
        np.testing.assert_allclose(discrete_base.y(), fl.array([y for y in xy.values()]))

    def test_activated(self) -> None:
        """Test the activated term."""
        TermAssert(
            self,
            fl.Activated(
                fl.Triangle("triangle", -0.400, 0.000, 0.400),
                1.0,
                fl.AlgebraicProduct(),
            ),
        ).repr_is(
            "fl.Activated(term=fl.Triangle('triangle', -0.4, 0.0, 0.4), "
            "degree=1.0, implication=fl.AlgebraicProduct())"
        ).exports_fll(
            "term: _ Activated AlgebraicProduct(1.000,triangle)"
        ).is_not_monotonic().has_membership(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.375,
                -0.1: 0.750,
                0.0: 1.000,
                0.1: 0.750,
                0.25: 0.375,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )

        TermAssert(
            self,
            fl.Activated(
                fl.Triangle("triangle", -0.400, 0.000, 0.400),
                0.5,
                fl.AlgebraicProduct(),
            ),
        ).exports_fll(
            "term: _ Activated AlgebraicProduct(0.500,triangle)"
        ).is_not_monotonic().has_membership(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.18750000000000003,
                -0.1: 0.37500000000000006,
                0.0: 0.5,
                0.1: 0.37500000000000006,
                0.25: 0.18750000000000003,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )

        TermAssert(
            self,
            fl.Activated(
                fl.Triangle("triangle", -0.400, 0.000, 0.400),
                fl.array([0.0, 0.5, 1.0]),
                fl.AlgebraicProduct(),
            ),
        ).exports_fll("term: _ Activated AlgebraicProduct([0.000, 0.500, 1.000],triangle)")

        activated = fl.Activated(fl.Triangle("x", 0, 1), degree=1.0)
        with self.assertRaisesRegex(ValueError, "expected an implication operator, but found none"):
            activated.membership(0.0)

    def test_aggregated(self) -> None:
        """Test the aggregated term."""
        aggregated = fl.Aggregated("fuzzy_output", -1.0, 1.0, fl.Maximum())
        low = fl.Triangle("LOW", -1.000, -0.500, 0.000)
        medium = fl.Triangle("MEDIUM", -0.500, 0.000, 0.500)
        aggregated.terms = [
            fl.Activated(low, 0.6, fl.Minimum()),
            fl.Activated(medium, 0.4, fl.Minimum()),
        ]

        TermAssert(self, aggregated).exports_fll(
            "term: fuzzy_output Aggregated Maximum[Minimum(0.600,LOW),Minimum(0.400,MEDIUM)]"
        ).repr_is(
            "fl.Aggregated(name='fuzzy_output', minimum=-1.0, maximum=1.0, "
            "aggregation=fl.Maximum(), terms=[fl.Activated(term=fl.Triangle('LOW', -1.0, "
            "-0.5, 0.0), degree=0.6, implication=fl.Minimum()), "
            "fl.Activated(term=fl.Triangle('MEDIUM', -0.5, 0.0, 0.5), degree=0.4, "
            "implication=fl.Minimum())])"
        ).is_not_monotonic().has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.6,
                -0.4: 0.6,
                -0.25: 0.5,
                -0.1: 0.4,
                0.0: 0.4,
                0.1: 0.4,
                0.25: 0.4,
                0.4: 0.2,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            },
            heights=[1.0],
        )

        self.assertEqual(aggregated.activation_degree(low), 0.6)
        self.assertEqual(aggregated.activation_degree(medium), 0.4)

        self.assertEqual(aggregated.highest_activated_term().term, low)  # type: ignore

        self.assertEqual(
            (
                "[fl.Activated(term=fl.Triangle('LOW', -1.0, -0.5, 0.0), degree=0.6, "
                "implication=None), fl.Activated(term=fl.Triangle('MEDIUM', -0.5, 0.0, 0.5), "
                "degree=0.4, implication=None)]"
            ),
            repr(list(aggregated.grouped_terms().values())),
        )

        aggregated.terms.append(fl.Activated(low, 0.4))
        # no change to grouped_terms because of fl.Maximum is 0.6
        self.assertEqual(
            (
                "[fl.Activated(term=fl.Triangle('LOW', -1.0, -0.5, 0.0), degree=0.6, "
                "implication=None), fl.Activated(term=fl.Triangle('MEDIUM', -0.5, 0.0, 0.5), "
                "degree=0.4, implication=None)]"
            ),
            repr(list(aggregated.grouped_terms().values())),
        )

        aggregated.aggregation = fl.UnboundedSum()
        self.assertEqual(aggregated.activation_degree(low), 0.6 + 0.4)
        self.assertEqual(
            (
                "[fl.Activated(term=fl.Triangle('LOW', -1.0, -0.5, 0.0), degree=1.0, "
                "implication=None), fl.Activated(term=fl.Triangle('MEDIUM', -0.5, 0.0, 0.5), "
                "degree=0.4, implication=None)]"
            ),
            repr(list(aggregated.grouped_terms().values())),
        )

        aggregated.aggregation = None
        TermAssert(self, aggregated).exports_fll(
            "term: fuzzy_output Aggregated [Minimum(0.600,LOW)+Minimum(0.400,MEDIUM)+(0.400*LOW)]"
        )

        with self.assertRaisesRegex(ValueError, "expected an aggregation operator, but found none"):
            aggregated.membership(0.0)

        self.assertEqual(aggregated.range(), 2.0)

        aggregated.terms = [
            fl.Activated(low, fl.array([0.6, 0.5]), fl.Minimum()),
            fl.Activated(medium, fl.array([0.5, 0.6]), fl.Minimum()),
        ]
        with self.assertRaises(ValueError) as value_error:
            aggregated.highest_activated_term()
        self.assertEqual(
            "expected a unit scalar, but got vector of size 2: activated.degree=array([0.6, 0.5])",
            str(value_error.exception),
        )

    def test_arc(self) -> None:
        """Test the concave term."""
        TermAssert(self, fl.Arc("arc")).exports_fll("term: arc Arc nan nan").repr_is(
            "fl.Arc('arc', fl.nan, fl.nan)"
        ).takes_parameters(2).is_monotonic().configured_as("-.50 .50").exports_fll(
            "term: arc Arc -0.500 0.500"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0,
                -0.4: 0.436,
                -0.25: 0.661,
                -0.1: 0.8,
                0.0: 0.866,
                0.1: 0.916,
                0.25: 0.968,
                0.4: 0.995,
                0.5: 1.0,
                1.0: 1.0,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        ).has_tsukamotos(
            {
                0.0: -0.5,
                0.25: -0.468,
                0.5: -0.366,
                0.75: -0.161,
                1.0: 0.5,
                # invalid values:
                -1.0: 0.5,
                -0.5: -0.366,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Arc("arc")).configured_as(".50 -.50").exports_fll(
            "term: arc Arc 0.500 -0.500"
        ).repr_is("fl.Arc('arc', 0.5, -0.5)").has_tsukamotos(
            {
                0.0: 0.5,
                0.25: 0.468,
                0.5: 0.366,
                0.75: 0.161,
                1.0: -0.5,
                # invalid values:
                -1.0: -0.5,
                -0.5: 0.366,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Arc("arc")).configured_as("-.50 .50 .5").exports_fll(
            "term: arc Arc -0.500 0.500 0.500"
        ).has_tsukamotos(
            {
                0.0: -0.5,
                0.25: -0.366,
                0.5: 0.5,
                # invalid values:
                0.75: nan,
                1.0: nan,
                -1.0: nan,
                -0.5: 0.5,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Arc("arc")).configured_as(".50 -.50 .5").exports_fll(
            "term: arc Arc 0.500 -0.500 0.500"
        ).has_tsukamotos(
            {
                0.0: 0.5,
                0.25: 0.366,
                0.5: -0.5,
                0.75: nan,
                1.0: nan,
                # invalid values:
                -1.0: nan,
                -0.5: -0.5,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

    def test_bell(self) -> None:
        """Test the bell term."""
        TermAssert(self, fl.Bell("bell")).exports_fll("term: bell Bell nan nan nan").repr_is(
            "fl.Bell('bell', fl.nan, fl.nan, fl.nan)"
        ).takes_parameters(3).is_not_monotonic().configured_as("0 0.25 3.0").exports_fll(
            "term: bell Bell 0.000 0.250 3.000"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.015384615384615385,
                -0.4: 0.05625177755617076,
                -0.25: 0.5,
                -0.1: 0.9959207087768499,
                0.0: 1.0,
                0.1: 0.9959207087768499,
                0.25: 0.5,
                0.4: 0.05625177755617076,
                0.5: 0.015384615384615385,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0 0.25 3.0 0.5"
        ).exports_fll(
            "term: bell Bell 0.000 0.250 3.000 0.500"
        ).repr_is(
            "fl.Bell('bell', 0.0, 0.25, 3.0, 0.5)"
        )

    def test_binary(self) -> None:
        """Test the binary term."""
        TermAssert(self, fl.Binary("binary")).exports_fll("term: binary Binary nan nan").repr_is(
            "fl.Binary('binary', fl.nan, fl.nan)"
        ).takes_parameters(2).is_not_monotonic().configured_as("0 inf").exports_fll(
            "term: binary Binary 0.000 inf"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.0,
                -0.25: 0.0,
                -0.1: 0.0,
                0.0: 1.0,
                0.1: 1.0,
                0.25: 1.0,
                0.4: 1.0,
                0.5: 1.0,
                1.0: 1.0,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Binary("binary")).configured_as("0 -inf 0.5").exports_fll(
            "term: binary Binary 0.000 -inf 0.500"
        ).repr_is("fl.Binary('binary', 0.0, -fl.inf, 0.5)").has_memberships(
            {
                -1: 1.0,
                -0.5: 1.0,
                -0.4: 1.0,
                -0.25: 1.0,
                -0.1: 1.0,
                0.0: 1.0,
                0.1: 0.0,
                0.25: 0.0,
                0.4: 0.0,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            }
        )

    def test_concave(self) -> None:
        """Test the concave term."""
        TermAssert(self, fl.Concave("concave")).exports_fll(
            "term: concave Concave nan nan"
        ).repr_is("fl.Concave('concave', fl.nan, fl.nan)").takes_parameters(
            2
        ).is_monotonic().configured_as(
            "0.00 0.50"
        ).exports_fll(
            "term: concave Concave 0.000 0.500"
        ).has_memberships(
            {
                -1.0: 0.25,
                -0.5: 0.3333333333333333,
                -0.4: 0.35714285714285715,
                -0.25: 0.4,
                -0.1: 0.45454545454545453,
                0.0: 0.5,
                0.1: 0.5555555555555556,
                0.25: 0.6666666666666666,
                0.4: 0.8333333333333334,
                0.5: 1.0,
                1.0: 1.0,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        ).has_tsukamotos(
            {
                0.0: -inf,
                0.25: -1.0,
                0.5: 0.0,
                0.75: 0.333,
                1.0: 0.5,
                # invalid values:
                -1.0: 1.5,
                -0.5: 2.0,
                nan: nan,
                inf: 1.0,
                -inf: 1.0,
            }
        )

        TermAssert(self, fl.Concave("concave")).configured_as("0.00 -0.50").exports_fll(
            "term: concave Concave 0.000 -0.500"
        ).has_memberships(
            {
                -1.0: 1.0,
                -0.5: 1.0,
                -0.4: 0.833,
                -0.25: 0.666,
                -0.1: 0.555,
                0.0: 0.5,
                0.1: 0.455,
                0.25: 0.4,
                0.4: 0.357,
                0.5: 0.333,
                1.0: 0.25,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            }
        ).has_tsukamotos(
            {
                0.0: inf,
                0.25: 1.0,
                0.5: 0.0,
                0.75: -0.333,
                1.0: -0.5,
                # invalid values
                -1.0: -1.5,
                -0.5: -2.0,
                nan: nan,
                inf: -1,
                -inf: -1,
            }
        )

        TermAssert(self, fl.Concave("concave")).configured_as("0.00 -0.500 0.5").repr_is(
            "fl.Concave('concave', 0.0, -0.5, 0.5)"
        ).exports_fll("term: concave Concave 0.000 -0.500 0.500").has_membership(
            {
                -1.0: 0.5,
                -0.5: 0.5,
                -0.4: 0.416,
                -0.25: 0.333,
                -0.1: 0.277,
                0.0: 0.25,
                0.1: 0.227,
                0.25: 0.2,
                0.4: 0.178,
                0.5: 0.166,
                1.0: 0.125,
                nan: nan,
                inf: 0.0,
                -inf: 0.5,
            }
        ).has_tsukamotos(
            {
                0.0: inf,
                0.125: 1.0,
                0.25: 0.0,
                0.375: -0.333,
                0.5: -0.5,
                # invalid values
                0.75: -0.666,
                1.0: -0.75,
                -1.0: -1.25,
                -0.5: -1.5,
                nan: nan,
                inf: -1,
                -inf: -1,
            }
        )

    def test_constant(self) -> None:
        """Test the constant term."""
        TermAssert(self, fl.Constant("constant")).exports_fll(
            "term: constant Constant nan"
        ).repr_is("fl.Constant('constant', fl.nan)").takes_parameters(
            1
        ).is_not_monotonic().configured_as(
            "0.5"
        ).exports_fll(
            "term: constant Constant 0.500"
        ).repr_is(
            "fl.Constant('constant', 0.5)"
        ).has_membership(
            {
                -1.0: 0.5,
                -0.5: 0.5,
                -0.4: 0.5,
                -0.25: 0.5,
                -0.1: 0.5,
                0.0: 0.5,
                0.1: 0.5,
                0.25: 0.5,
                0.4: 0.5,
                0.5: 0.5,
                1.0: 0.5,
                nan: 0.5,
                inf: 0.5,
                -inf: 0.5,
            }
        ).configured_as(
            "-0.500"
        ).repr_is(
            "fl.Constant('constant', -0.5)"
        ).exports_fll(
            "term: constant Constant -0.500"
        ).has_membership(
            {
                -1.0: -0.5,
                -0.5: -0.5,
                -0.4: -0.5,
                -0.25: -0.5,
                -0.1: -0.5,
                0.0: -0.5,
                0.1: -0.5,
                0.25: -0.5,
                0.4: -0.5,
                0.5: -0.5,
                1.0: -0.5,
                nan: -0.5,
                inf: -0.5,
                -inf: -0.5,
            }
        )

    def test_cosine(self) -> None:
        """Test the cosine term."""
        TermAssert(self, fl.Cosine("cosine")).exports_fll("term: cosine Cosine nan nan").repr_is(
            "fl.Cosine('cosine', fl.nan, fl.nan)"
        ).takes_parameters(2).is_not_monotonic().configured_as("0.0 1").exports_fll(
            "term: cosine Cosine 0.000 1.000"
        ).repr_is(
            "fl.Cosine('cosine', 0.0, 1.0)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.09549150281252633,
                -0.25: 0.5,
                -0.1: 0.9045084971874737,
                0.0: 1.0,
                0.1: 0.9045084971874737,
                0.25: 0.5,
                0.4: 0.09549150281252633,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0.0 1.0 0.5"
        ).exports_fll(
            "term: cosine Cosine 0.000 1.000 0.500"
        ).repr_is(
            "fl.Cosine('cosine', 0.0, 1.0, 0.5)"
        )

    def test_discrete(self) -> None:
        """Test the discrete term."""
        TermAssert(self, fl.Discrete("discrete")).exports_fll(
            "term: discrete Discrete"
        ).is_not_monotonic().configured_as("0 1 8 9 4 5 2 3 6 7").exports_fll(
            "term: discrete Discrete 0.000 1.000 8.000 9.000 4.000 5.000 2.000 3.000 6.000 7.000"
        ).apply(
            fl.Discrete.sort
        ).exports_fll(
            "term: discrete Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000"
        ).repr_is(
            "fl.Discrete('discrete', fl.array([fl.array([0.0, 1.0]), fl.array([2.0, "
            "3.0]), fl.array([4.0, 5.0]), fl.array([6.0, 7.0]), fl.array([8.0, 9.0])]))"
        ).configured_as(
            "0 1 8 9 4 5 2 3 6 7 0.5"
        ).apply(
            fl.Discrete.sort
        ).exports_fll(
            "term: discrete Discrete "
            "0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000 0.500"
        ).repr_is(
            "fl.Discrete('discrete', fl.array([fl.array([0.0, 1.0]), fl.array([2.0, "
            "3.0]), fl.array([4.0, 5.0]), fl.array([6.0, 7.0]), fl.array([8.0, 9.0])]), "
            "0.5)"
        ).configured_as(
            " -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000"
        ).exports_fll(
            "term: discrete Discrete "
            "-0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.3999999999999999,
                -0.25: 1.0,
                -0.1: 0.7,
                0.0: 0.5,
                0.1: 0.7,
                0.25: 1.0,
                0.4: 0.3999999999999999,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            " -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000 0.5"
        ).exports_fll(
            "term: discrete Discrete "
            "-0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000 0.500"
        )

        term = fl.Discrete()
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected xy to contain coordinate pairs, but it is empty"),
        ):
            term.membership(0.0)
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected xy to contain coordinate pairs, but it is empty"),
        ):
            term.values = fl.Discrete.to_xy([], [])
            term.membership(0.0)
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected xy to have with 2 columns, but got 1 in shape (1,): [1.]"),
        ):
            term.values = fl.array([1.0])
            term.membership(0.0)

    def test_create(self) -> None:
        """Test the discrete term creation."""
        x = fl.array([0, 2, 4, 6])
        y = fl.array([1, 3, 5, 7])
        xy_list = fl.array([x, y]).T.flatten().tolist()

        TermAssert(self, fl.Discrete.create("str", fl.Op.str(xy_list))).exports_fll(
            "term: str Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000"
        )
        TermAssert(self, fl.Discrete.create("list", xy_list)).exports_fll(
            "term: list Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000"
        )
        TermAssert(self, fl.Discrete.create("tuple", (x.tolist(), y.tolist()))).exports_fll(
            "term: tuple Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000"
        )
        TermAssert(self, fl.Discrete.create("dict", dict(zip(x.tolist(), y.tolist())))).exports_fll(
            "term: dict Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000"
        )
        # As strings
        TermAssert(self, fl.Discrete.create("list", fl.Op.str(xy_list).split())).exports_fll(
            "term: list Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000"
        )
        TermAssert(
            self,
            fl.Discrete.create("tuple", tuple((x.astype(str).tolist(), y.astype(str).tolist()))),
        ).exports_fll("term: tuple Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000")
        TermAssert(
            self, fl.Discrete.create("dict", dict(zip(x.astype(str), y.astype(str))))
        ).exports_fll("term: dict Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000")

    def test_discrete_to_xy(self) -> None:
        """Test the conversion to xy pairs."""
        xy = fl.Discrete("name", fl.Discrete.to_xy("0 2 4 6".split(), "1 3 5 7".split()))
        self.assertSequenceEqual(tuple(xy.x()), (0, 2, 4, 6))
        self.assertSequenceEqual(tuple(xy.y()), (1, 3, 5, 7))
        self.assertEqual(3, xy.membership(2))

        # Test iterators
        it = iter(xy.values)
        np.testing.assert_equal(next(it), (0, 1))
        np.testing.assert_equal(next(it), (2, 3))
        np.testing.assert_equal(next(it), (4, 5))
        np.testing.assert_equal(next(it), (6, 7))
        with self.assertRaisesRegex(StopIteration, ""):
            next(it)

        np.testing.assert_equal(fl.Discrete.to_xy([], []), fl.array([], ndmin=2).reshape((-1, 2)))

    def test_gaussian(self) -> None:
        """Test the gaussian term."""
        TermAssert(self, fl.Gaussian("gaussian")).exports_fll(
            "term: gaussian Gaussian nan nan"
        ).repr_is("fl.Gaussian('gaussian', fl.nan, fl.nan)").takes_parameters(
            2
        ).is_not_monotonic().configured_as(
            "0.0 0.25"
        ).exports_fll(
            "term: gaussian Gaussian 0.000 0.250"
        ).repr_is(
            "fl.Gaussian('gaussian', 0.0, 0.25)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.1353352832366127,
                -0.4: 0.2780373004531941,
                -0.25: 0.6065306597126334,
                -0.1: 0.9231163463866358,
                0.0: 1.0,
                0.1: 0.9231163463866358,
                0.25: 0.6065306597126334,
                0.4: 0.2780373004531941,
                0.5: 0.1353352832366127,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0.0 0.25 0.5"
        ).repr_is(
            "fl.Gaussian('gaussian', 0.0, 0.25, 0.5)"
        ).exports_fll(
            "term: gaussian Gaussian 0.000 0.250 0.500"
        )

    def test_gaussian_product(self) -> None:
        """Test the gaussian product term."""
        TermAssert(self, fl.GaussianProduct("gaussian_product")).exports_fll(
            "term: gaussian_product GaussianProduct nan nan nan nan"
        ).repr_is(
            "fl.GaussianProduct('gaussian_product', fl.nan, fl.nan, fl.nan, fl.nan)"
        ).takes_parameters(
            4
        ).is_not_monotonic().configured_as(
            "0.0 0.25 0.1 0.5"
        ).repr_is(
            "fl.GaussianProduct('gaussian_product', 0.0, 0.25, 0.1, 0.5)"
        ).exports_fll(
            "term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.1353352832366127,
                -0.4: 0.2780373004531941,
                -0.25: 0.6065306597126334,
                -0.1: 0.9231163463866358,
                0.0: 1.0,
                0.1: 1.0,
                0.25: 0.9559974818331,
                0.4: 0.835270211411272,
                0.5: 0.7261490370736908,
                1.0: 0.198,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0.0 0.25 0.1 0.5 0.5"
        ).exports_fll(
            "term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500 0.500"
        ).repr_is(
            "fl.GaussianProduct('gaussian_product', 0.0, 0.25, 0.1, 0.5, 0.5)"
        )

    def test_linear(self) -> None:
        """Test the linear term."""
        engine = fl.Engine(
            input_variables=[
                fl.InputVariable("A"),
                fl.InputVariable("B"),
                fl.InputVariable("C"),
            ]
        )
        engine.input_variables[0].value = 0
        engine.input_variables[1].value = 1
        engine.input_variables[2].value = 2

        with self.assertRaisesRegex(ValueError, "expected reference to an engine, but found none"):
            fl.Linear().membership(nan)

        linear = fl.Linear("linear", [1.0, 2.0])
        self.assertEqual(linear.engine, None)
        linear.update_reference(engine)
        self.assertEqual(linear.engine, engine)

        TermAssert(self, linear).exports_fll("term: linear Linear 1.000 2.000").repr_is(
            "fl.Linear('linear', [1.0, 2.0])"
        ).is_not_monotonic().configured_as("1.0 2.0 3").exports_fll(
            "term: linear Linear 1.000 2.000 3.000"
        ).repr_is(
            "fl.Linear('linear', [1.0, 2.0, 3.0])"
        ).has_membership(
            {
                -1.0: 1 * 0 + 2 * 1 + 3 * 2,  # = 8
                -0.5: 8,
                -0.4: 8,
                -0.25: 8,
                -0.1: 8,
                0.0: 8,
                0.1: 8,
                0.25: 8,
                0.4: 8,
                0.5: 8,
                1.0: 8,
                nan: 8,
                inf: 8,
                -inf: 8,
            }
        )
        TermAssert(self, linear).configured_as("1 2 3 5").exports_fll(
            "term: linear Linear 1.000 2.000 3.000 5.000"
        ).has_membership(
            {
                -1.0: 1 * 0 + 2 * 1 + 3 * 2 + 5,  # = 13
                -0.5: 13,
                -0.4: 13,
                -0.25: 13,
                -0.1: 13,
                0.0: 13,
                0.1: 13,
                0.25: 13,
                0.4: 13,
                0.5: 13,
                1.0: 13,
                nan: 13,
                inf: 13,
                -inf: 13,
            }
        )
        TermAssert(self, linear).configured_as("1 2 3 5 8").exports_fll(
            "term: linear Linear 1.000 2.000 3.000 5.000 8.000"
        ).membership_fails(
            nan,
            ValueError,
            "expected 3 (+1) coefficients (one for each input variable plus an optional constant), "
            "but found 5 coefficients: [1.0, 2.0, 3.0, 5.0, 8.0]",
        )

    def test_pi_shape(self) -> None:
        """Test the pi-shape term."""
        TermAssert(self, fl.PiShape("pi_shape")).exports_fll(
            "term: pi_shape PiShape nan nan nan nan"
        ).repr_is("fl.PiShape('pi_shape', fl.nan, fl.nan, fl.nan, fl.nan)").takes_parameters(
            4
        ).is_not_monotonic().configured_as(
            "-.9 -.1 .1 1"
        ).exports_fll(
            "term: pi_shape PiShape -0.900 -0.100 0.100 1.000"
        ).repr_is(
            "fl.PiShape('pi_shape', -0.9, -0.1, 0.1, 1.0)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.5,
                -0.4: 0.71875,
                -0.25: 0.9296875,
                -0.1: 1.0,
                0.0: 1.0,
                0.1: 1.0,
                0.25: 0.9444444444444444,
                0.4: 0.7777777777777777,
                0.5: 0.6049382716049383,
                0.95: 0.00617283950617285,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "-.9 -.1 .1 1 .5"
        ).exports_fll(
            "term: pi_shape PiShape -0.900 -0.100 0.100 1.000 0.500"
        ).repr_is(
            "fl.PiShape('pi_shape', -0.9, -0.1, 0.1, 1.0, 0.5)"
        )

    def test_ramp(self) -> None:
        """Test the ramp term."""
        TermAssert(self, fl.Ramp("ramp")).exports_fll("term: ramp Ramp nan nan").repr_is(
            "fl.Ramp('ramp', fl.nan, fl.nan)"
        ).takes_parameters(2).is_monotonic()

        TermAssert(self, fl.Ramp("ramp")).configured_as("0 0").exports_fll(
            "term: ramp Ramp 0.000 0.000"
        ).has_memberships(
            {
                -1.0: nan,
                -0.5: nan,
                -0.4: nan,
                -0.25: nan,
                -0.1: nan,
                0.0: nan,
                0.1: nan,
                0.25: nan,
                0.4: nan,
                0.5: nan,
                1.0: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        ).has_tsukamotos(
            {
                0.0: 0.0,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Ramp("ramp")).configured_as("-0.250 0.750").exports_fll(
            "term: ramp Ramp -0.250 0.750"
        ).repr_is("fl.Ramp('ramp', -0.25, 0.75)").has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.0,
                -0.25: 0.0,
                -0.1: 0.15,
                0.0: 0.25,
                0.1: 0.35,
                0.25: 0.50,
                0.4: 0.65,
                0.5: 0.75,
                1.0: 1.0,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        ).has_tsukamotos(
            {
                0.0: -0.25,
                0.25: 0.0,
                0.5: 0.25,
                0.75: 0.5,
                1.0: 0.75,
                # invalid values
                -1.0: -1.25,
                -0.5: -0.75,
                nan: nan,
                inf: inf,
                -inf: -inf,
            }
        )

        TermAssert(self, fl.Ramp("ramp")).configured_as("0.250 -0.750").exports_fll(
            "term: ramp Ramp 0.250 -0.750"
        ).has_memberships(
            {
                -1.0: 1.0,
                -0.5: 0.750,
                -0.4: 0.650,
                -0.25: 0.500,
                -0.1: 0.350,
                0.0: 0.250,
                0.1: 0.150,
                0.25: 0.0,
                0.4: 0.0,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            },
        ).has_tsukamotos(
            {
                0.0: 0.25,
                0.25: 0.0,
                0.5: -0.25,
                0.75: -0.5,
                1.0: -0.75,
                # invalid values
                -1.0: 1.25,
                -0.5: 0.75,
                nan: nan,
                inf: -inf,
                -inf: inf,
            }
        )

        TermAssert(self, fl.Ramp("ramp")).configured_as("0.250 -0.750 0.5").exports_fll(
            "term: ramp Ramp 0.250 -0.750 0.500"
        ).repr_is("fl.Ramp('ramp', 0.25, -0.75, 0.5)").has_tsukamotos(
            {
                0.0: 0.25,
                0.125: 0,
                0.25: -0.25,
                0.375: -0.5,
                0.5: -0.75,
                # invalid values
                0.75: -1.25,
                1.0: -1.75,
                -1.0: 2.25,
                -0.5: 1.25,
                nan: nan,
                inf: -inf,
                -inf: inf,
            }
        )

    def test_rectangle(self) -> None:
        """Test the rectangle term."""
        TermAssert(self, fl.Rectangle("rectangle")).exports_fll(
            "term: rectangle Rectangle nan nan"
        ).repr_is("fl.Rectangle('rectangle', fl.nan, fl.nan)").takes_parameters(
            2
        ).is_not_monotonic().configured_as(
            "-0.4 0.4"
        ).exports_fll(
            "term: rectangle Rectangle -0.400 0.400"
        ).repr_is(
            "fl.Rectangle('rectangle', -0.4, 0.4)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 1.0,
                -0.25: 1.0,
                -0.1: 1.0,
                0.0: 1.0,
                0.1: 1.0,
                0.25: 1.0,
                0.4: 1.0,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "-0.4 0.4 0.5"
        ).exports_fll(
            "term: rectangle Rectangle -0.400 0.400 0.500"
        ).repr_is(
            "fl.Rectangle('rectangle', -0.4, 0.4, 0.5)"
        )

    def test_semiellipse(self) -> None:
        """Test the spike term."""
        TermAssert(self, fl.SemiEllipse("semiellipse")).exports_fll(
            "term: semiellipse SemiEllipse nan nan"
        ).repr_is("fl.SemiEllipse('semiellipse', fl.nan, fl.nan)").takes_parameters(
            2
        ).is_not_monotonic().configured_as(
            "-0.5 0.5"
        ).exports_fll(
            "term: semiellipse SemiEllipse -0.500 0.500"
        ).repr_is(
            "fl.SemiEllipse('semiellipse', -0.5, 0.5)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.6,
                -0.25: 0.866,
                -0.1: 0.979,
                0.0: 1.0,
                0.1: 0.979,
                0.25: 0.866,
                0.4: 0.6,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0.5 -0.5"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.6,
                -0.25: 0.866,
                -0.1: 0.979,
                0.0: 1.0,
                0.1: 0.979,
                0.25: 0.866,
                0.4: 0.6,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )

        TermAssert(self, fl.SemiEllipse("semiellipse")).configured_as("-0.5 0.5 0.5").exports_fll(
            "term: semiellipse SemiEllipse -0.500 0.500 0.500"
        ).repr_is("fl.SemiEllipse('semiellipse', -0.5, 0.5, 0.5)")

    def test_sigmoid(self) -> None:
        """Test the sigmoid term."""
        TermAssert(self, fl.Sigmoid("sigmoid")).exports_fll(
            "term: sigmoid Sigmoid nan nan"
        ).repr_is("fl.Sigmoid('sigmoid', fl.nan, fl.nan)").takes_parameters(
            2
        ).is_monotonic().configured_as(
            "0 10"
        ).exports_fll(
            "term: sigmoid Sigmoid 0.000 10.000"
        ).repr_is(
            "fl.Sigmoid('sigmoid', 0.0, 10.0)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.007,
                -0.4: 0.018,
                -0.25: 0.076,
                -0.1: 0.269,
                0.0: 0.5,
                0.1: 0.731,
                0.25: 0.924,
                0.4: 0.982,
                0.5: 0.993,
                1.0: 0.999,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        ).has_tsukamotos(
            {
                0.0: -inf,
                0.25: -0.109,
                0.5: 0.0,
                0.75: 0.109,
                1.0: inf,
                # invalid values
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Sigmoid("sigmoid")).configured_as("0 -10").exports_fll(
            "term: sigmoid Sigmoid 0.000 -10.000"
        ).has_memberships(
            {
                -1.0: 0.999,
                -0.5: 0.994,
                -0.4: 0.982,
                -0.25: 0.924,
                -0.1: 0.731,
                0.0: 0.5,
                0.1: 0.269,
                0.25: 0.076,
                0.4: 0.018,
                0.5: 0.007,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            }
        ).has_tsukamotos(
            {
                0.0: inf,
                0.25: 0.109,
                0.5: 0.0,
                0.75: -0.109,
                1.0: -inf,
                # invalid values
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.Sigmoid("sigmoid")).configured_as("0 10 .5").exports_fll(
            "term: sigmoid Sigmoid 0.000 10.000 0.500"
        ).repr_is("fl.Sigmoid('sigmoid', 0.0, 10.0, 0.5)").has_tsukamotos(
            {
                0.0: -inf,
                0.125: -0.11,
                0.25: 0,
                0.375: 0.11,
                0.5: inf,
                # invalid values
                0.75: nan,
                1.0: nan,
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

    def test_sigmoid_difference(self) -> None:
        """Test the sigmoid difference term."""
        TermAssert(self, fl.SigmoidDifference("sigmoid_difference")).exports_fll(
            "term: sigmoid_difference SigmoidDifference nan nan nan nan"
        ).repr_is(
            "fl.SigmoidDifference('sigmoid_difference', fl.nan, fl.nan, fl.nan, fl.nan)"
        ).takes_parameters(
            4
        ).is_not_monotonic().configured_as(
            "-0.25 25.00 50.00 0.25"
        ).exports_fll(
            "term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250"
        ).repr_is(
            "fl.SigmoidDifference('sigmoid_difference', -0.25, 25.0, 50.0, 0.25)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0019267346633274238,
                -0.4: 0.022977369910017923,
                -0.25: 0.49999999998611205,
                -0.1: 0.9770226049799834,
                0.0: 0.9980695386973883,
                0.1: 0.9992887851439739,
                0.25: 0.49999627336071584,
                0.4: 0.000552690994449101,
                0.5: 3.7194451510957904e-06,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "-0.25 25.00 50.00 0.25 0.5"
        ).repr_is(
            "fl.SigmoidDifference('sigmoid_difference', -0.25, 25.0, 50.0, 0.25, 0.5)"
        ).exports_fll(
            "term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250 0.500"
        )

    def test_sigmoid_product(self) -> None:
        """Test the sigmoid product term."""
        TermAssert(self, fl.SigmoidProduct("sigmoid_product")).exports_fll(
            "term: sigmoid_product SigmoidProduct nan nan nan nan"
        ).repr_is(
            "fl.SigmoidProduct('sigmoid_product', fl.nan, fl.nan, fl.nan, fl.nan)"
        ).takes_parameters(
            4
        ).is_not_monotonic().configured_as(
            "-0.250 20.000 -20.000 0.250"
        ).exports_fll(
            "term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250"
        ).repr_is(
            "fl.SigmoidProduct('sigmoid_product', -0.25, 20.0, -20.0, 0.25)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.006692848876926853,
                -0.4: 0.04742576597971327,
                -0.25: 0.4999773010656488,
                -0.1: 0.9517062830264366,
                0.0: 0.9866590924049252,
                0.1: 0.9517062830264366,
                0.25: 0.4999773010656488,
                0.4: 0.04742576597971327,
                0.5: 0.006692848876926853,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "-0.250 20.000 -20.000 0.250 0.5"
        ).repr_is(
            "fl.SigmoidProduct('sigmoid_product', -0.25, 20.0, -20.0, 0.25, 0.5)"
        ).exports_fll(
            "term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250 0.500"
        )

    def test_spike(self) -> None:
        """Test the spike term."""
        TermAssert(self, fl.Spike("spike")).exports_fll("term: spike Spike nan nan").repr_is(
            "fl.Spike('spike', fl.nan, fl.nan)"
        ).takes_parameters(2).is_not_monotonic().configured_as("0 1.0").exports_fll(
            "term: spike Spike 0.000 1.000"
        ).repr_is(
            "fl.Spike('spike', 0.0, 1.0)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.006737946999085467,
                -0.4: 0.01831563888873418,
                -0.25: 0.0820849986238988,
                -0.1: 0.36787944117144233,
                0.0: 1.0,
                0.1: 0.36787944117144233,
                0.25: 0.0820849986238988,
                0.4: 0.01831563888873418,
                0.5: 0.006737946999085467,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        ).configured_as(
            "0 1.0 .5"
        ).repr_is(
            "fl.Spike('spike', 0.0, 1.0, 0.5)"
        ).exports_fll(
            "term: spike Spike 0.000 1.000 0.500"
        )

    def test_s_shape(self) -> None:
        """Test the s-shape term."""
        TermAssert(self, fl.SShape("s_shape")).exports_fll("term: s_shape SShape nan nan").repr_is(
            "fl.SShape('s_shape', fl.nan, fl.nan)"
        ).takes_parameters(2).is_monotonic().configured_as("-0.5 0.5").exports_fll(
            "term: s_shape SShape -0.500 0.500"
        ).repr_is(
            "fl.SShape('s_shape', -0.5, 0.5)"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.0,
                -0.4: 0.02,
                -0.25: 0.125,
                -0.1: 0.32,
                0.0: 0.5,
                0.1: 0.68,
                0.25: 0.875,
                0.4: 0.98,
                0.5: 1.0,
                1.0: 1.0,
                nan: nan,
                inf: 1.0,
                -inf: 0.0,
            }
        ).has_tsukamotos(
            {
                0.0: -0.5,
                0.25: -0.146,
                0.5: 0.0,
                0.75: 0.146,
                1.0: 0.5,
                # invalid values
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.SShape("s_shape")).configured_as("-0.5 0.5 0.5").repr_is(
            "fl.SShape('s_shape', -0.5, 0.5, 0.5)"
        ).exports_fll("term: s_shape SShape -0.500 0.500 0.500").has_tsukamotos(
            {
                0.0: -0.5,
                0.125: -0.146,
                0.25: 0,
                0.375: 0.146,
                0.5: 0.5,
                # invalid values
                0.75: nan,
                1.0: nan,
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

    def test_trapezoid(self) -> None:
        """Test the trapezoid term."""
        TermAssert(self, fl.Trapezoid("trapezoid", 0.0, 1.0)).exports_fll(
            "term: trapezoid Trapezoid 0.000 0.200 0.800 1.000"
        ).repr_is("fl.Trapezoid('trapezoid', 0.0, 0.2, 0.8, 1.0)")

        TermAssert(self, fl.Trapezoid("trapezoid")).exports_fll(
            "term: trapezoid Trapezoid nan nan nan nan"
        ).repr_is("fl.Trapezoid('trapezoid', fl.nan, fl.nan, fl.nan, fl.nan)").takes_parameters(
            4
        ).is_not_monotonic().configured_as(
            "-0.400 -0.100 0.100 0.400"
        ).exports_fll(
            "term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.500,
                -0.1: 1.000,
                0.0: 1.000,
                0.1: 1.000,
                0.25: 0.500,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Trapezoid("trapezoid")).configured_as(
            "-0.400 -0.100 0.100 0.400 .5"
        ).exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400 0.500").repr_is(
            "fl.Trapezoid('trapezoid', -0.4, -0.1, 0.1, 0.4, 0.5)"
        )

        TermAssert(self, fl.Trapezoid("trapezoid")).configured_as(
            "-0.400 -0.400 0.100 0.400"
        ).exports_fll("term: trapezoid Trapezoid -0.400 -0.400 0.100 0.400").has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 1.000,
                -0.25: 1.000,
                -0.1: 1.000,
                0.0: 1.000,
                0.1: 1.000,
                0.25: 0.500,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Trapezoid("trapezoid")).configured_as(
            "-0.400 -0.100 0.400 0.400"
        ).exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.400 0.400").has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.5,
                -0.1: 1.000,
                0.0: 1.000,
                0.1: 1.000,
                0.25: 1.000,
                0.4: 1.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Trapezoid("trapezoid")).configured_as(
            "-inf -0.100 0.100 .4"
        ).exports_fll("term: trapezoid Trapezoid -inf -0.100 0.100 0.400").has_memberships(
            {
                -1.0: 1.0,
                -0.5: 1.000,
                -0.4: 1.000,
                -0.25: 1.000,
                -0.1: 1.000,
                0.0: 1.000,
                0.1: 1.000,
                0.25: 0.500,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            }
        )
        TermAssert(self, fl.Trapezoid("trapezoid")).configured_as(
            "-.4 -0.100 0.100 inf .5"
        ).exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.100 inf 0.500").has_membership(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.500 * 0.5,
                -0.1: 1.000 * 0.5,
                0.0: 1.000 * 0.5,
                0.1: 1.000 * 0.5,
                0.25: 1.000 * 0.5,
                0.4: 1.000 * 0.5,
                0.5: 1.000 * 0.5,
                1.0: 1.0 * 0.5,
                nan: nan,
                inf: 1.0 * 0.5,
                -inf: 0.0,
            },
        )

    def test_triangle(self) -> None:
        """Test the triangle term."""
        TermAssert(self, fl.Triangle("triangle", 0.0, 1.0)).exports_fll(
            "term: triangle Triangle 0.000 0.500 1.000"
        ).repr_is("fl.Triangle('triangle', 0.0, 0.5, 1.0)")

        TermAssert(self, fl.Triangle("triangle")).exports_fll(
            "term: triangle Triangle nan nan nan"
        ).repr_is("fl.Triangle('triangle', fl.nan, fl.nan, fl.nan)").takes_parameters(
            3
        ).is_not_monotonic().configured_as(
            "-0.400 0.000 0.400"
        ).exports_fll(
            "term: triangle Triangle -0.400 0.000 0.400"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.37500000000000006,
                -0.1: 0.7500000000000001,
                0.0: 1.000,
                0.1: 0.7500000000000001,
                0.25: 0.37500000000000006,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Triangle("triangle")).configured_as("-0.400 0.000 0.400 .5").repr_is(
            "fl.Triangle('triangle', -0.4, 0.0, 0.4, 0.5)"
        ).exports_fll("term: triangle Triangle -0.400 0.000 0.400 0.500")
        TermAssert(self, fl.Triangle("triangle")).configured_as("-0.500 0.000 0.500").exports_fll(
            "term: triangle Triangle -0.500 0.000 0.500"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.2,
                -0.25: 0.5,
                -0.1: 0.8,
                0.0: 1.000,
                0.1: 0.8,
                0.25: 0.5,
                0.4: 0.2,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Triangle("triangle")).configured_as("-0.500 -0.500 0.500").exports_fll(
            "term: triangle Triangle -0.500 -0.500 0.500"
        ).has_memberships(
            {
                -1.0: 0.0,
                -0.5: 1.000,
                -0.4: 0.900,
                -0.25: 0.75,
                -0.1: 0.6,
                0.0: 0.5,
                0.1: 0.4,
                0.25: 0.25,
                0.4: 0.1,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Triangle("triangle")).configured_as("-0.500 0.500 0.500").exports_fll(
            "term: triangle Triangle -0.500 0.500 0.500"
        ).has_memberships(
            {
                1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.1,
                -0.25: 0.25,
                -0.1: 0.4,
                0.0: 0.5,
                0.1: 0.6,
                0.25: 0.75,
                0.4: 0.900,
                0.5: 1.000,
                nan: nan,
                inf: 0.0,
                -inf: 0.0,
            }
        )
        TermAssert(self, fl.Triangle("triangle")).configured_as("-inf 0.000 0.400").exports_fll(
            "term: triangle Triangle -inf 0.000 0.400"
        ).has_memberships(
            {
                -1.0: 1.0,
                -0.5: 1.000,
                -0.4: 1.000,
                -0.25: 1.000,
                -0.1: 1.000,
                0.0: 1.000,
                0.1: 0.75,
                0.25: 0.375,
                0.4: 0.000,
                0.5: 0.000,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.000,
            }
        )
        TermAssert(self, fl.Triangle("triangle")).configured_as("-0.400 0.000 inf .5").exports_fll(
            "term: triangle Triangle -0.400 0.000 inf 0.500"
        ).has_membership(
            {
                -1.0: 0.0,
                -0.5: 0.000,
                -0.4: 0.000,
                -0.25: 0.375 * 0.5,
                -0.1: 0.750 * 0.5,
                0.0: 1.000 * 0.5,
                0.1: 1.000 * 0.5,
                0.25: 1.000 * 0.5,
                0.4: 1.000 * 0.5,
                0.5: 1.000 * 0.5,
                1.0: 1.0 * 0.5,
                nan: nan,
                inf: 1.000 * 0.5,
                -inf: 0.0,
            },
        )

    def test_z_shape(self) -> None:
        """Test the z-shape term."""
        TermAssert(self, fl.ZShape("z_shape")).exports_fll("term: z_shape ZShape nan nan").repr_is(
            "fl.ZShape('z_shape', fl.nan, fl.nan)"
        ).takes_parameters(2).is_monotonic().configured_as("-0.5 0.5").exports_fll(
            "term: z_shape ZShape -0.500 0.500"
        ).repr_is(
            "fl.ZShape('z_shape', -0.5, 0.5)"
        ).has_memberships(
            {
                -1.0: 1.0,
                -0.5: 1.0,
                -0.4: 0.98,
                -0.25: 0.875,
                -0.1: 0.68,
                0.0: 0.5,
                0.1: 0.32,
                0.25: 0.125,
                0.4: 0.02,
                0.5: 0.0,
                1.0: 0.0,
                nan: nan,
                inf: 0.0,
                -inf: 1.0,
            }
        ).has_tsukamotos(
            {
                0.0: 0.5,
                0.25: 0.146,
                0.5: 0.0,
                0.75: -0.146,
                1.0: -0.5,
                # invalid values
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

        TermAssert(self, fl.ZShape("z_shape")).configured_as("-0.5 0.5 0.5").repr_is(
            "fl.ZShape('z_shape', -0.5, 0.5, 0.5)"
        ).exports_fll("term: z_shape ZShape -0.500 0.500 0.500").has_tsukamotos(
            {
                0.0: 0.5,
                0.125: 0.146,
                0.25: 0.0,
                0.375: -0.146,
                0.5: -0.5,
                # invalid values
                0.75: nan,
                1.0: nan,
                -1.0: nan,
                -0.5: nan,
                nan: nan,
                inf: nan,
                -inf: nan,
            }
        )

    def test_division_by_zero_does_not_fail_with_numpy_float(self) -> None:
        """Test the division by zero is not raised when using numpy floats."""
        import numpy as np

        with fl.settings.context(float_type=np.float64):
            np.seterr(divide="ignore")  # ignore "errors", (e.g., division by zero)
            TermAssert(self, fl.Function.create("dbz", "0.0/x")).has_membership(
                {0.0: fl.nan, fl.inf: 0.0, -fl.inf: -0.0, fl.nan: fl.nan}
            )

            TermAssert(self, fl.Function.create("dbz", "inf/x")).has_membership(
                {0.0: fl.inf, fl.inf: fl.nan, -fl.inf: fl.nan, -fl.nan: fl.nan}
            )

            TermAssert(self, fl.Function.create("dbz", "~inf/x")).has_membership(
                {0.0: -fl.inf, fl.inf: fl.nan, -fl.inf: fl.nan, -fl.nan: fl.nan}
            )

            TermAssert(self, fl.Function.create("dbz", "nan/x")).has_membership(
                {0.0: fl.nan, fl.inf: fl.nan, -fl.inf: fl.nan, -fl.nan: fl.nan}
            )

    def test_some_function(self) -> None:
        """Test function results."""
        f = fl.Function("X", "ge(x, 5)", load=True)
        np.testing.assert_allclose(0.0, f.membership(4))
        np.testing.assert_allclose(1.0, f.membership(5))

        f = fl.Function("X", "1 / (x-5)", load=True)
        np.testing.assert_allclose(fl.inf, f.membership(5))
        np.testing.assert_allclose(-1.0, f.membership(4))

        f = fl.Function("X", ".-1 / (x-5)", load=True)
        np.testing.assert_allclose(-fl.inf, f.membership(5))
        np.testing.assert_allclose(1.0, f.membership(4))

        f = fl.Function("X", "0 / (x-5)", load=True)
        np.testing.assert_allclose(fl.nan, f.membership(5))
        np.testing.assert_allclose(0.0, f.membership(4))


class FunctionNodeAssert(BaseAssert[fl.Function.Node]):
    """Function node assert."""

    def prefix_is(self, prefix: str) -> FunctionNodeAssert:
        """Assert the prefix notation of the node is the expected prefix notation."""
        self.test.assertEqual(prefix, self.actual.prefix())
        return self

    def infix_is(self, infix: str) -> FunctionNodeAssert:
        """Assert the infix notation of the node is the expected infix notation."""
        self.test.assertEqual(infix, self.actual.infix())
        return self

    def postfix_is(self, postfix: str) -> FunctionNodeAssert:
        """Assert the postfix notation of the node is the expected postfix notation."""
        self.test.assertEqual(postfix, self.actual.postfix())
        return self

    def value_is(self, expected: str) -> FunctionNodeAssert:
        """Assert the value of the node is the expected value."""
        self.test.assertEqual(expected, self.actual.value())
        return self

    def evaluates_to(
        self, expected: float, variables: dict[str, fl.Scalar] | None = None
    ) -> FunctionNodeAssert:
        """Assert the node evaluates to the expected value (optionally) given variables."""
        obtained = self.actual.evaluate(variables)
        np.testing.assert_allclose(
            obtained,
            expected,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )
        return self

    def fails_to_evaluate(self, exception: type[Exception], message: str) -> FunctionNodeAssert:
        """Assert the node raises the expection on evaluation."""
        with self.test.assertRaisesRegex(exception, message):
            self.actual.evaluate()
        return self


class TestFunction(unittest.TestCase):
    """Test the function term."""

    def test_function(self) -> None:
        """Test the function term."""
        with self.assertRaisesRegex(RuntimeError, re.escape("function 'f(x)=2x+1' is not loaded")):
            fl.Function("f(x)", "f(x)=2x+1").membership(nan)

        TermAssert(self, fl.Function("function", "", variables={"y": 1.5})).exports_fll(
            "term: function Function"
        ).repr_is("fl.Function('function', '', variables={'y': 1.5})").configured_as(
            "2*x**3 +2*y - 3"
        ).exports_fll(
            "term: function Function 2*x**3 +2*y - 3"
        ).repr_is(
            "fl.Function('function', '2*x**3 +2*y - 3', variables={'y': 1.5})"
        ).has_memberships(
            {
                -0.5: -0.25,
                -0.4: -0.1280000000000001,
                -0.25: -0.03125,
                -0.1: -0.0019999999999997797,
                0.0: 0.0,
                0.1: 0.0019999999999997797,
                0.25: 0.03125,
                0.4: 0.1280000000000001,
                0.5: 0.25,
                nan: nan,
                inf: inf,
                -inf: -inf,
            },
            heights=[1.0],
        )

        input_a = fl.InputVariable("i_A")
        output_a = fl.OutputVariable("o_A")
        engine_a = fl.Engine("A", "Engine A", [input_a], [output_a])
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "expected a map of variables containing the value for 'i_A', "
                "but the map contains: {'x': 0.0}"
            ),
        ):
            fl.Function.create("engine_a", "2*i_A + o_A + x").membership(0.0)

        function_a = fl.Function.create("f", "2*i_A + o_A + x", engine_a)
        assert_that = TermAssert(self, function_a)
        assert_that.exports_fll("term: f Function 2*i_A + o_A + x").repr_is(
            "fl.Function('f', '2*i_A + o_A + x')"
        ).has_memberships({0.0: nan})
        input_a.value = 3.0
        output_a.value = 1.0
        assert_that.has_memberships(
            {
                -1.0: 6.0,
                -0.5: 6.5,
                0.0: 7.0,
                0.5: 7.5,
                1.0: 8.0,
                nan: nan,
                inf: inf,
                -inf: -inf,
            },
            heights=[1.0],
        )

        function_a.variables = {"x": nan}
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "variable 'x' is reserved for internal use of Function term, "
                "please remove it from the map of variables: {'x': nan}"
            ),
        ):
            function_a.membership(0.0)
        del function_a.variables["x"]

        input_a.name = "x"
        TermAssert(self, function_a).membership_fails(
            0.0,
            ValueError,
            "variable 'x' is reserved for internal use of Function term, "
            f"please rename the engine variable: {str(input_a)}",
        )

        input_b = fl.InputVariable("i_B")
        output_b = fl.OutputVariable("o_B")
        engine_b = fl.Engine("B", "Engine B", [input_b], [output_b])
        self.assertEqual(engine_a, function_a.engine)
        self.assertTrue(function_a.is_loaded())

        function_a.update_reference(engine_b)
        TermAssert(self, function_a).membership_fails(
            0.0,
            ValueError,
            "expected a map of variables containing the value for 'i_A', "
            "but the map contains: {'i_B': fl.nan, 'o_B': fl.nan, 'x': 0.0}",
        )

    def test_element(self) -> None:
        """Test the function element."""
        element = fl.Function.Element(
            "function",
            "math function()",
            fl.Function.Element.Type.Function,
            method=any,
            arity=0,
            precedence=0,
            associativity=-1,
        )
        self.assertEqual(
            "fl.Element(name='function', description='math function()', "
            "type='Function', method=<built-in function any>, arity=0, "
            "precedence=0, associativity=-1)",
            str(element),
        )

        element = fl.Function.Element(
            "operator",
            "math operator",
            fl.Function.Element.Type.Operator,
            operator.add,
            2,
            10,
            1,
        )
        self.assertEqual(
            "fl.Element(name='operator', description='math operator', "
            "type='Operator', "
            "method=<built-in function add>, arity=2, "
            "precedence=10, associativity=1)",
            str(element),
        )

        self.assertEqual(str(element), str(copy.deepcopy(element)))

    def test_node_evaluation(self) -> None:
        """Test the function node evaluation."""
        type_function = fl.Function.Element.Type.Function

        functions = fl.FunctionFactory()
        node_pow = fl.Function.Node(
            element=functions.copy("**"),
            left=fl.Function.Node(constant=3.0),
            right=fl.Function.Node(constant=4.0),
        )
        FunctionNodeAssert(self, node_pow).postfix_is("3.000 4.000 **").prefix_is(
            "** 3.000 4.000"
        ).infix_is("3.000 ** 4.000").evaluates_to(81.0)

        node_sin = fl.Function.Node(element=functions.copy("sin"), right=node_pow)
        FunctionNodeAssert(self, node_sin).postfix_is("3.000 4.000 ** sin").prefix_is(
            "sin ** 3.000 4.000"
        ).infix_is("sin ( 3.000 ** 4.000 )").evaluates_to(-0.629887994274454)

        node_pow = fl.Function.Node(
            element=functions.copy("pow"),
            left=node_sin,
            right=fl.Function.Node(variable="two"),
        )

        FunctionNodeAssert(self, node_pow).postfix_is("3.000 4.000 ** sin two pow").prefix_is(
            "pow sin ** 3.000 4.000 two"
        ).infix_is("pow ( sin ( 3.000 ** 4.000 ) two )").fails_to_evaluate(
            ValueError,
            "expected a map of variables containing the value for 'two', "
            "but the map contains: None",
        ).evaluates_to(
            0.39675888533109455, {"two": 2}
        )

        node_sum = fl.Function.Node(element=functions.copy("+"), left=node_pow, right=node_pow)

        FunctionNodeAssert(self, node_sum).postfix_is(
            "3.000 4.000 ** sin two pow 3.000 4.000 ** sin two pow +"
        ).prefix_is("+ pow sin ** 3.000 4.000 two pow sin ** 3.000 4.000 two").infix_is(
            "pow ( sin ( 3.000 ** 4.000 ) two ) + pow ( sin ( 3.000 ** 4.000 ) two )"
        ).evaluates_to(
            0.7935177706621891, {"two": 2}
        )

        FunctionNodeAssert(
            self, fl.Function.Node(element=functions.copy("cos"), right=None)
        ).fails_to_evaluate(ValueError, re.escape("expected a node, but found none"))

        FunctionNodeAssert(
            self,
            fl.Function.Node(
                element=functions.copy("cos"),
                right=fl.Function.Node(constant=np.pi),
                left=None,
            ),
        ).evaluates_to(-1)

        FunctionNodeAssert(
            self, fl.Function.Node(element=functions.copy("pow"), left=None, right=None)
        ).fails_to_evaluate(ValueError, re.escape("expected a right node, but found none"))
        FunctionNodeAssert(
            self,
            fl.Function.Node(
                element=functions.copy("pow"),
                left=None,
                right=fl.Function.Node(constant=2.0),
            ),
        ).fails_to_evaluate(ValueError, re.escape("expected a left node, but found none"))
        FunctionNodeAssert(
            self,
            fl.Function.Node(
                element=functions.copy("pow"),
                left=fl.Function.Node(constant=2.0),
                right=None,
            ),
        ).fails_to_evaluate(ValueError, re.escape("expected a right node, but found none"))

        def raise_exception() -> NoReturn:
            raise ValueError("mocking testing exception")

        FunctionNodeAssert(
            self,
            fl.Function.Node(
                element=functions.copy("pow"),
                left=fl.Function.Node(constant=2.0),
                right=fl.Function.Node(
                    element=fl.Function.Element(
                        "raise", "exception", type_function, raise_exception
                    )
                ),
            ),
        ).fails_to_evaluate(ValueError, re.escape("mocking testing exception"))

    def test_node_deep_copy(self) -> None:
        """Test the deep copy on a function."""
        node_mult = fl.Function.Node(
            element=fl.Function.Element(
                "*",
                "multiplication",
                fl.Function.Element.Type.Operator,
                operator.mul,
                2,
                80,
            ),
            left=fl.Function.Node(constant=3.0),
            right=fl.Function.Node(constant=4.0),
        )
        node_sin = fl.Function.Node(
            element=fl.Function.Element(
                "sin", "sine", fl.Function.Element.Type.Function, np.sin, 1
            ),
            right=node_mult,
        )
        FunctionNodeAssert(self, node_sin).infix_is("sin ( 3.000 * 4.000 )").evaluates_to(
            -0.5365729180004349
        )

        node_copy = copy.deepcopy(node_sin)

        FunctionNodeAssert(self, node_copy).infix_is("sin ( 3.000 * 4.000 )").evaluates_to(
            -0.5365729180004349
        )

        # if we change the original object
        node_sin.right.element.name = "?"  # type: ignore
        # the copy cannot be affected
        FunctionNodeAssert(self, node_copy).infix_is("sin ( 3.000 * 4.000 )").evaluates_to(
            -0.5365729180004349
        )

    def test_node_str(self) -> None:
        """Test Operator nodes to string return the expected value."""
        some_type = fl.Function.Element.Type.Operator
        FunctionNodeAssert(
            self,
            fl.Function.Node(element=fl.Function.Element("+", "sum", some_type, sum)),
        ).value_is("+")
        FunctionNodeAssert(
            self,
            fl.Function.Node(element=fl.Function.Element("+", "sum", some_type, sum), variable="x"),
        ).value_is("+")
        FunctionNodeAssert(
            self,
            fl.Function.Node(
                element=fl.Function.Element("+", "sum", some_type, sum),
                variable="x",
                constant=1,
            ),
        ).value_is("+")

        FunctionNodeAssert(self, fl.Function.Node(variable="x")).value_is("x")
        FunctionNodeAssert(self, fl.Function.Node(variable="x", constant=1.0)).value_is("x")
        FunctionNodeAssert(self, fl.Function.Node(constant=1)).value_is("1")
        negate = fuzzylite.library.settings.factory_manager.function.copy("~")
        (
            FunctionNodeAssert(self, fl.Function.Node(negate, right=fl.Function.Node(constant=5.0)))
            .prefix_is("~ 5.000")
            .infix_is("~ 5.000")
            .postfix_is("5.000 ~")
            .evaluates_to(-5.000)
        )
        (
            FunctionNodeAssert(self, fl.Function.Node(negate, left=fl.Function.Node(constant=5.0)))
            .prefix_is("~ 5.000")
            .infix_is("~ 5.000")
            .postfix_is("5.000 ~")
            .evaluates_to(-5.000)
        )

    def test_function_format_infix(self) -> None:
        """Test formatting infix equations."""
        self.assertEqual(
            "a + b * 1 ( True or True ) / ( False and False )",
            fl.Function.format_infix(f"a+b*1(True {fl.Rule.OR} True)/(False {fl.Rule.AND} False)"),
        )
        self.assertEqual(
            "sqrt ( a + b * 1 + sin ( pi / 2 ) - ~ 3 )",
            fl.Function.format_infix("sqrt(a+b*1+sin(pi/2)-~3)"),
        )

    def test_function_postfix(self) -> None:
        """Test infix to postfix equations."""
        infix_postfix = {
            "a+b": "a b +",
            "a+b*2": "a b 2 * +",
            "a+b*2^3": "a b 2 3 ^ * +",
            "a+b*2^3/(4 - 2)": "a b 2 3 ^ * 4 2 - / +",
            "a+b*2^3/(4 - 2)*sin(pi/4)": "a b 2 3 ^ * 4 2 - / pi 4 / sin * +",
            ".-.-a + .+.+b": "a .- .- b .+ .+ +",
            "a*.-b**3": "a b 3 ** .- *",
            ".-(a)**.-b": "a b .- ** .-",
            ".+a**.-b": "a b .- ** .+",
            ".-a**b + .+a**.-b - .-a ** .-b + .-(a**b) - .-(a)**.-b": "a b ** .- a b .- ** .+ + a b .- ** .- - a b ** .- + a b .- ** .- -",
            "a+~b": "a b ~ +",
            "~a*~b": "a ~ b ~ *",
            "(sin(pi()/4) + cos(pi/4)) / (~sin(pi()/4) - ~cos(pi/4))": "pi 4 / sin pi 4 / cos + pi 4 / sin ~ pi 4 / cos ~ - /",
        }
        for infix, postfix in infix_postfix.items():
            self.assertEqual(postfix, fl.Function.infix_to_postfix(infix))

    def test_function_parse(self) -> None:
        """Test parsing postfix equations."""
        infix_postfix = {
            "a+b": "a b +",
            "a+b*2": "a b 2.000 * +",
            "a+b*2^3": "a b 2.000 3.000 ^ * +",
            "a+b*2^3/(4 - 2)": "a b 2.000 3.000 ^ * 4.000 2.000 - / +",
            "a+b*2^3/(4 - 2)*sin(pi/4)": "a b 2.000 3.000 ^ * 4.000 2.000 - / pi 4.000 / sin * +",
            "a+~b": "a b ~ +",
            "~a*~b": "a ~ b ~ *",
            "(sin(pi()/4) + cos(pi/4)) / (~sin(pi()/4) - ~cos(pi/4))": "pi 4.000 / sin pi 4.000 / cos + "
            "pi 4.000 / sin ~ pi 4.000 / cos ~ - /",
        }
        for infix, postfix in infix_postfix.items():
            self.assertEqual(postfix, fl.Function.parse(infix).postfix())


if __name__ == "__main__":
    unittest.main()
