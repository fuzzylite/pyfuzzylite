"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import re
import unittest
from typing import Any

import numpy as np

import fuzzylite as fl
from fuzzylite.types import Scalar, Self
from tests.assert_component import BaseAssert


class DefuzzifierAssert(BaseAssert[fl.Defuzzifier]):
    """Defuzzifier assert."""

    def configured_as(self, parameters: str) -> Self:
        """Configures the actual defuzzifier with the parameters."""
        self.actual.configure(parameters)
        return self

    def has_parameters(self, parameters: str) -> Self:
        """Assert that the defuzzifier has the given parameters."""
        self.test.assertEqual(self.actual.parameters(), parameters)
        return self

    def has_attribute(self, **kwargs: Any) -> Self:
        """Assert that the defuzzifier has the given attributes."""
        for key, value in kwargs.items():
            self.test.assertIn(key, vars(self.actual))
            self.test.assertEqual(value, vars(self.actual)[key])
        return self

    def defuzzifies(
        self,
        minimum: float,
        maximum: float,
        terms: dict[fl.Term, float],
        vectorized: bool = True,
    ) -> Self:
        """Assert that the defuzzification of the given terms result in the expected values."""
        for term, expected in terms.items():
            obtained = self.actual.defuzzify(term, minimum, maximum)
            np.testing.assert_allclose(
                obtained,
                expected,
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
                err_msg=f"{fl.Op.class_name(self.actual)}({term}) = {obtained}, but expected {expected}",
            )
        if vectorized:

            class StackTerm(fl.Term):
                def __init__(self, terms: list[fl.Term]) -> None:
                    super().__init__("_")
                    self.terms = terms

                def membership(self, x: Scalar) -> Scalar:
                    return np.vstack([term.membership(x) for term in self.terms])

            expected_vector = np.atleast_1d(fl.array([x for x in terms.values()]))
            obtained_vector = np.atleast_1d(
                self.actual.defuzzify(StackTerm(terms=list(terms.keys())), minimum, maximum)
            )
            np.testing.assert_allclose(
                obtained_vector,
                expected_vector,
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
            )
        return self


NaN = fl.Constant


class TestDefuzzifier(unittest.TestCase):
    """Tests the defuzzifiers."""

    def test_bisector(self) -> None:
        """Test the bisector defuzzifier."""
        DefuzzifierAssert(self, fl.Bisector()).exports_fll("Bisector").has_attribute(
            resolution=1000
        ).configured_as("200").exports_fll("Bisector 200")

        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            0,
            1,
            {fl.Triangle("", 0, 1, 1): 0.7065},
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            0,
            1,
            {fl.Triangle("", 0, 0, 1): 0.2925},
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            0,
            1,
            {fl.Triangle("", 0, 0.5, 1): 0.4995},
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            0,
            1,
            {fl.Rectangle("", 0, 1): 0.4995},
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            -1,
            1,
            {fl.Rectangle("", -1, 1): -0.001},
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            -1,
            1,
            {
                fl.Aggregated(
                    "",
                    -1,
                    1,
                    aggregation=fl.UnboundedSum(),
                    terms=[
                        fl.Activated(
                            fl.Triangle("", -1, -1, -0.5),
                            implication=fl.AlgebraicProduct(),
                        ),
                        fl.Activated(
                            fl.Triangle("", 0.5, 1, 1),
                            implication=fl.AlgebraicProduct(),
                        ),
                    ],
                ): -0.001
            },
        )

        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            0,
            2,
            {NaN(): np.nan},  # mean of range of variable
        )

    def test_centroid(self) -> None:
        """Test the centroid defuzzifier."""
        DefuzzifierAssert(self, fl.Centroid()).exports_fll("Centroid").has_attribute(
            resolution=1000
        ).configured_as("200").has_parameters("200").exports_fll("Centroid 200")

        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            -fl.inf,
            0.0,
            {fl.Triangle(): fl.nan},
        )
        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            0.0,
            fl.inf,
            {fl.Triangle(): fl.nan},
        )
        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            fl.nan,
            0.0,
            {fl.Triangle(): fl.nan},
        )

        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            -1,
            1,
            {
                fl.Triangle("", -fl.inf, 0): fl.nan,
                fl.Triangle("", 0, fl.inf): fl.nan,
                fl.Triangle("", fl.nan, 0): fl.nan,
                fl.Triangle("", -1, 0): -0.5,
                fl.Triangle("", -1, 1): 0.0,
                fl.Triangle("", 0, 1): 0.5,
                fl.Aggregated(
                    "",
                    0,
                    1,
                    fl.Maximum(),
                    [
                        fl.Activated(fl.Triangle("Medium", 0.25, 0.5, 0.75), 0.2, fl.Minimum()),
                        fl.Activated(fl.Triangle("High", 0.5, 0.75, 1.0), 0.8, fl.Minimum()),
                    ],
                ): 0.6896552,
            },
        )
        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            -1,
            1,
            {NaN(): np.nan},
        )

    def test_som_defuzzifier(self) -> None:
        """Test the Smallest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).exports_fll(
            "SmallestOfMaximum"
        ).has_attribute(resolution=1000).configured_as("200").exports_fll("SmallestOfMaximum 200")

        # Test case:
        #            ______
        #      _____/      \
        # ____/             \
        # |                   \____
        term = fl.Discrete.create(
            "test",
            {
                0.0: 0.25,
                0.1: 0.25,
                0.2: 0.5,
                0.4: 0.5,
                0.5: 1.0,  # SOM
                0.7: 1.0,
                0.9: 1.0,
                1.0: 0.0,
            },
        )

        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            -fl.inf,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            -fl.inf,
            0.0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            0.0,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            fl.nan,
            fl.nan,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            fl.nan,
            0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            0,
            fl.nan,
            {term: fl.nan},
        )

        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            0,
            1,
            {
                term: 0.5,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.2,
            },
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            -1,
            1,
            {NaN(): np.nan},
        )

    def test_lom_defuzzifier(self) -> None:
        """Test the Largest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.LargestOfMaximum()).exports_fll(
            "LargestOfMaximum"
        ).has_attribute(resolution=1000).configured_as("200").has_parameters("200").exports_fll(
            "LargestOfMaximum 200"
        )

        # Test case:
        #            ______
        #      _____/      \
        # ____/             \
        # |                   \____
        term = fl.Discrete.create(
            "test",
            {
                0.0: 0.25,
                0.1: 0.25,
                0.2: 0.5,
                0.4: 0.5,
                0.5: 1.0,
                0.7: 1.0,
                0.9: 1.0,  # LOM
                1.0: 0.0,
            },
        )

        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            -fl.inf,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            -fl.inf,
            0.0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            0.0,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            fl.nan,
            fl.nan,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            fl.nan,
            0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            0,
            fl.nan,
            {term: fl.nan},
        )

        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            0,
            1,
            {
                term: 0.9,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.4,
            },
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            -1,
            1,
            {NaN(): np.nan},
        )

    def test_mom_defuzzifier(self) -> None:
        """Test the Largest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.MeanOfMaximum()).exports_fll("MeanOfMaximum").has_attribute(
            resolution=1000
        ).configured_as("200").has_parameters("200").exports_fll("MeanOfMaximum 200")

        # Test case:
        #            ______
        #      _____/      \
        # ____/             \
        # |                   \____
        term = fl.Discrete.create(
            "test",
            {
                0.0: 0.25,
                0.1: 0.25,
                0.2: 0.5,
                0.4: 0.5,
                0.5: 1.0,
                0.7: 1.0,
                # 0.7: 1.0 , # MOM: (0.5 + 0.9)/2=0.7
                0.9: 1.0,
                1.0: 0.0,
            },
        )

        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            -fl.inf,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            -fl.inf,
            0.0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            0.0,
            fl.inf,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            fl.nan,
            fl.nan,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            fl.nan,
            0,
            {term: fl.nan},
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            0,
            fl.nan,
            {term: fl.nan},
        )

        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            0,
            1,
            {
                term: 0.7,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.3,
            },
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            -1,
            1,
            {NaN(): np.nan},
        )

    def test_weighted_defuzzifier(self) -> None:
        """Test the weighted defuzzifier and its methods."""

        class BaseWeightedDefuzzifier(fl.WeightedDefuzzifier):
            def defuzzify(
                self, term: fl.Term, minimum: float = fl.nan, maximum: float = fl.nan
            ) -> Scalar:
                return fl.nan

        self.assertEqual(
            {BaseWeightedDefuzzifier().type},
            {fl.WeightedDefuzzifier.Type.Automatic},
        )

        defuzzifier = BaseWeightedDefuzzifier()
        defuzzifier.configure("TakagiSugeno")
        self.assertEqual(defuzzifier.type, fl.WeightedDefuzzifier.Type.TakagiSugeno)

        self.assertEqual(
            defuzzifier.infer_type(fl.Constant()),
            fl.WeightedDefuzzifier.Type.TakagiSugeno,
        )

        # Inverse Tsukamoto
        self.assertEqual(
            defuzzifier.infer_type(fl.Triangle()),
            fl.WeightedDefuzzifier.Type.Automatic,
        )
        with self.assertRaises(TypeError) as type_error:
            defuzzifier.infer_type(fl.OutputVariable(terms=[fl.Triangle(), fl.Constant()]))
        self.assertEqual(
            "cannot infer type of BaseWeightedDefuzzifier, got multiple types: "
            "['Type.Automatic', 'Type.TakagiSugeno']",
            str(type_error.exception),
        )

    def test_infer_type(self) -> None:
        """Test the inference of weighted defuzzifier based on terms."""
        takagi_sugeno_terms = [fl.Constant(), fl.Linear(), fl.Function()]
        tsukamoto_terms = [
            fl.Concave(),
            fl.Ramp(),
            fl.SShape(),
            fl.Sigmoid(),
            fl.ZShape(),
        ]

        # Takagi-Sugeno
        for term in takagi_sugeno_terms:
            self.assertEqual(
                fl.WeightedDefuzzifier.infer_type(term),
                fl.WeightedDefuzzifier.Type.TakagiSugeno,
            )
        # Tsukamoto
        for term in tsukamoto_terms:
            self.assertTrue(term.is_monotonic())
            self.assertEqual(
                fl.WeightedDefuzzifier.infer_type(term),
                fl.WeightedDefuzzifier.Type.Tsukamoto,
            )
        # Activated: TakagiSugeno
        for term in takagi_sugeno_terms:
            self.assertEqual(
                fl.WeightedDefuzzifier.infer_type(fl.Activated(term)),
                fl.WeightedDefuzzifier.Type.TakagiSugeno,
            )
        # Activated: Tsukamoto
        for term in tsukamoto_terms:
            self.assertEqual(
                fl.WeightedDefuzzifier.infer_type(fl.Activated(term)),
                fl.WeightedDefuzzifier.Type.Tsukamoto,
            )

        # Aggregated: None
        self.assertEqual(
            fl.WeightedDefuzzifier.infer_type(fl.Aggregated()),
            fl.WeightedDefuzzifier.Type.Automatic,
        )
        # Aggregated: TakagiSugeno
        self.assertEqual(
            fl.WeightedDefuzzifier.infer_type(
                fl.Aggregated(terms=[fl.Activated(term) for term in takagi_sugeno_terms])
            ),
            fl.WeightedDefuzzifier.Type.TakagiSugeno,
        )
        # Aggregated: Tsukamoto
        self.assertEqual(
            fl.WeightedDefuzzifier.infer_type(
                fl.Aggregated(terms=[fl.Activated(term) for term in tsukamoto_terms])
            ),
            fl.WeightedDefuzzifier.Type.Tsukamoto,
        )
        # Aggregated: Mixed
        with self.assertRaises(TypeError) as error:
            fl.WeightedDefuzzifier.infer_type(
                fl.Aggregated(terms=[fl.Activated(term) for term in [fl.Constant(), fl.Concave()]])
            )
        self.assertEqual(
            str(error.exception),
            "cannot infer type of WeightedDefuzzifier, "
            "got multiple types: ['Type.TakagiSugeno', 'Type.Tsukamoto']",
        )

    def test_weighted_average(self) -> None:
        """Test the weighted average defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedAverage()).has_attribute(
            type=fl.WeightedDefuzzifier.Type.Automatic
        ).exports_fll("WeightedAverage")
        DefuzzifierAssert(self, fl.WeightedAverage()).configured_as("TakagiSugeno").exports_fll(
            "WeightedAverage TakagiSugeno"
        )
        with self.assertRaises(KeyError):
            fl.WeightedAverage().configure("SugenoTakagi")

        defuzzifier = fl.WeightedAverage()
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>"),
        ):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedAverage("TakagiSugeno")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", 2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", 2.0), 0.5),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", -1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", -3.0), 0.5),
                    ]
                ): -1.0,
            },
            vectorized=False,
        )

    def test_weighted_average_grouped(self) -> None:
        """Test the weighted average defuzzifier in the presence of multiple activations of the same term."""
        DefuzzifierAssert(self, fl.WeightedAverage("TakagiSugeno")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 0.5),
                        fl.Activated(fl.Constant("A", 1.0), 0.5),
                        fl.Activated(fl.Constant("B", 2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", 2.0), 0.25),
                        fl.Activated(fl.Constant("B", 2.0), 0.25),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", -1.0), 1.0),
                        fl.Activated(fl.Constant("A", -1.0), 1.0),
                        fl.Activated(fl.Constant("A", -1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ],
                    aggregation=fl.Maximum(),
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", -3.0), 0.5),
                        fl.Activated(fl.Constant("C", -3.0), 0.5),
                    ],
                    aggregation=fl.AlgebraicSum(),
                ): -1.181818,  # ((1 * 1 + 1 * -2 + (0.5 + 0.5 - (0.25)) * -3) / (1 + 1 + (0.5 + 0.5 - 0.25))),
            },
            vectorized=False,
        )

    def test_weighted_sum(self) -> None:
        """Test the weighted sum defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedSum()).exports_fll("WeightedSum")
        DefuzzifierAssert(self, fl.WeightedSum()).configured_as("TakagiSugeno").exports_fll(
            "WeightedSum TakagiSugeno"
        )
        DefuzzifierAssert(self, fl.WeightedSum()).configured_as("Tsukamoto").exports_fll(
            "WeightedSum Tsukamoto"
        )
        with self.assertRaises(KeyError):
            fl.WeightedSum().configure("SugenoTakagi")

        defuzzifier = fl.WeightedSum()

        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>"),
        ):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedSum("TakagiSugeno")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", 2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 6.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", 2.0), 0.5),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 5.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", -1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("B", -2.0), 1.0),
                        fl.Activated(fl.Constant("C", -3.0), 0.5),
                    ]
                ): -2.5,
            },
            vectorized=False,
        )

    def test_weighted_sum_grouped(self) -> None:
        """Test the weighted sum defuzzifier in the presence of multiple activations of the same term."""
        DefuzzifierAssert(self, fl.WeightedSum("TakagiSugeno")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 0.0),
                        fl.Activated(fl.Constant("A", 1.0), 0.3),
                        fl.Activated(fl.Constant("A", 1.0), 0.6),
                    ]
                ): (0.9 * 1.0),
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 1.0),
                        fl.Activated(fl.Constant("A", 1.0), 0.3),
                        fl.Activated(fl.Constant("A", 1.0), 0.6),
                    ]
                ): (1.9 * 1.0),
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 0.0),
                        fl.Activated(fl.Constant("A", 1.0), 0.3),
                        fl.Activated(fl.Constant("A", 1.0), 0.6),
                    ],
                    aggregation=fl.Maximum(),
                ): (0.6 * 1.0),
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("A", 1.0), 0.0),
                        fl.Activated(fl.Constant("A", 1.0), 0.3),
                        fl.Activated(fl.Constant("A", 1.0), 0.6),
                    ],
                    aggregation=fl.AlgebraicSum(),
                ): (0.72 * 1.0),
            },
            vectorized=False,
        )

    def test_weighted_sum_tsukamoto(self) -> None:
        """Test the Tsukamoto defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedSum("Tsukamoto")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.015),
                        fl.Activated(fl.Ramp("b", 0.6, 0.4), 1.0),
                        fl.Activated(fl.Ramp("c", 0.7, 1.0), 0.015),
                    ]
                ): 0.410,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Sigmoid("a", 0.13, 30), 0.015),
                        fl.Activated(fl.Sigmoid("b", 0.5, -30), 1.0),
                        fl.Activated(fl.Sigmoid("c", 0.83, 30), 0.015),
                    ]
                ): -fl.inf,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Concave("a", 0.24, 0.25), 0.015),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 1.0),
                        fl.Activated(fl.Concave("c", 0.9, 1.0), 0.015),
                    ]
                ): 0.310,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.SShape("a", 0.000, 0.250), 0.015),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.SShape("c", 0.700, 1.000), 0.015),
                    ]
                ): 0.311,
            },
            vectorized=False,
        )

    def test_weighted_sum_tsukamoto_grouped(self) -> None:
        """Test the Tsukamoto defuzzifier in the presence of multiple activations of the same term."""
        DefuzzifierAssert(self, fl.WeightedSum("Tsukamoto")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.0075),
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.0075),
                        fl.Activated(fl.Ramp("b", 0.6, 0.4), 1.0),
                        fl.Activated(fl.Ramp("c", 0.7, 1.0), 0.015),
                    ]
                ): 0.410,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Concave("a", 0.24, 0.25), 0.015),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 0.5),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 0.5),
                        fl.Activated(fl.Concave("c", 0.9, 1.0), 0.015),
                    ]
                ): 0.310,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.SShape("a", 0.000, 0.250), 0.015),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.SShape("c", 0.700, 1.000), 0.015),
                    ],
                    aggregation=fl.Maximum(),
                ): 0.311,
            },
            vectorized=False,
        )

    def test_weighted_average_tsukamoto(self) -> None:
        """Test the Tsukamoto defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedAverage("Tsukamoto")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.015),
                        fl.Activated(fl.Ramp("b", 0.6, 0.4), 1.0),
                        fl.Activated(fl.Ramp("c", 0.7, 1.0), 0.015),
                    ]
                ): 0.398,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Sigmoid("a", 0.13, 30), 0.015),
                        fl.Activated(fl.Sigmoid("b", 0.5, -30), 1.0),
                        fl.Activated(fl.Sigmoid("c", 0.83, 30), 0.015),
                    ]
                ): -fl.inf,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Concave("a", 0.24, 0.25), 0.015),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 1.0),
                        fl.Activated(fl.Concave("c", 0.9, 1.0), 0.015),
                    ]
                ): 0.301,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.SShape("a", 0.000, 0.250), 0.015),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.SShape("c", 0.700, 1.000), 0.015),
                    ]
                ): 0.302,
            },
            vectorized=False,
        )

    def test_weighted_average_tsukamoto_grouped(self) -> None:
        """Test the Tsukamoto defuzzifier in the presence of multiple activations of the same term."""
        DefuzzifierAssert(self, fl.WeightedAverage("Tsukamoto")).defuzzifies(
            -fl.inf,
            fl.inf,
            {
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.0075),
                        fl.Activated(fl.Ramp("a", 0, 0.25), 0.0075),
                        fl.Activated(fl.Ramp("b", 0.6, 0.4), 1.0),
                        fl.Activated(fl.Ramp("c", 0.7, 1.0), 0.015),
                    ]
                ): 0.398,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Concave("a", 0.24, 0.25), 0.015),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 0.5),
                        fl.Activated(fl.Concave("b", 0.5, 0.4), 0.5),
                        fl.Activated(fl.Concave("c", 0.9, 1.0), 0.015),
                    ]
                ): 0.301,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.SShape("a", 0.000, 0.250), 0.015),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.ZShape("b", 0.300, 0.600), 1.0),
                        fl.Activated(fl.SShape("c", 0.700, 1.000), 0.015),
                    ],
                    aggregation=fl.Maximum(),
                ): 0.302,
            },
            vectorized=False,
        )

    def test_all_defuzzifiers_return_nan_when_empty_output(self) -> None:
        """Test that all defuzzifiers return NaN when the output is empty."""
        self.assertTrue(bool(fl.settings.factory_manager.defuzzifier.constructors))
        defuzzifiers = fl.settings.factory_manager.defuzzifier.constructors.values()
        for defuzzifier in defuzzifiers:
            expected = np.nan
            obtained = defuzzifier().defuzzify(fl.Aggregated(), -1, 1)
            np.testing.assert_equal(expected, obtained)


if __name__ == "__main__":
    unittest.main()
