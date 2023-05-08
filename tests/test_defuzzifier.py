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
import re
import unittest

import numpy as np

import fuzzylite as fl
from tests.assert_component import BaseAssert


class DefuzzifierAssert(BaseAssert[fl.Defuzzifier]):
    """Defuzzifier assert."""

    def configured_as(self, parameters: str) -> "DefuzzifierAssert":
        """Configures the actual defuzzifier with the parameters."""
        self.actual.configure(parameters)
        return self

    def has_parameters(self, parameters: str) -> "DefuzzifierAssert":
        """Assert that the defuzzifier has the given parameters."""
        self.test.assertEqual(self.actual.parameters(), parameters)
        return self

    def defuzzifies(
        self,
        terms: dict[fl.Term, float],
        minimum: float = -fl.inf,
        maximum: float = fl.inf,
    ) -> "DefuzzifierAssert":
        """Assert that the defuzzification of the given terms result in the expected values."""
        # todo: do range parameters first, terms next
        for term, result in terms.items():
            np.testing.assert_almost_equal(
                self.actual.defuzzify(term, minimum, maximum), result, decimal=3
            )

        return self


class TestDefuzzifier(unittest.TestCase):
    """Tests the defuzzifiers."""

    def test_defuzzifier(self) -> None:
        """Test the base class methods."""
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().configure("")
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().parameters()
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().defuzzify(fl.Term(), fl.nan, fl.nan)

    def test_integral_defuzzifier(self) -> None:
        """Test integral defuzzifier default values and methods."""
        DefuzzifierAssert(self, fl.IntegralDefuzzifier()).exports_fll(
            "IntegralDefuzzifier 1000"
        ).has_parameters("1000").configured_as("300").exports_fll(
            "IntegralDefuzzifier 300"
        )
        with self.assertRaises(NotImplementedError):
            fl.IntegralDefuzzifier().defuzzify(fl.Term(), fl.nan, fl.nan)

    def test_bisector(self) -> None:
        """Test the bisector defuzzifier."""
        DefuzzifierAssert(self, fl.Bisector()).exports_fll(
            "Bisector 1000"
        ).has_parameters("1000").configured_as("200").exports_fll("Bisector 200")

        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            {fl.Triangle("", 0, 1, 1): 0.7065}, 0, 1
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            {fl.Triangle("", 0, 0, 1): 0.2925}, 0, 1
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            {fl.Triangle("", 0, 0.5, 1): 0.4995}, 0, 1
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            {fl.Rectangle("", 0, 1): 0.4995}, 0, 1
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
            {fl.Rectangle("", -1, 1): -0.001}, -1, 1
        )
        DefuzzifierAssert(self, fl.Bisector()).defuzzifies(
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
            -1,
            1,
        )

    def test_centroid(self) -> None:
        """Test the centroid defuzzifier."""
        DefuzzifierAssert(self, fl.Centroid()).exports_fll(
            "Centroid 1000"
        ).has_parameters("1000").configured_as("200").exports_fll("Centroid 200")

        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            {fl.Triangle(): fl.nan}, -fl.inf, 0.0
        )
        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            {fl.Triangle(): fl.nan}, 0.0, fl.inf
        )
        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
            {fl.Triangle(): fl.nan}, fl.nan, 0.0
        )

        DefuzzifierAssert(self, fl.Centroid()).defuzzifies(
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
                        fl.Activated(
                            fl.Triangle("Medium", 0.25, 0.5, 0.75), 0.2, fl.Minimum()
                        ),
                        fl.Activated(
                            fl.Triangle("High", 0.5, 0.75, 1.0), 0.8, fl.Minimum()
                        ),
                    ],
                ): 0.6896552,
            },
            -1,
            1,
        )

    def test_som_defuzzifier(self) -> None:
        """Test the Smallest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).exports_fll(
            "SmallestOfMaximum 1000"
        ).has_parameters("1000").configured_as("200").exports_fll(
            "SmallestOfMaximum 200"
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
                0.5: 1.0,  # SOM
                0.7: 1.0,
                0.9: 1.0,
                1.0: 0.0,
            },
        )

        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, -fl.inf, fl.inf
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, -fl.inf, 0.0
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, 0.0, fl.inf
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, fl.nan
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, 0
        )
        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {term: fl.nan}, 0, fl.nan
        )

        DefuzzifierAssert(self, fl.SmallestOfMaximum()).defuzzifies(
            {
                term: 0.5,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.2,
            },
            0,
            1,
        )

    def test_lom_defuzzifier(self) -> None:
        """Test the Largest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.LargestOfMaximum()).exports_fll(
            "LargestOfMaximum 1000"
        ).has_parameters("1000").configured_as("200").exports_fll(
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
            {term: fl.nan}, -fl.inf, fl.inf
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {term: fl.nan}, -fl.inf, 0.0
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {term: fl.nan}, 0.0, fl.inf
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, fl.nan
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, 0
        )
        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {term: fl.nan}, 0, fl.nan
        )

        DefuzzifierAssert(self, fl.LargestOfMaximum()).defuzzifies(
            {
                term: 0.9,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.4,
            },
            0,
            1,
        )

    def test_mom_defuzzifier(self) -> None:
        """Test the Largest of Maximum defuzzifier."""
        DefuzzifierAssert(self, fl.MeanOfMaximum()).exports_fll(
            "MeanOfMaximum 1000"
        ).has_parameters("1000").configured_as("200").exports_fll("MeanOfMaximum 200")

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
            {term: fl.nan}, -fl.inf, fl.inf
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {term: fl.nan}, -fl.inf, 0.0
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {term: fl.nan}, 0.0, fl.inf
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, fl.nan
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {term: fl.nan}, fl.nan, 0
        )
        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {term: fl.nan}, 0, fl.nan
        )

        DefuzzifierAssert(self, fl.MeanOfMaximum()).defuzzifies(
            {
                term: 0.7,
                fl.Trapezoid("", 0.0, 0.2, 0.4, 0.6): 0.3,
            },
            0,
            1,
        )

    def test_weighted_defuzzifier(self) -> None:
        """Test the weighted defuzzifier and its methods."""
        self.assertEqual(
            fl.WeightedDefuzzifier().type, fl.WeightedDefuzzifier.Type.Automatic
        )

        defuzzifier = fl.WeightedDefuzzifier()
        defuzzifier.configure("TakagiSugeno")
        self.assertEqual(defuzzifier.type, fl.WeightedDefuzzifier.Type.TakagiSugeno)

        defuzzifier.type = None  # type: ignore
        defuzzifier.configure("")
        self.assertEqual(defuzzifier.type, None)

        with self.assertRaises(KeyError):
            defuzzifier.configure("ABC")

        with self.assertRaises(NotImplementedError):
            defuzzifier.defuzzify(fl.Term(), fl.nan, fl.nan)

        self.assertEqual(
            defuzzifier.infer_type(fl.Constant()),
            fl.WeightedDefuzzifier.Type.TakagiSugeno,
        )
        self.assertEqual(
            defuzzifier.infer_type(fl.Triangle()), fl.WeightedDefuzzifier.Type.Tsukamoto
        )

    def test_weighted_average(self) -> None:
        """Test the weighted average defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedAverage()).exports_fll(
            "WeightedAverage Automatic"
        )
        DefuzzifierAssert(self, fl.WeightedAverage()).configured_as(
            "TakagiSugeno"
        ).exports_fll("WeightedAverage TakagiSugeno")
        with self.assertRaises(KeyError):
            fl.WeightedAverage().configure("SugenoTakagi")

        defuzzifier = fl.WeightedAverage()
        defuzzifier.type = None  # type: ignore
        with self.assertRaisesRegex(
            ValueError, "expected a type of defuzzifier, but found none"
        ):
            defuzzifier.defuzzify(fl.Aggregated(terms=[fl.Activated(fl.Term())]))
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>"
            ),
        ):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedAverage()).defuzzifies(
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 0.5),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", -1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", -3.0), 0.5),
                    ]
                ): -1.0,
            }
        )
        DefuzzifierAssert(self, fl.WeightedAverage()).configured_as(
            "Tsukamoto"
        ).defuzzifies(
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 0.5),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 2.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", -1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", -3.0), 0.5),
                    ]
                ): -1.0,
            }
        )

    def test_weighted_sum(self) -> None:
        """Test the weighted sum defuzzifier."""
        DefuzzifierAssert(self, fl.WeightedSum()).exports_fll("WeightedSum Automatic")
        DefuzzifierAssert(self, fl.WeightedSum()).configured_as(
            "TakagiSugeno"
        ).exports_fll("WeightedSum TakagiSugeno")
        with self.assertRaises(KeyError):
            fl.WeightedSum().configure("SugenoTakagi")

        defuzzifier = fl.WeightedSum()
        defuzzifier.type = None  # type: ignore
        with self.assertRaisesRegex(
            ValueError, "expected a type of defuzzifier, but found none"
        ):
            defuzzifier.defuzzify(fl.Aggregated(terms=[fl.Activated(fl.Term())]))
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>"
            ),
        ):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedSum()).defuzzifies(
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 6.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 0.5),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 5.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", -1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", -3.0), 0.5),
                    ]
                ): -2.5,
            }
        )
        DefuzzifierAssert(self, fl.WeightedSum()).configured_as(
            "Tsukamoto"
        ).defuzzifies(
            {
                fl.Aggregated(): fl.nan,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 6.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", 2.0), 0.5),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 5.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", -1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", 3.0), 1.0),
                    ]
                ): 0.0,
                fl.Aggregated(
                    terms=[
                        fl.Activated(fl.Constant("", 1.0), 1.0),
                        fl.Activated(fl.Constant("", -2.0), 1.0),
                        fl.Activated(fl.Constant("", -3.0), 0.5),
                    ]
                ): -2.5,
            }
        )


if __name__ == "__main__":
    unittest.main()
