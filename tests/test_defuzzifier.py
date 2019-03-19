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

import re
import unittest
from typing import Dict

import fuzzylite as fl
from tests.assert_component import BaseAssert


class DefuzzifierAssert(BaseAssert[fl.Defuzzifier]):

    def configured_as(self, parameters: str) -> 'DefuzzifierAssert':
        self.actual.configure(parameters)
        return self

    def has_parameters(self, parameters: str) -> 'DefuzzifierAssert':
        self.test.assertEqual(self.actual.parameters(), parameters)
        return self

    def defuzzifies(self, terms: Dict[fl.Term, float], minimum: float = -fl.inf,
                    maximum: float = fl.inf) -> 'DefuzzifierAssert':
        for term, result in terms.items():
            if fl.isnan(result):
                self.test.assertEqual(fl.isnan(self.actual.defuzzify(term, minimum, maximum)),
                                      True, f"for {str(term)}")
            else:
                self.test.assertAlmostEqual(self.actual.defuzzify(term, minimum, maximum), result,
                                            places=15, msg=f"for {str(term)}")
        return self


class TestDefuzzifier(unittest.TestCase):

    def test_defuzzifier(self) -> None:
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().configure("")
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().parameters()
        with self.assertRaises(NotImplementedError):
            fl.Defuzzifier().defuzzify(fl.Term(), fl.nan, fl.nan)

    def test_integral_defuzzifier(self) -> None:
        DefuzzifierAssert(self, fl.IntegralDefuzzifier()) \
            .exports_fll("IntegralDefuzzifier 100") \
            .has_parameters("100") \
            .configured_as("300") \
            .exports_fll("IntegralDefuzzifier 300")
        with self.assertRaises(NotImplementedError):
            fl.IntegralDefuzzifier().defuzzify(fl.Term(), fl.nan, fl.nan)

    @unittest.skip("Need to manually compute bisectors of triangles")
    def test_bisector(self) -> None:
        DefuzzifierAssert(self, fl.Bisector()) \
            .exports_fll("Bisector 100") \
            .has_parameters("100") \
            .configured_as("200") \
            .exports_fll("Bisector 200")

        DefuzzifierAssert(self, fl.Bisector()) \
            .defuzzifies(
            {
                fl.Triangle("", -1, -1, 0): -0.5,
                fl.Triangle("", -1, 1, 2): 0.0,
                fl.Triangle("", 0, 0, 3): 0.5,
                fl.Aggregated("", 0, 1, fl.Maximum(), [
                    fl.Activated(fl.Triangle("Medium", 0.25, 0.5, 0.75), 0.2, fl.Minimum()),
                    fl.Activated(fl.Triangle("High", 0.5, 0.75, 1.0), 0.8, fl.Minimum())
                ]): 0.7200552486187846
            }, -1, 0)

    def test_centroid(self) -> None:
        DefuzzifierAssert(self, fl.Centroid()) \
            .exports_fll("Centroid 100") \
            .has_parameters("100") \
            .configured_as("200") \
            .exports_fll("Centroid 200")

        DefuzzifierAssert(self, fl.Centroid()) \
            .defuzzifies(
            {
                fl.Triangle("", -1, 0): -0.5,
                fl.Triangle("", -1, 1): 0.0,
                fl.Triangle("", 0, 1): 0.5,
                fl.Aggregated("", 0, 1, fl.Maximum(), [
                    fl.Activated(fl.Triangle("Medium", 0.25, 0.5, 0.75), 0.2, fl.Minimum()),
                    fl.Activated(fl.Triangle("High", 0.5, 0.75, 1.0), 0.8, fl.Minimum())
                ]): 0.6900552486187845
            }, -1, 1)

    def test_weighted_defuzzifier(self) -> None:
        self.assertEqual(fl.WeightedDefuzzifier().type, fl.WeightedDefuzzifier.Type.Automatic)

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

        self.assertEqual(defuzzifier.infer_type(fl.Constant()),
                         fl.WeightedDefuzzifier.Type.TakagiSugeno)
        self.assertEqual(defuzzifier.infer_type(fl.Triangle()),
                         fl.WeightedDefuzzifier.Type.Tsukamoto)

    def test_weighted_average(self) -> None:
        DefuzzifierAssert(self, fl.WeightedAverage()).exports_fll("WeightedAverage Automatic")
        DefuzzifierAssert(self, fl.WeightedAverage()) \
            .configured_as("TakagiSugeno") \
            .exports_fll("WeightedAverage TakagiSugeno")
        with self.assertRaises(KeyError):
            fl.WeightedAverage().configure("SugenoTakagi")

        defuzzifier = fl.WeightedAverage()
        defuzzifier.type = None  # type: ignore
        with self.assertRaisesRegex(ValueError, "expected a type of defuzzifier, but found none"):
            defuzzifier.defuzzify(fl.Aggregated(terms=[fl.Activated(fl.Term())]))
        with self.assertRaisesRegex(ValueError, re.escape(
                "expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>")):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedAverage()) \
            .defuzzifies({fl.Aggregated(): fl.nan,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 2.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 0.5),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 2.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", -1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 0.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", -3.0), 0.5),
                                               ]): -1.0
                          })
        DefuzzifierAssert(self, fl.WeightedAverage()) \
            .configured_as("Tsukamoto") \
            .defuzzifies({fl.Aggregated(): fl.nan,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 2.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 0.5),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 2.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", -1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 0.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", -3.0), 0.5),
                                               ]): -1.0
                          })

    def test_weighted_sum(self) -> None:
        DefuzzifierAssert(self, fl.WeightedSum()).exports_fll("WeightedSum Automatic")
        DefuzzifierAssert(self, fl.WeightedSum()) \
            .configured_as("TakagiSugeno") \
            .exports_fll("WeightedSum TakagiSugeno")
        with self.assertRaises(KeyError):
            fl.WeightedSum().configure("SugenoTakagi")

        defuzzifier = fl.WeightedSum()
        defuzzifier.type = None  # type: ignore
        with self.assertRaisesRegex(ValueError, "expected a type of defuzzifier, but found none"):
            defuzzifier.defuzzify(fl.Aggregated(terms=[fl.Activated(fl.Term())]))
        with self.assertRaisesRegex(ValueError, re.escape(
                "expected an Aggregated term, but found <class 'fuzzylite.term.Triangle'>")):
            defuzzifier.defuzzify(fl.Triangle())

        DefuzzifierAssert(self, fl.WeightedSum()) \
            .defuzzifies({fl.Aggregated(): fl.nan,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 6.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 0.5),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 5.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", -1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 0.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", -3.0), 0.5),
                                               ]): -2.5
                          })
        DefuzzifierAssert(self, fl.WeightedSum()) \
            .configured_as("Tsukamoto") \
            .defuzzifies({fl.Aggregated(): fl.nan,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 6.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", 2.0), 0.5),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 5.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", -1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", 3.0), 1.0),
                                               ]): 0.0,
                          fl.Aggregated(terms=[fl.Activated(fl.Constant("", 1.0), 1.0),
                                               fl.Activated(fl.Constant("", -2.0), 1.0),
                                               fl.Activated(fl.Constant("", -3.0), 0.5),
                                               ]): -2.5
                          })


if __name__ == '__main__':
    unittest.main()
