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

from fuzzylite import *


class DefuzzifierAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Defuzzifier):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def exports_fll(self, fll: str):
        self.test.assertEqual(str(self.actual), fll)
        return self

    def configured_as(self, parameters: str):
        self.actual.configure(parameters)
        return self

    def has_parameters(self, parameters: str):
        self.test.assertEqual(self.actual.parameters(), parameters)
        return self

    def defuzzifies(self, terms: Dict[Term, float], minimum=-inf, maximum=inf):
        for term, result in terms.items():
            if isnan(result):
                self.test.assertEqual(isnan(self.actual.defuzzify(term, minimum, maximum)), True, f"for {str(term)}")
            else:
                self.test.assertAlmostEqual(self.actual.defuzzify(term, minimum, maximum), result, places=15,
                                            msg=f"for {str(term)}")
        return result


class TestDefuzzifier(unittest.TestCase):
    def test_defuzzifier(self):
        with self.assertRaises(NotImplementedError):
            Defuzzifier().configure("")
        with self.assertRaises(NotImplementedError):
            Defuzzifier().parameters()
        with self.assertRaises(NotImplementedError):
            Defuzzifier().defuzzify(None, nan, nan)

    def test_integral_defuzzifier(self):
        DefuzzifierAssert(self, IntegralDefuzzifier(100)) \
            .exports_fll("IntegralDefuzzifier 100") \
            .has_parameters("100") \
            .configured_as("300") \
            .exports_fll("IntegralDefuzzifier 300")
        with self.assertRaises(NotImplementedError):
            IntegralDefuzzifier(nan).defuzzify(None, nan, nan)

    @unittest.skip("Need to manually compute bisectors of triangles")
    def test_bisector(self):
        DefuzzifierAssert(self, BiSector()) \
            .exports_fll("BiSector 100") \
            .has_parameters("100") \
            .configured_as("200") \
            .exports_fll("BiSector 200")

        DefuzzifierAssert(self, BiSector()) \
            .defuzzifies(
            {
                Triangle("", -1, -1, 0): -0.5,
                Triangle("", -1, 1, 2): 0.0,
                Triangle("", 0, 0, 3): 0.5,
                Aggregated("", 0, 1, Maximum(),
                           [Activated(Triangle("Medium", 0.25, 0.5, 0.75), 0.2, Minimum()),
                            Activated(Triangle("High", 0.5, 0.75, 1.0), 0.8, Minimum())])
                : 0.7200552486187846
            }, -1, 0)

    def test_centroid(self):
        DefuzzifierAssert(self, Centroid()) \
            .exports_fll("Centroid 100") \
            .has_parameters("100") \
            .configured_as("200") \
            .exports_fll("Centroid 200")

        DefuzzifierAssert(self, Centroid()) \
            .defuzzifies(
            {
                Triangle("", -1, 0): -0.5,
                Triangle("", -1, 1): 0.0,
                Triangle("", 0, 1): 0.5,
                Aggregated("", 0, 1, Maximum(),
                           [Activated(Triangle("Medium", 0.25, 0.5, 0.75), 0.2, Minimum()),
                            Activated(Triangle("High", 0.5, 0.75, 1.0), 0.8, Minimum())])
                : 0.6900552486187845
            }, -1, 1)

    def test_weighted_defuzzifier(self):
        self.assertEqual(WeightedDefuzzifier(None).type, WeightedDefuzzifier.Type.Automatic)
        self.assertEqual(WeightedDefuzzifier("TakagiSugeno").type, WeightedDefuzzifier.Type.TakagiSugeno)

        defuzzifier = WeightedDefuzzifier(None)
        defuzzifier.configure("TakagiSugeno")
        self.assertEqual(defuzzifier.type, WeightedDefuzzifier.Type.TakagiSugeno)

        defuzzifier.type = None
        defuzzifier.configure("")
        self.assertEqual(defuzzifier.type, None)

        with self.assertRaises(KeyError):
            defuzzifier.configure("ABC")

        with self.assertRaises(NotImplementedError):
            defuzzifier.defuzzify(None, nan, nan)

        self.assertEqual(defuzzifier.infer_type(Constant()), WeightedDefuzzifier.Type.TakagiSugeno)
        self.assertEqual(defuzzifier.infer_type(Triangle()), WeightedDefuzzifier.Type.Tsukamoto)

    def test_weighted_average(self):
        DefuzzifierAssert(self, str(WeightedAverage())).exports_fll("WeightedAverage Automatic")
        DefuzzifierAssert(self, str(WeightedAverage("TakagiSugeno"))).exports_fll("WeightedAverage TakagiSugeno")
        with self.assertRaises(KeyError):
            DefuzzifierAssert(self, str(WeightedAverage("SugenoTakagi")))

        defuzzifier = WeightedAverage()
        defuzzifier.type = None
        with self.assertRaisesRegex(ValueError, "expected a type of defuzzifier, but found none"):
            defuzzifier.defuzzify(Aggregated(terms=[None]))
        with self.assertRaisesRegex(AttributeError, "'Triangle' object has no attribute 'terms'"):
            defuzzifier.defuzzify(Triangle())

        DefuzzifierAssert(self, WeightedAverage()) \
            .defuzzifies({Aggregated(): nan,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 2.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 0.5),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 2.0,
                          Aggregated(terms=[Activated(Constant("", -1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 0.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", -3.0), 0.5),
                                            ]): -1.0
                          })
        DefuzzifierAssert(self, WeightedAverage("Tsukamoto")) \
            .defuzzifies({Aggregated(): nan,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 2.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 0.5),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 2.0,
                          Aggregated(terms=[Activated(Constant("", -1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 0.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", -3.0), 0.5),
                                            ]): -1.0
                          })

    def test_weighted_sum(self):
        DefuzzifierAssert(self, str(WeightedSum())).exports_fll("WeightedSum Automatic")
        DefuzzifierAssert(self, str(WeightedSum("TakagiSugeno"))).exports_fll("WeightedSum TakagiSugeno")
        with self.assertRaises(KeyError):
            DefuzzifierAssert(self, str(WeightedSum("SugenoTakagi")))

        defuzzifier = WeightedSum()
        defuzzifier.type = None
        with self.assertRaisesRegex(ValueError, "expected a type of defuzzifier, but found none"):
            defuzzifier.defuzzify(Aggregated(terms=[None]))
        with self.assertRaisesRegex(AttributeError, "'Triangle' object has no attribute 'terms'"):
            defuzzifier.defuzzify(Triangle())

        DefuzzifierAssert(self, WeightedSum()) \
            .defuzzifies({Aggregated(): nan,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 6.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 0.5),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 5.0,
                          Aggregated(terms=[Activated(Constant("", -1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 0.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", -3.0), 0.5),
                                            ]): -2.5
                          })
        DefuzzifierAssert(self, WeightedSum("Tsukamoto")) \
            .defuzzifies({Aggregated(): nan,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 6.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", 2.0), 0.5),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 5.0,
                          Aggregated(terms=[Activated(Constant("", -1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", 3.0), 1.0),
                                            ]): 0.0,
                          Aggregated(terms=[Activated(Constant("", 1.0), 1.0),
                                            Activated(Constant("", -2.0), 1.0),
                                            Activated(Constant("", -3.0), 0.5),
                                            ]): -2.5
                          })


if __name__ == '__main__':
    unittest.main()
