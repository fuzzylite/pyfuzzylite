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
from typing import Callable

from fuzzylite import *


class TermAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Term):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def has_name(self, name: str, height: float = 1.0):
        self.test.assertEqual(self.actual.name, name)
        self.test.assertEqual(self.actual.height, height)
        return self

    def takes_parameters(self, parameters: int):
        with self.test.assertRaisesRegex(ValueError,
                                         r"not enough values to unpack \(expected %s, got 0\)" % parameters):
            self.actual.__class__().configure("")
        return self

    def is_monotonic(self, monotonic: bool = True):
        self.test.assertEqual(self.actual.is_monotonic(), monotonic)
        return self

    def is_not_monotonic(self):
        self.test.assertEqual(self.actual.is_monotonic(), False)
        return self

    def exports_to(self, fll: str):
        self.test.assertEqual(str(self.actual), fll)
        return self

    def configured_as(self, parameters: str):
        self.actual.configure(parameters)
        return self

    def has_membership(self, x: float, mf: float):
        if isnan(mf):
            self.test.assertEqual(isnan(self.actual.membership(x)), True, "when x=%.3f" % x)
        else:
            self.test.assertEqual(self.actual.membership(x), mf, "when x=%.3f" % x)
        return self

    def has_memberships(self, x_mf: Dict[float, float], height: float = 1.0):
        for x in x_mf.keys():
            self.has_membership(x, height * x_mf[x])
        return self

    def has_tsukamoto(self, x: float, mf: float, minimum: float = -1.0, maximum: float = 1.0):
        self.test.assertEqual(self.actual.is_monotonic(), True)
        if isnan(mf):
            self.test.assertEqual(isnan(self.actual.tsukamoto(x, minimum, maximum)), True, "when x=%.3f" % x)
        else:
            self.test.assertEqual(self.actual.tsukamoto(x, minimum, maximum), mf, "when x=%.3f" % x)
        return self

    def has_tsukamotos(self, x_mf: Dict[float, float], minimum: float = -1.0, maximum: float = 1.0):
        for x in x_mf.keys():
            self.has_tsukamoto(x, x_mf[x], minimum, maximum)
        return self

    def apply(self, func: Callable[..., None], args=(), **kwds):
        func(self.actual, *args, **kwds)
        return self


class TestTerm(unittest.TestCase):
    def test_term(self):
        self.assertEqual(Term().name, "")
        self.assertEqual(Term("X").name, "X")
        self.assertEqual(Term("X").height, 1.0)
        self.assertEqual(Term("X", .5).height, .5)

        self.assertEqual(str(Term("xxx", 0.5)), "term: xxx Term 0.500")
        self.assertEqual(Term().is_monotonic(), False)

        discrete_triangle = Triangle("triangle", -1, 0, 1).discretize(-1, 1, 10)
        self.assertEqual(Discrete.dict_from(discrete_triangle.xy),
                         {-1.0: 0.0,
                          -0.8: 0.19999999999999996,
                          -0.6: 0.4,
                          -0.3999999999999999: 0.6000000000000001,
                          -0.19999999999999996: 0.8,
                          0.0: 1.0,
                          0.20000000000000018: 0.7999999999999998,
                          0.40000000000000013: 0.5999999999999999,
                          0.6000000000000001: 0.3999999999999999,
                          0.8: 0.19999999999999996,
                          1.0: 0.0})

    def test_activated(self):
        TermAssert(self, Activated(Triangle("activated", -0.400, 0.000, 0.400), 1.0, AlgebraicProduct())) \
            .exports_to("AlgebraicProduct(1.000,activated)") \
            .is_not_monotonic() \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.37500000000000006,
                              -0.1: 0.7500000000000001,
                              0.0: 1.000,
                              0.1: 0.7500000000000001,
                              0.25: 0.37500000000000006,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0})

        TermAssert(self, Activated(Triangle("activated", -0.400, 0.000, 0.400), 0.5, AlgebraicProduct())) \
            .exports_to("AlgebraicProduct(0.500,activated)") \
            .is_not_monotonic() \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.18750000000000003,
                              -0.1: 0.37500000000000006,
                              0.0: 0.5,
                              0.1: 0.37500000000000006,
                              0.25: 0.18750000000000003,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0})

        activated = Activated()
        self.assertEqual(str(activated), "(1.000*none)")
        self.assertEqual(isnan(activated.membership(nan)), True, "when x=%f" % nan)
        activated.configure("")

        with self.assertRaisesRegex(ValueError, "expected a term to activate, but none found"):
            activated.membership(0.0)

        activated = Activated(Triangle("x", 0, 1), degree=1.0)
        with self.assertRaisesRegex(ValueError, "expected an implication operator, but none found"):
            activated.membership(0.0)

    def test_aggregated(self):
        aggregated = Aggregated("fuzzy_output", -1.0, 1.0, Maximum())
        low = Triangle("LOW", -1.000, -0.500, 0.000)
        medium = Triangle("MEDIUM", -0.500, 0.000, 0.500)
        aggregated.terms.extend([Activated(low, 0.6, Minimum()), Activated(medium, 0.4, Minimum())])

        TermAssert(self, aggregated) \
            .exports_to("fuzzy_output: Aggregated Maximum[Minimum(0.600,LOW),Minimum(0.400,MEDIUM)]") \
            .is_not_monotonic() \
            .has_memberships({-0.5: 0.6,
                              -0.4: 0.6,
                              -0.25: 0.5,
                              -0.1: 0.4,
                              0.0: 0.4,
                              0.1: 0.4,
                              0.25: 0.4,
                              0.4: 0.19999999999999996,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0})

        self.assertEqual(aggregated.activation_degree(low), 0.6)
        self.assertEqual(aggregated.activation_degree(medium), 0.4)

        self.assertEqual(aggregated.highest_activated_term().term, low)

        aggregated.terms.append(Activated(low, 0.4))
        aggregated.aggregation = UnboundedSum()
        self.assertEqual(aggregated.activation_degree(low), 0.6 + 0.4)

        aggregated.aggregation = None
        with self.assertRaisesRegex(ValueError, "expected an aggregation operator, but none found"):
            aggregated.membership(0.0)

    def test_bell(self):
        TermAssert(self, Bell("bell")) \
            .exports_to("term: bell Bell nan nan nan") \
            .takes_parameters(3) \
            .is_not_monotonic() \
            .configured_as("0 0.25 3.0") \
            .exports_to("term: bell Bell 0.000 0.250 3.000") \
            .has_memberships({-0.5: 0.015384615384615385,
                              -0.4: 0.05625177755617076,
                              -0.25: 0.5,
                              -0.1: 0.9959207087768499,
                              0.0: 1.0,
                              0.1: 0.9959207087768499,
                              0.25: 0.5,
                              0.4: 0.05625177755617076,
                              0.5: 0.015384615384615385,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0 0.25 3.0 0.5") \
            .exports_to("term: bell Bell 0.000 0.250 3.000 0.500") \
            .has_memberships({-0.5: 0.015384615384615385,
                              -0.4: 0.05625177755617076,
                              -0.25: 0.5,
                              -0.1: 0.9959207087768499,
                              0.0: 1.0,
                              0.1: 0.9959207087768499,
                              0.25: 0.5,
                              0.4: 0.05625177755617076,
                              0.5: 0.015384615384615385,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_binary(self):
        TermAssert(self, Binary("binary")) \
            .exports_to("term: binary Binary nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0 inf") \
            .exports_to("term: binary Binary 0.000 inf") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.0,
                              -0.25: 0.0,
                              -0.1: 0.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 1.0,
                              0.4: 1.0,
                              0.5: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("0 -inf 0.5") \
            .exports_to("term: binary Binary 0.000 -inf 0.500") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.5,
                              -0.25: 0.5,
                              -0.1: 0.5,
                              0.0: 0.5,
                              0.1: 0.0,
                              0.25: 0.0,
                              0.4: 0.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.5})

    def test_concave(self):
        TermAssert(self, Concave("concave")) \
            .exports_to("term: concave Concave nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("0.00 0.50") \
            .exports_to("term: concave Concave 0.000 0.500") \
            .has_memberships({-0.5: 0.3333333333333333,
                              -0.4: 0.35714285714285715,
                              -0.25: 0.4,
                              -0.1: 0.45454545454545453,
                              0.0: 0.5,
                              0.1: 0.5555555555555556,
                              0.25: 0.6666666666666666,
                              0.4: 0.8333333333333334,
                              0.5: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("0.00 -0.500 0.5") \
            .exports_to("term: concave Concave 0.000 -0.500 0.500") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.4166666666666667,
                              -0.25: 0.3333333333333333,
                              -0.1: 0.2777777777777778,
                              0.0: 0.25,
                              0.1: 0.22727272727272727,
                              0.25: 0.2,
                              0.4: 0.17857142857142858,
                              0.5: 0.16666666666666666,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.5})

    def test_constant(self):
        TermAssert(self, Constant("constant")) \
            .exports_to("term: constant Constant nan") \
            .takes_parameters(1) \
            .is_not_monotonic() \
            .configured_as("0.5") \
            .exports_to("term: constant Constant 0.500") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.5,
                              -0.25: 0.5,
                              -0.1: 0.5,
                              0.0: 0.5,
                              0.1: 0.5,
                              0.25: 0.5,
                              0.4: 0.5,
                              0.5: 0.5,
                              nan: 0.5,
                              inf: 0.5,
                              -inf: 0.5}) \
            .configured_as("-0.500 0.5") \
            .exports_to("term: constant Constant -0.500") \
            .has_memberships({-0.5: -0.5,
                              -0.4: -0.5,
                              -0.25: -0.5,
                              -0.1: -0.5,
                              0.0: -0.5,
                              0.1: -0.5,
                              0.25: -0.5,
                              0.4: -0.5,
                              0.5: -0.5,
                              nan: -0.5,
                              inf: -0.5,
                              -inf: -0.5})

    def test_cosine(self):
        TermAssert(self, Cosine("cosine")) \
            .exports_to("term: cosine Cosine nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.0 1") \
            .exports_to("term: cosine Cosine 0.000 1.000") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.09549150281252633,
                              -0.25: 0.5,
                              -0.1: 0.9045084971874737,
                              0.0: 1.0,
                              0.1: 0.9045084971874737,
                              0.25: 0.5,
                              0.4: 0.09549150281252633,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0.0 1.0 0.5") \
            .exports_to("term: cosine Cosine 0.000 1.000 0.500") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.09549150281252633,
                              -0.25: 0.5,
                              -0.1: 0.9045084971874737,
                              0.0: 1.0,
                              0.1: 0.9045084971874737,
                              0.25: 0.5,
                              0.4: 0.09549150281252633,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_discrete(self):
        TermAssert(self, Discrete("discrete")) \
            .exports_to("term: discrete Discrete ") \
            .is_not_monotonic() \
            .configured_as("0 1 8 9 4 5 2 3 6 7") \
            .exports_to("term: discrete Discrete 0.000 1.000 8.000 9.000 4.000 5.000 2.000 3.000 6.000 7.000") \
            .apply(Discrete.sort) \
            .exports_to("term: discrete Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000") \
            .configured_as("0 1 8 9 4 5 2 3 6 7 0.5") \
            .apply(Discrete.sort) \
            .exports_to("term: discrete Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000 0.500") \
            .configured_as(" -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000") \
            .exports_to("term: discrete Discrete -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.3999999999999999,
                              -0.25: 1.0,
                              -0.1: 0.7,
                              0.0: 0.5,
                              0.1: 0.7,
                              0.25: 1.0,
                              0.4: 0.3999999999999999,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as(" -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000 0.5") \
            .exports_to("term: discrete Discrete -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000 0.500") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.3999999999999999,
                              -0.25: 1.0,
                              -0.1: 0.7,
                              0.0: 0.5,
                              0.1: 0.7,
                              0.25: 1.0,
                              0.4: 0.3999999999999999,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

        self.assertEqual(Discrete.pairs_from([]), [])
        self.assertEqual(Discrete.values_from(Discrete.pairs_from([1, 2, 3, 4])), [1, 2, 3, 4])
        self.assertEqual(Discrete.values_from(Discrete.pairs_from({1: 2, 3: 4})), [1, 2, 3, 4])

        with self.assertRaisesRegex(ValueError,
                                    r"not enough values to unpack \(expected even number, got 3\)"):
            Discrete.pairs_from([1, 2, 3])

    def test_gaussian(self):
        TermAssert(self, Gaussian("gaussian")) \
            .exports_to("term: gaussian Gaussian nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.0 0.25") \
            .exports_to("term: gaussian Gaussian 0.000 0.250") \
            .has_memberships({-0.5: 0.1353352832366127,
                              -0.4: 0.2780373004531941,
                              -0.25: 0.6065306597126334,
                              -0.1: 0.9231163463866358,
                              0.0: 1.0,
                              0.1: 0.9231163463866358,
                              0.25: 0.6065306597126334,
                              0.4: 0.2780373004531941,
                              0.5: 0.1353352832366127,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0.0 0.25 0.5") \
            .exports_to("term: gaussian Gaussian 0.000 0.250 0.500") \
            .has_memberships({-0.5: 0.1353352832366127,
                              -0.4: 0.2780373004531941,
                              -0.25: 0.6065306597126334,
                              -0.1: 0.9231163463866358,
                              0.0: 1.0,
                              0.1: 0.9231163463866358,
                              0.25: 0.6065306597126334,
                              0.4: 0.2780373004531941,
                              0.5: 0.1353352832366127,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_gaussian_product(self):
        TermAssert(self, GaussianProduct("gaussian_product")) \
            .exports_to("term: gaussian_product GaussianProduct nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("0.0 0.25 0.1 0.5") \
            .exports_to("term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500") \
            .has_memberships({-0.5: 0.1353352832366127,
                              -0.4: 0.2780373004531941,
                              -0.25: 0.6065306597126334,
                              -0.1: 0.9231163463866358,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9559974818331,
                              0.4: 0.835270211411272,
                              0.5: 0.7261490370736908,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0.0 0.25 0.1 0.5 0.5") \
            .exports_to("term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500 0.500") \
            .has_memberships({-0.5: 0.1353352832366127,
                              -0.4: 0.2780373004531941,
                              -0.25: 0.6065306597126334,
                              -0.1: 0.9231163463866358,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9559974818331,
                              0.4: 0.835270211411272,
                              0.5: 0.7261490370736908,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_pi_shape(self):
        TermAssert(self, PiShape("pi_shape")) \
            .exports_to("term: pi_shape PiShape nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-.9 -.1 .1 1") \
            .exports_to("term: pi_shape PiShape -0.900 -0.100 0.100 1.000") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.71875,
                              -0.25: 0.9296875,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9444444444444444,
                              0.4: 0.7777777777777777,
                              0.5: 0.6049382716049383,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-.9 -.1 .1 1 .5") \
            .exports_to("term: pi_shape PiShape -0.900 -0.100 0.100 1.000 0.500") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.71875,
                              -0.25: 0.9296875,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9444444444444444,
                              0.4: 0.7777777777777777,
                              0.5: 0.6049382716049383,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_ramp(self):
        TermAssert(self, Ramp("ramp")) \
            .exports_to("term: ramp Ramp nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("-0.250 0.750") \
            .exports_to("term: ramp Ramp -0.250 0.750") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.0,
                              -0.25: 0.0,
                              -0.1: 0.150,
                              0.0: 0.250,
                              0.1: 0.350,
                              0.25: 0.500,
                              0.4: 0.650,
                              0.5: 0.750,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .has_tsukamotos({0.0: -0.250,
                             0.1: -0.150,
                             0.25: 0.0,
                             0.4: 0.15000000000000002,
                             0.5: 0.25,
                             0.6: 0.35,
                             0.75: 0.5,
                             0.9: 0.65,
                             1.0: 0.75,
                             nan: nan,
                             inf: inf,
                             -inf: -inf
                             }) \
            .configured_as("0.250 -0.750 0.5") \
            .exports_to("term: ramp Ramp 0.250 -0.750 0.500") \
            .has_memberships({-0.5: 0.750,
                              -0.4: 0.650,
                              -0.25: 0.500,
                              -0.1: 0.350,
                              0.0: 0.250,
                              0.1: 0.150,
                              0.25: 0.0,
                              0.4: 0.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 1.0}, height=0.5) \
            .has_tsukamotos({0.0: 0.250,
                             0.1: 0.04999999999999999,
                             0.25: -0.25,
                             0.4: -0.550,
                             0.5: -0.75,  # maximum \mu(x)=0.5.
                             # 0.6: -0.75,
                             # 0.75: -0.75,
                             # 0.9: -0.75,
                             # 1.0: -0.75,
                             nan: nan,
                             inf: -inf,
                             -inf: inf
                             })

    def test_rectangle(self):
        TermAssert(self, Rectangle("rectangle")) \
            .exports_to("term: rectangle Rectangle nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_to("term: rectangle Rectangle -0.400 0.400") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 1.0,
                              -0.25: 1.0,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 1.0,
                              0.4: 1.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.4 0.4 0.5") \
            .exports_to("term: rectangle Rectangle -0.400 0.400 0.500") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 1.0,
                              -0.25: 1.0,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 1.0,
                              0.4: 1.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_s_shape(self):
        TermAssert(self, SShape("s_shape")) \
            .exports_to("term: s_shape SShape nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_to("term: s_shape SShape -0.400 0.400") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.0,
                              -0.25: 0.07031250000000001,
                              -0.1: 0.28125000000000006,
                              0.0: 0.5,
                              0.1: 0.71875,
                              0.25: 0.9296875,
                              0.4: 1.0,
                              0.5: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("-0.4 0.4 0.5") \
            .exports_to("term: s_shape SShape -0.400 0.400 0.500") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.0,
                              -0.25: 0.07031250000000001,
                              -0.1: 0.28125000000000006,
                              0.0: 0.5,
                              0.1: 0.71875,
                              0.25: 0.9296875,
                              0.4: 1.0,
                              0.5: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}, height=0.5)

    def test_sigmoid(self):
        TermAssert(self, Sigmoid("sigmoid")) \
            .exports_to("term: sigmoid Sigmoid nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("0 10") \
            .exports_to("term: sigmoid Sigmoid 0.000 10.000") \
            .has_memberships({-0.5: 0.0066928509242848554,
                              -0.4: 0.01798620996209156,
                              -0.25: 0.07585818002124355,
                              -0.1: 0.2689414213699951,
                              0.0: 0.5,
                              0.1: 0.7310585786300049,
                              0.25: 0.9241418199787566,
                              0.4: 0.9820137900379085,
                              0.5: 0.9933071490757153,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("0 10 .5") \
            .exports_to("term: sigmoid Sigmoid 0.000 10.000 0.500") \
            .has_memberships({-0.5: 0.0066928509242848554,
                              -0.4: 0.01798620996209156,
                              -0.25: 0.07585818002124355,
                              -0.1: 0.2689414213699951,
                              0.0: 0.5,
                              0.1: 0.7310585786300049,
                              0.25: 0.9241418199787566,
                              0.4: 0.9820137900379085,
                              0.5: 0.9933071490757153,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}, height=0.5)

    def test_sigmoid_difference(self):
        TermAssert(self, SigmoidDifference("sigmoid_difference")) \
            .exports_to("term: sigmoid_difference SigmoidDifference nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.25 25.00 50.00 0.25") \
            .exports_to("term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250") \
            .has_memberships({-0.5: 0.0019267346633274238,
                              -0.4: 0.022977369910017923,
                              -0.25: 0.49999999998611205,
                              -0.1: 0.9770226049799834,
                              0.0: 0.9980695386973883,
                              0.1: 0.9992887851439739,
                              0.25: 0.49999627336071584,
                              0.4: 0.000552690994449101,
                              0.5: 3.7194451510957904e-06,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.25 25.00 50.00 0.25 0.5") \
            .exports_to("term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250 0.500") \
            .has_memberships({-0.5: 0.0019267346633274238,
                              -0.4: 0.022977369910017923,
                              -0.25: 0.49999999998611205,
                              -0.1: 0.9770226049799834,
                              0.0: 0.9980695386973883,
                              0.1: 0.9992887851439739,
                              0.25: 0.49999627336071584,
                              0.4: 0.000552690994449101,
                              0.5: 3.7194451510957904e-06,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_sigmoid_product(self):
        TermAssert(self, SigmoidProduct("sigmoid_product")) \
            .exports_to("term: sigmoid_product SigmoidProduct nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.250 20.000 -20.000 0.250") \
            .exports_to("term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250") \
            .has_memberships({-0.5: 0.006692848876926853,
                              -0.4: 0.04742576597971327,
                              -0.25: 0.4999773010656488,
                              -0.1: 0.9517062830264366,
                              0.0: 0.9866590924049252,
                              0.1: 0.9517062830264366,
                              0.25: 0.4999773010656488,
                              0.4: 0.04742576597971327,
                              0.5: 0.006692848876926853,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.250 20.000 -20.000 0.250 0.5") \
            .exports_to("term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250 0.500") \
            .has_memberships({-0.5: 0.006692848876926853,
                              -0.4: 0.04742576597971327,
                              -0.25: 0.4999773010656488,
                              -0.1: 0.9517062830264366,
                              0.0: 0.9866590924049252,
                              0.1: 0.9517062830264366,
                              0.25: 0.4999773010656488,
                              0.4: 0.04742576597971327,
                              0.5: 0.006692848876926853,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_spike(self):
        TermAssert(self, Spike("spike")) \
            .exports_to("term: spike Spike nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0 1.0") \
            .exports_to("term: spike Spike 0.000 1.000") \
            .has_memberships({-0.5: 0.006737946999085467,
                              -0.4: 0.01831563888873418,
                              -0.25: 0.0820849986238988,
                              -0.1: 0.36787944117144233,
                              0.0: 1.0,
                              0.1: 0.36787944117144233,
                              0.25: 0.0820849986238988,
                              0.4: 0.01831563888873418,
                              0.5: 0.006737946999085467,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0 1.0 .5") \
            .exports_to("term: spike Spike 0.000 1.000 0.500") \
            .has_memberships({-0.5: 0.006737946999085467,
                              -0.4: 0.01831563888873418,
                              -0.25: 0.0820849986238988,
                              -0.1: 0.36787944117144233,
                              0.0: 1.0,
                              0.1: 0.36787944117144233,
                              0.25: 0.0820849986238988,
                              0.4: 0.01831563888873418,
                              0.5: 0.006737946999085467,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_trapezoid(self):
        TermAssert(self, Trapezoid("trapezoid")) \
            .exports_to("term: trapezoid Trapezoid nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.400 -0.100 0.100 0.400") \
            .exports_to("term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.500,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 0.500,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.400 -0.100 0.100 0.400 .5") \
            .exports_to("term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.500,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 0.500,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_triangle(self):
        TermAssert(self, Triangle("triangle")) \
            .exports_to("term: triangle Triangle nan nan nan") \
            .takes_parameters(3) \
            .is_not_monotonic() \
            .configured_as("-0.400 0.000 0.400") \
            .exports_to("term: triangle Triangle -0.400 0.000 0.400") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.37500000000000006,
                              -0.1: 0.7500000000000001,
                              0.0: 1.000,
                              0.1: 0.7500000000000001,
                              0.25: 0.37500000000000006,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.400 0.000 0.400 .5") \
            .exports_to("term: triangle Triangle -0.400 0.000 0.400 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.37500000000000006,
                              -0.1: 0.7500000000000001,
                              0.0: 1.000,
                              0.1: 0.7500000000000001,
                              0.25: 0.37500000000000006,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_z_shape(self):
        TermAssert(self, ZShape("z_shape")) \
            .exports_to("term: z_shape ZShape nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_to("term: z_shape ZShape -0.400 0.400") \
            .has_memberships({-0.5: 1.0,
                              -0.4: 1.0,
                              -0.25: 0.9296875,
                              -0.1: 0.71875,
                              0.0: 0.5,
                              0.1: 0.28125000000000006,
                              0.25: 0.07031250000000001,
                              0.4: 0.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 1.0}) \
            .configured_as("-0.4 0.4 0.5") \
            .exports_to("term: z_shape ZShape -0.400 0.400 0.500") \
            .has_memberships({-0.5: 1.0,
                              -0.4: 1.0,
                              -0.25: 0.9296875,
                              -0.1: 0.71875,
                              0.0: 0.5,
                              0.1: 0.28125000000000006,
                              0.25: 0.07031250000000001,
                              0.4: 0.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 1.0}, height=0.5)

    def test_function(self):
        # todo
        pass


if __name__ == '__main__':
    unittest.main()
