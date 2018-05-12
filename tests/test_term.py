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

import math
import operator
import platform
import unittest

from fuzzylite import *
from tests.assert_component import ComponentAssert, BaseAssert


class TermAssert(ComponentAssert):
    def has_name(self, name: str, height: float = 1.0):
        super().has_name(name)
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

    def configured_as(self, parameters: str):
        self.actual.configure(parameters)
        return self

    def has_membership(self, x: float, mf: float):
        if isnan(mf):
            self.test.assertEqual(isnan(self.actual.membership(x)), True, "when x=%.3f" % x)
            return self
        # TODO: Find out why we get different values in different platforms
        # compare against exact values on Mac OSX
        if platform.system() == 'Darwin':
            self.test.assertEqual(self.actual.membership(x), mf, "when x=%.3f" % x)
        else:  # use approximate values in other platforms
            self.test.assertAlmostEqual(self.actual.membership(x), mf, places=15,
                                        msg="when x=%.3f" % x)
        return self

    def has_memberships(self, x_mf: Dict[float, float], height: float = 1.0):
        for x in x_mf.keys():
            self.has_membership(x, height * x_mf[x])
        return self

    def has_tsukamoto(self, x: float, mf: float, minimum: float = -1.0, maximum: float = 1.0):
        self.test.assertEqual(self.actual.is_monotonic(), True)
        if isnan(mf):
            self.test.assertEqual(isnan(self.actual.tsukamoto(x, minimum, maximum)), True,
                                  "when x=%.3f" % x)
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

        with self.assertRaisesRegex(NotImplementedError, ""):
            Term().membership(nan)
        with self.assertRaisesRegex(NotImplementedError, ""):
            Term().tsukamoto(nan, nan, nan)

        Term().update_reference(None)  # does nothing, for test coverage

        discrete_triangle = Triangle("triangle", -1.0, 0.0, 1.0).discretize(-1, 1, 10)
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
        TermAssert(self,
                   Activated(Triangle("triangle", -0.400, 0.000, 0.400), 1.0, AlgebraicProduct())) \
            .exports_fll("term: _ Activated AlgebraicProduct(1.000,triangle)") \
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

        TermAssert(self,
                   Activated(Triangle("triangle", -0.400, 0.000, 0.400), 0.5, AlgebraicProduct())) \
            .exports_fll("term: _ Activated AlgebraicProduct(0.500,triangle)") \
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
        self.assertEqual(str(activated), "term: _ Activated (1.000*none)")
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
            .exports_fll(
            "term: fuzzy_output Aggregated Maximum[Minimum(0.600,LOW),Minimum(0.400,MEDIUM)]") \
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
        TermAssert(self, aggregated) \
            .exports_fll(
            "term: fuzzy_output Aggregated [Minimum(0.600,LOW)+Minimum(0.400,MEDIUM)+(0.400*LOW)]")

        with self.assertRaisesRegex(ValueError, "expected an aggregation operator, but none found"):
            aggregated.membership(0.0)

        self.assertEqual(aggregated.range(), 2.0)

    def test_bell(self):
        TermAssert(self, Bell("bell")) \
            .exports_fll("term: bell Bell nan nan nan") \
            .takes_parameters(3) \
            .is_not_monotonic() \
            .configured_as("0 0.25 3.0") \
            .exports_fll("term: bell Bell 0.000 0.250 3.000") \
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
            .exports_fll("term: bell Bell 0.000 0.250 3.000 0.500") \
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
            .exports_fll("term: binary Binary nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0 inf") \
            .exports_fll("term: binary Binary 0.000 inf") \
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
            .exports_fll("term: binary Binary 0.000 -inf 0.500") \
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
            .exports_fll("term: concave Concave nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("0.00 0.50") \
            .exports_fll("term: concave Concave 0.000 0.500") \
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
            .exports_fll("term: concave Concave 0.000 -0.500 0.500") \
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
            .exports_fll("term: constant Constant nan") \
            .takes_parameters(1) \
            .is_not_monotonic() \
            .configured_as("0.5") \
            .exports_fll("term: constant Constant 0.500") \
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
            .exports_fll("term: constant Constant -0.500") \
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
            .exports_fll("term: cosine Cosine nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.0 1") \
            .exports_fll("term: cosine Cosine 0.000 1.000") \
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
            .exports_fll("term: cosine Cosine 0.000 1.000 0.500") \
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
            .exports_fll("term: discrete Discrete") \
            .is_not_monotonic() \
            .configured_as("0 1 8 9 4 5 2 3 6 7") \
            .exports_fll(
            "term: discrete Discrete 0.000 1.000 8.000 9.000 4.000 5.000 2.000 3.000 6.000 7.000") \
            .apply(Discrete.sort) \
            .exports_fll(
            "term: discrete Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000") \
            .configured_as("0 1 8 9 4 5 2 3 6 7 0.5") \
            .apply(Discrete.sort) \
            .exports_fll(
            "term: discrete Discrete 0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000 0.500") \
            .configured_as(" -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000") \
            .exports_fll(
            "term: discrete Discrete -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000") \
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
            .exports_fll(
            "term: discrete Discrete -0.500 0.000 -0.250 1.000 0.000 0.500 0.250 1.000 0.500 0.000 0.500") \
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

        xy = Discrete("name", Discrete.pairs_from("0 1 2 3 4 5 6 7".split()))
        self.assertSequenceEqual(xy.x(), (0, 2, 4, 6))
        self.assertSequenceEqual(xy.y(), (1, 3, 5, 7))

        self.assertEqual(Discrete.pairs_from([]), [])
        self.assertEqual(Discrete.values_from(Discrete.pairs_from([1, 2, 3, 4])), [1, 2, 3, 4])
        self.assertEqual(Discrete.values_from(Discrete.pairs_from({1: 2, 3: 4})), [1, 2, 3, 4])

        with self.assertRaisesRegex(ValueError,
                                    r"not enough values to unpack \(expected even number, got 3\)"):
            Discrete.pairs_from([1, 2, 3])

        term = Discrete()
        with self.assertRaisesRegex(ValueError,
                                    r"expected a list of \(x,y\)-pairs, but found none"):
            term.membership(0.0)
        with self.assertRaisesRegex(ValueError,
                                    r"expected a list of \(x,y\)-pairs, but found none"):
            term.xy = None
            term.membership(0.0)

    def test_gaussian(self):
        TermAssert(self, Gaussian("gaussian")) \
            .exports_fll("term: gaussian Gaussian nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.0 0.25") \
            .exports_fll("term: gaussian Gaussian 0.000 0.250") \
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
            .exports_fll("term: gaussian Gaussian 0.000 0.250 0.500") \
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
            .exports_fll("term: gaussian_product GaussianProduct nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("0.0 0.25 0.1 0.5") \
            .exports_fll("term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500") \
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
            .exports_fll("term: gaussian_product GaussianProduct 0.000 0.250 0.100 0.500 0.500") \
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

    def test_linear(self):
        engine = Engine(inputs=[InputVariable("A"), InputVariable("B"), InputVariable("C")])
        engine.inputs[0].value = 0
        engine.inputs[1].value = 1
        engine.inputs[2].value = 2

        with self.assertRaisesRegex(ValueError,
                                    r"expected the reference to an engine, but found none"):
            Linear().membership(nan)

        linear = Linear("linear", [1.0, 2.0])
        self.assertEqual(linear.engine, None)
        linear.update_reference(engine)
        self.assertEqual(linear.engine, engine)

        TermAssert(self, linear) \
            .exports_fll("term: linear Linear 1.000 2.000") \
            .is_not_monotonic() \
            .configured_as("1.0 2.0 3") \
            .exports_fll("term: linear Linear 1.000 2.000 3.000") \
            .has_memberships({-0.5: 1 * 0 + 2 * 1 + 3 * 2,  # = 8
                              -0.4: 8,
                              -0.25: 8,
                              -0.1: 8,
                              0.0: 8,
                              0.1: 8,
                              0.25: 8,
                              0.4: 8,
                              0.5: 8,
                              nan: 8,
                              inf: 8,
                              -inf: 8}) \
            .configured_as("1 2 3 5") \
            .exports_fll("term: linear Linear 1.000 2.000 3.000 5.000") \
            .has_memberships({-0.5: 1 * 0 + 2 * 1 + 3 * 2 + 5,  # = 13
                              -0.4: 13,
                              -0.25: 13,
                              -0.1: 13,
                              0.0: 13,
                              0.1: 13,
                              0.25: 13,
                              0.4: 13,
                              0.5: 13,
                              nan: 13,
                              inf: 13,
                              -inf: 13}) \
            .configured_as("1 2 3 5 8") \
            .exports_fll("term: linear Linear 1.000 2.000 3.000 5.000 8.000") \
            .has_memberships({-0.5: 1 * 0 + 2 * 1 + 3 * 2 + 5,  # = 13
                              -0.4: 13,
                              -0.25: 13,
                              -0.1: 13,
                              0.0: 13,
                              0.1: 13,
                              0.25: 13,
                              0.4: 13,
                              0.5: 13,
                              nan: 13,
                              inf: 13,
                              -inf: 13})

    def test_pi_shape(self):
        TermAssert(self, PiShape("pi_shape")) \
            .exports_fll("term: pi_shape PiShape nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-.9 -.1 .1 1") \
            .exports_fll("term: pi_shape PiShape -0.900 -0.100 0.100 1.000") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.71875,
                              -0.25: 0.9296875,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9444444444444444,
                              0.4: 0.7777777777777777,
                              0.5: 0.6049382716049383,
                              0.95: 0.00617283950617285,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-.9 -.1 .1 1 .5") \
            .exports_fll("term: pi_shape PiShape -0.900 -0.100 0.100 1.000 0.500") \
            .has_memberships({-0.5: 0.5,
                              -0.4: 0.71875,
                              -0.25: 0.9296875,
                              -0.1: 1.0,
                              0.0: 1.0,
                              0.1: 1.0,
                              0.25: 0.9444444444444444,
                              0.4: 0.7777777777777777,
                              0.5: 0.6049382716049383,
                              0.95: 0.00617283950617285,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}, height=0.5)

    def test_ramp(self):
        TermAssert(self, Ramp("ramp")) \
            .exports_fll("term: ramp Ramp nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("1 1") \
            .exports_fll("term: ramp Ramp 1.000 1.000") \
            .has_memberships({-0.5: 0.0,
                              -0.4: 0.0,
                              -0.25: 0.0,
                              -0.1: 0.0,
                              0.0: 0.0,
                              0.1: 0.0,
                              0.25: 0.0,
                              0.4: 0.0,
                              0.5: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.250 0.750") \
            .exports_fll("term: ramp Ramp -0.250 0.750") \
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
            .exports_fll("term: ramp Ramp 0.250 -0.750 0.500") \
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
            .exports_fll("term: rectangle Rectangle nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_fll("term: rectangle Rectangle -0.400 0.400") \
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
            .exports_fll("term: rectangle Rectangle -0.400 0.400 0.500") \
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
            .exports_fll("term: s_shape SShape nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_fll("term: s_shape SShape -0.400 0.400") \
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
            .exports_fll("term: s_shape SShape -0.400 0.400 0.500") \
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
            .exports_fll("term: sigmoid Sigmoid nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("0 10") \
            .exports_fll("term: sigmoid Sigmoid 0.000 10.000") \
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
            .exports_fll("term: sigmoid Sigmoid 0.000 10.000 0.500") \
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
            .exports_fll("term: sigmoid_difference SigmoidDifference nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.25 25.00 50.00 0.25") \
            .exports_fll("term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250") \
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
            .exports_fll(
            "term: sigmoid_difference SigmoidDifference -0.250 25.000 50.000 0.250 0.500") \
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
            .exports_fll("term: sigmoid_product SigmoidProduct nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.250 20.000 -20.000 0.250") \
            .exports_fll("term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250") \
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
            .exports_fll("term: sigmoid_product SigmoidProduct -0.250 20.000 -20.000 0.250 0.500") \
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
            .exports_fll("term: spike Spike nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0 1.0") \
            .exports_fll("term: spike Spike 0.000 1.000") \
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
            .exports_fll("term: spike Spike 0.000 1.000 0.500") \
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
        TermAssert(self, Trapezoid("trapezoid", 0.0, 1.0)).exports_fll(
            "term: trapezoid Trapezoid 0.000 0.200 0.800 1.000")

        TermAssert(self, Trapezoid("trapezoid")) \
            .exports_fll("term: trapezoid Trapezoid nan nan nan nan") \
            .takes_parameters(4) \
            .is_not_monotonic() \
            .configured_as("-0.400 -0.100 0.100 0.400") \
            .exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400") \
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
            .exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.100 0.400 0.500") \
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
                              -inf: 0.0}, height=0.5) \
            .configured_as("-0.400 -0.400 0.100 0.400") \
            .exports_fll("term: trapezoid Trapezoid -0.400 -0.400 0.100 0.400") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 1.000,
                              -0.25: 1.000,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 0.500,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.400 -0.100 0.400 0.400") \
            .exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.400 0.400") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.5,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 1.000,
                              0.4: 1.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-inf -0.100 0.100 .4") \
            .exports_fll("term: trapezoid Trapezoid -inf -0.100 0.100 0.400") \
            .has_memberships({-0.5: 1.000,
                              -0.4: 1.000,
                              -0.25: 1.000,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 0.500,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 1.0}) \
            .configured_as("-.4 -0.100 0.100 inf .5") \
            .exports_fll("term: trapezoid Trapezoid -0.400 -0.100 0.100 inf 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.500,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 1.000,
                              0.4: 1.000,
                              0.5: 1.000,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}, height=0.5)

    def test_triangle(self):
        TermAssert(self, Triangle("triangle", 0.0, 1.0)).exports_fll(
            "term: triangle Triangle 0.000 0.500 1.000")

        TermAssert(self, Triangle("triangle")) \
            .exports_fll("term: triangle Triangle nan nan nan") \
            .takes_parameters(3) \
            .is_not_monotonic() \
            .configured_as("-0.400 0.000 0.400") \
            .exports_fll("term: triangle Triangle -0.400 0.000 0.400") \
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
            .exports_fll("term: triangle Triangle -0.400 0.000 0.400 0.500") \
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
                              -inf: 0.0}, height=0.5) \
            .configured_as("-0.500 0.000 0.500") \
            .exports_fll("term: triangle Triangle -0.500 0.000 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.19999999999999996,
                              -0.25: 0.5,
                              -0.1: 0.8,
                              0.0: 1.000,
                              0.1: 0.8,
                              0.25: 0.5,
                              0.4: 0.19999999999999996,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.500 -0.500 0.500") \
            .exports_fll("term: triangle Triangle -0.500 -0.500 0.500") \
            .has_memberships({-0.5: 1.000,
                              -0.4: 0.900,
                              -0.25: 0.75,
                              -0.1: 0.6,
                              0.0: 0.5,
                              0.1: 0.4,
                              0.25: 0.25,
                              0.4: 0.09999999999999998,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-0.500 0.500 0.500") \
            .exports_fll("term: triangle Triangle -0.500 0.500 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.09999999999999998,
                              -0.25: 0.25,
                              -0.1: 0.4,
                              0.0: 0.5,
                              0.1: 0.6,
                              0.25: 0.75,
                              0.4: 0.900,
                              0.5: 1.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("-inf 0.000 0.400") \
            .exports_fll("term: triangle Triangle -inf 0.000 0.400") \
            .has_memberships({-0.5: 1.000,
                              -0.4: 1.000,
                              -0.25: 1.000,
                              -0.1: 1.000,
                              0.0: 1.000,
                              0.1: 0.7500000000000001,
                              0.25: 0.37500000000000006,
                              0.4: 0.000,
                              0.5: 0.000,
                              nan: nan,
                              inf: 0.0,
                              -inf: 1.000}) \
            .configured_as("-0.400 0.000 inf .5") \
            .exports_fll("term: triangle Triangle -0.400 0.000 inf 0.500") \
            .has_memberships({-0.5: 0.000,
                              -0.4: 0.000,
                              -0.25: 0.37500000000000006,
                              -0.1: 0.7500000000000001,
                              0.0: 1.000,
                              0.1: 1.000,
                              0.25: 1.000,
                              0.4: 1.000,
                              0.5: 1.000,
                              nan: nan,
                              inf: 1.000,
                              -inf: 0.0}, height=0.5)

    def test_z_shape(self):
        TermAssert(self, ZShape("z_shape")) \
            .exports_fll("term: z_shape ZShape nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("-0.4 0.4") \
            .exports_fll("term: z_shape ZShape -0.400 0.400") \
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
            .exports_fll("term: z_shape ZShape -0.400 0.400 0.500") \
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

    @unittest.skip("Testing of Function")
    def test_function(self):
        # http://www.bestcode.com/html/evaluate_math_expressions_pyth.html
        # https://www.geeksforgeeks.org/eval-in-python/
        pass

    @unittest.skip("Testing of Tsukamoto")
    def test_tsukamoto(self):
        pass


class FunctionNodeAssert(BaseAssert):
    def prefix_is(self, prefix: str):
        self.test.assertEqual(self.actual.prefix(), prefix)
        return self

    def infix_is(self, infix: str):
        self.test.assertEqual(self.actual.infix(), infix)
        return self

    def postfix_is(self, postfix: str):
        self.test.assertEqual(self.actual.postfix(), postfix)
        return self

    def to_string_is(self, expected: str):
        self.test.assertEqual(str(self.actual), expected)
        return self

    def evaluates_to(self, value: float, variables: Dict[str, float] = None):
        self.test.assertAlmostEqual(self.actual.evaluate(variables), value, places=15,
                                    msg="when value is %.3f" % value)
        return self

    def fails_to_evaluate(self, exception: Exception, message: str):
        with self.test.assertRaisesRegex(exception, message):
            self.actual.evaluate()
        return self


class TestFunction(unittest.TestCase):
    def test_element(self):
        element = Function.Element("function", "math function()", Function.Element.Type.Function,
                                   None, 0, 0, -1)
        self.assertEqual(str(element), "Element: name='function', description='math function()', "
                                       "element_type='Type.Function', method='None', arity=0, "
                                       "precedence=0, associativity=-1")

        element = Function.Element("operator", "math operator", Function.Element.Type.Operator,
                                   operator.add, 2, 10, 1)
        self.assertEqual(str(element), "Element: name='operator', description='math operator', "
                                       "element_type='Type.Operator', "
                                       "method='<built-in function add>', arity=2, "
                                       "precedence=10, associativity=1")

        self.assertEqual(str(element), str(element.clone()))

    def test_node_evaluation(self):
        FunctionNodeAssert(self, Function.Node(
            element=Function.Element("undefined", "undefined method", None)
        )).fails_to_evaluate(ValueError, "expected a method reference, but found none")

        node_mult = Function.Node(
            element=Function.Element("*", "multiplication", Function.Element.Type.Operator,
                                     operator.mul, 2, 80),
            left=Function.Node(value=3.0),
            right=Function.Node(value=4.0)
        )
        FunctionNodeAssert(self, node_mult) \
            .postfix_is("3.000 4.000 *") \
            .prefix_is("* 3.000 4.000") \
            .infix_is("3.000 * 4.000") \
            .evaluates_to(12.0)

        node_sin = Function.Node(
            element=Function.Element("sin", "sine", Function.Element.Type.Function,
                                     math.sin, 1),
            left=node_mult
        )
        FunctionNodeAssert(self, node_sin) \
            .postfix_is("3.000 4.000 * sin") \
            .prefix_is("sin * 3.000 4.000") \
            .infix_is("sin ( 3.000 * 4.000 )") \
            .evaluates_to(-0.5365729180004349)

        node_pow = Function.Node(
            element=Function.Element("pow", "pow", Function.Element.Type.Function,
                                     math.pow, 2),
            left=node_sin,
            right=Function.Node(variable="two")
        )

        FunctionNodeAssert(self, node_pow) \
            .postfix_is("3.000 4.000 * sin two pow") \
            .prefix_is("pow sin * 3.000 4.000 two") \
            .infix_is("pow ( sin ( 3.000 * 4.000 ) two )") \
            .fails_to_evaluate(ValueError,
                               "expected a map of variables containing the value for 'two', "
                               "but the map contains: None") \
            .evaluates_to(0.28791049633150145, {'two': 2})

        node_sum = Function.Node(
            element=Function.Element("+", "sum", Function.Element.Type.Operator,
                                     operator.add, 2, 90),
            left=node_pow,
            right=node_pow
        )

        FunctionNodeAssert(self, node_sum) \
            .postfix_is("3.000 4.000 * sin two pow 3.000 4.000 * sin two pow +") \
            .prefix_is("+ pow sin * 3.000 4.000 two pow sin * 3.000 4.000 two") \
            .infix_is("pow ( sin ( 3.000 * 4.000 ) two ) + pow ( sin ( 3.000 * 4.000 ) two )") \
            .evaluates_to(0.5758209926630029, {'two': 2})

    def test_node_deep_clone(self):
        node_mult = Function.Node(
            element=Function.Element("*", "multiplication", Function.Element.Type.Operator,
                                     operator.mul, 2, 80),
            left=Function.Node(value=3.0),
            right=Function.Node(value=4.0)
        )
        node_sin = Function.Node(
            element=Function.Element("sin", "sine", Function.Element.Type.Function,
                                     math.sin, 1),
            left=node_mult
        )
        FunctionNodeAssert(self, node_sin) \
            .infix_is("sin ( 3.000 * 4.000 )") \
            .evaluates_to(-0.5365729180004349)

        clone = node_sin.clone()

        FunctionNodeAssert(self, clone) \
            .infix_is("sin ( 3.000 * 4.000 )") \
            .evaluates_to(-0.5365729180004349)

        # if we change the original object
        node_sin.left.element.name = "?"
        # the clone cannot be affected
        FunctionNodeAssert(self, clone) \
            .infix_is("sin ( 3.000 * 4.000 )") \
            .evaluates_to(-0.5365729180004349)

    def test_node_str(self):
        FunctionNodeAssert(self, Function.Node(
            element=Function.Element("+", "sum", None))) \
            .to_string_is("+")
        FunctionNodeAssert(self, Function.Node(
            element=Function.Element("+", "sum", None), variable="x")) \
            .to_string_is("+")
        FunctionNodeAssert(self, Function.Node(
            element=Function.Element("+", "sum", None), variable="x", value="1")) \
            .to_string_is("+")

        FunctionNodeAssert(self, Function.Node(variable="x")) \
            .to_string_is("x")
        FunctionNodeAssert(self, Function.Node(variable="x", value="1.0")) \
            .to_string_is("x")

        FunctionNodeAssert(self, Function.Node(value="1")) \
            .to_string_is("1")


if __name__ == '__main__':
    unittest.main()
