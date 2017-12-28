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
from typing import Dict, Tuple

from fuzzylite.norm import *


class NormAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Norm):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def is_t_norm(self):
        self.test.assertIsInstance(self.actual, TNorm)
        return self

    def is_s_norm(self):
        self.test.assertIsInstance(self.actual, SNorm)
        return self

    def evaluates(self, abz: Dict[Tuple, float], commutative: bool = True):
        for ab, z in abz.items():
            self.test.assertEqual(self.actual.compute(*ab), z, "in (%f %f)" % (ab))
            if commutative:
                self.test.assertEqual(self.actual.compute(*reversed(ab)), z, "when (%f %f)" % tuple(reversed(ab)))
        return self


class TNormTerm(unittest.TestCase):
    def test_algebraic_product(self):
        NormAssert(self, AlgebraicProduct()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_bounded_difference(self):
        NormAssert(self, BoundedDifference()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_drastic_product(self):
        NormAssert(self, DrasticProduct()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_einstein_product(self):
        NormAssert(self, EinsteinProduct()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_hamacher_product(self):
        NormAssert(self, HamacherProduct()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_minimum(self):
        NormAssert(self, Minimum()) \
            .is_t_norm() \
            .evaluates(
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
            })

    def test_nilpotent_minimum(self):
        NormAssert(self, NilpotentMinimum()) \
            .is_t_norm() \
            .evaluates(
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
            })

    @unittest.skip("Testing of TNormFunction")
    def test_t_function(self):
        raise NotImplemented()


class SNormTerm(unittest.TestCase):
    def test_algebraic_sum(self):
        NormAssert(self, AlgebraicSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_bounded_sum(self):
        NormAssert(self, BoundedSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_drastic_sum(self):
        NormAssert(self, DrasticSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_einstein_sum(self):
        NormAssert(self, EinsteinSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_hamacher_sum(self):
        NormAssert(self, HamacherSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_maximum(self):
        NormAssert(self, Maximum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_nilpotent_maximum(self):
        NormAssert(self, NilpotentMaximum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_normalized_sum(self):
        NormAssert(self, NormalizedSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    def test_unbounded_sum(self):
        NormAssert(self, UnboundedSum()) \
            .is_s_norm() \
            .evaluates(
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
            })

    @unittest.skip("Testing of SNormFunction")
    def test_s_function(self):
        raise NotImplemented()


if __name__ == '__main__':
    unittest.main()
