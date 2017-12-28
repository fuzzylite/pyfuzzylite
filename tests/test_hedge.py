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
from math import inf, isnan, nan
from typing import Dict

from fuzzylite.hedge import *


class HedgeAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Hedge):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def evaluates(self, az: Dict[float, float]):
        for a, z in az.items():
            if isnan(z):
                self.test.assertEqual(isnan(self.actual.hedge(a)), True, "when x=%f" % (a))
            else:
                self.test.assertEqual(self.actual.hedge(a), z, "when x=%f" % (a))
        return self


class TestHedge(unittest.TestCase):
    def test_any(self):
        HedgeAssert(self, Any()) \
            .evaluates({-1.0: 1.0,
                        -0.5: 1.0,
                        0.00: 1.0,
                        0.25: 1.0,
                        0.50: 1.0,
                        0.75: 1.0,
                        1.00: 1.0,
                        inf: 1.0,
                        -inf: 1.0,
                        nan: 1.0})

    def test_extremely(self):
        HedgeAssert(self, Extremely()) \
            .evaluates({-1.0: 2.0,
                        -0.5: 0.5,
                        0.00: 0.0,
                        0.25: 0.125,
                        0.50: 0.5,
                        0.75: 0.875,
                        1.00: 1.0,
                        inf: -inf,
                        -inf: inf,
                        nan: nan})

    def test_not(self):
        HedgeAssert(self, Not()) \
            .evaluates({-1.0: 2.0,
                        -0.5: 1.5,
                        0.00: 1.0,
                        0.25: 0.75,
                        0.50: 0.50,
                        0.75: 0.25,
                        1.00: 0.00,
                        inf: -inf,
                        -inf: inf,
                        nan: nan})

    def test_seldom(self):
        with self.assertRaisesRegex(ValueError,
                                    r"math domain error"):
            HedgeAssert(self, Seldom()).evaluates({-1.0: nan})
            HedgeAssert(self, Seldom()).evaluates({-0.5: nan})
            HedgeAssert(self, Seldom()).evaluates({inf: nan})
            HedgeAssert(self, Seldom()).evaluates({-inf: nan})

        HedgeAssert(self, Seldom()) \
            .evaluates({0.00: 0.0,
                        0.25: 0.3535533905932738,
                        0.50: 0.5,
                        0.75: 0.6464466094067263,
                        1.00: 1.0,
                        nan: nan})

    def test_somewhat(self):
        with self.assertRaisesRegex(ValueError,
                                    r"math domain error"):
            HedgeAssert(self, Somewhat()).evaluates({-1.0: nan})
            HedgeAssert(self, Somewhat()).evaluates({-0.5: nan})
            HedgeAssert(self, Somewhat()).evaluates({-inf: nan})

        HedgeAssert(self, Somewhat()) \
            .evaluates({0.00: 0.0,
                        0.25: 0.5,
                        0.50: 0.7071067811865476,
                        0.75: 0.8660254037844386,
                        1.00: 1.0,
                        inf: inf,
                        nan: nan})

    def test_very(self):
        HedgeAssert(self, Very()) \
            .evaluates({-1.0: 1.0,
                        -0.5: 0.25,
                        0.00: 0.0,
                        0.25: 0.0625,
                        0.50: 0.25,
                        0.75: 0.5625,
                        1.00: 1.0,
                        inf: inf,
                        -inf: inf,
                        nan: nan})

    @unittest.skip
    def test_function(self):
        HedgeAssert(self, HedgeFunction()) \
            .evaluates({})
