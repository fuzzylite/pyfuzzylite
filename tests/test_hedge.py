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

import fuzzylite as fl
from tests.assert_component import BaseAssert


class HedgeAssert(BaseAssert[fl.Hedge]):

    def evaluates(self, az: Dict[float, float]) -> 'HedgeAssert':
        for a, z in az.items():
            if isnan(z):
                self.test.assertEqual(isnan(self.actual.hedge(a)), True, f"when x={a}")
            else:
                self.test.assertEqual(self.actual.hedge(a), z, f"when x={a}")
        return self


class TestHedge(unittest.TestCase):

    def test_any(self) -> None:
        HedgeAssert(self, fl.Any()) \
            .has_name("any") \
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

    def test_extremely(self) -> None:
        HedgeAssert(self, fl.Extremely()) \
            .has_name("extremely") \
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

    def test_not(self) -> None:
        HedgeAssert(self, fl.Not()) \
            .has_name("not") \
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

    def test_seldom(self) -> None:
        with self.assertRaisesRegex(ValueError,
                                    r"math domain error"):
            HedgeAssert(self, fl.Seldom()).evaluates({-1.0: nan})
            HedgeAssert(self, fl.Seldom()).evaluates({-0.5: nan})
            HedgeAssert(self, fl.Seldom()).evaluates({inf: nan})
            HedgeAssert(self, fl.Seldom()).evaluates({-inf: nan})

        HedgeAssert(self, fl.Seldom()) \
            .has_name("seldom") \
            .evaluates({0.00: 0.0,
                        0.25: 0.3535533905932738,
                        0.50: 0.5,
                        0.75: 0.6464466094067263,
                        1.00: 1.0,
                        nan: nan})

    def test_somewhat(self) -> None:
        with self.assertRaisesRegex(ValueError,
                                    r"math domain error"):
            HedgeAssert(self, fl.Somewhat()).evaluates({-1.0: nan})
            HedgeAssert(self, fl.Somewhat()).evaluates({-0.5: nan})
            HedgeAssert(self, fl.Somewhat()).evaluates({-inf: nan})

        HedgeAssert(self, fl.Somewhat()) \
            .has_name("somewhat") \
            .evaluates({0.00: 0.0,
                        0.25: 0.5,
                        0.50: 0.7071067811865476,
                        0.75: 0.8660254037844386,
                        1.00: 1.0,
                        inf: inf,
                        nan: nan})

    def test_very(self) -> None:
        HedgeAssert(self, fl.Very()) \
            .has_name("very") \
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

    def test_function(self) -> None:
        HedgeAssert(self, fl.HedgeFunction(fl.Function.create("my_hedge", "x**2"))) \
            .has_name("my_hedge") \
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

    def test_lambda(self) -> None:
        HedgeAssert(self, fl.HedgeLambda("my_hedge", lambda x: (
            2.0 * x * x if x <= 0.5 else (1.0 - 2.0 * (1.0 - x) * (1.0 - x))))) \
            .has_name("my_hedge") \
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

        if __name__ == '__main__':
            unittest.main()
