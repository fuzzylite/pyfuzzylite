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
import unittest

import fuzzylite as fl


class TestOperation(unittest.TestCase):

    def test_valid_identifier(self) -> None:
        self.assertEqual(fl.Op.as_identifier("  xx  "), "xx")  # trims
        self.assertEqual(fl.Op.as_identifier("   ~!@#$%^&*()+{}[]:;\"'<>?/,   "), "unnamed")
        self.assertEqual(fl.Op.as_identifier("abc123_.ABC"), "abc123_.ABC")
        self.assertEqual(fl.Op.as_identifier("      "), "unnamed")

    def test_str(self) -> None:
        fl.lib.decimals = 3
        self.assertEqual(fl.Op.str(0.3), "0.300")
        self.assertEqual(fl.Op.str(-0.3), "-0.300")
        self.assertEqual(fl.Op.str(3), "3")
        self.assertEqual(fl.Op.str(3.0001), "3.000")

        self.assertEqual(fl.Op.str(math.inf), "inf")
        self.assertEqual(fl.Op.str(-math.inf), "-inf")
        self.assertEqual(fl.Op.str(math.nan), "nan")

        fl.lib.decimals = 5
        self.assertEqual(fl.Op.str(0.3), "0.30000")

        fl.lib.decimals = 0
        self.assertEqual(fl.Op.str(0.3), "0")

        fl.lib.decimals = 3

    def test_scale(self) -> None:
        self.assertEqual(fl.Op.scale(0, 0, 1, -10, 10), -10.0)
        self.assertEqual(fl.Op.scale(.5, 0, 1, -10, 10), 0.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, -10, 10), 10)

        self.assertEqual(fl.Op.scale(0, 0, 1, 0, 10), 0.0)
        self.assertEqual(fl.Op.scale(.5, 0, 1, 0, 10), 5.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, 0, 10), 10)

        self.assertEqual(fl.Op.scale(-1, 0, 1, 0, 10), -10.0)
        self.assertEqual(fl.Op.scale(2, 0, 1, 0, 10), 20)

        self.assertEqual(math.isnan(fl.Op.scale(math.nan, 0, 1, 0, 10)), True)
        self.assertEqual(fl.Op.scale(math.inf, 0, 1, 0, 10), math.inf)
        self.assertEqual(fl.Op.scale(-math.inf, 0, 1, 0, 10), -math.inf)

    @unittest.skip("Revisit describe() method")
    def test_describe(self) -> None:
        self.assertEqual("OutputVariable[{"
                         "'default_value': 'nan', 'defuzzifier': 'None', "
                         "'fuzzy': 'term: x Aggregated []', "
                         "'lock_previous': 'False', 'previous_value': 'nan'"
                         "}]",
                         fl.Op.describe(
                             fl.OutputVariable("x", "an x", terms=[fl.Triangle("t")])))
        self.assertEqual("InputVariable[{}]",
                         fl.Op.describe(
                             fl.InputVariable("x", "an x", terms=[fl.Triangle("t")])))


if __name__ == '__main__':
    unittest.main()
