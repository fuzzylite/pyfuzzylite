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
from fuzzylite.operation import Operation as Op


class TestOperation(unittest.TestCase):
    def test_validName(self):
        self.assertEqual(Op.valid_name("  xx  "), "xx")  # trims
        self.assertEqual(Op.valid_name("   ~!@#$%^&*()+{}[]:;\"'<>?/,   "), "unnamed")
        self.assertEqual(Op.valid_name("abc123_.ABC"), "abc123_.ABC")
        self.assertEqual(Op.valid_name("      "), "unnamed")

    def test_str(self):
        fl.DECIMALS = 3
        self.assertEqual(Op.str(0.3), "0.300")
        self.assertEqual(Op.str(-0.3), "-0.300")
        self.assertEqual(Op.str(3), "3")
        self.assertEqual(Op.str(3.0001), "3.000")

        self.assertEqual(Op.str(math.inf), "inf")
        self.assertEqual(Op.str(-math.inf), "-inf")
        self.assertEqual(Op.str(math.nan), "nan")

        fl.DECIMALS = 5
        self.assertEqual(Op.str(0.3), "0.30000")

        fl.DECIMALS = 0
        self.assertEqual(Op.str(0.3), "0")

        fl.DECIMALS = 3

    def test_scale(self):
        self.assertEqual(Op.scale(0, 0, 1, -10, 10), -10.0)
        self.assertEqual(Op.scale(.5, 0, 1, -10, 10), 0.0)
        self.assertEqual(Op.scale(1, 0, 1, -10, 10), 10)

        self.assertEqual(Op.scale(0, 0, 1, 0, 10), 0.0)
        self.assertEqual(Op.scale(.5, 0, 1, 0, 10), 5.0)
        self.assertEqual(Op.scale(1, 0, 1, 0, 10), 10)

        self.assertEqual(Op.scale(-1, 0, 1, 0, 10), -10.0)
        self.assertEqual(Op.scale(2, 0, 1, 0, 10), 20)

        self.assertEqual(math.isnan(Op.scale(math.nan, 0, 1, 0, 10)), True)
        self.assertEqual(Op.scale(math.inf, 0, 1, 0, 10), math.inf)
        self.assertEqual(Op.scale(-math.inf, 0, 1, 0, 10), -math.inf)


if __name__ == '__main__':
    unittest.main()
