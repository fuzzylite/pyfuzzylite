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


class TestFllExporter(unittest.TestCase):
    def test_to_string(self):
        self.assertEqual(FllExporter().term(Term("X", 1.0)), "term: X Term")
        self.assertEqual(FllExporter().rule(Rule("if x then y")), "rule: if x then y")

    def test_variables(self):
        self.assertEqual(FllExporter().input_variable(Variable("name", terms=[Triangle("A"), Trapezoid("B")])),
                         "\n".join(
                                 ["InputVariable: name",
                                  "  description: ",
                                  "  enabled: true",
                                  "  range: -inf inf",
                                  "  lock-range: true",
                                  "  term: A Triangle nan nan nan",
                                  "  term: B Trapezoid nan nan nan nan",
                                  ]))


if __name__ == '__main__':
    unittest.main()
