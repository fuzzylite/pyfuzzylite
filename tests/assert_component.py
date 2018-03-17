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

from fuzzylite.exporter import FllExporter


class ComponentAssert(object):
    def __init__(self, test: unittest.TestCase, actual: object):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def has_name(self, name: str):
        self.test.assertEqual(self.actual.name, name)
        return self

    def has_description(self, description: str):
        self.test.assertEqual(self.actual.description, description)
        return self

    def exports_fll(self, fll: str):
        self.test.assertEqual(FllExporter().to_string(self.actual), fll)
        self.test.assertEqual(str(self.actual), fll)
        return self
