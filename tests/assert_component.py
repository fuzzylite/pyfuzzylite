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
from typing import Any, Generic, TypeVar

import fuzzylite as fl

T = TypeVar('T')


class BaseAssert(Generic[T]):

    def __init__(self, test: unittest.TestCase, actual: T) -> None:
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def exports_fll(self, fll: str) -> Any:
        self.test.assertEqual(fll, fl.FllExporter().to_string(self.actual))
        self.test.assertEqual(fll, str(self.actual))
        return self

    def has_name(self, name: str) -> Any:
        self.test.assertTrue(hasattr(self.actual, "name"),
                             f"{type(self.actual)} does not have a 'name' attribute")
        self.test.assertEqual(getattr(self.actual, "name"), name)
        return self

    def has_description(self, description: str) -> Any:
        self.test.assertTrue(hasattr(self.actual, "description"),
                             f"{type(self.actual)} does not have a 'description' attribute")
        self.test.assertEqual(getattr(self.actual, "description"), description)
        return self
