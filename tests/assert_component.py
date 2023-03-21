"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited
fuzzylite is a registered trademark of FuzzyLite Limited.
"""
import unittest
from typing import Any, Generic, TypeVar

import fuzzylite as fl

T = TypeVar("T")


class BaseAssert(Generic[T]):
    """Basic assert of FuzzyLite objects."""

    def __init__(self, test: unittest.TestCase, actual: T) -> None:
        """Create the assert.
        @param test is the test instance
        @param actual is the obtained value
        #TODO rename actual to obtained or others to expected?
        """
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def exports_fll(self, fll: str) -> Any:
        """Asserts that the given fll is equal to the obtained fll.
        @param fll is the expected fll.
        """
        self.test.assertEqual(fll, fl.FllExporter().to_string(self.actual))
        self.test.assertEqual(fll, str(self.actual))
        return self

    def has_name(self, name: str) -> Any:
        """Asserts that the obtained object's name is equal to the expected name.
        @param name is the expected name.
        """
        self.test.assertTrue(
            hasattr(self.actual, "name"),
            f"{type(self.actual)} does not have a 'name' attribute",
        )
        self.test.assertEqual(self.actual.name, name)  # type: ignore
        return self

    def has_description(self, description: str) -> Any:
        """Asserts that the obtained object's description is equal to the expected description.
        @param description is the expected description.
        """
        self.test.assertTrue(
            hasattr(self.actual, "description"),
            f"{type(self.actual)} does not have a 'description' attribute",
        )
        self.test.assertEqual(self.actual.description, description)  # type: ignore
        return self
