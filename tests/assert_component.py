"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest
from typing import Any, Generic, TypeVar

import numpy as np

import fuzzylite as fl
from fuzzylite.types import Self

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

    def repr_is(
        self,
        representation: str,
        /,
        with_alias: str | None = None,
        validate: bool = True,
    ) -> Self:
        """Asserts that the obtained object's representation is equal to the expected representation."""

        def run_test() -> None:
            self.test.assertEqual(representation, repr(self.actual))
            if validate:
                self.test.assertEqual(repr(eval(representation)), repr(self.actual))

        if with_alias is None:
            run_test()
        else:
            with fl.settings.context(alias=with_alias):
                run_test()
        return self

    def exports_fll(self, fll: str) -> Self:
        """Asserts that the given fll is equal to the obtained fll.

        @param fll is the expected fll.
        """
        self.test.assertEqual(fll, fl.Op.to_fll(self.actual))
        self.test.assertEqual(fll, str(self.actual))
        return self

    def has_name(self, name: str) -> Self:
        """Asserts that the obtained object's name is equal to the expected name.

        @param name is the expected name.
        """
        self.test.assertTrue(
            hasattr(self.actual, "name"),
            f"{type(self.actual)} does not have a 'name' attribute",
        )
        self.test.assertEqual(self.actual.name, name)  # type: ignore
        return self

    def has_description(self, description: str) -> Self:
        """Asserts that the obtained object's description is equal to the expected description.

        @param description is the expected description.
        """
        self.test.assertTrue(
            hasattr(self.actual, "description"),
            f"{type(self.actual)} does not have a 'description' attribute",
        )
        self.test.assertEqual(self.actual.description, description)  # type: ignore
        return self

    def when(self, **settings: Any) -> Self:
        """Set the settings of the output variable."""
        for key, value in settings.items():
            if hasattr(self.actual, key):
                setattr(self.actual, key, value)
            else:
                raise KeyError(f"OutputVariable has no attribute '{key}'")
        return self

    def then(self, **settings: Any) -> Self:
        """Assert the settings of the output variable."""
        for key, expected in settings.items():
            self.test.assertTrue(hasattr(self.actual, key))
            obtained = getattr(self.actual, key)
            if isinstance(expected, (np.ndarray, float)):
                np.testing.assert_allclose(
                    obtained,
                    expected,
                    atol=fl.settings.atol,
                    rtol=fl.settings.rtol,
                    err_msg=f"in attribute '{key}'",
                )
            else:
                self.test.assertEqual(getattr(self.actual, key), expected)
        return self
