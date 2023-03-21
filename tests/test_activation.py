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
from typing import Dict, Tuple
from unittest.mock import MagicMock

import fuzzylite as fl
from tests.assert_component import BaseAssert


class ActivationAssert(BaseAssert[fl.Activation]):
    """Activation Assert."""

    def fuzzy_values(self, fuzzification: Dict[float, str]) -> "ActivationAssert":
        """Not implemented."""
        raise NotImplementedError()
        # for x in fuzzification:
        #     # self.test.assertEqual(self.actual.fuzzify(x), fuzzification[x], f"when x={x}")
        #     pass
        # return self

    def highest_memberships(
        self, x_mf: Dict[float, Tuple[float, fl.Term]]
    ) -> "ActivationAssert":
        """Not implemented."""
        raise NotImplementedError()
        for x in x_mf:
            # self.test.assertEqual(self.actual.highest_membership(x), x_mf[x], f"when x={x}")
            pass
        return self


class TestActivation(unittest.TestCase):
    """Tests the base activation class."""

    def test_class_name(self) -> None:
        """Asserts the class name is correct."""
        self.assertEqual("Activation", fl.Activation().class_name)

    def test_activation(self) -> None:
        """Asserts that activate method is not implemented."""
        with self.assertRaises(NotImplementedError):
            fl.Activation().activate(fl.RuleBlock())

    def test_parameters(self) -> None:
        """Asserts parameters are empty."""
        self.assertEqual("", fl.Activation().parameters())

    def test_str(self) -> None:
        """Asserts the base exporting to string is correct."""
        self.assertEqual("Activation", str(fl.Activation()))

        activation = fl.Activation()
        activation.parameters = MagicMock(return_value="param1 param2")  # type: ignore
        self.assertEqual("Activation param1 param2", str(activation))


if __name__ == "__main__":
    unittest.main()
