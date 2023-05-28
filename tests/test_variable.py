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
from __future__ import annotations

import math
import unittest
from collections.abc import Sequence
from unittest.mock import MagicMock

import fuzzylite as fl
from tests.assert_component import BaseAssert


class VariableAssert(BaseAssert[fl.Variable]):
    """Variable assert."""

    def fuzzy_values(self, fuzzification: dict[float, str]) -> VariableAssert:
        """Test the fuzzification of the given keys result in their expected values."""
        for x in fuzzification:
            self.test.assertEqual(
                self.actual.fuzzify(x), fuzzification[x], f"when x={x}"
            )
        return self

    def highest_memberships(
        self, x_mf: dict[float, tuple[float, fl.Term | None]]
    ) -> VariableAssert:
        """Test the highest memberships for the given keys result in the expected activation values and terms."""
        for x in x_mf:
            self.test.assertEqual(
                self.actual.highest_membership(x), x_mf[x], f"when x={x}"
            )
        return self


class TestVariable(unittest.TestCase):
    """Test variables."""

    def test_constructor(self) -> None:
        """Test the base constructor."""
        VariableAssert(self, fl.Variable("name", "description")).exports_fll(
            "\n".join(
                [
                    "Variable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -inf inf",
                    "  lock-range: false",
                ]
            )
        )
        VariableAssert(
            self,
            fl.Variable(
                name="name",
                description="description",
                minimum=-1.0,
                maximum=1.0,
                terms=[fl.Triangle("A", -1.0, 1.0), fl.Triangle("B", -10.0, 10.0)],
            ),
        ).exports_fll(
            "\n".join(
                [
                    "Variable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -1.000 1.000",
                    "  lock-range: false",
                    "  term: A Triangle -1.000 0.000 1.000",
                    "  term: B Triangle -10.000 0.000 10.000",
                ]
            )
        )

    def test_lock_range(self) -> None:
        """Test the lock range."""
        variable = fl.Variable("name", "description")
        variable.range = (-1.0, 1.0)

        variable.lock_range = False
        variable.value = -10.0
        self.assertEqual(variable.value, -10.0)
        variable.value = 10.0
        self.assertEqual(variable.value, 10.0)

        minimum, maximum = variable.range
        self.assertEqual(minimum, -1.0)
        self.assertEqual(maximum, 1.0)

        variable.lock_range = True
        variable.value = -10.0
        self.assertEqual(variable.value, -1.0)
        variable.value = 10.0
        self.assertEqual(variable.value, 1.0)

    def test_fuzzify(self) -> None:
        """Test the fuzzification of values."""
        VariableAssert(
            self,
            fl.Variable(
                name="name",
                description="description",
                minimum=-1.0,
                maximum=1.0,
                terms=[
                    fl.Triangle("Low", -1.0, -1.0, 0.0),
                    fl.Triangle("Medium", -0.5, 0.0, 0.5),
                    fl.Triangle("High", 0.0, 1.0, 1.0),
                ],
            ),
        ).fuzzy_values(
            {
                -1.00: "1.000/Low + 0.000/Medium + 0.000/High",
                -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
                -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
                0.00: "0.000/Low + 1.000/Medium + 0.000/High",
                0.25: "0.000/Low + 0.500/Medium + 0.250/High",
                0.50: "0.000/Low + 0.000/Medium + 0.500/High",
                0.75: "0.000/Low + 0.000/Medium + 0.750/High",
                1.00: "0.000/Low + 0.000/Medium + 1.000/High",
                math.nan: "nan/Low + nan/Medium + nan/High",
                math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
            }
        )

    def test_highest_membership(self) -> None:
        """Test the computation of highest memberships."""
        low, medium, high = (
            fl.Triangle("Low", -1.0, -0.5, 0.0),
            fl.Triangle("Medium", -0.5, 0.0, 0.5),
            fl.Triangle("High", 0.0, 0.5, 1.0),
        )
        VariableAssert(
            self,
            fl.Variable(
                name="name",
                description="description",
                minimum=-1.0,
                maximum=1.0,
                terms=[low, medium, high],
            ),
        ).highest_memberships(
            {
                -1.00: (0.0, None),
                -0.75: (0.5, low),
                -0.50: (1.0, low),
                -0.25: (0.5, low),
                0.00: (1.0, medium),
                0.25: (0.5, medium),
                0.50: (1.0, high),
                0.75: (0.5, high),
                1.00: (0.0, None),
                math.nan: (0.0, None),
                math.inf: (0.0, None),
                -math.inf: (0.0, None),
            }
        )


class InputVariableAssert(BaseAssert[fl.InputVariable]):
    """Input variable assert."""

    def exports_fll(self, fll: str) -> InputVariableAssert:
        """Asserts exporting the input variable to FLL yields the expected FLL."""
        self.test.assertEqual(fl.FllExporter().input_variable(self.actual), fll)
        return self

    def fuzzy_values(self, fuzzification: dict[float, str]) -> InputVariableAssert:
        """Assert the fuzzification of the given keys result in their expected fuzzy values."""
        for x in fuzzification:
            self.actual.value = x
            self.test.assertEqual(
                self.actual.fuzzy_value(), fuzzification[x], f"when x={x}"
            )
        return self


class TestInputVariable(unittest.TestCase):
    """Test input variable."""

    def test_constructor(self) -> None:
        """Test constructor."""
        InputVariableAssert(self, fl.InputVariable("name", "description")).exports_fll(
            "\n".join(
                [
                    "InputVariable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -inf inf",
                    "  lock-range: false",
                ]
            )
        )
        InputVariableAssert(
            self,
            fl.InputVariable(
                name="name",
                description="description",
                minimum=-1.0,
                maximum=1.0,
                terms=[fl.Triangle("A", -1.0, 1.0), fl.Triangle("B", -10.0, 10.0)],
            ),
        ).exports_fll(
            "\n".join(
                [
                    "InputVariable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -1.000 1.000",
                    "  lock-range: false",
                    "  term: A Triangle -1.000 0.000 1.000",
                    "  term: B Triangle -10.000 0.000 10.000",
                ]
            )
        )

    def test_fuzzy_value(self) -> None:
        """Test fuzzy values."""
        InputVariableAssert(
            self,
            fl.InputVariable(
                name="name",
                description="description",
                minimum=-1.0,
                maximum=1.0,
                terms=[
                    fl.Triangle("Low", -1.0, -1.0, 0.0),
                    fl.Triangle("Medium", -0.5, 0.0, 0.5),
                    fl.Triangle("High", 0.0, 1.0, 1.0),
                ],
            ),
        ).fuzzy_values(
            {
                -1.00: "1.000/Low + 0.000/Medium + 0.000/High",
                -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
                -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
                0.00: "0.000/Low + 1.000/Medium + 0.000/High",
                0.25: "0.000/Low + 0.500/Medium + 0.250/High",
                0.50: "0.000/Low + 0.000/Medium + 0.500/High",
                0.75: "0.000/Low + 0.000/Medium + 0.750/High",
                1.00: "0.000/Low + 0.000/Medium + 1.000/High",
                math.nan: "nan/Low + nan/Medium + nan/High",
                math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
            }
        )


class OutputVariableAssert(BaseAssert[fl.OutputVariable]):
    """Output variable assert."""

    def clear(self) -> OutputVariableAssert:
        """Clear the output variable."""
        self.actual.clear()
        return self

    def when_fuzzy_output(
        self, *, is_empty: bool, implication: fl.TNorm | None = None
    ) -> OutputVariableAssert:
        """Set the output variable to the given terms."""
        if is_empty:
            self.actual.fuzzy.terms = []
        else:
            self.actual.fuzzy.terms = [
                fl.Activated(term, degree=1.0, implication=implication)
                for term in self.actual.terms
            ]
        return self

    def defuzzify(self, raises: Exception | None = None) -> OutputVariableAssert:
        """Defuzzify the output variable."""
        if raises:
            with self.test.assertRaises(type(raises)) as error:
                self.actual.defuzzify()
            self.test.assertEqual(str(error.exception), str(raises))
        else:
            self.actual.defuzzify()
        return self

    def then_fuzzy_value_is(self, value: str | list[str]) -> OutputVariableAssert:
        """Assert the fuzzy value of the output variable."""
        self.test.assertEqual(self.actual.fuzzy_value(), value)
        return self

    def exports_fll(self, fll: str) -> OutputVariableAssert:
        """Assert exporting the output variable results in the expected FLL."""
        self.test.assertEqual(fl.FllExporter().output_variable(self.actual), fll)
        return self

    def activated_values(
        self, fuzzification: dict[Sequence[fl.Activated], str]
    ) -> OutputVariableAssert:
        """Assert the list of activated terms results in the expected fuzzy value."""
        for x in fuzzification:
            self.actual.fuzzy.terms.clear()
            self.actual.fuzzy.terms.extend(x)
            self.test.assertEqual(
                self.actual.fuzzy_value(), fuzzification[x], f"when x={x}"
            )
        return self


class TestOutputVariable(unittest.TestCase):
    """Test the output variable."""

    def output_variable(
        self,
        enabled: bool = True,
        name: str = "name",
        description: str = "description",
        minimum: float = -1.0,
        maximum: float = 1.0,
        terms: list[fl.Term] | None = None,
    ) -> fl.OutputVariable:
        """Create an output variable."""
        return fl.OutputVariable(
            enabled=enabled,
            name=name,
            description=description,
            minimum=minimum,
            maximum=maximum,
            terms=terms
            if terms is not None
            else [
                fl.Triangle("low", -1.0, -1.0, 0.0),
                fl.Triangle("medium", -0.5, 0.0, 0.5),
                fl.Triangle("high", 0.0, 1.0, 1.0),
            ],
        )

    def test_constructor(self) -> None:
        """Test the constructor."""
        OutputVariableAssert(
            self, fl.OutputVariable("name", "description")
        ).exports_fll(
            "\n".join(
                [
                    "OutputVariable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -inf inf",
                    "  lock-range: false",
                    "  aggregation: none",
                    "  defuzzifier: none",
                    "  default: nan",
                    "  lock-previous: false",
                ]
            )
        )
        OutputVariableAssert(
            self,
            self.output_variable(
                terms=[fl.Triangle("A", -1.0, 1.0), fl.Triangle("B", -10.0, 10.0)],
            ),
        ).exports_fll(
            "\n".join(
                [
                    "OutputVariable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -1.000 1.000",
                    "  lock-range: false",
                    "  aggregation: none",
                    "  defuzzifier: none",
                    "  default: nan",
                    "  lock-previous: false",
                    "  term: A Triangle -1.000 0.000 1.000",
                    "  term: B Triangle -10.000 0.000 10.000",
                ]
            )
        )

    def test_fuzzy_value(self) -> None:
        """Test the fuzzy values."""
        low, medium, high = [
            fl.Triangle("Low", -1.0, -1.0, 0.0),
            fl.Triangle("Medium", -0.5, 0.0, 0.5),
            fl.Triangle("High", 0.0, 1.0, 1.0),
        ]
        OutputVariableAssert(
            self,
            self.output_variable(
                terms=[low, medium, high],
            ),
        ).activated_values(
            {
                tuple(): "0.000/Low + 0.000/Medium + 0.000/High",
                tuple(
                    [fl.Activated(low, 0.5)]
                ): "0.500/Low + 0.000/Medium + 0.000/High",
                tuple(
                    [
                        fl.Activated(low, -1.0),
                        fl.Activated(medium, -0.5),
                        fl.Activated(high, -0.1),
                    ]
                ): "-1.000/Low - 0.500/Medium - 0.100/High",
            }
        )

    def test_clear(self) -> None:
        """Test the output variable can be cleared."""
        low, medium, high = [
            fl.Triangle("Low", -1.0, -1.0, 0.0),
            fl.Triangle("Medium", -0.5, 0.0, 0.5),
            fl.Triangle("High", 0.0, 1.0, 1.0),
        ]
        variable = self.output_variable(
            terms=[low, medium, high],
        )
        variable.value = 0.0
        variable.previous_value = -1.0
        variable.fuzzy.terms.extend(
            [fl.Activated(term, 0.5) for term in variable.terms]
        )
        OutputVariableAssert(self, variable).exports_fll(
            "\n".join(
                [
                    "OutputVariable: name",
                    "  description: description",
                    "  enabled: true",
                    "  range: -1.000 1.000",
                    "  lock-range: false",
                    "  aggregation: none",
                    "  defuzzifier: none",
                    "  default: nan",
                    "  lock-previous: false",
                    "  term: Low Triangle -1.000 -1.000 0.000",
                    "  term: Medium Triangle -0.500 0.000 0.500",
                    "  term: High Triangle 0.000 1.000 1.000",
                ]
            )
        )

        self.assertEqual(variable.value, 0.0)
        self.assertEqual(variable.previous_value, -1.0)
        self.assertSequenceEqual(
            [term.parameters() for term in variable.fuzzy.terms],
            ["(0.500*Low)", "(0.500*Medium)", "(0.500*High)"],
        )
        variable.clear()
        self.assertEqual(math.isnan(variable.value), True)
        self.assertEqual(math.isnan(variable.previous_value), True)
        self.assertSequenceEqual(variable.fuzzy.terms, [])

    def test_defuzzification(self) -> None:
        """Test the defuzzification of the output variable with multiple values."""
        # Test that nothing changes when the output variable is disabled
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=False,
            value=1.0,
            previous_value=fl.nan,
        ).defuzzify().then(
            enabled=False,
            value=1.0,
            previous_value=fl.nan,
        )

        # Test the default value and previous value are used when the output variable is enabled
        # but there is nothing to defuzzify
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            value=1.0,
            previous_value=fl.nan,
            defuzzifier=None,
            default_value=0.123,
        ).defuzzify().then(
            value=0.123,
            previous_value=1.0,
        )

        # Test the previous value is used when locking the previous value is enabled and the output value is invalid
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            value=fl.nan,
            previous_value=1.0,
            lock_previous=True,
            default_value=0.123,
        ).defuzzify().then(
            value=1.0,
            previous_value=1.0,
        )

        # When the default value is used when the previous value is invalid
        # and locking the previous value is enabled
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            value=fl.nan,
            previous_value=fl.nan,
            lock_previous=True,
            default_value=0.123,
        ).defuzzify().then(
            value=0.123,
            previous_value=fl.nan,
        )

        # When the defuzzifier is None and there are terms to defuzzify
        # then the default value is used and previous value is stored
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            defuzzifier=None,
            value=1.0,
            previous_value=fl.nan,
            default_value=0.123,
        ).when_fuzzy_output(is_empty=False).defuzzify(
            raises=ValueError(
                "expected a defuzzifier in output variable 'Output', but found None"
            )
        ).then(
            value=0.123,
            previous_value=1.0,
        )

        # When the defuzzifier raises an exception and there are terms to defuzzify
        # then the default value is used and previous value is stored
        defuzzifier = fl.Defuzzifier()
        defuzzifier.defuzzify = MagicMock(  # type: ignore
            side_effect=ValueError("mocking exception during defuzzification")
        )
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            defuzzifier=defuzzifier,
            value=1.0,
            previous_value=fl.nan,
            default_value=0.123,
        ).when_fuzzy_output(is_empty=False).defuzzify(
            raises=ValueError("mocking exception during defuzzification")
        ).then(
            value=0.123,
            previous_value=1.0,
        )


if __name__ == "__main__":
    unittest.main()
