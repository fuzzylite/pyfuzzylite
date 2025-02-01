"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any, TypeVar
from unittest.mock import MagicMock

import math
import numpy as np

import fuzzylite as fl
from fuzzylite.types import Self
from tests.assert_component import BaseAssert

T_Variable = TypeVar("T_Variable", bound=fl.Variable)


class VariableAssert(BaseAssert[T_Variable]):
    """Variable assert."""

    def fuzzy_values(self, fuzzification: dict[float, str]) -> Self:
        """Test the fuzzification of the given keys result in their expected values."""
        for x in fuzzification:
            self.test.assertEqual(fuzzification[x], str(self.actual.fuzzify(x)), f"when x={x}")

        # test vectorization
        values = fl.array(list(fuzzification.keys()))
        expected = fl.array(list(fuzzification.values()), dtype=np.str_)
        obtained = self.actual.fuzzify(values)
        np.testing.assert_equal(expected, obtained)
        return self

    def highest_memberships(self, x_mf: dict[float, fl.Activated | None]) -> Self:
        """Test the highest memberships for the given keys result in the expected activation values and terms."""
        for x in x_mf:
            obtained = self.actual.highest_membership(x)
            expected = x_mf[x]
            if expected is None:
                self.test.assertIsNone(obtained, msg=f"when x={x}")
            else:
                self.test.assertEqual(str(expected), str(obtained), f"when x={x}")
        return self


class InputVariableAssert(VariableAssert[fl.InputVariable]):
    """Input variable assert."""


class OutputVariableAssert(VariableAssert[fl.OutputVariable]):
    """Output variable assert."""

    def clear(self) -> Self:
        """Clear the output variable."""
        self.actual.clear()
        return self

    def when_fuzzy_output(self, *, is_empty: bool, implication: fl.TNorm | None = None) -> Self:
        """Set the output variable to the given terms."""
        if is_empty:
            self.actual.fuzzy.terms = []
        else:
            self.actual.fuzzy.terms = [
                fl.Activated(term, degree=1.0, implication=implication)
                for term in self.actual.terms
            ]
        return self

    def defuzzify(self, raises: Exception | None = None) -> Self:
        """Defuzzify the output variable."""
        if raises:
            with self.test.assertRaises(type(raises)) as error:
                self.actual.defuzzify()
            self.test.assertEqual(str(error.exception), str(raises))
        else:
            self.actual.defuzzify()
        return self

    def then_fuzzy_value_is(self, value: str) -> Self:
        """Assert the fuzzy value of the output variable."""
        self.test.assertEqual(value, self.actual.fuzzy_value().item())
        return self

    def activated_values(self, fuzzification: dict[Sequence[fl.Activated], str]) -> Self:
        """Assert the list of activated terms results in the expected fuzzy value."""
        for x in fuzzification:
            self.actual.fuzzy.terms.clear()
            self.actual.fuzzy.terms.extend(x)
            self.test.assertEqual(fuzzification[x], self.actual.fuzzy_value().item(), f"when x={x}")
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

    @unittest.skip(reason="__getattr__ has not been implemented properly yet.")
    def test_getattr(self) -> None:
        """Test accessing term with getattr and getitem."""
        a, b = [fl.Constant("a", 0), fl.Constant("b", 1)]
        variable = fl.Variable("test", terms=[a, b])

        # test getattr
        self.assertEqual(variable.a, a)  # type:ignore
        self.assertEqual(variable.b, b)  # type:ignore

    def test_getitem_len_iter(self) -> None:
        """Test accessing term with getattr and getitem."""
        a, b = [fl.Constant("a", 0), fl.Constant("b", 1)]
        variable = fl.Variable("test", terms=[a, b])

        # test getitem
        self.assertEqual(variable["a"], a)
        self.assertEqual(variable["b"], b)

        # side-effect
        self.assertEqual(variable[0], a)
        self.assertEqual(variable[1], b)

        self.assertEqual(0, len(fl.Variable()))
        self.assertEqual(2, len(variable))

        # iterating over terms
        ## via list comprehension
        self.assertListEqual([t for t in variable], variable.terms)

        ## via enumeration
        terms = [a, b]
        for index, term in enumerate(variable):
            self.assertEqual(terms[index], term)

        ## via iterators
        iterator = iter(variable)
        self.assertEqual(next(iterator), a)
        self.assertEqual(next(iterator), b)

    def test_deepcopy(self) -> None:
        """Test the variable can be deeply copied."""
        import copy

        variable = fl.Variable("name", "description", terms=[fl.Triangle("a"), fl.Triangle("b")])
        variable_copy = copy.deepcopy(variable)

        self.assertNotEqual(variable, variable_copy)
        self.assertEqual(repr(variable), repr(variable_copy))

        variable_copy["a"].name = "z"
        self.assertEqual(variable_copy.term(0).name, "z")
        self.assertEqual(variable.term(0).name, "a")

    def test_term_by_index_or_name(self) -> None:
        """Test finding term by index or name."""
        a, b = [fl.Constant("a", 0), fl.Constant("b", 1)]
        variable = fl.Variable("test", terms=[a, b])

        self.assertEqual(variable.term(0), a)
        self.assertEqual(variable.term(1), b)
        with self.assertRaises(IndexError) as index_error:
            variable.term(2)
        self.assertEqual("list index out of range", str(index_error.exception))

        self.assertEqual(variable.term("a"), a)
        self.assertEqual(variable.term("b"), b)
        with self.assertRaises(ValueError) as value_error:
            variable.term("z")
        self.assertEqual("term 'z' not found in ['a', 'b']", str(value_error.exception))

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
                math.nan: "0.000/Low + 0.000/Medium + 0.000/High",
                math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
            }
        )

        VariableAssert(
            self,
            fl.Variable(
                terms=[
                    fl.Constant("A", -1),
                    fl.Constant("B", 0),
                    fl.Constant("C", 1),
                    fl.Constant("D", -fl.inf),
                    fl.Constant("E", fl.inf),
                    fl.Constant("F", fl.nan),
                ]
            ),
        ).fuzzy_values({fl.nan: "-1.000/A + 0.000/B + 1.000/C + 0.000/D + 1.000/E + 0.000/F"})

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
                -1.00: None,
                -0.75: fl.Activated(low, 0.5),
                -0.50: fl.Activated(low, 1.0),
                -0.25: fl.Activated(low, 0.5),
                0.00: fl.Activated(medium, 1.0),
                0.25: fl.Activated(medium, 0.5),
                0.50: fl.Activated(high, 1.0),
                0.75: fl.Activated(high, 0.5),
                1.00: None,
                math.nan: None,
                math.inf: None,
                -math.inf: None,
            }
        )


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
                math.nan: "0.000/Low + 0.000/Medium + 0.000/High",
                math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
            }
        )


class TestOutputVariable(unittest.TestCase):
    """Test the output variable."""

    def output_variable(
        self,
        enabled: bool = True,
        name: str = "name",
        description: str = "description",
        minimum: float = -1.0,
        maximum: float = 1.0,
        default_value: float = fl.nan,
        terms: list[fl.Term] | None = None,
    ) -> fl.OutputVariable:
        """Create an output variable."""
        return fl.OutputVariable(
            enabled=enabled,
            name=name,
            description=description,
            minimum=minimum,
            maximum=maximum,
            default_value=default_value,
            terms=(
                terms
                if terms is not None
                else [
                    fl.Triangle("low", -1.0, -1.0, 0.0),
                    fl.Triangle("medium", -0.5, 0.0, 0.5),
                    fl.Triangle("high", 0.0, 1.0, 1.0),
                ]
            ),
        )

    def test_constructor(self) -> None:
        """Test the constructor."""
        OutputVariableAssert(self, fl.OutputVariable("name", "description")).exports_fll(
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
                tuple([fl.Activated(low, 0.5)]): "0.500/Low + 0.000/Medium + 0.000/High",
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
        variable.fuzzy.terms.extend([fl.Activated(term, 0.5) for term in variable.terms])
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
        # When the output variable is disabled
        # Then nothing changes on defuzzify
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=False,
            value=1.0,
            previous_value=fl.nan,
        ).defuzzify().then(
            enabled=False,
            value=1.0,
            previous_value=fl.nan,
        )

        # When the output variable is enabled
        #   And the defuzzifier is None
        # Then an error is raised
        #   And nothing changes on defuzzify
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            value=1.0,
            previous_value=fl.nan,
            defuzzifier=None,
            default_value=0.123,
        ).defuzzify(
            raises=ValueError("expected a defuzzifier in output variable 'Output', but found None")
        ).then(
            value=1.0,
            previous_value=fl.nan,
        )

        # When the output variable is enabled
        #   And there is nothing to defuzzify
        #   And there is a valid defuzzifier
        # Then the default value is used
        #   And the previous value is stored
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            value=1.0,
            previous_value=fl.nan,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=0.123,
            previous_value=1.0,
        )

        # When the output variable is enabled
        #   And there is nothing to defuzzify
        #   And there is a valid defuzzifier
        #   And it is locking the previous value
        # Then the previous value is used
        #   And the previous value is updated
        #   And the default value is not used
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=2.0,
            previous_value=-1.0,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=2.0,
            previous_value=2.0,
        )

        # When the output variable is enabled
        #   And there is nothing to defuzzify
        #   And there is a valid defuzzifier
        #   And it is locking the previous value
        #   And the previous value is invalid
        # Then the default value is used
        #   And the previous value is updated
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=fl.nan,
            previous_value=0.5,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=0.123,
            previous_value=fl.nan,
        )

        # When the defuzzifier raises an exception
        #   and there are terms to defuzzify
        # then nothing changes on defuzzify
        defuzzifier_exception = fl.Centroid()
        defuzzifier_exception.defuzzify = MagicMock(  # type: ignore
            side_effect=ValueError("exception during defuzzification")
        )
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            defuzzifier=defuzzifier_exception,
            value=1.0,
            previous_value=fl.nan,
            default_value=0.123,
        ).when_fuzzy_output(is_empty=False).then_fuzzy_value_is(
            "1.000/low + 1.000/medium + 1.000/high"
        ).defuzzify(
            raises=ValueError("exception during defuzzification")
        ).then(
            value=1.0,
            previous_value=fl.nan,
        ).then_fuzzy_value_is(
            "1.000/low + 1.000/medium + 1.000/high"
        )

    def test_defuzzification_arrays(self) -> None:
        """Test the defuzzification of the output variable with multiple values."""
        # When the output variable is disabled
        # Then nothing changes on defuzzify
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=False,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
        ).defuzzify().then(
            enabled=False,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
        )

        # When the output variable is enabled
        #   And the defuzzifier is None
        # Then an error is raised
        #   And nothing changes on defuzzify
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
            defuzzifier=None,
            default_value=0.123,
        ).defuzzify(
            raises=ValueError("expected a defuzzifier in output variable 'Output', but found None")
        ).then(
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
        )

        # When the defuzzifier raises an exception
        #   and there are terms to defuzzify
        # then nothing changes on defuzzify
        defuzzifier_exception = fl.Centroid()
        defuzzifier_exception.defuzzify = MagicMock(  # type: ignore
            side_effect=ValueError("exception during defuzzification")
        )
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            name="Output",
            defuzzifier=defuzzifier_exception,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
            default_value=0.123,
        ).when_fuzzy_output(is_empty=False).then_fuzzy_value_is(
            "1.000/low + 1.000/medium + 1.000/high"
        ).defuzzify(
            raises=ValueError("exception during defuzzification")
        ).then(
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
        ).then_fuzzy_value_is(
            "1.000/low + 1.000/medium + 1.000/high"
        )

        # When the output variable is enabled
        #   And there is nothing to defuzzify
        #   And there is a valid defuzzifier
        #   And it is not locking the previous value
        # Then the default value is used
        #   And the previous value is updated to the last value
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=0.123,
            previous_value=3.0,
        )

        # When the fuzzy output is empty
        # Cases:
        # default_value = nan | 0.123
        # previous_value = nan | 1.0

        # (a) when default value is nan and previous value is nan
        #     then value is nan and previous value is nan
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=fl.array([1.0, 2.0, fl.nan]),
            previous_value=-1.0,
            defuzzifier=fl.Centroid(),
            default_value=fl.nan,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=fl.nan,
            previous_value=fl.nan,
        )

        # (b) when default value is nan and previous value is not nan
        #     then previous value is updated and value is previous_value
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=fl.array([1.0, 2.0, fl.inf]),
            previous_value=-1.0,
            defuzzifier=fl.Centroid(),
            default_value=fl.nan,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=fl.inf,
            previous_value=fl.inf,
        )

        # (c) when default value is not nan and previous value is nan
        #     then previous value is updated and value is default_value
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=fl.array([1.0, 2.0, fl.nan]),
            previous_value=-1.0,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=0.123,
            previous_value=fl.nan,
        )

        # (d) when default value is not nan and previous value is not nan
        #     then previous value is updated and value is previous value
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            lock_previous=True,
            value=fl.array([1.0, 2.0, 4.0]),
            previous_value=-1.0,
            defuzzifier=fl.Centroid(),
            default_value=0.123,
        ).when_fuzzy_output(is_empty=True).defuzzify().then(
            value=4.0,
            previous_value=4.0,
        )

        def mock_defuzzify(*_: Any, **__: Any) -> fl.Scalar:
            return fl.array([np.nan, 1.0, np.nan, 2.0, np.nan, 3.0, -np.inf, np.inf, 0.0])

        # When the fuzzy output is not empty
        mock_defuzzifier = fl.Centroid()
        mock_defuzzifier.defuzzify = MagicMock(side_effect=mock_defuzzify)  # type: ignore
        # And locking the previous value is not enabled
        # And default value is nan
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            defuzzifier=mock_defuzzifier,
            lock_previous=False,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
            default_value=fl.nan,
        ).defuzzify().then(
            value=fl.array([np.nan, 1.0, np.nan, 2.0, np.nan, 3.0, -np.inf, np.inf, 0.0]),
            previous_value=3.0,
        )

        # And locking the previous value is not enabled
        # And default value is not nan
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            defuzzifier=mock_defuzzifier,
            lock_previous=False,
            value=fl.array([1.0, 2.0, 3.0]),
            previous_value=fl.nan,
            default_value=0.123,
        ).defuzzify().then(
            value=fl.array([0.123, 1.0, 0.123, 2.0, 0.123, 3.0, -np.inf, np.inf, 0.0]),
            previous_value=3.0,
        )

        # And locking the previous value is enabled
        # And previous_value is nan
        # And default value is nan
        OutputVariableAssert(self, self.output_variable()).when(
            enabled=True,
            defuzzifier=mock_defuzzifier,
            lock_previous=True,
            value=fl.array([1.0, 2.0, fl.nan]),
            previous_value=fl.nan,
            default_value=fl.nan,
        ).when_fuzzy_output(is_empty=False).defuzzify().then(
            value=fl.array([np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, -np.inf, np.inf, 0.0]),
            previous_value=fl.nan,
        )


if __name__ == "__main__":
    unittest.main()
