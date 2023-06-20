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

import unittest
from unittest.mock import MagicMock

import numpy as np

import fuzzylite as fl
from tests.assert_component import BaseAssert


class ActivationAssert(BaseAssert[fl.RuleBlock]):
    """Activation Assert."""

    def mock_rule(
        self,
        id: str,
        activation_degree: fl.Scalar,
        weight: float = 1.0,
        loaded: bool = True,
    ) -> fl.Rule:
        """Mocks a rule with the given activation degree and weight."""
        rule = fl.Rule()
        rule.text = f"if {id.upper()} is {id.lower()} then {id.upper()} is {id.lower()} with {weight}"
        rule.antecedent.activation_degree = MagicMock(return_value=activation_degree)  # type: ignore
        rule.consequent.modify = MagicMock()  # type: ignore
        rule.is_loaded = MagicMock(return_value=loaded)  # type: ignore
        return rule

    def given(
        self, rule: str, activation_degree: fl.Scalar, weight: float = 1.0
    ) -> ActivationAssert:
        """Given a rule with an activation degree and weight."""
        self.actual.rules.append(
            self.mock_rule(rule, activation_degree=activation_degree, weight=weight)
        )
        return self

    def activate(self, method: fl.Activation) -> ActivationAssert:
        """Activates the actual rule block with the given method."""
        method.activate(self.actual)
        return self

    def activate_fails(self, method: fl.Activation) -> ActivationAssert:
        """Activates the actual rule block with the given method."""
        with self.test.assertRaises(TypeError) as error:
            method.activate(self.actual)
        self.test.assertTrue(
            str(error.exception).startswith(
                "expected activation degree to be a single scalar, "
            )
        )
        return self

    def then_triggers(self, rules: dict[str, bool | list[bool]]) -> ActivationAssert:
        """Asserts the rules triggered are the given ones."""

        def rule_text(rule_id: str) -> str:
            return f"if {rule_id.upper()} is {rule_id.lower()} then {rule_id.upper()} is {rule_id.lower()}"

        expected = {
            rule_text(rule_id): np.array(activation)
            for rule_id, activation in rules.items()
        }
        obtained = {rule.text: rule.triggered for rule in self.actual.rules}

        self.test.assertEqual(expected.keys(), obtained.keys())

        for rule_id, activation in rules.items():
            np.testing.assert_array_equal(
                activation,
                obtained[rule_text(rule_id)],
                err_msg=f"in rule'{rule_id}' when rules={rules}",
            )
        return self

    def with_activation(self, rules: dict[str, float]) -> ActivationAssert:
        """Asserts the rules triggered are the given ones."""

        def rule_text(rule_id: str) -> str:
            return f"if {rule_id.upper()} is {rule_id.lower()} then {rule_id.upper()} is {rule_id.lower()}"

        expected = {
            rule_text(rule_id): np.array(activation)
            for rule_id, activation in rules.items()
        }
        obtained = {rule.text: rule.activation_degree for rule in self.actual.rules}

        self.test.assertEqual(expected.keys(), obtained.keys())

        for rule_id, activation in rules.items():
            np.testing.assert_allclose(
                activation,
                obtained[rule_text(rule_id)],
                err_msg=f"in rule'{rule_id}' when rules={rules}",
                atol=fl.lib.atol,
                rtol=fl.lib.rtol,
            )
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

    def test_general_activation(self) -> None:
        """Asserts the general activation is correct."""
        self.assertEqual("General", str(fl.General()))
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.General()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.General()
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.General()
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

    def test_activation_vectors(self) -> None:
        """Asserts the general activation is correct."""
        # All combinations
        expected_a = [True] * 4 + [False] * 4
        expected_b = 2 * ([True] * 2 + [False] * 2)
        expected_c = [True, False] * 4

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=np.array(expected_a, dtype=np.float_)
        ).given(
            rule="b", activation_degree=np.array(expected_b, dtype=np.float_)
        ).given(
            rule="c", activation_degree=np.array(expected_c, dtype=np.float_)
        ).activate(
            fl.General()
        ).then_triggers(
            rules={"a": expected_a, "b": expected_b, "c": expected_c}
        )

        unsupported_activations = [
            fl.First(),
            fl.Last(),
            fl.Highest(),
            fl.Lowest(),
            fl.Proportional(),
            fl.Threshold(),
        ]
        for activation in unsupported_activations:
            ActivationAssert(self, fl.RuleBlock()).given(
                rule="a", activation_degree=np.array(expected_a, dtype=np.float_)
            ).given(
                rule="b", activation_degree=np.array(expected_b, dtype=np.float_)
            ).given(
                rule="c", activation_degree=np.array(expected_c, dtype=np.float_)
            ).activate_fails(
                activation
            )

    def test_first_activation(self) -> None:
        """Asserts the first activation is correct."""
        self.assertEqual("First 1 0.000", str(fl.First()))
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.First()
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First()
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        # First two rules
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First(rules=2)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.First(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # First two rules with threshold > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5
        ).given(rule="b", activation_degree=0.49).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

    def test_last_activation(self) -> None:
        """Asserts the last activation is correct."""
        self.assertEqual("Last 1 0.000", str(fl.Last()))
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Last()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Last()
        ).then_triggers(
            rules={"a": False, "b": False, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Last()
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Last two rules
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Last(rules=2)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Last(rules=2)
        ).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Last(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # Last two rules with threshold > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.49).given(
            rule="c", activation_degree=0.5
        ).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": True}
        )

    def test_highest_activation(self) -> None:
        """Asserts the highest activation is correct."""
        self.assertEqual("Highest 1", str(fl.Highest()))
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Highest()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Highest()
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5
        ).given(rule="b", activation_degree=1).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Highest()
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Highest two rules
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Highest(rules=2)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Highest(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.25
        ).activate(
            fl.Highest(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

    def test_lowest_activation(self) -> None:
        """Asserts the lowest activation is correct."""
        self.assertEqual("Lowest 1", str(fl.Lowest()))
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Lowest()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Lowest()
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Lowest()
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Lowest two rules
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Lowest(rules=2)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Lowest(rules=2)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.25
        ).activate(
            fl.Lowest(rules=2)
        ).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

    def test_proportional_activation(self) -> None:
        """Asserts the proportional activation is correct."""
        self.assertEqual("Proportional", str(fl.Proportional()))
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Proportional()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Proportional()
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        ).with_activation(
            {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Proportional()
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        ).with_activation(
            {"a": 2 / 3, "b": 1 / 3, "c": 0.0}
        )

    def test_threshold_activation(self) -> None:
        """Asserts the threshold activation is correct."""
        # Default: > 0.0
        self.assertEqual("Threshold > 0.000", str(fl.Threshold()))

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold()
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold()
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold()
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold(">", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold(">", 0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold(">", 0.5)
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        # >= 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold(">=", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold(">=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold(">=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # < 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("<", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold("<", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5
        ).given(rule="b", activation_degree=0.25).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("<", 0.5)
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # <= 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("<=", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold("<=", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5
        ).given(rule="b", activation_degree=0.25).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("<=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # == 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("==", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold("==", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5 - 1e-10
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.5 + 1e-10
        ).activate(
            fl.Threshold("==", 0.5)
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # != 0.5
        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.0
        ).given(rule="b", activation_degree=0.0).given(
            rule="c", activation_degree=0.0
        ).activate(
            fl.Threshold("!=", 0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=1.0
        ).given(rule="b", activation_degree=1.0).given(
            rule="c", activation_degree=1.0
        ).activate(
            fl.Threshold("!=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=0.5 - 1e-10
        ).given(rule="b", activation_degree=0.5).given(
            rule="c", activation_degree=0.5 + 1e-10
        ).activate(
            fl.Threshold("!=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": False, "c": True}
        )


if __name__ == "__main__":
    unittest.main()
