"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

import fuzzylite as fl
from fuzzylite import RuleBlock
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
        rule.text = (
            f"if {id.upper()} is {id.lower()} then {id.upper()} is {id.lower()} with {weight}"
        )
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
        with self.test.assertRaises(ValueError) as error:
            method.activate(self.actual)
        self.test.assertTrue(
            str(error.exception).startswith("expected a unit scalar, but got vector of size ")
        )
        return self

    def then_triggers(self, rules: dict[str, bool | list[bool]]) -> ActivationAssert:
        """Asserts the rules triggered are the given ones."""

        def rule_text(rule_id: str) -> str:
            return f"if {rule_id.upper()} is {rule_id.lower()} then {rule_id.upper()} is {rule_id.lower()}"

        expected = {
            rule_text(rule_id): np.array(activation) for rule_id, activation in rules.items()
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
            rule_text(rule_id): fl.array(activation) for rule_id, activation in rules.items()
        }
        obtained = {rule.text: rule.activation_degree for rule in self.actual.rules}

        self.test.assertEqual(expected.keys(), obtained.keys())

        for rule_id, activation in rules.items():
            np.testing.assert_allclose(
                activation,
                obtained[rule_text(rule_id)],
                err_msg=f"in rule'{rule_id}' when rules={rules}",
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
            )
        return self


class TestActivation(unittest.TestCase):
    """Tests the base activation class."""

    def test_base_activation(self) -> None:
        """Tests the base activation class."""

        class BaseActivation(fl.Activation):
            """Base Activation."""

            def activate(self, rule_block: RuleBlock) -> None:
                """Do nothing."""
                pass

        self.assertEqual("", BaseActivation().parameters())
        self.assertEqual("BaseActivation", str(BaseActivation()))

        activation = BaseActivation()
        activation.parameters = MagicMock(return_value="param1 param2")  # type: ignore
        self.assertEqual("BaseActivation param1 param2", str(activation))

    def test_general_activation(self) -> None:
        """Asserts the general activation is correct."""
        BaseAssert(self, fl.General()).repr_is("fl.General()").exports_fll("General")

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.General()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.General()).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.General()).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

    def test_activation_vectors(self) -> None:
        """Asserts the general activation is correct."""
        # All combinations
        expected_a = [True] * 4 + [False] * 4
        expected_b = 2 * ([True] * 2 + [False] * 2)
        expected_c = [True, False] * 4

        ActivationAssert(self, fl.RuleBlock()).given(
            rule="a", activation_degree=fl.array(expected_a, dtype=np.float64)
        ).given(rule="b", activation_degree=fl.array(expected_b, dtype=np.float64)).given(
            rule="c", activation_degree=fl.array(expected_c, dtype=np.float64)
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
                rule="a", activation_degree=fl.array(expected_a, dtype=np.float64)
            ).given(rule="b", activation_degree=fl.array(expected_b, dtype=np.float64)).given(
                rule="c", activation_degree=fl.array(expected_c, dtype=np.float64)
            ).activate_fails(
                activation
            )

    def test_first_activation(self) -> None:
        """Asserts the first activation is correct."""
        BaseAssert(self, fl.First()).repr_is("fl.First(rules=1, threshold=0.0)").exports_fll(
            "First 1 0.000"
        )
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.First()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.First()).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.First()).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        # First two rules
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.First(rules=2)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.First(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.First(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # First two rules with threshold > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5).given(
            rule="b", activation_degree=0.49
        ).given(rule="c", activation_degree=0.0).activate(
            fl.First(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

    def test_last_activation(self) -> None:
        """Asserts the last activation is correct."""
        BaseAssert(self, fl.Last()).repr_is("fl.Last(rules=1, threshold=0.0)").exports_fll(
            "Last 1 0.000"
        )
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Last()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Last()).then_triggers(
            rules={"a": False, "b": False, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Last()).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Last two rules
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Last(rules=2)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Last(rules=2)).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Last(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # Last two rules with threshold > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.49
        ).given(rule="c", activation_degree=0.5).activate(
            fl.Last(rules=2, threshold=0.5)
        ).then_triggers(
            rules={"a": False, "b": False, "c": True}
        )

    def test_highest_activation(self) -> None:
        """Asserts the highest activation is correct."""
        BaseAssert(self, fl.Highest()).repr_is("fl.Highest(rules=1)").exports_fll("Highest 1")
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Highest()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Highest()).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5).given(
            rule="b", activation_degree=1
        ).given(rule="c", activation_degree=0.0).activate(fl.Highest()).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Highest two rules
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Highest(rules=2)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Highest(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.25).activate(fl.Highest(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

    def test_lowest_activation(self) -> None:
        """Asserts the lowest activation is correct."""
        BaseAssert(self, fl.Lowest()).repr_is("fl.Lowest(rules=1)").exports_fll("Lowest 1")
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Lowest()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Lowest()).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Lowest()).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # Lowest two rules
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Lowest(rules=2)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Lowest(rules=2)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.25).activate(fl.Lowest(rules=2)).then_triggers(
            rules={"a": False, "b": True, "c": True}
        )

    def test_proportional_activation(self) -> None:
        """Asserts the proportional activation is correct."""
        BaseAssert(self, fl.Proportional()).repr_is("fl.Proportional()").exports_fll("Proportional")
        # Default activation
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Proportional()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Proportional()).then_triggers(
            rules={"a": True, "b": True, "c": True}
        ).with_activation(
            {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Proportional()).then_triggers(
            rules={"a": True, "b": True, "c": False}
        ).with_activation(
            {"a": 2 / 3, "b": 1 / 3, "c": 0.0}
        )

    def test_threshold_activation(self) -> None:
        """Asserts the threshold activation is correct."""
        # Default: > 0.0
        BaseAssert(self, fl.Threshold()).repr_is(
            "fl.Threshold(comparator='>', threshold=0.0)"
        ).exports_fll("Threshold > 0.000")

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold()).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold()).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold()).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # > 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold(">", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold(">", 0.5)).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold(">", 0.5)).then_triggers(
            rules={"a": True, "b": False, "c": False}
        )

        # >= 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold(">=", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold(">=", 0.5)).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold(">=", 0.5)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # < 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("<", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold("<", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5).given(
            rule="b", activation_degree=0.25
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("<", 0.5)).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # <= 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("<=", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold("<=", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5).given(
            rule="b", activation_degree=0.25
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("<=", 0.5)).then_triggers(
            rules={"a": True, "b": True, "c": False}
        )

        # == 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("==", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold("==", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5 - 1e-10).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.5 + 1e-10).activate(
            fl.Threshold("==", 0.5)
        ).then_triggers(
            rules={"a": False, "b": True, "c": False}
        )

        # != 0.5
        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.0).given(
            rule="b", activation_degree=0.0
        ).given(rule="c", activation_degree=0.0).activate(fl.Threshold("!=", 0.5)).then_triggers(
            rules={"a": False, "b": False, "c": False}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=1.0).given(
            rule="b", activation_degree=1.0
        ).given(rule="c", activation_degree=1.0).activate(fl.Threshold("!=", 0.5)).then_triggers(
            rules={"a": True, "b": True, "c": True}
        )

        ActivationAssert(self, fl.RuleBlock()).given(rule="a", activation_degree=0.5 - 1e-10).given(
            rule="b", activation_degree=0.5
        ).given(rule="c", activation_degree=0.5 + 1e-10).activate(
            fl.Threshold("!=", 0.5)
        ).then_triggers(
            rules={"a": True, "b": False, "c": True}
        )


if __name__ == "__main__":
    unittest.main()
