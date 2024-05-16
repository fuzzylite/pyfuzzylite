"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = [
    "Activation",
    "General",
    "First",
    "Last",
    "Highest",
    "Lowest",
    "Proportional",
    "Threshold",
]

import enum
import heapq
import operator
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .library import representation, scalar, to_float
from .operation import Op
from .rule import Rule, RuleBlock
from .types import Array, Scalar


class Activation(ABC):
    """Abstract class for activation methods.

    Activation methods implement the criteria to activate the rules in a rule block.
    Activation methods should process every rule and determine whether the rule is to be activated or deactivated.

    info: related
        - [fuzzylite.activation.General][]
        - [fuzzylite.activation.First][]
        - [fuzzylite.activation.Last][]
        - [fuzzylite.activation.Highest][]
        - [fuzzylite.activation.Lowest][]
        - [fuzzylite.activation.Proportional][]
        - [fuzzylite.activation.Threshold][]
        - [fuzzylite.factory.ActivationFactory][]
    """

    def __str__(self) -> str:
        """Return the code to construct the activation method in the FuzzyLite Language.

        Returns:
            code to construct the activation method in the FuzzyLite Language.
        """
        return representation.fll.activation(self)

    def __repr__(self) -> str:
        """Returns the code to construct the activation method in Python.

        Returns:
            code to construct the activation method in Python.
        """
        return representation.as_constructor(self)

    @abstractmethod
    def activate(self, rule_block: RuleBlock) -> None:
        """Implement the activation method of the rule block.

        Args:
             rule_block: rule block to activate
        """

    def parameters(self) -> str:
        """Return the parameters of the activation method.

        Returns:
            parameters of the activation method.
        """
        return ""

    def configure(  # noqa: B027  empty method in an abstract base class
        self, parameters: str
    ) -> None:
        """Configure the activation method with the parameters.

        Args:
             parameters: list of space-separated parameter values
        """
        pass

    def assert_is_not_vector(self, activation_degree: Scalar) -> None:
        """Assert that the activation degree is not a vector.

        Args:
            activation_degree: activation degree to assert

        Raises:
            ValueError: if the activation degree contains more than one element
        """
        if (size := np.size(activation_degree)) > 1:
            raise ValueError(
                f"expected a unit scalar, but got vector of size {size}: {activation_degree=}"
            )


class General(Activation):
    """Activation method that activates every rule of a rule block in insertion order.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate every rule in the rule block in the order they were added.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        for rule in rule_block.rules:
            rule.deactivate()
            if rule.is_loaded():
                rule.activate_with(conjunction, disjunction)
                rule.trigger(implication)


class First(Activation):
    """Activate the first $n$ rules (in insertion order) whose activation degrees are greater than or equal to the threshold.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.Last][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        """Constructor.

        Args:
            rules: maximum number of rules to activate
            threshold: minimum activation degree required to activate the rule.
        """
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        """Return the number of rules and threshold.

        Returns:
            number of rules and threshold.
        """
        return f"{Op.str(self.rules)} {Op.str(self.threshold)}"

    def configure(self, parameters: str) -> None:
        """Configure the activation method with the parameters.

        Args:
            parameters: number of rules and threshold (eg, `3 0.5`).
        """
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = to_float(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate the first $n$ rules (in insertion order) whose activation degrees are greater than or equal to the threshold.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activated = 0

        for rule in iter(rule_block.rules):
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if (
                    activated < self.rules
                    and activation_degree > 0.0
                    and activation_degree >= self.threshold
                ):
                    rule.trigger(implication)
                    activated += 1


class Last(Activation):
    """Activation method that activates the first $n$ rules (in reverse insertion order) whose activation degrees are greater than or equal to the threshold.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.First][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        """Constructor.

        Args:
            rules: maximum number of rules to activate
            threshold: minimum activation degree required to activate the rule.
        """
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        """Return the number of rules and threshold.

        Returns:
            number of rules and threshold.
        """
        return f"{Op.str(self.rules)} {Op.str(self.threshold)}"

    def configure(self, parameters: str) -> None:
        """Configure the activation method with the parameters.

        Args:
            parameters: number of rules and threshold (eg, `3 0.5`).
        """
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = to_float(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate the last $n$ rules (in reverse insertion order) whose activation degrees are greater than or equal to the threshold.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activated = 0

        for rule in reversed(rule_block.rules):
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if (
                    activated < self.rules
                    and activation_degree > 0.0
                    and activation_degree >= self.threshold
                ):
                    rule.trigger(implication)
                    activated += 1


class Highest(Activation):
    """Activation method that activates only the rules with the highest activation degrees in descending order.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.Lowest][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def __init__(self, rules: int = 1) -> None:
        """Constructor.

        Args:
            rules: number of rules to activate.
        """
        self.rules = rules

    def parameters(self) -> str:
        """Return the number of rules.

        Returns:
            number of rules.
        """
        return str(self.rules)

    def configure(self, parameters: str) -> None:
        """Configure the activation method with the parameters.

        Args:
            parameters: number of rules (eg, `3`).
        """
        if parameters:
            self.rules = int(parameters)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate the rules with the highest activation degrees.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: list[tuple[Scalar, int]] = []

        for index, rule in enumerate(rule_block.rules):
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if activation_degree > 0.0:
                    heapq.heappush(activate, (-activation_degree, index))

        activated = 0
        while activate and activated < self.rules:
            index = heapq.heappop(activate)[1]
            rule_block.rules[index].trigger(implication)
            activated += 1


class Lowest(Activation):
    """Activation method that activates only the rules with the lowest activation degrees in ascending order.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.Highest][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def __init__(self, rules: int = 1) -> None:
        """Constructor.

        Args:
            rules: number of rules to activate.
        """
        self.rules = rules

    def parameters(self) -> str:
        """Return the number of rules.

        Returns:
            number of rules.
        """
        return str(self.rules)

    def configure(self, parameters: str) -> None:
        """Configure the activation method with the parameters.

        Args:
            parameters: number of rules (eg, `3`).
        """
        if parameters:
            self.rules = int(parameters)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate the rules with the lowest activation degrees.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: list[tuple[Scalar, int]] = []

        for index, rule in enumerate(rule_block.rules):
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if activation_degree > 0.0:
                    heapq.heappush(activate, (activation_degree, index))

        activated = 0
        while activate and activated < self.rules:
            index = heapq.heappop(activate)[1]
            rule_block.rules[index].trigger(implication)
            activated += 1


class Proportional(Activation):
    """Activation method that activates the rules utilizing normalized activation degrees, thus the sum of the activation degrees is equal to one.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.General][]
        - [fuzzylite.activation.Threshold][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    def activate(self, rule_block: RuleBlock) -> None:
        """Activate the rules using normalized activation degrees.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: list[Rule] = []
        sum_degrees = scalar(0.0)
        for rule in rule_block.rules:
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if activation_degree > 0.0:
                    activate.append(rule)
                    sum_degrees += activation_degree

        for rule in activate:
            rule.activation_degree /= sum_degrees
            rule.trigger(implication)


class Threshold(Activation):
    """Activation method that activates the rules whose activation degrees satisfy the comparison operator and the threshold, and deactivates the rest.

    info: related
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.activation.General][]
        - [fuzzylite.activation.Proportional][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.rule.RuleBlock][]
    """

    @enum.unique
    class Comparator(enum.Enum):
        r"""Six comparison operators between the activation degree $a$ and the threshold $\theta$."""

        # $a < \theta$
        LessThan = "<"
        # $a \leq \theta$
        LessThanOrEqualTo = "<="
        # $a = \theta$
        EqualTo = "=="
        # $a \neq \theta$
        NotEqualTo = "!="
        # $a \geq \theta$
        GreaterThanOrEqualTo = ">="
        # $a > \theta$
        GreaterThan = ">"

        __operator__: dict[
            str,
            Callable[[Scalar, Scalar], bool | Array[np.bool_]],
        ] = {  # pyright: ignore
            LessThan: operator.lt,
            LessThanOrEqualTo: operator.le,
            EqualTo: operator.eq,
            NotEqualTo: operator.ne,
            GreaterThanOrEqualTo: operator.ge,
            GreaterThan: operator.gt,
        }

        def __repr__(self) -> str:
            """Return the code to construct the comparator in Python.

            Returns:
                code to construct the comparator in Python.
            """
            return f"'{self.value}'"

        @property
        def operator(self) -> Callable[[Scalar, Scalar], bool | Array[np.bool_]]:
            """Return the function reference for the operator.

            Returns:
                function reference for the operator.
            """
            return Threshold.Comparator.__operator__[self.value]

    def __init__(
        self,
        comparator: Comparator | str = Comparator.GreaterThan,
        threshold: float = 0.0,
    ) -> None:
        """Constructor.

        Args:
            comparator: comparison operator
            threshold: value for activation degrees.
        """
        if isinstance(comparator, str):
            comparator = Threshold.Comparator(comparator)
        self.comparator = comparator
        self.threshold = threshold

    def parameters(self) -> str:
        """Return the comparator and threshold.

        Returns:
            comparator and threshold.
        """
        return f"{self.comparator.value} {Op.str(self.threshold)}"

    def configure(self, parameters: str) -> None:
        """Configure the activation method with the parameters.

        Args:
            parameters: comparator and threshold (eg, `> 0.5`).
        """
        if parameters:
            comparator, threshold = parameters.split()
            self.comparator = Threshold.Comparator(comparator)
            self.threshold = to_float(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the rules whose activation degrees satisfy the comparator and threshold, and deactivate the rest.

        Args:
            rule_block: rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        for rule in rule_block.rules:
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                self.assert_is_not_vector(activation_degree)
                if self.comparator.operator(activation_degree, self.threshold):
                    rule.trigger(implication)
