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
from typing import Callable, Dict, List, Tuple, Union

from .operation import Op
from .rule import Rule, RuleBlock


class Activation:
    """The Activation class is the abstract class for RuleBlock activation
    methods. An activation method implements the criteria to activate the
    rules within a given rule block. An activation method needs to process
    every rule and determine whether the rule is to be activated or
    deactivated. The activation methods were first introduced in version 6.0,
    but in earlier versions the term `activation` referred to the TNorm that
    modulated the consequent of a rule, which is now referred to as the
    `implication` operator.

    @author Juan Rada-Vilela, Ph.D.
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    @property
    def class_name(self) -> str:
        """Returns the name of the activation method, which is also utilized to
        register the activation method in the ActivationFactory.
        @return the name of the activation method
        @see ActivationFactory.
        """
        return self.__class__.__name__

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the rule block
        @param rule_block is the rule block to activate.
        """
        raise NotImplementedError()

    def parameters(self) -> str:
        """Returns the parameters of the activation method, which can be used to
        configure other instances of the activation method.
        @return the parameters of the activation method.
        """
        return ""

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the given parameters.
        @param parameters contains a list of space-separated parameter values.
        """

    def __str__(self) -> str:
        """Returns the FLL code for the activation method
        @return FLL code for the activation method.
        """
        result = self.class_name
        parameters = self.parameters()
        if parameters:
            result += f" {parameters}"
        return result


class General(Activation):
    """The General class is a RuleBlock Activation method that activates every
    rule following the order in which the rules were added to the rule block.

    @author Juan Rada-Vilela, Ph.D.
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates every rule in the given rule block following the order in
        which the rules were added.
        @param rule_block is the rule block to activate.
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
    """The First class is a RuleBlock Activation method that activates the first
    $n$ rules whose activation degrees are greater than or equal to the given
    threshold. The rules are iterated in the order they were added to the rule block.

    @author Juan Rada-Vilela, Ph.D.
    @see Last
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        """Constructs a First activation method
        @param rules is the number of rules for the activation degree
        @param threshold is the threshold for the activation degree.
        """
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        """Returns the number of rules and the threshold of the activation method
        @return "rules threshold"
        TODO: convert to f-string.
        """
        return " ".join([Op.str(self.rules), Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the given number of rules and
        threshold
        @param parameters as "rules threshold".
        """
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = Op.scalar(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the first $n$ rules whose activation degrees are greater than or
        equal to the given threshold. The rules are iterated in the order the
        rules were added to the rule block.
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activated = 0

        for rule in iter(rule_block.rules):
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if (
                    activated < self.rules
                    and Op.gt(activation_degree, 0.0)
                    and activation_degree >= self.threshold
                ):
                    rule.trigger(implication)
                    activated += 1


class Last(Activation):
    """The Last class is a RuleBlock Activation method that activates the last
    $n$ rules whose activation degrees are greater than or equal to the given
    threshold. The rules are iterated in the reverse order in which they were
    added to the rule block.

    @author Juan Rada-Vilela, Ph.D.
    @see First
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        """Constructs the Last activtion method
        @param rules is the number of rules for the activation degree
        @param threshold is the threshold for the activation degree.
        """
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        """Returns the number of rules and the threshold of the activation method
        @return "rules threshold".
        """
        return " ".join([Op.str(self.rules), Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the given number of rules and
        threshold
        @param parameters as "rules threshold".
        """
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = Op.scalar(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the last $n$ rules whose activation degrees are greater
        than the given threshold. The rules are iterated in the reverse order
        that the rules were added to the rule block.
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activated = 0

        for rule in reversed(rule_block.rules):
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if (
                    activated < self.rules
                    and Op.gt(activation_degree, 0.0)
                    and activation_degree >= self.threshold
                ):
                    rule.trigger(implication)
                    activated += 1


class Highest(Activation):
    """The Highest class is a RuleBlock Activation method that activates a given
    number of rules with the highest activation degrees in descending order.

    @author Juan Rada-Vilela, Ph.D.
    @see Lowest
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def __init__(self, rules: int = 1) -> None:
        """Creates the Highest activation method
        @param rules is the number of rules to activate.
        """
        self.rules = rules

    def parameters(self) -> str:
        """Returns the number of rules to activate.
        @return number of rules to activate.
        """
        return str(self.rules)

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the number of rules to activate.
        @param parameters contains the number of rules to activate.
        """
        if parameters:
            self.rules = int(parameters)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the given number of rules with the highest activation
        degrees
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: List[Tuple[float, int]] = []

        for index, rule in enumerate(rule_block.rules):
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if Op.gt(activation_degree, 0.0):
                    heapq.heappush(activate, (-activation_degree, index))

        activated = 0
        while activate and activated < self.rules:
            index = heapq.heappop(activate)[1]
            rule_block.rules[index].trigger(implication)
            activated += 1


class Lowest(Activation):
    """The Lowest class is a RuleBlock Activation method that activates a given
    number of rules with the lowest activation degrees in ascending order.

    @author Juan Rada-Vilela, Ph.D.
    @see Highest
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def __init__(self, rules: int = 1) -> None:
        """Creates the Lowest activation method.
        @param rules is the number of rules to activate.
        """
        self.rules = rules

    def parameters(self) -> str:
        """Returns the number of rules to activate
        @return number of rules to activate.
        """
        return str(self.rules)

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the number of rules to activate.
        @param parameters contains the number of rules to activate.
        """
        if parameters:
            self.rules = int(parameters)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the rules with the lowest activation degrees in the given
        rule block
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: List[Tuple[float, int]] = []

        for index, rule in enumerate(rule_block.rules):
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if Op.gt(activation_degree, 0.0):
                    heapq.heappush(activate, (activation_degree, index))

        activated = 0
        while activate and activated < self.rules:
            index = heapq.heappop(activate)[1]
            rule_block.rules[index].trigger(implication)
            activated += 1


class Proportional(Activation):
    """The Proportional class is a RuleBlock Activation method that activates
    the rules utilizing activation degrees proportional to the activation
    degrees of the other rules, thus the sum of the activation degrees is
    equal to one.

    @author Juan Rada-Vilela, Ph.D.
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the rules utilizing activation degrees proportional to
        the activation degrees of the other rules in the rule block.
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activate: List[Rule] = []
        sum_degrees = 0.0
        for rule in rule_block.rules:
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if Op.gt(activation_degree, 0.0):
                    activate.append(rule)
                    sum_degrees += activation_degree

        for rule in activate:
            rule.activation_degree /= sum_degrees
            rule.trigger(implication)


class Threshold(Activation):
    """The Threshold class is a RuleBlock Activation method that activates the
    rules whose activation degrees satisfy the equation given by the
    comparison operator and the threshold, and deactivates the rules which do
    not satisfy the equation.

    @author Juan Rada-Vilela, Ph.D.
    @see Rule
    @see RuleBlock
    @see ActivationFactory
    @since 6.0
    """

    @enum.unique
    class Comparator(enum.Enum):
        r"""Comparator is an enumerator that provides six comparison operators
        between the activation degree $a$ and the threshold $\theta$.
        """

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

        __operator__: Dict[str, Callable[[float, float], bool]] = {
            LessThan: Op.lt,
            LessThanOrEqualTo: Op.le,
            EqualTo: Op.eq,
            NotEqualTo: Op.neq,
            GreaterThanOrEqualTo: Op.ge,
            GreaterThan: Op.gt,
        }

        @property
        def operator(self) -> Callable[[float, float], bool]:
            """Gets the function reference for the operator."""
            return Threshold.Comparator.__operator__[self.value]

    def __init__(
        self,
        comparator: Union[Comparator, str] = Comparator.GreaterThanOrEqualTo,
        threshold: float = 0.0,
    ) -> None:
        """Creates a Threshold activation method
        @param comparator is a valid comparison operator
        @param threshold is the threshold for activation degrees.
        """
        if isinstance(comparator, str):
            comparator = Threshold.Comparator(comparator)
        self.comparator = comparator
        self.threshold = threshold

    def parameters(self) -> str:
        """Returns the comparator followed by the threshold.
        @return comparator and threshold
        TODO: convert to f-string.
        """
        return " ".join([self.comparator.value, Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        """Configures the activation method with the comparator and the
        threshold.
        @param parameters is the comparator and threshold.
        """
        if parameters:
            comparator, threshold = parameters.split()
            self.comparator = Threshold.Comparator(comparator)
            self.threshold = Op.scalar(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        """Activates the rules whose activation degrees satisfy the comparison
        equation with the given threshold, and deactivate the rules which do
        not.
        @param rule_block is the rule block to activate.
        """
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        for rule in rule_block.rules:
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if self.comparator.operator(activation_degree, self.threshold):
                    rule.trigger(implication)
