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

__all__ = ["Activation", "General", "First", "Last", "Highest", "Lowest", "Proportional",
           "Threshold"]

import enum
import heapq
import operator
from typing import Callable, List, Tuple, Union

from .operation import Op
from .rule import Rule, RuleBlock


class Activation:

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def activate(self, rule_block: RuleBlock) -> None:
        raise NotImplementedError()

    def parameters(self) -> str:
        return ""

    def configure(self, parameters: str) -> None:
        pass

    def __str__(self) -> str:
        result = self.class_name
        parameters = self.parameters()
        if parameters:
            result += f" {parameters}"
        return result


class General(Activation):

    def activate(self, rule_block: RuleBlock) -> None:
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        for rule in rule_block.rules:
            rule.deactivate()
            if rule.is_loaded():
                rule.activate_with(conjunction, disjunction)
                rule.trigger(implication)


def _activate_positional(activation: Union['First', 'Last'], rule_block: RuleBlock) -> None:
    conjunction = rule_block.conjunction
    disjunction = rule_block.disjunction
    implication = rule_block.implication

    activated = 0
    if isinstance(activation, First):
        rules = iter(rule_block.rules)
    elif isinstance(activation, Last):
        rules = reversed(rule_block.rules)
    else:
        raise ValueError()

    for rule in rules:
        rule.deactivate()

        if rule.is_loaded():
            activation_degree = rule.activate_with(conjunction, disjunction)
            if (activated < activation.rules
                    and Op.gt(activation_degree, 0.0)
                    and activation_degree >= activation.threshold):
                rule.trigger(implication)
                activated += 1


class First(Activation):

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        return " ".join([Op.str(self.rules),
                         Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = Op.scalar(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        _activate_positional(self, rule_block)


class Last(Activation):

    def __init__(self, rules: int = 1, threshold: float = 0.0) -> None:
        self.rules = rules
        self.threshold = threshold

    def parameters(self) -> str:
        return " ".join([Op.str(self.rules),
                         Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        if parameters:
            rules, threshold = parameters.split()
            self.rules = int(rules)
            self.threshold = Op.scalar(threshold)

    def activate(self, rule_block: RuleBlock) -> None:
        _activate_positional(self, rule_block)


def _activate_ranking(activation: Union['Highest', 'Lowest'], rule_block: RuleBlock) -> None:
    conjunction = rule_block.conjunction
    disjunction = rule_block.disjunction
    implication = rule_block.implication

    activate: List[Tuple[float, int]] = []

    if isinstance(activation, Highest):
        sign = -1
    elif isinstance(activation, Lowest):
        sign = 1
    else:
        raise ValueError()

    for index, rule in enumerate(rule_block.rules):
        rule.deactivate()
        if rule.is_loaded():
            activation_degree = rule.activate_with(conjunction, disjunction)
            if Op.gt(activation_degree, 0.0):
                heapq.heappush(activate, (sign * activation_degree, index))

    activated = 0
    while activate and activated < activation.rules:
        index = heapq.heappop(activate)[1]
        rule_block.rules[index].trigger(implication)
        activated += 1


class Highest(Activation):

    def __init__(self, rules: int = 1) -> None:
        self.rules = rules

    def configure(self, parameters: str) -> None:
        if parameters:
            self.rules = int(parameters)

    def parameters(self) -> str:
        return str(self.rules)

    def activate(self, rule_block: RuleBlock) -> None:
        _activate_ranking(self, rule_block)


class Lowest(Activation):

    def __init__(self, rules: int = 1) -> None:
        self.rules = rules

    def configure(self, parameters: str) -> None:
        if parameters:
            self.rules = int(parameters)

    def parameters(self) -> str:
        return str(self.rules)

    def activate(self, rule_block: RuleBlock) -> None:
        _activate_ranking(self, rule_block)


class Proportional(Activation):

    def activate(self, rule_block: RuleBlock) -> None:
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
    @enum.unique
    class Comparator(enum.Enum):
        LessThan = "<"
        LessThanOrEqualTo = "<="
        EqualTo = "=="
        NotEqualTo = "!="
        GreaterThanOrEqualTo = ">="
        GreaterThan = ">"

        __operator__ = {
            LessThan: operator.lt,
            LessThanOrEqualTo: operator.le,
            EqualTo: operator.eq,
            NotEqualTo: operator.ne,
            GreaterThanOrEqualTo: operator.ge,
            GreaterThan: operator.gt
        }

        @property
        def operator(self) -> Callable[[object, object], bool]:
            return Threshold.Comparator.__operator__[self.value]  # type:ignore

    def __init__(self, comparator: Union[Comparator, str] = Comparator.GreaterThanOrEqualTo,
                 threshold: float = 0.0) -> None:
        if isinstance(comparator, str):
            comparator = Threshold.Comparator(comparator)
        self.comparator = comparator
        self.threshold = threshold

    def configure(self, parameters: str) -> None:
        if parameters:
            comparator, threshold = parameters.split()
            self.comparator = Threshold.Comparator(comparator)
            self.threshold = Op.scalar(threshold)

    def parameters(self) -> str:
        return " ".join([self.comparator.value, Op.str(self.threshold)])

    def activate(self, rule_block: RuleBlock) -> None:
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        for rule in rule_block.rules:
            rule.deactivate()
            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if self.comparator.operator(activation_degree, self.threshold):
                    rule.trigger(implication)
