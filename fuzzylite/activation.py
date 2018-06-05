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

from .operation import Op
from .rule import RuleBlock


class Activation(object):
    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def activate(self, rule_block: RuleBlock) -> None:
        raise NotImplementedError()

    def parameters(self) -> str:
        return ""

    def configure(self, parameters: str) -> None:
        pass


class First(Activation):
    __slots__ = ("number_of_rules", "threshold")

    def __init__(self, number_of_rules: int = 1, threshold: float = 0.0) -> None:
        self.number_of_rules = number_of_rules
        self.threshold = threshold

    def parameters(self) -> str:
        return " ".join([Op.str(self.number_of_rules),
                         Op.str(self.threshold)])

    def configure(self, parameters: str) -> None:
        if not parameters:
            return
        values = parameters.split()
        required = 2
        if len(values) < required:
            raise ValueError(f"activation <{self.class_name}> requires {required} parameters, "
                             f"but only {len(values)} were provided")
        self.number_of_rules = int(values[0])
        self.threshold = float(values[1])

    def activate(self, rule_block: RuleBlock) -> None:
        conjunction = rule_block.conjunction
        disjunction = rule_block.disjunction
        implication = rule_block.implication

        activated = 0
        for rule in rule_block.rules:
            rule.deactivate()

            if rule.is_loaded():
                activation_degree = rule.activate_with(conjunction, disjunction)
                if (activated < self.number_of_rules
                        and activation_degree > 0.0
                        and activation_degree >= self.threshold):
                    rule.trigger(implication)
                    activated += 1


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


class Highest(Activation):
    pass


class Last(Activation):
    pass


class Lowest(Activation):
    pass


class Proportional(Activation):
    pass


class Threshold(Activation):
    pass
