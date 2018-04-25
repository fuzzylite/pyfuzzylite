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

import logging
from math import nan
from typing import Iterable

from .exporter import FllExporter
from .hedge import Any
from .norm import SNorm, TNorm
from .operation import Operation as Op
from .variable import InputVariable, OutputVariable


class Expression(object):
    pass


class Proposition(Expression):
    __slots__ = ["variable", "hedges", "term"]

    def __init__(self):
        self.variable = None
        self.hedges = []
        self.term = None

    def __str__(self):
        result = []

        result.append(self.variable.name if self.variable else "?")

        result.append(Rule.IS)

        if self.hedges:
            for hedge in self.hedges:
                result.append(hedge.name)

        result.append(self.term.name if self.term else "?")

        return " ".join(result)


class Operator(Expression):
    __slots__ = ["name", "left", "right"]

    def __init__(self):
        self.name = ""
        self.left = None
        self.right = None

    def __str__(self):
        return self.name


class Antecedent(object):
    __slots__ = ["text", "expression"]

    def __init__(self):
        self.text = ""
        self.expression = None

    def is_loaded(self) -> bool:
        return bool(self.expression)

    def activation_degree(self, conjunction: TNorm, disjunction: SNorm) -> float:
        if not self.is_loaded():
            raise ValueError(f"antecedent <{self.text}> is not loaded")

        return self._activation_degree(conjunction, disjunction, self.expression)

    def _activation_degree(self, conjunction: TNorm, disjunction: SNorm, node: Expression):
        if not node:
            raise ValueError("expected an expression node, but found none")

        if isinstance(node, Proposition):
            proposition = node
            if not proposition.variable.enabled:
                return 0.0

            if proposition.hedges:
                # if last hedge is "Any", apply hedges in reverse order and return degree
                if isinstance(proposition.hedges[-1], Any):
                    result = nan
                    for hedge in reversed(proposition.hedges):
                        result = hedge.hedge(result)
                    return result

            result = nan
            if isinstance(proposition.variable, InputVariable):
                result = node.term.membership(proposition.variable.value)
            elif isinstance(proposition.variable, OutputVariable):
                result = node.variable.fuzzy.activation_degree(proposition.term)

            if proposition.hedges:
                for hedge in reversed(proposition.hedges):
                    result = hedge.hedge(result)

            return result

        if isinstance(node, Operator):
            operator = node
            if not (operator.left and operator.right):
                raise ValueError("expected left and right operands")

            if operator.name == Rule.AND:
                if not conjunction:
                    raise ValueError(f"rule requires a conjunction operator: '{self.text}'")
                return conjunction.compute(self._activation_degree(conjunction, disjunction, operator.left),
                                           self._activation_degree(conjunction, disjunction, operator.right))

            if operator.name == Rule.OR:
                if not disjunction:
                    raise ValueError(f"rule requires a disjunction operator: '{self.text}'")
                return disjunction.compute(self._activation_degree(conjunction, disjunction, operator.left),
                                           self._activation_degree(conjunction, disjunction, operator.right))

            raise ValueError(f"operator <{operator.name}> not recognized")

        raise ValueError(f"expected a Proposition or an Operator, but found <{str(node)}>")

    def unload(self):
        self.expression = None

    def load(self, engine: 'Engine'):
        self.load(self.text, engine)

    def load(self, antecedent: str, engine: 'Engine'):
        logging.debug(f"Antecedent: {antecedent}")
        self.unload()
        self.text = antecedent
        if not antecedent:
            raise ValueError("expected the antecedent of a rule, but found none")


class Consequent(object):
    pass


class Rule(object):
    __slots__ = ["enabled", "weight", "activation_degree", "triggered", "antecedent", "consequent"]

    IF = 'if'
    IS = 'is'
    THEN = 'then'
    AND = 'and'
    OR = 'or'
    WITH = 'with'

    def __init__(self):
        self.enabled = True
        self.weight = 1.0
        self.activation_degree = 0.0
        self.triggered = False
        self.antecedent = None
        self.consequent = None

    def text(self) -> str:
        result = [Rule.IF]
        if self.antecedent:
            result.append(self.antecedent.text)
        result.append(Rule.THEN)
        if self.consequent:
            result.append(self.consequent.text)
        if self.weight != 1.0:
            result.append(Rule.WITH)
            result.append(Op.str(self.weight))
        return " ".join(result)

    def deactivate(self):
        pass

    def activate_with(self, conjunction: TNorm, disjunction: SNorm):
        pass

    def trigger(self, implication: TNorm):
        pass

    def is_triggered(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return (self.antecedent and self.consequent and
                self.antecedent.is_loaded() and self.consequent.is_loaded())

    def unload(self) -> None:
        self.deactivate()
        if self.antecedent: self.antecedent.unload()
        if self.consequent: self.consequent.unload()

    def load(self, rule: str, engine: "Engine"):
        pass

    def load(self, rule: str):
        pass

    @staticmethod
    def parse(text: str, engine: "Engine") -> "Rule":
        rule = Rule()
        rule.load(text, engine)
        return rule


class RuleBlock(object):
    __slots__ = ["name", "description", "enabled", "conjunction", "disjunction", "implication", "activation", "rules"]

    def __init__(self, name: str = "", description: str = "", enabled: bool = True,
                 conjunction: TNorm = None, disjunction: SNorm = None,
                 implication: TNorm = None, activation: "Activation" = None,
                 rules: Iterable["Rule"] = None):
        self.name = name
        self.description = description
        self.enabled = enabled
        self.conjunction = conjunction
        self.disjunction = disjunction
        self.implication = implication
        self.activation = activation
        self.rules = []
        if rules:
            self.rules.extend(rules)

    def __str__(self):
        return FllExporter().rule_block(self)

    def unload_rules(self):
        pass

    def load_rules(self):
        pass

    def reload_rules(self):
        pass
