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

__all__ = ["Expression", "Proposition", "Operator", "Antecedent", "Consequent", "Rule", "RuleBlock"]

import typing
from math import nan
from typing import Deque, Iterable, List, Optional

from .exporter import FllExporter
from .hedge import Any
from .norm import SNorm, TNorm
from .operation import Op
from .variable import InputVariable, OutputVariable

if typing.TYPE_CHECKING:
    from .activation import Activation  # noqa: F401
    from .engine import Engine
    from .hedge import Hedge
    from .term import Term  # noqa: F401
    from .variable import Variable  # noqa: F401


class Expression:
    pass


class Proposition(Expression):

    def __init__(self, variable: Optional['Variable'] = None,
                 hedges: Optional[Iterable['Hedge']] = None,
                 term: Optional['Term'] = None) -> None:
        self.variable = variable
        self.hedges: List[Hedge] = []
        if hedges:
            self.hedges.extend(hedges)
        self.term = term

    def __str__(self) -> str:
        result = []

        if self.variable:
            result.append(self.variable.name)

            result.append(Rule.IS)

        if self.hedges:
            for hedge in self.hedges:
                result.append(hedge.name)

        if self.term:
            result.append(self.term.name)

        return " ".join(result)


class Operator(Expression):

    def __init__(self, name: str = "",
                 right: Optional[Expression] = None,
                 left: Optional[Expression] = None) -> None:
        self.name = name
        self.right = right
        self.left = left

    def __str__(self) -> str:
        return self.name


class Antecedent(object):

    def __init__(self, text: str = "") -> None:
        self.text: str = text
        self.expression: Optional[Expression] = None

    def __str__(self) -> str:
        return self.text

    def is_loaded(self) -> bool:
        return bool(self.expression)

    def unload(self) -> None:
        self.expression = None

    def activation_degree(self,  # noqa C901 'Antecedent.activation_degree' is too complex (20)
                          conjunction: Optional[TNorm] = None,
                          disjunction: Optional[SNorm] = None,
                          node: Optional[Expression] = None) -> float:
        if not node:
            if self.expression:
                return self.activation_degree(conjunction, disjunction, self.expression)
            raise RuntimeError(f"antecedent '{self.text}' is not loaded")

        # PROPOSITION
        if isinstance(node, Proposition):
            if not node.variable:
                raise ValueError(f"expected a variable in proposition '{node}', "
                                 f"but found none in antecedent: '{self.text}'")
            if not node.variable.enabled:
                return 0.0

            if node.hedges:
                # if last hedge is "Any", apply hedges in reverse order and return degree
                if isinstance(node.hedges[-1], Any):
                    result = nan
                    for hedge in reversed(node.hedges):
                        result = hedge.hedge(result)
                    return result

            if not node.term:
                raise ValueError(f"expected a term in proposition '{node}', "
                                 f"but found none for antecedent: '{self.text}'")

            result = nan
            if isinstance(node.variable, InputVariable):
                result = node.term.membership(node.variable.value)
            elif isinstance(node.variable, OutputVariable):
                result = node.variable.fuzzy.activation_degree(node.term)

            for hedge in reversed(node.hedges):
                result = hedge.hedge(result)

            return result

        # OPERATOR
        if isinstance(node, Operator):
            if not (node.left and node.right):
                raise ValueError(f"expected left and right operands for operator '{node}' "
                                 f"in antecedent: '{self.text}'")

            if node.name == Rule.AND:
                if not conjunction:
                    raise ValueError(f"expected a conjunction operator, "
                                     f"but found none for antecedent: '{self.text}'")
                return conjunction.compute(
                    self.activation_degree(conjunction, disjunction, node.left),
                    self.activation_degree(conjunction, disjunction, node.right))

            if node.name == Rule.OR:
                if not disjunction:
                    raise ValueError(f"expected a disjunction operator, "
                                     f"but found none for antecedent: '{self.text}'")
                return disjunction.compute(
                    self.activation_degree(conjunction, disjunction, node.left),
                    self.activation_degree(conjunction, disjunction, node.right))

            raise ValueError(f"operator '{node}' not recognized in antecedent: '{self.text}'")

        raise RuntimeError(f"unexpected type of node '{node}': {type(node)}")

    def load(self, engine: 'Engine') -> None:  # noqa: C901 'Antecedent.load' is too complex (23)
        from collections import deque
        from . import lib
        from .term import Function

        self.unload()
        if not self.text:
            raise SyntaxError("expected the antecedent of a rule, but found none")

        postfix = Function().infix_to_postfix(self.text)
        if lib.debugging:
            lib.logger.debug(f"antecedent={self.text}\npostfix={postfix}")

        # Build a proposition tree from the antecedent of a fuzzy rule. The rules are:
        # (1) After a variable comes 'is',
        # (2) After 'is' comes a hedge or a term
        # (3) After a hedge comes a hedge or a term
        # (4) After a term comes a variable or an operator

        s_variable, s_is, s_hedge, s_term, s_and_or = (2 ** i for i in range(5))
        state = s_variable

        stack: Deque[Expression] = deque()

        proposition: Optional[Proposition] = None
        variables = {v.name: v for v in engine.variables}
        for token in postfix.split():
            if state & s_variable:
                variable = variables.get(token, None)
                if variable:
                    proposition = Proposition(variable)
                    stack.append(proposition)
                    state = s_is
                    lib.logger.debug(f"token '{token}' is a variable")
                    continue

            if state & s_is:
                if Rule.IS == token:
                    state = s_hedge | s_term
                    lib.logger.debug(f"token '{token}' is a keyword")
                    continue

            if state & s_hedge:
                factory = lib.factory_manager.hedge
                if token in factory:
                    hedge = factory.construct(token)
                    proposition.hedges.append(hedge)  # type: ignore
                    if isinstance(hedge, Any):
                        state = s_variable | s_and_or
                    else:
                        state = s_hedge | s_term
                    lib.logger.debug(f"token '{token} is hedge")
                    continue

            if state & s_term:
                terms = {t.name: t for t in proposition.variable.terms}  # type: ignore
                term = terms.get(token, None)
                if term:
                    proposition.term = term  # type: ignore
                    state = s_variable | s_and_or
                    lib.logger.debug(f"token '{token} is term")
                    continue

            if state & s_and_or:
                if token in {Rule.AND, Rule.OR}:
                    if len(stack) < 2:
                        raise SyntaxError(f"operator '{token}' expects 2 operands, "
                                          f"but found {len(stack)}")
                    operator = Operator(token)
                    operator.right = stack.pop()
                    operator.left = stack.pop()
                    stack.append(operator)
                    state = s_variable | s_and_or
                    lib.logger.debug(f"token '{token} is logical operator '{operator}'")
                    continue

            # if reached this point, there was an error in the current state
            if state & (s_variable | s_and_or):
                raise SyntaxError(f"expected variable or logical operator, but found '{token}'")

            if state & s_is:
                raise SyntaxError(f"expected keyword '{Rule.IS}', but found '{token}'")

            if state & (s_hedge | s_term):
                raise SyntaxError(f"expected hedge or term, but found '{token}'")

            raise SyntaxError(f"unexpected token '{token}'")

        # check final state for errors (outside of for-loop)
        if not (state & (s_variable | s_and_or)):  # only acceptable final states
            if state & s_is:
                raise SyntaxError(f"expected keyword '{Rule.IS}' after '{token}'")
            if stack & (s_hedge | s_term):
                raise SyntaxError(f"expected hedge or term, but found '{token}'")

        if len(stack) != 1:
            errors = " ".join(str(element) for element in stack)
            raise SyntaxError(f"unable to parse the following expressions: {errors}")

        self.expression = stack.pop()

    def prefix(self, node: Optional[Expression] = None) -> str:
        if not node:
            if self.expression:
                return self.prefix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: List[str] = [node.name]
            if node.left:
                result.append(self.prefix(node.left))
            if node.right:
                result.append(self.prefix(node.right))
            return " ".join(result)

        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")

    def infix(self, node: Optional[Expression] = None) -> str:
        # TODO: enclose propositions in parentheses
        if not node:
            if self.expression:
                return self.infix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: List[str] = []
            if node.left:
                result.append(self.infix(node.left))
            result.append(node.name)
            if node.right:
                result.append(self.infix(node.right))
            return " ".join(result)

        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")

    def postfix(self, node: Optional[Expression] = None) -> str:
        if not node:
            if self.expression:
                return self.postfix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: List[str] = []
            if node.left:
                result.append(self.postfix(node.left))
            if node.right:
                result.append(self.postfix(node.right))
            result.append(node.name)
            return " ".join(result)
        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")


class Consequent:

    def __init__(self, text: str = "") -> None:
        self.text: str = text
        self.conclusions: List[Proposition] = []

    def __str__(self) -> str:
        return self.text

    def is_loaded(self) -> bool:
        return bool(self.conclusions)

    def unload(self) -> None:
        self.conclusions.clear()

    def modify(self, activation_degree: float, implication: Optional[TNorm]) -> None:
        from .term import Activated

        if not self.conclusions:
            raise RuntimeError(f"consequent is not loaded")

        for proposition in self.conclusions:
            if not proposition.variable:
                raise ValueError(f"expected a variable in '{proposition}', "
                                 f"but found none in consequent")
            if proposition.variable.enabled:
                for hedge in reversed(proposition.hedges):
                    # TODO: Revisit because hedging like this stage would decrease the importance
                    # TODO: What about any?
                    activation_degree = hedge.hedge(activation_degree)

                if not proposition.term:
                    raise ValueError(f"expected a term in proposition '{proposition}', "
                                     f"but found none")
                activated_term = Activated(proposition.term, activation_degree, implication)
                if isinstance(proposition.variable, OutputVariable):
                    proposition.variable.fuzzy.terms.append(activated_term)
                else:
                    raise RuntimeError(f"expected an output variable, but found "
                                       f"'{type(proposition.variable)}'")

    def load(self, engine: 'Engine') -> None:  # noqa C901 'Consequent.load' is too complex (21)
        from . import lib

        self.unload()
        if not self.text:
            raise SyntaxError("expected the consequent of a rule, but found none")

        if lib.debugging:
            lib.logger.debug(f"consequent={self.text}")

        # Extracts the list of propositions from the consequent
        #  The rules are:
        #  (1) After a variable comes 'is' or '=',
        #  (2) After 'is' comes a hedge or a term
        #  (3) After a hedge comes a hedge or a term
        #  (4) After a term comes operators 'and' or 'with'
        #  (5) After operator 'and' comes a variable
        #  (6) After operator 'with' comes a float

        s_variable, s_is, s_hedge, s_term, s_and, s_with = (2 ** i for i in range(6))
        state = s_variable

        proposition: Optional[Proposition] = None
        conclusions: List[Proposition] = []
        output_variables = {v.name: v for v in engine.output_variables}
        for token in self.text.split():
            if state & s_variable:
                variable = output_variables.get(token, None)
                if variable:
                    proposition = Proposition(variable)
                    conclusions.append(proposition)
                    state = s_is
                    continue

            if state & s_is:
                if Rule.IS == token:
                    state = s_hedge | s_term
                    continue

            if state & s_hedge:
                factory = lib.factory_manager.hedge
                if token in factory:
                    hedge = factory.construct(token)
                    proposition.hedges.append(hedge)  # type: ignore
                    state = s_hedge | s_term
                    continue

            if state & s_term:
                terms = {t.name: t for t in proposition.variable.terms}  # type: ignore
                term = terms.get(token, None)
                if term:
                    proposition.term = term  # type: ignore
                    state = s_and | s_with
                    continue

            if state & s_and:
                if Rule.AND == token:
                    state = s_variable
                    continue

            # if reached this point, there was an error:
            if state & s_variable:
                raise SyntaxError(f"consequent expected an output variable, "
                                  f"but found '{token}'")
            if state & s_is:
                raise SyntaxError(f"consequent expected keyword '{Rule.IS}', "
                                  f"but found '{token}'")
            if state & (s_hedge | s_term):
                raise SyntaxError(f"consequent expected a hedge or term, "
                                  f"but found '{token}'")

            raise SyntaxError(f"unexpected token '{token}'")

        # final states
        if not (state & (s_and | s_with)):
            if state & s_variable:
                raise SyntaxError(f"consequent expected output variable after '{token}'")
            if state & s_is:
                raise SyntaxError(f"consequent expected keyword '{Rule.IS}' after '{token}'")
            if state & (s_hedge | s_term):
                raise SyntaxError(f"consequent expected hedge or term after '{token}' ")

        self.conclusions = conclusions


class Rule(object):
    IF = 'if'
    IS = 'is'
    THEN = 'then'
    AND = 'and'
    OR = 'or'
    WITH = 'with'

    def __init__(self) -> None:
        self.enabled: bool = True
        self.weight: float = 1.0
        self.activation_degree: float = 0.0
        self.triggered: bool = False
        self.antecedent: Antecedent = Antecedent()
        self.consequent: Consequent = Consequent()

    def __str__(self) -> str:
        return FllExporter().rule(self)

    @property
    def text(self) -> str:
        result = [Rule.IF,
                  self.antecedent.text,
                  Rule.THEN,
                  self.consequent.text]
        if not Op.eq(self.weight, 1.0):
            result.extend([
                Rule.WITH,
                Op.str(self.weight)])
        return " ".join(result)

    @text.setter
    def text(self, text: str) -> None:
        self.parse(text)

    def parse(self, text: str) -> None:
        comment_index = text.find("#")
        rule = text if comment_index == -1 else text[0:comment_index]

        antecedent: List[str] = []
        consequent: List[str] = []
        weight: float = Op.scalar(1.0)

        s_begin, s_if, s_then, s_with, s_end = range(5)
        state = s_begin
        for token in rule.split():
            if state == s_begin:
                if token == Rule.IF:
                    state = s_if
                else:
                    raise SyntaxError(f"expected keyword '{Rule.IF}', "
                                      f"but found '{token}' in rule '{text}'")
            elif state == s_if:
                if token == Rule.THEN:
                    state = s_then
                else:
                    antecedent.append(token)
            elif state == s_then:
                if token == Rule.WITH:
                    state = s_with
                else:
                    consequent.append(token)
            elif state == s_with:
                weight = Op.scalar(token)
                state = s_end
            elif state == s_end:
                raise SyntaxError(f"unexpected token '{token}' in rule '{text}'")
            else:
                raise SyntaxError(f"unexpected state '{state}' in finite state machine")

        if state == s_begin:
            raise SyntaxError(f"expected an if-then rule, but found '{text}'")
        if state == s_if:
            raise SyntaxError(f"expected keyword '{Rule.THEN}' in rule '{text}'")
        if state == s_with:
            raise SyntaxError(f"expected the rule weight in rule '{text}'")

        if not antecedent:
            raise SyntaxError(f"expected an antecedent in rule '{text}'")
        if not consequent:
            raise SyntaxError(f"expected a consequent in rule '{text}'")

        self.antecedent.text = " ".join(antecedent)
        self.consequent.text = " ".join(consequent)
        self.weight = weight

    def deactivate(self) -> None:
        self.activation_degree = 0.0
        self.triggered = False

    def activate_with(self, conjunction: Optional[TNorm], disjunction: Optional[SNorm]) -> float:
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        self.activation_degree = (self.weight
                                  * self.antecedent.activation_degree(conjunction, disjunction))
        return self.activation_degree

    def trigger(self, implication: Optional[TNorm]) -> None:
        self.triggered = False
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        if self.enabled and Op.gt(self.activation_degree, 0.0):
            self.consequent.modify(self.activation_degree, implication)
            self.triggered = True

    def is_loaded(self) -> bool:
        return self.antecedent.is_loaded() and self.consequent.is_loaded()

    def unload(self) -> None:
        self.deactivate()
        self.antecedent.unload()
        self.consequent.unload()

    def load(self, engine: 'Engine') -> None:
        self.deactivate()
        self.antecedent.load(engine)
        self.consequent.load(engine)

    @staticmethod
    def create(text: str, engine: Optional['Engine'] = None) -> 'Rule':
        rule = Rule()
        rule.parse(text)
        if engine:
            rule.load(engine)
        return rule


class RuleBlock:

    def __init__(self,
                 name: str = "",
                 description: str = "",
                 enabled: bool = True,
                 conjunction: Optional[TNorm] = None,
                 disjunction: Optional[SNorm] = None,
                 implication: Optional[TNorm] = None,
                 activation: Optional['Activation'] = None,
                 rules: Optional[Iterable[Rule]] = None) -> None:
        self.name = name
        self.description = description
        self.enabled = enabled
        self.conjunction = conjunction
        self.disjunction = disjunction
        self.implication = implication
        self.activation = activation
        self.rules: List[Rule] = []
        if rules:
            self.rules.extend(rules)

    def __str__(self) -> str:
        return FllExporter().rule_block(self)

    def activate(self) -> None:
        if not self.activation:
            raise ValueError(f"expected an activation method, "
                             f"but found none in rule block:\n{str(self)}")
        return self.activation.activate(self)

    def unload_rules(self) -> None:
        for rule in self.rules:
            rule.unload()

    def load_rules(self, engine: 'Engine') -> None:
        exceptions: List[str] = []  # noqa E701 (False Positive)
        for rule in self.rules:
            rule.unload()
            try:
                rule.load(engine)
            except Exception as ex:
                exceptions.append(f"['{str(rule)}']: {str(ex)}")
        if exceptions:
            raise RuntimeError("failed to load the following rules:\n"
                               + "\n".join(exceptions))

    def reload_rules(self, engine: 'Engine') -> None:
        self.unload_rules()
        self.load_rules(engine)
