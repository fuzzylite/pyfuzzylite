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

__all__ = [
    "Expression",
    "Proposition",
    "Operator",
    "Antecedent",
    "Consequent",
    "Rule",
    "RuleBlock",
]

import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .exporter import FllExporter
from .hedge import Any
from .library import array, nan, representation, scalar, settings
from .norm import SNorm, TNorm
from .operation import Op
from .variable import InputVariable, OutputVariable

if typing.TYPE_CHECKING:
    from .activation import Activation
    from .engine import Engine
    from .hedge import Hedge
    from .term import Term
    from .types import Scalar
    from .variable import Variable


class Expression(ABC):
    """The Expression class is the base class to build an expression tree.
    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Consequent
    @see Rule
    @since 4.0.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Create the expression."""
        pass


class Proposition(Expression):
    """The Proposition class is an Expression that represents a terminal node in
    the expression tree as `variable is [hedge]* term`.
    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Consequent
    @see Rule
    @since 4.0.
    """

    def __init__(
        self,
        variable: Variable | None = None,
        hedges: Iterable[Hedge] | None = None,
        term: Term | None = None,
    ) -> None:
        """Create the proposition.
        @param variable is the variable in the proposition
        @param hedges is the list of hedges that apply to the term of the variable
        @param term is the term in the proposition.
        """
        self.variable = variable
        self.hedges: list[Hedge] = []
        if hedges:
            self.hedges.extend(hedges)
        self.term = term

    def __str__(self) -> str:
        """Returns a string representation of the proposition
        @return a string representation of the proposition.
        """
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
    """The Operator class is an Expression that represents a non-terminal node
    in the expression tree as a binary operator (i.e., `and` or `or`) on two
    Expression nodes.
    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Consequent
    @see Rule
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        right: Expression | None = None,
        left: Expression | None = None,
    ) -> None:
        """Create operator with the given parameters.
        @param name is the name of the operator
        @param left is the left expression in the binary tree
        @param right is the right expression in the binary tree.
        """
        self.name = name
        self.right = right
        self.left = left

    def __str__(self) -> str:
        """Returns the name of the operator
        @return the name of the operator.
        """
        return self.name


class Antecedent:
    """The Antecedent class is an expression tree that represents and evaluates
     the antecedent of a Rule. The structure of a rule is: `if (antecedent)
     then (consequent)`. The structure of the antecedent of a rule is:
    `if variable is [hedge]* term [(and|or) variable is [hedge]* term]*`
     where `*`-marked elements may appear zero or more times, elements in
     brackets are optional, and elements in parentheses are compulsory.
     @author Juan Rada-Vilela, Ph.D.
     @see Consequent
     @see Rule
     @since 4.0.
    """

    def __init__(self, text: str = "") -> None:
        """Create antecedent from the text.
        @param text is the text of the antecedent.
        """
        self.text = text
        self.expression: Expression | None = None

    def __str__(self) -> str:
        """Return the text of the antecedent."""
        return self.text

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        fields = vars(self).copy()
        fields.pop("expression")
        return representation.as_constructor(self, fields)

    def is_loaded(self) -> bool:
        """Indicates whether the antecedent is loaded
        @return whether the antecedent is loaded.
        """
        return bool(self.expression)

    def unload(self) -> None:
        """Unloads the antecedent."""
        self.expression = None

    def activation_degree(
        self,
        conjunction: TNorm | None = None,
        disjunction: SNorm | None = None,
        node: Expression | None = None,
    ) -> Scalar:
        """Computes the activation degree of the antecedent on the expression
        tree from the given node
        @param conjunction is the conjunction operator from the RuleBlock
        @param disjunction is the disjunction operator from the RuleBlock
        @param node is a node in the expression tree of the antecedent
        @return the activation degree of the antecedent.
        """
        if not node:
            if self.expression:
                return self.activation_degree(conjunction, disjunction, self.expression)
            raise RuntimeError(f"antecedent '{self.text}' is not loaded")

        # PROPOSITION
        if isinstance(node, Proposition):
            if not node.variable:
                raise ValueError(
                    f"expected a variable in proposition '{node}', "
                    f"but found none in antecedent: '{self.text}'"
                )
            if not node.variable.enabled:
                return scalar(0.0)

            if node.hedges:
                # if last hedge is "Any", apply hedges in reverse order and return degree
                if isinstance(node.hedges[-1], Any):
                    result = scalar(nan)
                    for hedge in reversed(node.hedges):
                        result = hedge.hedge(result)
                    return result

            if not node.term:
                raise ValueError(
                    f"expected a term in proposition '{node}', "
                    f"but found none for antecedent: '{self.text}'"
                )

            result = scalar(nan)
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
                raise ValueError(
                    f"expected left and right operands for operator '{node}' "
                    f"in antecedent: '{self.text}'"
                )

            if node.name == Rule.AND:
                if not conjunction:
                    raise ValueError(
                        f"expected a conjunction operator, "
                        f"but found none for antecedent: '{self.text}'"
                    )
                return conjunction.compute(
                    self.activation_degree(conjunction, disjunction, node.left),
                    self.activation_degree(conjunction, disjunction, node.right),
                )

            if node.name == Rule.OR:
                if not disjunction:
                    raise ValueError(
                        f"expected a disjunction operator, "
                        f"but found none for antecedent: '{self.text}'"
                    )
                return disjunction.compute(
                    self.activation_degree(conjunction, disjunction, node.left),
                    self.activation_degree(conjunction, disjunction, node.right),
                )

            raise ValueError(
                f"operator '{node}' not recognized in antecedent: '{self.text}'"
            )

        raise RuntimeError(f"unexpected type of node '{node}': {type(node)}")

    def load(self, engine: Engine) -> None:
        """Loads the antecedent with the given text and uses the engine to
        identify and retrieve references to the input variables and output
        variables as required
        @param antecedent is the antecedent of the rule in text
        @param engine is the engine from which the rules are part of.
        """
        from collections import deque

        from .term import Function

        self.unload()
        if not self.text:
            raise SyntaxError("expected the antecedent of a rule, but found none")

        postfix = Function.infix_to_postfix(self.text)
        if settings.debugging:
            settings.logger.debug(f"antecedent={self.text}\npostfix={postfix}")

        # Build a proposition tree from the antecedent of a fuzzy rule. The rules are:
        # (1) After a variable comes 'is',
        # (2) After 'is' comes a hedge or a term
        # (3) After a hedge comes a hedge or a term
        # (4) After a term comes a variable or an operator

        # TODO: replace with enum.Flag("State", "VARIABLE IS HEDGE TERM AND_OR".split())
        s_variable, s_is, s_hedge, s_term, s_and_or = (2**i for i in range(5))
        state = s_variable

        stack: deque[Expression] = deque()

        proposition: Proposition | None = None
        variables = {v.name: v for v in engine.variables}
        token: str | None = None
        for token in postfix.split():
            if state & s_variable:
                variable = variables.get(token, None)
                if variable:
                    proposition = Proposition(variable)
                    stack.append(proposition)
                    state = s_is
                    settings.logger.debug(f"token '{token}' is a variable")
                    continue

            if state & s_is:
                if Rule.IS == token:
                    state = s_hedge | s_term
                    settings.logger.debug(f"token '{token}' is a keyword")
                    continue

            if state & s_hedge:
                factory = settings.factory_manager.hedge
                if token in factory:
                    hedge = factory.construct(token)
                    proposition.hedges.append(hedge)  # type: ignore
                    if isinstance(hedge, Any):
                        state = s_variable | s_and_or
                    else:
                        state = s_hedge | s_term
                    settings.logger.debug(f"token '{token} is hedge")
                    continue

            if state & s_term:
                terms = {t.name: t for t in proposition.variable.terms}  # type: ignore
                term = terms.get(token, None)
                if term:
                    proposition.term = term  # type: ignore
                    state = s_variable | s_and_or
                    settings.logger.debug(f"token '{token} is term")
                    continue

            if state & s_and_or:
                if token in {Rule.AND, Rule.OR}:
                    if len(stack) < 2:
                        raise SyntaxError(
                            f"operator '{token}' expects 2 operands, "
                            f"but found {len(stack)}"
                        )
                    operator = Operator(token)
                    operator.right = stack.pop()
                    operator.left = stack.pop()
                    stack.append(operator)
                    state = s_variable | s_and_or
                    settings.logger.debug(
                        f"token '{token} is logical operator '{operator}'"
                    )
                    continue

            # if reached this point, there was an error in the current state
            if state & (s_variable | s_and_or):
                raise SyntaxError(
                    f"expected variable or logical operator, but found '{token}'"
                )

            if state & s_is:
                raise SyntaxError(f"expected keyword '{Rule.IS}', but found '{token}'")

            if state & (s_hedge | s_term):
                raise SyntaxError(f"expected hedge or term, but found '{token}'")

            raise SyntaxError(f"unexpected token '{token}'")

        # check final state for errors (outside of for-loop)
        if not state & (s_variable | s_and_or):  # only acceptable final states
            if state & s_is:
                raise SyntaxError(f"expected keyword '{Rule.IS}' after '{token}'")
            if stack & (s_hedge | s_term):
                raise SyntaxError(f"expected hedge or term, but found '{token}'")

        if len(stack) != 1:
            errors = " ".join(str(element) for element in stack)
            raise SyntaxError(f"unable to parse the following expressions: {errors}")

        self.expression = stack.pop()

    def prefix(self, node: Expression | None = None) -> str:
        """Returns a string represention of the given expression tree utilizing
        prefix notation
        @param node is a node in the expression tree of the antecedent
        @return a string represention of the given expression tree utilizing
        prefix notation.
        """
        if not node:
            if self.expression:
                return self.prefix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: list[str] = [node.name]
            if node.left:
                result.append(self.prefix(node.left))
            if node.right:
                result.append(self.prefix(node.right))
            return " ".join(result)

        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")

    def infix(self, node: Expression | None = None) -> str:
        """Returns a string represention of the given expression tree utilizing
        infix notation
        @param node is a node in the expression tree of the antecedent
        @return a string represention of the given expression tree utilizing
        infix notation.

        """
        # TODO: enclose propositions in parentheses
        if not node:
            if self.expression:
                return self.infix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: list[str] = []
            if node.left:
                result.append(self.infix(node.left))
            result.append(node.name)
            if node.right:
                result.append(self.infix(node.right))
            return " ".join(result)

        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")

    def postfix(self, node: Expression | None = None) -> str:
        """Returns a string represention of the given expression tree utilizing
        postfix notation
        @param node is a node in the expression tree of the antecedent
        @return a string represention of the given expression tree utilizing
        postfix notation.
        """
        if not node:
            if self.expression:
                return self.postfix(self.expression)
            raise RuntimeError(f"antecedent is not loaded in rule: '{self.text}'")

        if isinstance(node, Proposition):
            return str(node)

        if isinstance(node, Operator):
            result: list[str] = []
            if node.left:
                result.append(self.postfix(node.left))
            if node.right:
                result.append(self.postfix(node.right))
            result.append(node.name)
            return " ".join(result)
        raise RuntimeError(f"unexpected instance '{type(node)}': {str(node)}")


class Consequent:
    """The Consequent class is a proposition set that represents and evaluates
    the consequent of a Rule.. The structure of a rule is: `if (antecedent)
    then (consequent)`. The structure of the consequent of a rule is:
    `then variable is [hedge]* term [and variable is [hedge]* term]* [with
    w]?`
    where `*`-marked elements may appear zero or more times, elements in
    brackets are optional, elements in parentheses are compulsory, and
    `?`-marked elements may appear once or not at all.
    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Rule
    @since 4.0.
    """

    def __init__(self, text: str = "") -> None:
        """Create the consequent from the text.
        @param text is the text of the consequent.

        """
        self.text: str = text
        self.conclusions: list[Proposition] = []

    def __str__(self) -> str:
        """Return the text of the consequent."""
        return self.text

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        fields = vars(self).copy()
        fields.pop("conclusions")
        return representation.as_constructor(self, fields)

    def is_loaded(self) -> bool:
        """Indicates whether the consequent is loaded
        @return whether the consequent is loaded.
        """
        return bool(self.conclusions)

    def unload(self) -> None:
        """Unloads the consequent."""
        self.conclusions.clear()

    def modify(self, activation_degree: Scalar, implication: TNorm | None) -> None:
        """Modifies the proposition set according to the activation degree
        (computed in the Antecedent of the Rule) and the implication operator
        (given in the RuleBlock)
        @param activationDegree is the activation degree computed in the
        Antecedent of the Rule
        @param implication is the implication operator configured in the
        RuleBlock.
        """
        from .term import Activated

        if not self.conclusions:
            raise RuntimeError("consequent is not loaded")

        for proposition in self.conclusions:
            if not proposition.variable:
                raise ValueError(
                    f"expected a variable in '{proposition}', "
                    f"but found none in consequent"
                )
            if proposition.variable.enabled:
                for hedge in reversed(proposition.hedges):
                    activation_degree = hedge.hedge(activation_degree)

                if not proposition.term:
                    raise ValueError(
                        f"expected a term in proposition '{proposition}', "
                        f"but found none"
                    )
                activated_term = Activated(
                    proposition.term, activation_degree, implication
                )
                if isinstance(proposition.variable, OutputVariable):
                    proposition.variable.fuzzy.terms.append(activated_term)
                else:
                    raise RuntimeError(
                        f"expected an output variable, but found "
                        f"'{type(proposition.variable)}'"
                    )

    def load(self, engine: Engine) -> None:
        """Loads the consequent with the given text and uses the engine to
        identify and retrieve references to the input variables and output
        variables as required
        @param consequent is the consequent of the rule in text
        @param engine is the engine from which the rules are part of.
        """
        self.unload()
        if not self.text:
            raise SyntaxError("expected the consequent of a rule, but found none")

        if settings.debugging:
            settings.logger.debug(f"consequent={self.text}")

        # Extracts the list of propositions from the consequent
        #  The rules are:
        #  (1) After a variable comes 'is' or '=',
        #  (2) After 'is' comes a hedge or a term
        #  (3) After a hedge comes a hedge or a term
        #  (4) After a term comes operators 'and' or 'with'
        #  (5) After operator 'and' comes a variable
        #  (6) After operator 'with' comes a scalar

        s_variable, s_is, s_hedge, s_term, s_and, s_with = (2**i for i in range(6))
        state = s_variable

        proposition: Proposition | None = None
        conclusions: list[Proposition] = []
        output_variables = {v.name: v for v in engine.output_variables}
        token: str | None = None
        for token in self.text.split():
            if state & s_variable:
                variable = output_variables.get(token, None)
                if variable:
                    proposition = Proposition(variable)
                    conclusions.append(proposition)
                    state = s_is
                    continue

            if state & s_is and Rule.IS == token:
                state = s_hedge | s_term
                continue

            if state & s_hedge:
                factory = settings.factory_manager.hedge
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

            if state & s_and and Rule.AND == token:
                state = s_variable
                continue

            # if reached this point, there was an error:
            if state & s_variable:
                raise SyntaxError(
                    f"consequent expected an output variable, but found '{token}'"
                )
            if state & s_is:
                raise SyntaxError(
                    f"consequent expected keyword '{Rule.IS}', but found '{token}'"
                )
            if state & (s_hedge | s_term):
                raise SyntaxError(
                    f"consequent expected a hedge or term, but found '{token}'"
                )

            raise SyntaxError(f"unexpected token '{token}'")

        # final states
        if not state & (s_and | s_with):
            if state & s_variable:
                raise SyntaxError(
                    f"consequent expected output variable after '{token}'"
                )
            if state & s_is:
                raise SyntaxError(
                    f"consequent expected keyword '{Rule.IS}' after '{token}'"
                )
            if state & (s_hedge | s_term):
                raise SyntaxError(f"consequent expected hedge or term after '{token}' ")

        self.conclusions = conclusions


class Rule:
    r"""The Rule class is a conditional statement that contributes to the control
     of an Engine. Each rule consists of an Antecedent and a Consequent, each
     of which comprises propositions in the form `variable is term`. The
     propositions in the Antecedent can be connected by the conjunctive `and`
     or the disjunctive `or`, both of which are fuzzy logic operators (TNorm
     and SNorm, respectively). Differently, the propositions in the Consequent
     are independent from each other and are separated with a symbolic `and`.
     The Term in any proposition can be preceded by a Hedge that modifies its
     membership function to model cases such as Very, Somewhat, Seldom and
     Not. Additionally, the contribution of a rule to the control of the
     engine can be determined by its weight $w \in [0.0, 1.0]$, which is
     equal to 1.0 if omitted. The structure of a rule is the following: `if
     (antecedent) then (consequent) [with weight]`. The structures of
     the antecedent and the consequent are:
    `if variable is [hedge]* term [(and|or) variable is [hedge]* term]*`
    `then variable is [hedge]* term [and variable is [hedge]* term]* [with w]?`
     where elements in brackets are optional, elements in parentheses are
     compulsory, `*`-marked elements may appear zero or more times, and
    `?`-marked elements may appear once or not at all.
     @author Juan Rada-Vilela, Ph.D.
     @see Antecedent
     @see Consequent
     @see Hedge
     @see RuleBlock
     @since 4.0.
    """

    IF = "if"
    IS = "is"
    THEN = "then"
    AND = "and"
    OR = "or"
    WITH = "with"

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        activation_degree: Scalar = 0.0,
        antecedent: Antecedent | None = None,
        consequent: Consequent | None = None,
    ) -> None:
        """Create the rule."""
        self.enabled = enabled
        self.weight = weight
        self.activation_degree = activation_degree
        self.triggered = array(False)
        self.antecedent = antecedent or Antecedent()
        self.consequent = consequent or Consequent()

    def __str__(self) -> str:
        """Gets a string representation of the rule in the FuzzyLite Language."""
        return FllExporter().rule(self)

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        fields = vars(self).copy()
        fields.pop("triggered")
        return representation.as_constructor(self, fields)

    @property
    def text(self) -> str:
        """Gets the text of the rule
        @return the text of the rule.
        """
        result = [Rule.IF, self.antecedent.text, Rule.THEN, self.consequent.text]
        if not Op.is_close(self.weight, 1.0):
            result.extend([Rule.WITH, Op.str(self.weight)])
        return " ".join(result)

    @text.setter
    def text(self, text: str) -> None:
        """Sets the text of the rule
        @param text is the text of the rule.
        """
        self.parse(text)

    def parse(self, text: str) -> None:
        """Parses and creates a new rule based on the text passed
        @param rule is the rule in text
        @param engine is the engine from which the rule is part of
        @return a new rule parsed from the given text.
        """
        comment_index = text.find("#")
        rule = text if comment_index == -1 else text[0:comment_index]

        antecedent: list[str] = []
        consequent: list[str] = []
        weight = 1.0

        s_begin, s_if, s_then, s_with, s_end = range(5)
        state = s_begin
        for token in rule.split():
            if state == s_begin:
                if token == Rule.IF:
                    state = s_if
                else:
                    raise SyntaxError(
                        f"expected keyword '{Rule.IF}', "
                        f"but found '{token}' in rule '{text}'"
                    )
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
                weight = float(token)
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
        """Deactivates the rule."""
        self.activation_degree = scalar(0.0)
        self.triggered = array(False)

    def activate_with(
        self, conjunction: TNorm | None, disjunction: SNorm | None
    ) -> Scalar:
        """Activates the rule by computing its activation degree using the given
        conjunction and disjunction operators
        @param conjunction is the conjunction operator
        @param disjunction is the disjunction operator
        @return the activation degree of the rule.
        """
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        self.activation_degree = self.weight * self.antecedent.activation_degree(
            conjunction, disjunction
        )
        return self.activation_degree

    def trigger(self, implication: TNorm | None) -> None:
        """Triggers the rule's implication (if the rule is enabled) using the
        given implication operator and the underlying activation degree
        @param implication is the implication operator.
        """
        self.triggered = array(False)
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        if self.enabled:
            self.consequent.modify(self.activation_degree, implication)
            self.triggered = array(self.activation_degree > 0.0)

    def is_loaded(self) -> bool:
        """Indicates whether the rule is loaded
        @return whether the rule is loaded.
        """
        return self.antecedent.is_loaded() and self.consequent.is_loaded()

    def unload(self) -> None:
        """Unloads the rule."""
        self.deactivate()
        self.antecedent.unload()
        self.consequent.unload()

    def load(self, engine: Engine) -> None:
        """Loads the rule with the text from Rule::getText(), and uses the
        engine to identify and retrieve references to the input variables and
        output variables as required
        @param engine is the engine from which the rule is part of.
        """
        self.deactivate()
        self.antecedent.load(engine)
        self.consequent.load(engine)

    @staticmethod
    def create(text: str, engine: Engine | None = None) -> Rule:
        """Create the rule from the text for the engine
        @param text is the text of the rule
        @param engine is the engine.

        """
        rule = Rule()
        rule.parse(text)
        if engine:
            rule.load(engine)
        return rule


class RuleBlock:
    """The RuleBlock class contains a set of Rule%s and fuzzy logic
    operators required to control an Engine.
    @author Juan Rada-Vilela, Ph.D.
    @see Engine
    @see Rule
    @see Antecedent
    @see Consequent
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        conjunction: TNorm | None = None,
        disjunction: SNorm | None = None,
        implication: TNorm | None = None,
        activation: Activation | None = None,
        rules: Iterable[Rule] | None = None,
    ) -> None:
        """Create the rule block.
        @param name is the name of the rule block
        @param description is the description of the rule block
        @param enabled is whether the rule block is enabled
        @param conjunction is the conjunction operator
        @param disjunction is the disjunction operator
        @param implication is the implication operator
        @param activation is the activation method
        @param rules is the list of rules.
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.conjunction = conjunction
        self.disjunction = disjunction
        self.implication = implication
        self.activation = activation
        self.rules = list(rules) if rules else []

    def __str__(self) -> str:
        """Returns a string representation of the rule block in the FuzzyLite
        Language
        @return a string representation of the rule block in the  FuzzyLite
        Language.
        """
        return Op.to_fll(self)

    def __repr__(self) -> str:
        """Returns a string representation of the object in the FuzzyLite Language."""
        return representation.as_constructor(self)

    def activate(self) -> None:
        """Activates the rule block."""
        if not self.activation:
            raise ValueError(
                f"expected an activation method, "
                f"but found none in rule block:\n{str(self)}"
            )
        return self.activation.activate(self)

    def unload_rules(self) -> None:
        """Unloads all the rules in the rule block."""
        for rule in self.rules:
            rule.unload()

    def load_rules(self, engine: Engine) -> None:
        """Loads all the rules into the rule block
        @param engine is the engine where this rule block is registered.
        """
        exceptions: list[str] = []
        for rule in self.rules:
            rule.unload()
            try:
                rule.load(engine)
            except Exception as ex:
                exceptions.append(f"['{str(rule)}']: {str(ex)}")
        if exceptions:
            raise RuntimeError(
                "failed to load the following rules:\n" + "\n".join(exceptions)
            )

    def reload_rules(self, engine: Engine) -> None:
        """Unloads all the rules in the rule block and then loads each rule again
        @param engine is the engine where this rule block is registered.
        """
        self.unload_rules()
        self.load_rules(engine)
