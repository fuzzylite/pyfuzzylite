"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
from collections.abc import Iterable, Iterator
from typing import overload

from .hedge import Any
from .library import array, nan, representation, scalar, settings
from .norm import SNorm, TNorm
from .operation import Op
from .types import Scalar
from .variable import InputVariable, OutputVariable

if typing.TYPE_CHECKING:
    from .activation import Activation
    from .engine import Engine
    from .hedge import Hedge
    from .term import Term
    from .variable import Variable


class Expression(ABC):
    """Base class to build an expression tree.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
    """

    @abstractmethod
    def __init__(self) -> None:
        """Constructor."""


class Proposition(Expression):
    """Expression that represents a terminal node in the expression tree as `variable is [hedge]* term`.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
    """

    def __init__(
        self,
        variable: Variable | None = None,
        hedges: Iterable[Hedge] | None = None,
        term: Term | None = None,
    ) -> None:
        """Constructor.

        Args:
            variable: variable in the proposition
            hedges: list of hedges that apply to the term of the variable
            term: term in the proposition.
        """
        self.variable = variable
        self.hedges: list[Hedge] = []
        if hedges:
            self.hedges.extend(hedges)
        self.term = term

    def __str__(self) -> str:
        """Return proposition as text.

        Returns:
            proposition as text.
        """
        result = []

        if self.variable is not None:
            result.append(self.variable.name)
            result.append(Rule.IS)

        if self.hedges:
            for hedge in self.hedges:
                result.append(hedge.name)

        if self.term is not None:
            result.append(self.term.name)

        return " ".join(result)


class Operator(Expression):
    """Expression that represents a non-terminal node in the expression tree as a binary operator (i.e., `and` or `or`) on two Expression nodes.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
    """

    def __init__(
        self,
        name: str = "",
        right: Expression | None = None,
        left: Expression | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: name of the operator
            left: left expression in the binary tree
            right: right expression in the binary tree.
        """
        self.name = name
        self.right = right
        self.left = left

    def __str__(self) -> str:
        """Return the name of the operator.

        Returns:
            name of the operator.
        """
        return self.name


class Antecedent:
    """Expression tree that represents and evaluates the antecedent of a rule.

    info: structure
        The structure of a rule is: <br/>
        ```if (antecedent) then (consequent)```

        The structure of the antecedent of a rule is: <br/>
        ```if variable is [hedge]* term [(and|or) variable is [hedge]* term]*```

        ---

        `*`-marked elements may appear zero or more times, <br/>
        elements in brackets are optional, and <br/>
        elements in parentheses are compulsory.

    info: related
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
    """

    def __init__(self, text: str = "") -> None:
        """Constructor.

        Args:
            text: antecedent as text.
        """
        self.text = text
        self.expression: Expression | None = None

    def __str__(self) -> str:
        """Return the antecedent as text.

        Returns:
            antecedent as text
        """
        return self.text

    def __repr__(self) -> str:
        """Return the code to construct the antecedent in Python.

        Returns:
            code to construct the antecedent in Python.
        """
        fields = vars(self).copy()
        fields.pop("expression")
        return representation.as_constructor(self, fields, positional=True)

    def is_loaded(self) -> bool:
        """Return whether the antecedent is loaded.

        Returns:
             antecedent is loaded.
        """
        return bool(self.expression)

    def unload(self) -> None:
        """Unload the antecedent."""
        self.expression = None

    def activation_degree(
        self,
        conjunction: TNorm | None = None,
        disjunction: SNorm | None = None,
        node: Expression | None = None,
    ) -> Scalar:
        """Compute the activation degree of the antecedent on the expression tree from the given node.

        Args:
            conjunction: conjunction operator from the rule block
            disjunction: disjunction operator from the rule block
            node: node in the expression tree of the antecedent

        Returns:
             activation degree of the antecedent.
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

            raise ValueError(f"operator '{node}' not recognized in antecedent: '{self.text}'")

        raise RuntimeError(f"unexpected type of node '{node}': {type(node)}")

    def load(self, engine: Engine) -> None:
        """Load the antecedent using the engine to identify and get references to the input and output variables.

        Args:
            engine: engine to get references in the antecedent.
        """
        from collections import deque

        from .term import Function

        self.unload()
        if not self.text:
            raise SyntaxError("expected the antecedent of a rule, but found none")

        postfix = Function.infix_to_postfix(self.text)

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
                variable = variables.get(token)
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
                    state = s_variable | s_and_or if isinstance(hedge, Any) else s_hedge | s_term
                    settings.logger.debug(f"token '{token} is hedge")
                    continue

            if state & s_term:
                terms = {t.name: t for t in proposition.variable.terms}  # type: ignore
                term = terms.get(token)
                if term:
                    proposition.term = term  # type: ignore
                    state = s_variable | s_and_or
                    settings.logger.debug(f"token '{token} is term")
                    continue

            if state & s_and_or:
                if token in {Rule.AND, Rule.OR}:
                    if len(stack) < 2:
                        raise SyntaxError(
                            f"operator '{token}' expects 2 operands, but found {len(stack)}"
                        )
                    operator = Operator(token)
                    operator.right = stack.pop()
                    operator.left = stack.pop()
                    stack.append(operator)
                    state = s_variable | s_and_or
                    settings.logger.debug(f"token '{token} is logical operator '{operator}'")
                    continue

            # if reached this point, there was an error in the current state
            if state & (s_variable | s_and_or):
                raise SyntaxError(f"expected variable or logical operator, but found '{token}'")

            if state & s_is:
                raise SyntaxError(f"expected keyword '{Rule.IS}', but found '{token}'")

            if state & (s_hedge | s_term):
                raise SyntaxError(f"expected hedge or term, but found '{token}'")

            raise SyntaxError(f"unexpected token '{token}'")

        # check final state for errors (outside for-loop)
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
        """Return the prefix notation of the node.

        Args:
             node: node in the expression tree of the antecedent

        Returns:
             prefix notation of the node.
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
        """Return the infix notation of the node.

        Args:
             node: node in the expression tree of the antecedent

        Returns:
             infix notation of the node.
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
        """Return the postfix notation of the node.

        Args:
             node: node in the expression tree of the antecedent

        Returns:
             postfix notation of the node.
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
    """Proposition set that represents and evaluates the consequent of a rule.

    info: structure
        The structure of a rule is: <br/>
        ```if (antecedent) then (consequent)```

        The structure of the consequent of a rule is: <br/>
        ```then variable is [hedge]* term [and variable is [hedge]* term]* [with w]?```

        ---

        `*`-marked elements may appear zero or more times, <br/>
        elements in brackets are optional, <br/>
        elements in parentheses are compulsory, and <br/>
        `?`-marked elements may appear once or not at all.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Rule][]
    """

    def __init__(self, text: str = "") -> None:
        """Constructor.

        text: consequent as text.

        """
        self.text: str = text
        self.conclusions: list[Proposition] = []

    def __str__(self) -> str:
        """Return the consequent as text.

        Returns:
            consequent as text
        """
        return self.text

    def __repr__(self) -> str:
        """Return the code to construct the consequent in Python.

        Returns:
            code to construct the consequent in Python.
        """
        fields = vars(self).copy()
        fields.pop("conclusions")
        return representation.as_constructor(self, fields, positional=True)

    def is_loaded(self) -> bool:
        """Return whether the consequent is loaded.

        Returns:
             consequent is loaded.
        """
        return bool(self.conclusions)

    def unload(self) -> None:
        """Unload the consequent."""
        self.conclusions.clear()

    def modify(self, activation_degree: Scalar, implication: TNorm | None) -> None:
        """Modify the consequent with the activation degree and the implication operator.

        Args:
            activation_degree: activation degree computed in the antecedent of the rule
            implication: implication operator configured in the rule block.
        """
        from .term import Activated

        if not self.conclusions:
            raise RuntimeError("consequent is not loaded")

        for proposition in self.conclusions:
            if not proposition.variable:
                raise ValueError(
                    f"expected a variable in '{proposition}', but found none in consequent"
                )
            if proposition.variable.enabled:
                for hedge in reversed(proposition.hedges):
                    activation_degree = hedge.hedge(activation_degree)

                if not proposition.term:
                    raise ValueError(
                        f"expected a term in proposition '{proposition}', but found none"
                    )
                activated_term = Activated(proposition.term, activation_degree, implication)
                if isinstance(proposition.variable, OutputVariable):
                    proposition.variable.fuzzy.terms.append(activated_term)
                else:
                    raise RuntimeError(
                        f"expected an output variable, but found '{type(proposition.variable)}'"
                    )

    def load(self, engine: Engine) -> None:
        """Load the consequent using the engine to identify and get references to the input and output variables.

        Args:
            engine: engine to get references in the consequent.
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
        #  (6) After operator 'with' comes a float

        s_variable, s_is, s_hedge, s_term, s_and, s_with = (2**i for i in range(6))
        state = s_variable

        proposition: Proposition | None = None
        conclusions: list[Proposition] = []
        output_variables = {v.name: v for v in engine.output_variables}
        token: str | None = None
        for token in self.text.split():
            if state & s_variable:
                variable = output_variables.get(token)
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
                term = terms.get(token)
                if term:
                    proposition.term = term  # type: ignore
                    state = s_and | s_with
                    continue

            if state & s_and and Rule.AND == token:
                state = s_variable
                continue

            # if reached this point, there was an error:
            if state & s_variable:
                raise SyntaxError(f"consequent expected an output variable, but found '{token}'")
            if state & s_is:
                raise SyntaxError(f"consequent expected keyword '{Rule.IS}', but found '{token}'")
            if state & (s_hedge | s_term):
                raise SyntaxError(f"consequent expected a hedge or term, but found '{token}'")

            raise SyntaxError(f"unexpected token '{token}'")

        # final states
        if not state & (s_and | s_with):
            if state & s_variable:
                raise SyntaxError(f"consequent expected output variable after '{token}'")
            if state & s_is:
                raise SyntaxError(f"consequent expected keyword '{Rule.IS}' after '{token}'")
            if state & (s_hedge | s_term):
                raise SyntaxError(f"consequent expected hedge or term after '{token}' ")

        self.conclusions = conclusions


class Rule:
    r"""Conditional statement that contributes to the control of an Engine.

    A rule consists of an Antecedent and a Consequent, each with propositions in the form `variable is term`.

    The propositions in the Antecedent are connected by the conjunctive `and` or the disjunctive `or`,
    which are fuzzy logic operators represented as TNorm and SNorm (respectively).

    The propositions in the Consequent are independent and separated by a symbolic `and`.

    The term in any proposition can be preceded by a hedge that modifies its membership function value of the term.

    The contribution of a rule to the control of the engine can be determined by its weight $w \in [0.0, 1.0]$,
    which is equal to 1.0 if omitted.

    info: structure
        The structure of a rule is: <br/>
        ```if (antecedent) then (consequent) [with weight]```

        The structure of the antecedent is: <br/>
            ```if variable is [hedge]* term [(and|or) variable is [hedge]* term]*```

        The structure of the consequent is: <br/>
            ```then variable is [hedge]* term [and variable is [hedge]* term]* [with w]?```

        ---
        where elements in brackets are optional,
        elements in parentheses are compulsory,
        `*`-marked elements may appear zero or more times, and
        `?`-marked elements may appear once or not at all.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.rule.RuleBlock][]
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
        antecedent: Antecedent | None = None,
        consequent: Consequent | None = None,
    ) -> None:
        """Constructor.

        Args:
            enabled: enable the rule
            weight: weight of the rule
            antecedent: antecedent of the rule
            consequent: consequent of the rule
        """
        self.enabled = enabled
        self.weight = weight
        self.activation_degree = scalar(0.0)
        self.triggered = array(False)
        self.antecedent = antecedent or Antecedent()
        self.consequent = consequent or Consequent()

    def __str__(self) -> str:
        """Return the code to construct the rule in the FuzzyLite Language.

        Returns:
            code to construct the rule in the FuzzyLite Language.
        """
        return representation.fll.rule(self)

    def __repr__(self) -> str:
        """Return the code to construct the rule in Python.

        Returns:
            code to construct the rule in Python.
        """
        return f"{Op.class_name(self, qualname=True)}.{Rule.create.__name__}('{self.text}')"

    @property
    def text(self) -> str:
        """Get/Set the rule as text.

        # Getter

        Returns:
            rule as text

        # Setter

        Args:
            text (str): rule as text

        """
        result = [Rule.IF, self.antecedent.text, Rule.THEN, self.consequent.text]
        if not Op.is_close(self.weight, 1.0):
            result.extend([Rule.WITH, Op.str(self.weight)])
        return " ".join(result)

    @text.setter
    def text(self, text: str) -> None:
        """Set the rule as text.

        Args:
            text (str): rule as text
        """
        self.parse(text)

    def parse(self, text: str) -> None:
        """Parse and load the rule based on the text.

        Args:
            text: rule as text.
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
                        f"expected keyword '{Rule.IF}', but found '{token}' in rule '{text}'"
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
        """Deactivate the rule."""
        self.activation_degree = scalar(0.0)
        self.triggered = array(False)

    def activate_with(self, conjunction: TNorm | None, disjunction: SNorm | None) -> Scalar:
        """Compute and set activation degree of the rule with the conjunction and disjunction operators.

        Args:
            conjunction: conjunction operator
            disjunction: disjunction operator

        Returns:
             activation degree of the rule.
        """
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        self.activation_degree = self.weight * self.antecedent.activation_degree(
            conjunction, disjunction
        )
        return self.activation_degree

    def trigger(self, implication: TNorm | None) -> None:
        """Trigger the rule using the implication operator and the previously computed activation degree.

        Args:
            implication: implication operator.

        Raises:
            RuntimeError: when the rule is not loaded
        """
        self.triggered = array(False)
        if not self.is_loaded():
            raise RuntimeError(f"rule is not loaded: '{self.text}'")
        if self.enabled:
            self.consequent.modify(self.activation_degree, implication)
            self.triggered = array(self.activation_degree > 0.0)

    def is_loaded(self) -> bool:
        """Return whether the rule is loaded.

        Returns:
             rule is loaded.
        """
        return self.antecedent.is_loaded() and self.consequent.is_loaded()

    def unload(self) -> None:
        """Unload the rule."""
        self.deactivate()
        self.antecedent.unload()
        self.consequent.unload()

    def load(self, engine: Engine) -> None:
        """Load the rule using the engine to identify and get references to the input and output variables.

        Args:
            engine: engine that the rule (partially) controls
        """
        self.deactivate()
        self.antecedent.load(engine)
        self.consequent.load(engine)

    @staticmethod
    def create(text: str, engine: Engine | None = None) -> Rule:
        """Create rule from the text.

        Args:
            text: rule as text
            engine: engine that the rule (partially) controls
        """
        rule = Rule()
        rule.parse(text)
        if engine:
            rule.load(engine)
        return rule


class RuleBlock:
    """Block of rules and fuzzy logic operators required to control an engine.

    info: related
        - [fuzzylite.engine.Engine][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.activation.Activation][]
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
        """Constructor.

        Args:
            name: name of the rule block
            description: description of the rule block
            enabled: enable the rule block
            conjunction: conjunction operator
            disjunction: disjunction operator
            implication: implication operator
            activation: activation method
            rules: list of rules.
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.conjunction = conjunction
        self.disjunction = disjunction
        self.implication = implication
        self.activation = activation
        self.rules = list(rules or [])

    def __iter__(self) -> Iterator[Rule]:
        """Return the iterator of the rules.

        Returns:
            iterator of the rules
        """
        return iter(self.rules)

    @overload
    def __getitem__(self, item: int) -> Rule: ...

    @overload
    def __getitem__(self, item: slice) -> list[Rule]: ...

    def __getitem__(self, item: int | slice) -> Rule | list[Rule]:
        """Allow indexing rules in rule block (eg, `rule_block[0]`).

        Args:
            item: rule index or slice

        Returns:
            rule at index or slice of rules
        """
        return self.rules[item]

    def __len__(self) -> int:
        """Return the number of rules.

        Returns:
            number of rules
        """
        return len(self.rules)

    def __str__(self) -> str:
        """Return the code to construct the rule block in the FuzzyLite Language.

        Returns:
            code to construct the rule block in the FuzzyLite Language.
        """
        return representation.fll.rule_block(self)

    def __repr__(self) -> str:
        """Return the code to construct the rule block in Python.

        Returns:
            code to construct the rule block in Python.
        """
        fields = vars(self).copy()
        if not self.description:
            fields.pop("description")
        if self.enabled:
            fields.pop("enabled")
        return representation.as_constructor(self, fields)

    def activate(self) -> None:
        """Activate the rule block."""
        if not self.activation:
            raise ValueError(
                f"expected an activation method, but found none in rule block:\n{str(self)}"
            )
        return self.activation.activate(self)

    def unload_rules(self) -> None:
        """Unload all the rules in the rule block."""
        for rule in self.rules:
            rule.unload()

    def load_rules(self, engine: Engine) -> None:
        """Load all the rules in the rule block.

        Args:
            engine: engine where this rule block is registered.
        """
        exceptions: list[str] = []
        for rule in self.rules:
            rule.unload()
            try:
                rule.load(engine)
            except Exception as ex:
                exceptions.append(f"['{str(rule)}']: {str(ex)}")
        if exceptions:
            raise RuntimeError("failed to load the following rules:\n" + "\n".join(exceptions))

    def reload_rules(self, engine: Engine) -> None:
        """Reload all the rules in the rule block.

        Args:
            engine: engine where this rule block is registered.
        """
        self.unload_rules()
        self.load_rules(engine)

    def rule(self, index: int, /) -> Rule:
        """Get the rule at the index.

        Args:
            index: index of the rule.

        Returns:
            rule at the index
        """
        return self.rules[index]
