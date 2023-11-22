"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited.

fuzzylite is a registered trademark of FuzzyLite Limited.
"""
from __future__ import annotations

__all__ = ["Variable", "InputVariable", "OutputVariable"]

import contextlib
from collections.abc import Iterable, Iterator
from typing import overload

import numpy as np

from .defuzzifier import Defuzzifier
from .library import inf, nan, representation, scalar
from .norm import SNorm
from .operation import Op
from .term import Aggregated, Term
from .types import Scalar


class Variable:
    """Base class for linguistic variables.

    info: related
        - [fuzzylite.variable.InputVariable][]
        - [fuzzylite.variable.OutputVariable][]
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        terms: Iterable[Term] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: name of the variable
            description: description of the variable
            enabled: enable the variable
            minimum: minimum value of the range
            maximum: maximum value of the range
            lock_range: lock the value to the range of the variable
            terms: list of terms
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.minimum = minimum
        self.maximum = maximum
        self.lock_range = lock_range
        self.terms = list(terms or [])
        self.value = scalar(nan)

    # TODO: implement properly. Currently, RecursionError when using self[item]
    # def __getattr__(self, item: str) -> Term:
    #     """@return the term with the given name, so it can be used like `engine.power.low`."""
    #     try:
    #         return self[item]
    #     except ValueError:
    #         raise AttributeError(
    #             f"'{self.__class__.__name__}' object has no attribute '{item}'"
    #         ) from None

    @overload
    def __getitem__(self, item: int | str) -> Term:
        ...

    @overload
    def __getitem__(self, item: slice) -> list[Term]:
        ...

    def __getitem__(self, item: int | str | slice) -> Term | list[Term]:
        """Allow indexing terms by index, name, or slices (eg, `engine["power"]["low"]`).

        Args:
            item: index, name, or slice of terms

        Returns:
            term by index or name, or slice of terms
        """
        if isinstance(item, slice):
            return self.terms[item]
        return self.term(item)

    def __iter__(self) -> Iterator[Term]:
        """Return the iterator of the terms.

        Returns:
            iterator of the terms
        """
        return iter(self.terms)

    def __len__(self) -> int:
        """Return the number of terms.

        Returns:
            number of terms
        """
        return len(self.terms)

    def __str__(self) -> str:
        """Return the code to construct the variable in the FuzzyLite Language.

        Returns:
            code to construct the variable in the FuzzyLite Language.
        """
        return representation.fll.variable(self)

    def __repr__(self) -> str:
        """Return the code to construct the variable in Python.

        Returns:
            code to construct the variable in Python.
        """
        fields = vars(self).copy()

        fields.pop("_value")

        if not self.description:
            fields.pop("description")
        if self.enabled:
            fields.pop("enabled")

        return representation.as_constructor(self, fields)

    def clear(self) -> None:
        """Clear the variable to its initial state."""
        self.value = nan

    def term(self, name_or_index: str | int, /) -> Term:
        """Find the term by the name or index.

        The best performance is $O(1)$ when using indices,
        and the worst performance is $O(n)$ when using names, where $n$ is the number terms.

        Args:
            name_or_index: name or index of the term

        Returns:
            term by the name or index

        Raises:
             ValueError: when there is no term by the given name.
             IndexError: when the index is out of range
        """
        if isinstance(name_or_index, int):
            return self.terms[name_or_index]
        for term in self.terms:
            if term.name == name_or_index:
                return term
        raise ValueError(
            f"term '{name_or_index}' not found in {[t.name for t in self.terms]}"
        )

    @property
    def drange(self) -> float:
        """Return the magnitude of the range of the variable.

        Returns:
            `maximum - minimum`
        """
        return self.maximum - self.minimum

    @property
    def range(self) -> tuple[float, float]:
        """Return the range of the variable.

        # Getter

        Returns:
            tuple of (minimum, maximum).

        # Setter

        Args:
            min_max (tuple[float, float]): range of the variable
        """
        return self.minimum, self.maximum

    @range.setter
    def range(self, min_max: tuple[float, float]) -> None:
        """Set the range of the variable.

        Args:
            min_max: range of the variable (minimum, maximum).
        """
        self.minimum, self.maximum = min_max

    @property
    def value(self) -> Scalar:
        """Get/Set the value of the variable.

        # Getter

        Returns:
            value of the variable

        # Setter

        when `lock_range = true`, the value is clipped to the range of the variable

        Args:
            value (Scalar): value of the variable
        """
        return self._value

    @value.setter
    def value(self, value: Scalar) -> None:
        """Set the value of the variable.

        If `lock_range = true`, the value will be clipped to the range of the variable.

        Args:
            value: value of the variable
        """
        self._value = (
            np.clip(value, self.minimum, self.maximum) if self.lock_range else value
        )

    def fuzzify(self, x: Scalar) -> str:
        r"""Return the fuzzy representation of $x$.

        The fuzzy representation is computed by evaluating the membership function of $x$ for each
        term $i$, resulting in a fuzzy value in the form $\tilde{x}=\sum_i{\mu_i(x)/i}$

        Args:
            x: value to fuzzify

        Returns:
            fuzzy value expressed as $\sum_i{\mu_i(x)/i}$.
        """
        result: list[str] = []
        for term in self.terms:
            fx = scalar(nan)
            with contextlib.suppress(ValueError):
                fx = term.membership(x)
            if not result:
                result.append(f"{Op.str(fx)}/{term.name}")
            else:
                pm = "+" if fx >= 0.0 or Op.isnan(fx) else "-"
                result.append(f" {pm} {Op.str(fx)}/{term.name}")

        return "".join(result)

    def highest_membership(self, x: Scalar) -> tuple[Scalar, Term | None]:
        r"""Return the term that has the highest membership function value for $x$.

        Args:
            x: value

        Returns:
             term $i$ that maximimizes $\mu_i(x)$.
        """
        result: tuple[Scalar, Term | None] = (0.0, None)
        for term in self.terms:
            y = scalar(nan)
            with contextlib.suppress(ValueError):
                y = term.membership(x)
            if y > result[0]:
                result = (y, term)
        return result


class InputVariable(Variable):
    """Variable to represent the input of a fuzzy logic controller.

    info: related
        - [fuzzylite.variable.Variable][]
        - [fuzzylite.variable.OutputVariable][]
        - [fuzzylite.term.Term][]
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        terms: Iterable[Term] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: name of the variable
            description: description of the variable
            enabled: enable the variable
            minimum: minimum value of the variable
            maximum: maximum value of the variable
            lock_range: lock the value to the range of the variable
            terms: list of terms.
        """
        super().__init__(
            name=name,
            description=description,
            enabled=enabled,
            minimum=minimum,
            maximum=maximum,
            lock_range=lock_range,
            terms=terms,
        )

    def __str__(self) -> str:
        """Return the code to construct the input variable in the FuzzyLite Language.

        Returns:
            code to construct the input variable in the FuzzyLite Language.
        """
        return representation.fll.input_variable(self)

    def fuzzy_value(self) -> str:
        r"""Return the current fuzzy input value.

        The fuzzy value is computed by evaluating the membership function of the current input value $x$
        for each term $i$, resulting in a fuzzy input value in the form $\tilde{x}=\sum_i{\mu_i(x)/i}$.

        Returns:
            current fuzzy value expressed as $\sum_i{\mu_i(x)/i}$.
        """
        return super().fuzzify(self.value)


class OutputVariable(Variable):
    r"""Variable to represents the output of a fuzzy logic controller.

    During the activation of a rule block, the activated terms of each rule are aggregated in the
    fuzzy output, which represents a fuzzy set hereinafter referred to as $\tilde{y}$.

    The defuzzification of $\tilde{y}$ converts the fuzzy output value $\tilde{y}$ into a crisp output value $y$,
    which is stored as the value of this variable.

    info: related
        - [fuzzylite.variable.Variable][]
        - [fuzzylite.variable.InputVariable][]
        - [fuzzylite.term.Term][]
        - [fuzzylite.rule.RuleBlock][]
        - [fuzzylite.norm.SNorm][]
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        lock_previous: bool = False,
        default_value: float = nan,
        aggregation: SNorm | None = None,
        defuzzifier: Defuzzifier | None = None,
        terms: Iterable[Term] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: name of the variable
            description: description of the variable
            enabled: enable the variable
            minimum: minimum value of the variable
            maximum: maximum value of the variable
            lock_range: lock the value to the range of the variable
            lock_previous: lock the previous value of the output variable
            default_value: default value of the output variable
            aggregation: aggregation operator
            defuzzifier: defuzzifier of the output variable
            terms: list of terms.
        """
        self.fuzzy = Aggregated(
            name=name, minimum=minimum, maximum=maximum, aggregation=aggregation
        )
        # initialize parent members
        super().__init__(
            name=name,
            description=description,
            enabled=enabled,
            minimum=minimum,
            maximum=maximum,
            lock_range=lock_range,
            terms=terms,
        )
        # set values of output variable
        self.defuzzifier = defuzzifier
        self.lock_previous = lock_previous
        self.default_value = default_value
        self.previous_value = nan

    def __str__(self) -> str:
        """Return the code to construct the output variable in the FuzzyLite Language.

        Returns:
            code to construct the output variable in the FuzzyLite Language.
        """
        return representation.fll.output_variable(self)

    def __repr__(self) -> str:
        """Return the code to construct the output variable in Python.

        Returns:
            code to construct the output variable in Python.
        """
        fields = vars(self).copy()

        fields["minimum"] = self.minimum
        fields["maximum"] = self.maximum
        fields["aggregation"] = self.aggregation

        fields.pop("fuzzy")
        fields.pop("_value")
        fields.pop("previous_value")

        if not self.description:
            fields.pop("description")
        if self.enabled:
            fields.pop("enabled")

        return representation.as_constructor(self, fields)

    @property
    def minimum(self) -> float:
        """Get/Set the minimum value of the range of the output variable.

        # Getter

        Returns:
            minimum value of the range of the output variable.

        # Setter

        Args:
            value (float): minimum value of the output variable.
        """
        return self.fuzzy.minimum

    @minimum.setter
    def minimum(self, value: float) -> None:
        """Set the minimum value of the range of the output variable.

        Args:
            value: minimum value of the output variable.
        """
        self.fuzzy.minimum = value

    @property
    def maximum(self) -> float:
        """Get/Set the maximum value of the range of the output variable.

        # Getter

        Returns:
            maximum value of the range of the output variable.

        # Setter

        Args:
            value (float): maximum value of the output variable.
        """
        return self.fuzzy.maximum

    @maximum.setter
    def maximum(self, value: float) -> None:
        """Set the maximum value of the range of the output variable.

        Args:
            value: maximum value of the output variable.
        """
        self.fuzzy.maximum = value

    @property
    def aggregation(self) -> SNorm | None:
        """Get/Set the aggregation operator.

        # Getter

        Returns:
             aggregation operator.

        # Setter

        Args:
            value (SNorm): aggregation operator
        """
        return self.fuzzy.aggregation

    @aggregation.setter
    def aggregation(self, value: SNorm | None) -> None:
        """Set the aggregation operator.

        Args:
            value (SNorm): aggregation operator
        """
        self.fuzzy.aggregation = value

    def defuzzify(self) -> None:
        """Defuzzify the output variable and store the output value and the previous output value.

        The final value $y$ depends on the following cascade of properties in order of precedence expressed in the FuzzyLite Language:

        - `lock-previous: boolean`: when the output value is not finite (ie, `nan` or `inf`) and `lock-previous: true`,
        then the output value is replaced with the value defuzzified in the previous iteration.

        - `default: scalar`: when the output value is (still) not finite and the default value is not `nan`,
        then the output value is replaced with the `default` value.

        - `lock-range: boolean`: when `lock-range: true`, the output value is clipped to the range of the variable.

        """
        if not self.enabled:
            return

        if not self.defuzzifier:
            raise ValueError(
                f"expected a defuzzifier in output variable '{self.name}', "
                "but found None"
            )
        # value at t+1
        value = self.defuzzifier.defuzzify(self.fuzzy, self.minimum, self.maximum)

        # previous value is the last element of the value at t
        self.previous_value = np.take(self.value, -1).astype(float)

        # Locking previous values
        if self.lock_previous:
            with np.nditer(value, op_flags=[["readwrite"]]) as iterator:
                previous_value = self.previous_value
                for value_i in iterator:
                    if np.isnan(value_i):
                        value_i[...] = previous_value  # type:ignore
                    else:
                        previous_value = value_i  # type: ignore

        # Applying default values
        if not np.isnan(self.default_value):
            value[np.isnan(value)] = self.default_value  # type: ignore

        # Committing the value
        self.value = value

    def clear(self) -> None:
        """Clear the output variable."""
        self.fuzzy.clear()
        self.previous_value = nan
        self.value = nan

    def fuzzy_value(self) -> str:
        """Return the current fuzzy output value.

        Returns:
            current fuzzy output value.
        """
        # TODO: fix for vectorisation
        result: list[str] = []
        for term in self.terms:
            degree = self.fuzzy.activation_degree(term)

            if not result:
                result.append(f"{Op.str(degree)}/{term.name}")
            else:
                result.append(
                    f" {'+' if Op.isnan(degree) or degree >= 0 else '-'} {Op.str(np.fabs(degree))}/{term.name}"
                )
        return "".join(result)
