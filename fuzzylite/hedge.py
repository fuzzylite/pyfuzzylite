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
    "Hedge",
    "Any",
    "Extremely",
    "Not",
    "Seldom",
    "Somewhat",
    "Very",
    "HedgeLambda",
    "HedgeFunction",
]

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .library import representation, scalar
from .term import Function
from .types import Scalar


class Hedge(ABC):
    """Abstract class for hedges.

    Hedges are used in the antecedent and consequent of a rule to modify the membership function of the term it precedes.

    The hedges in the library can be ordered based on the closeness of the result and the membership function value
    as follows:

    1. [fuzzylite.hedge.Not][],
    2. [fuzzylite.hedge.Seldom][],
    3. [fuzzylite.hedge.Somewhat][],
    4. [fuzzylite.hedge.Very][],
    5. [fuzzylite.hedge.Extremely][], and
    6. [fuzzylite.hedge.Any][], being this a special case

    info: related
        - [fuzzylite.hedge.Not][]
        - [fuzzylite.hedge.Seldom][]
        - [fuzzylite.hedge.Somewhat][]
        - [fuzzylite.hedge.Very][]
        - [fuzzylite.hedge.Extremely][]
        - [fuzzylite.hedge.Any][]
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __str__(self) -> str:
        """Return the name of the hedge.

        Returns:
            name of the hedge.
        """
        return self.name

    def __repr__(self) -> str:
        """Return the Python code to construct the hedge.

        Returns:
            Python code to construct the hedge.
        """
        return representation.as_constructor(self)

    @property
    def name(self) -> str:
        """Return the name of the hedge.

        Returns:
            name of the hedge.
        """
        return self.__class__.__name__.lower()

    @abstractmethod
    def hedge(self, x: Scalar) -> Scalar:
        """Implement the hedge for the membership function value $x$.

        Args:
            x: membership function value

        Returns:
           hedge of $x$.
        """
        pass


class Any(Hedge):
    """Special hedge that always returns `1.0`.

    The antecedent of a rule considers `Any` to be a syntactically special hedge because it is not
    followed by a term (e.g., `if Variable is any then...` vs `if Variable is very term then...`)

    The hedge is useful for better documenting rules.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        """Return scalar of same shape of `x` filled with `1.0`.

        Args:
            x: irrelevant except for its shape

        Returns:
            scalar of same shape of `x` filled with `1.0`
        """
        x = scalar(x)
        y = np.full_like(x, 1.0)
        return y


class Extremely(Hedge):
    r"""Hedge that modifies the membership function value of a term as follows.

    $$
    \begin{cases}
        2x^2 & \mbox{if $x \le 0.5$} \cr
        1-2(1-x)^2 & \mbox{otherwise} \cr
    \end{cases}
    $$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Extremely}(x)$.

        Args:
             x: membership function value

        Returns:
            $$\begin{cases} 2x^2 & \mbox{if $x \le 0.5$} \cr 1-2(1-x)^2 & \mbox{otherwise} \cr \end{cases}$$
        """
        x = scalar(x)
        y = np.where(x <= 0.5, 2 * x**2, 1 - 2 * (1 - x) ** 2)
        return y


class Not(Hedge):
    """Hedge that modifies the membership function value of a term by $1-x$.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Not}(x)$.

        Args:
            x: membership function value

        Returns:
             $1-x$.
        """
        x = scalar(x)
        y = 1 - x
        return y


class Seldom(Hedge):
    r"""Hedge that modifies the membership function value of a term as follows.

    $$
    \begin{cases}
        \sqrt{\dfrac{x}{2}} & \mbox{if $x \le 0.5$} \cr
        1-\sqrt{\dfrac{(1-x)}{2}} & \mbox{otherwise}\cr
    \end{cases}
    $$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]

    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Seldom(x)}$.

        Args:
            x: membership function value

        Returns:
            $$\begin{cases} \sqrt{\dfrac{x}{2}} & \mbox{if $x \le 0.5$} \cr 1-\sqrt{\dfrac{(1-x)}{2}} & \mbox{otherwise}\cr \end{cases}$$
        """
        x = scalar(x)
        y = np.where(x <= 0.5, np.sqrt(0.5 * x), 1 - np.sqrt(0.5 * (1 - x)))
        return y


class Somewhat(Hedge):
    r"""Hedge that modifies the membership function value of a term by $\sqrt{x}$.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Somewhat}(x)$.

        Args:
            x: membership function value

        Returns:
            $\sqrt{x}$
        """
        x = scalar(x)
        y = np.sqrt(x)
        return y


class Very(Hedge):
    r"""Hedge that modifies the membership function value of a term by $x^2$.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Very}(x)$.

        Args:
            x: membership function value

        Returns:
             $x^2$.
        """
        x = scalar(x)
        y = x**2
        return y


class HedgeLambda(Hedge):
    r"""Hedge that modifies the membership function value of a term according to a $\lambda$ function.

    This hedge is not registered with the HedgeFactory because the $\lambda$ function cannot be easily configured.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.hedge.HedgeFunction][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __init__(self, name: str, function: Callable[[Scalar], Scalar]) -> None:
        r"""Constructor.

        Args:
            name: name of the hedge
            function: $\lambda$ function.
        """
        self._name = name
        self.function = function

    @property
    def name(self) -> str:
        """Get the name of the hedge.

        Returns:
            name of the hedge
        """
        return self._name

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\lambda(x)$.

        Args:
            x: membership function value

        Returns:
            $\lambda(x)$
        """
        return self.function(x)


class HedgeFunction(Hedge):
    r"""Hedge that modifies the membership function value of a term according to the term Function.

    This hedge is not registered with the HedgeFactory because the Function cannot be easily configured.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.hedge.HedgeLambda][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __init__(self, function: Function) -> None:
        """Constructor.

        Args:
            function: function $f$.
        """
        self.function = function

    @property
    def name(self) -> str:
        """Get the name of the hedge.

        Returns:
            name of the hedge
        """
        return self.function.name

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\f(x)$.

        Args:
            x: membership function value

        Returns:
            $\f(x)$
        """
        return self.function.membership(x)
