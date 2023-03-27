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

import math
import typing
from typing import Callable

if typing.TYPE_CHECKING:
    from .term import Function


class Hedge:
    """The Hedge class is the abstract class for hedges. Hedges are utilized
    within the Antecedent and Consequent of a Rule in order to modify the
    membership function of a linguistic Term.

    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Consequent
    @see Rule
    @see HedgeFactory
    @since 4.0
    """

    @property
    def name(self) -> str:
        """Returns the name of the hedge
        @return the name of the hedge.
        """
        return self.__class__.__name__.lower()

    def hedge(self, x: float) -> float:
        """Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return the hedge of $x$.
        """
        raise NotImplementedError()


class Any(Hedge):
    """The Any class is a special Hedge that always returns `1.0`. Its
    position with respect to the other hedges is last in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any). The Antecedent of a Rule
    considers Any to be a syntactically special hedge because it is not
    followed by a Term (e.g., `if Variable is any then...`). Amongst hedges,
    only Any has virtual methods to be overridden due to its particular case.

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        """Computes the hedge for the given value
        @param x is irrelevant
        @return `1.0`.
        """
        return 1.0


class Extremely(Hedge):
    """The Extremely class is a Hedge located fifth in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any).

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        r"""Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return $
        \begin{cases}
        2x^2 & \mbox{if $x \le 0.5$} \cr
        1-2(1-x)^2 & \mbox{otherwise} \cr
        \end{cases}$.
        """
        return 2.0 * x * x if x <= 0.5 else (1.0 - 2.0 * (1.0 - x) * (1.0 - x))


class Not(Hedge):
    """The Not class is a Hedge located first in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any).

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        """Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return $1-x$.
        """
        return 1.0 - x


class Seldom(Hedge):
    """The Seldom class is a Hedge located second in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any).

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        r"""Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return $
        \begin{cases}
        \sqrt{0.5x} & \mbox{if $x \le 0.5$} \cr
        1-\sqrt{0.5(1-x)} & \mbox{otherwise}\cr
        \end{cases}
        $.
        """
        return math.sqrt(0.5 * x) if x <= 0.5 else (1.0 - math.sqrt(0.5 * (1.0 - x)))


class Somewhat(Hedge):
    """The Somewhat class is a Hedge located third in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any).

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        r"""Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return $\sqrt{x}$.
        """
        return math.sqrt(x)


class Very(Hedge):
    """The Very class is a Hedge located fourth in the ordered set
    (Not, Seldom, Somewhat, Very, Extremely, Any).

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see HedgeFactory
    @since 4.0
    """

    def hedge(self, x: float) -> float:
        """Computes the hedge for the membership function value $x$
        @param x is a membership function value
        @return $x^2$.
        """
        return x * x


class HedgeLambda(Hedge):
    """The HedgeLambda class is a customizable Hedge via Lambda, which
    computes any function based on the $x$ value. This hedge is not
    registered with the HedgeFactory due to issues configuring the formula
    within. To register the hedge, a static method with the
    constructor needs to be manually created and registered.

    @author Juan Rada-Vilela, Ph.D.
    @see Function
    @see Hedge
    @see HedgeFactory
    @since 7.0
    """

    def __init__(self, name: str, function: Callable[[float], float]) -> None:
        """Create the hedge.
        @param name is the name of the hedge
        @param function is the lambda function.
        """
        self._name = name
        self.function = function

    @property
    def name(self) -> str:
        """Gets the name of the hedge."""
        return self._name

    def hedge(self, x: float) -> float:
        """Computes the hedge for the membership function value $x$ utilizing
        the HedgeFunction::function
        @param x is a membership function value
        @return the evaluation of the function.
        """
        return self.function(x)


class HedgeFunction(Hedge):
    """The HedgeFunction class is a customizable Hedge via Function, which
    computes any function based on the $x$ value. This hedge is not
    registered with the HedgeFactory due to issues configuring the formula
    within. To register the hedge, a static method with the
    constructor needs to be manually created and registered. Please, check the
    file `test/hedge/HedgeFunction.cpp` for further details.

    @author Juan Rada-Vilela, Ph.D.
    @see Function
    @see Hedge
    @see HedgeFactory
    @since 6.0
    """

    def __init__(self, function: "Function") -> None:
        """Create the hedge.
        @param function is the function.
        """
        self.function = function

    @property
    def name(self) -> str:
        """Gets the name of the function."""
        return self.function.name

    def hedge(self, x: float) -> float:
        """Computes the hedge for the membership function value $x$ utilizing
        the HedgeFunction::function
        @param x is a membership function value
        @return the evaluation of the function.
        """
        return self.function.membership(x)
