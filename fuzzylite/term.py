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
    "Activated",
    "Aggregated",
    "Bell",
    "Binary",
    "Concave",
    "Constant",
    "Cosine",
    "Discrete",
    "Function",
    "Gaussian",
    "GaussianProduct",
    "Linear",
    "PiShape",
    "Ramp",
    "Rectangle",
    "SShape",
    "Sigmoid",
    "SigmoidDifference",
    "SigmoidProduct",
    "Spike",
    "Term",
    "Trapezoid",
    "Triangle",
    "ZShape",
]

import enum
import re
import typing
from collections.abc import Iterable, Sequence
from math import inf, isnan, nan
from typing import Any, Callable, SupportsFloat, TypeVar

import numpy as np

from .exporter import FllExporter
from .norm import SNorm, TNorm
from .operation import Op, scalar, scalars
from .types import Array, Scalar

if typing.TYPE_CHECKING:
    from .engine import Engine


# TODO: optimise numpy expressions, maybe using indexes instead of np.where
class Term:
    """The Term class is the abstract class for linguistic terms. The linguistic
    terms in this library can be divided in four groups as: `basic`,
    `extended`, `edge`, and `function`. The `basic` terms are Triangle,
    Trapezoid, Rectangle, and Discrete. The `extended` terms are Bell,
    Binary, Cosine, Gaussian, GaussianProduct, PiShape, SigmoidDifference,
    SigmoidProduct, and Spike. The `edge` terms are Concave, Ramp, Sigmoid,
    SShape, and ZShape. The `function` terms are Constant, Linear, and
    Function.

    In the figure below, the `basic` terms are represented in the first
    column, and the `extended` terms in the second and third columns. The
    `edge` terms are represented in the fifth and sixth rows, and the
    `function` terms in the last row.

    @image html terms.svg

    @author Juan Rada-Vilela, Ph.D.
    @see Variable
    @see InputVariable
    @see OutputVariable
    @since 4.0

    Attributes:
        name is the name of the term
        height is the height of the term
    """

    def __init__(self, name: str = "", height: float = 1.0) -> None:
        """Create the term.
        @param name is the name of the term
        @param height is the height of the term.

        """
        self.name = name
        self.height = height

    def __str__(self) -> str:
        """Returns the representation of the term in the FuzzyLite Language
        :return the representation of the term in FuzzyLite Language
        @see FllExporter.
        """
        return FllExporter().term(self)

    @property
    def class_name(self) -> str:
        """Gets the name of the class of the term."""
        return self.__class__.__name__

    def parameters(self) -> str:
        """Returns the parameters to configure the term. The parameters are
        separated by spaces. If there is one additional parameter, the
        parameter will be considered as the height of the term; otherwise,
        the height will be set to $1.0$
        :return the parameters to configure the term (@see Term::configure()).
        """
        return self._parameters()

    def _parameters(self, *args: object) -> str:
        """Concatenates the parameters given, and the height if it is different from 1.0 or None
        :param args: is the parameters to configure the term
        :return: the parameters concatenated and an optional height at the end.
        """
        result: list[str] = []
        if args:
            result.extend(map(Op.str, args))
        if self.height and self.height != 1.0:
            result.append(Op.str(self.height))
        return " ".join(result)

    def configure(self, parameters: str) -> None:
        """Configures the term with the given takes_parameters. The takes_parameters are
        separated by spaces. If there is one additional parameter, the
        parameter will be considered as the height of the term; otherwise,
        the height will be set to $1.0$
        :param parameters is the takes_parameters to configure the term.
        """
        pass

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the has_membership function value at $x$
        :param x
        :return the has_membership function value $\mu(x)$.
        """
        raise NotImplementedError()

    def update_reference(self, engine: Engine | None) -> None:
        """Updates the references (if any) to point to the current engine (useful
        when cloning engines or creating terms within Importer objects
        :param engine: is the engine to which this term belongs to.
        """
        pass

    def tsukamoto(
        self,
        activation_degree: Scalar,
        minimum: float,
        maximum: float,
    ) -> Scalar:
        r"""For monotonic terms, computes the tsukamoto value of the term for the
        given activation degree $\alpha$, that is,
        $ g_j(\alpha) = \{ z \in\mathbb{R} : \mu_j(z) = \alpha \} $@f. If
        the term is not monotonic (or does not override this method) the
        method computes the has_membership function $\mu(\alpha)$.
        :param activation_degree: is the activationDegree
        :param minimum is the minimum value of the range of the term
        :param maximum is the maximum value of the range of the term
        :return the tsukamoto value of the term for the given activation degree
                if the term is monotonic (or overrides this method), or
                the has_membership function for the activation degree otherwise.
        """
        return self.membership(activation_degree)

    def is_monotonic(self) -> bool:
        """Indicates whether the term is monotonic.
        :return whether the term is monotonic.
        """
        return False

    def discretize(
        self, start: float, end: float, resolution: int = 100, bounded_mf: bool = True
    ) -> Discrete:
        """Discretise the term.
        @param start is the start of the range
        @param end is the end of the range
        @param resolution is the number of points to discretise
        @param bounded_mf whether to bound the membership values to [0.0, 1.0].
        """
        x = np.linspace(start, end, resolution)
        y = self.membership(x)
        if bounded_mf:
            y = np.clip(y, 0.0, 1.0)
        xy = Discrete.to_xy(x, y)
        return Discrete(self.name, xy)


class Activated(Term):
    """The Activated class is a special Term that contains pointers to the
    necessary information of a term that has been activated as part of the
    Antecedent of a Rule. The ownership of the pointers is not transferred to
    objects of this class. The Activated class was named
    `Thresholded` in versions 4.0 and earlier.
    @author Juan Rada-Vilela, Ph.D.
    @see OutputVariable
    @see Term
    @since 5.0.
    """

    def __init__(
        self, term: Term, degree: Scalar = 1.0, implication: TNorm | None = None
    ) -> None:
        """Create the term.
        @param term is the activated term
        @param degree is the activation degree of the term
        @param implication is the implication operator.
        """
        super().__init__("_")
        self.term = term
        self.degree = degree
        self.implication = implication

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"degree implication term"`.
        """
        name = self.term.name if self.term else "none"
        if self.implication:
            result = (
                f"{FllExporter().norm(self.implication)}({Op.str(self.degree)},{name})"
            )
        else:
            result = f"({Op.str(self.degree)}*{name})"
        return result

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the implication of the activation degree and the membership
        function value of $x$
        @param x is a value
        @return $d \otimes \mu(x)$, where $d$ is the activation degree.
        """
        if not self.term:
            raise ValueError("expected a term to activate, but none found")
        if not self.implication:
            raise ValueError("expected an implication operator, but none found")
        return self.implication.compute(
            self.term.membership(x), np.atleast_2d(self.degree).T
        )


class Aggregated(Term):
    """The Aggregated class is a special Term that stores a fuzzy set with the
    Activated terms from the Antecedent%s of a Rule, thereby serving mainly
    as the fuzzy output value of the OutputVariable%s. The ownership of the
    activated terms will be transfered to objects of this class, and
    therefore their destructors will be called upon destruction of this term
    (or calling Aggregated::clear()).
    @author Juan Rada-Vilela, Ph.D.
    @see Antecedent
    @see Rule
    @see OutputVariable
    @see Activated
    @see Term
    @since 6.0.
    """

    def __init__(
        self,
        name: str = "",
        minimum: float = nan,
        maximum: float = nan,
        aggregation: SNorm | None = None,
        terms: Iterable[Activated] | None = None,
    ) -> None:
        """Create the term.
        @param name is the name of the aggregated term
        @param minimum is the minimum of the range of the fuzzy set
        @param maximum is the maximum of the range of the fuzzy set
        @param aggregation is the aggregation operator
        @param terms is the list of activated terms.
        """
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum
        self.aggregation = aggregation
        # TODO: convert list of terms to dictionary
        self.terms: list[Activated] = []
        if terms:
            self.terms.extend(terms)

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"aggregation minimum maximum terms"`.
        """
        result = []
        activated = [term.parameters() for term in self.terms]
        if self.aggregation:
            result.append(
                f"{FllExporter().norm(self.aggregation)}[{','.join(activated)}]"
            )
        else:
            result.append(f"[{'+'.join(activated)}]")

        return " ".join(result)

    def range(self) -> float:
        """Returns the magnitude of the range of the fuzzy set,
        @return the magnitude of the range of the fuzzy set,
        i.e., `maximum - minimum`.
        """
        return self.maximum - self.minimum

    def membership(self, x: Scalar) -> Scalar:
        r"""Aggregates the membership function values of $x$ utilizing the
        aggregation operator
        @param x is a value
        @return $\sum_i{\mu_i(x)}, i \in \mbox{terms}$.
        """
        if self.terms and not self.aggregation:
            raise ValueError("expected an aggregation operator, but none found")

        result = scalar(0.0)
        for term in self.terms:
            result = self.aggregation.compute(result, term.membership(x))  # type: ignore
        return result

    def activation_degree(self, term: Term) -> Scalar:
        """Computes the aggregated activation degree for the given term.
        If the same term is present multiple times, the aggregation operator
        is utilized to sum the activation degrees of the term. If the
        aggregation operator is fl::null, a regular sum is performed.
        @param erm is the term for which to compute the aggregated
        activation degree
        @return the aggregated activation degree for the given term.
        """
        result = scalar(0.0)
        for activation in self.terms:
            if activation.term == term:
                if self.aggregation:
                    result = self.aggregation.compute(result, activation.degree)
                else:
                    result += activation.degree

        return result

    def highest_activated_term(self) -> Activated | None:
        """Iterates over the Activated terms to find the term with the maximum
        activation degree
        @return the term with the maximum activation degree.
        """
        result = None
        maximum_activation = -inf
        for activated in self.terms:
            if activated.degree > maximum_activation:
                maximum_activation = activated.degree  # type: ignore
                result = activated
        return result

    def clear(self) -> None:
        """Clears the list of activated terms."""
        self.terms.clear()


class Bell(Term):
    """The Bell class is an extended Term that represents the generalized bell
    curve membership function.
    @image html bell.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        slope: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param center is the center of the bell curve
        @param width is the width of the bell curve
        @param slope is the slope of the bell curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.center = center
        self.width = width
        self.slope = slope

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $h / (1 + \left(|x-c|/w\right)^{2s}$
        where $h$ is the height of the Term,
              $c$ is the center of the Bell,
              $w$ is the width of the Bell,
              $s$ is the slope of the Bell.
        """
        x = scalar(x)
        return (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * (
                1.0
                / (1.0 + (np.abs((x - self.center) / self.width) ** (2.0 * self.slope)))
            )
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width slope [height]"`.
        """
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width slope [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width, self.slope = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


class Binary(Term):
    """The Binary class is an edge Term that represents the binary membership
    function.
    @image html binary.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 6.0.
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        direction: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term
        @param name is the name of the term
        @param start is the start of the binary edge
        @param direction is the direction of the binary edge.
        """
        super().__init__(name, height)
        self.start = start
        self.direction = direction

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        1h & \mbox{if $ \left(s < d \vedge x \in [s, d)\right) \wedge
        \left( s > d \vedge x \in (d, s] \right) $} \cr
        0h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $s$ is the start of the Binary edge,
              $d$ is the direction of the Binary edge.
        """
        x = scalar(x)
        right = (self.direction > self.start) & (x >= self.start)
        left = (self.direction < self.start) & (x <= self.start)
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(right | left, 1.0, 0.0)
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start direction [height]"`.
        """
        return super()._parameters(self.start, self.direction)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start direction [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.direction = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Concave(Term):
    """The Concave class is an edge Term that represents the concave membership
    function.
    @image html concave.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 5.0.
    """

    def __init__(
        self,
        name: str = "",
        inflection: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param inflection is the inflection of the curve
        @param end is the end of the curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.inflection = inflection
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        h \times (e - i) / (2e - i - x) & \mbox{if $i \leq e \wedge x < e$
        (increasing concave)} \cr
        h \times (i - e) / (-2e + i + x) & \mbox{if $i > e \wedge x > e$
        (decreasing concave)} \cr
        h & \mbox{otherwise} \cr
        \end{cases}$
        where $h$ is the height of the Term,
              $i$ is the inflection of the Concave,
              $e$ is the end of the Concave.
        """
        x = scalar(x)
        increasing = (self.inflection <= self.end) & (x < self.end)
        decreasing = (self.inflection >= self.end) & (x > self.end)
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                increasing,
                (self.end - self.inflection) / (2.0 * self.end - self.inflection - x),
                np.where(
                    decreasing,
                    (self.inflection - self.end)
                    / (self.inflection - 2.0 * self.end + x),
                    1.0,
                ),
            )
        )

    def is_monotonic(self) -> bool:
        """Returns True because this term is monotonic."""
        return True

    def tsukamoto(
        self, activation_degree: Scalar, minimum: float, maximum: float
    ) -> Scalar:
        """Returns the tsukamoto value of the term."""
        i = self.inflection
        e = self.end
        return (i - e) / self.membership(activation_degree) + 2 * e - i

    def parameters(self) -> str:
        """Returns the parameters of the term as
        @return `"inflection end [height]"`.
        """
        return super()._parameters(self.inflection, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters given
        @param parameters as `"inflection end [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.inflection, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Constant(Term):
    """The Constant class is a (zero) polynomial Term that represents a constant
    value $ f(x) = k $
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(self, name: str = "", value: Scalar = nan) -> None:
        """Create the term.
        @param name is the name of the term
        @param value is the value of the term.
        """
        super().__init__(name)
        self.value = value

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function evaluated at $x$
        @param x is irrelevant
        @return $c$, where $c$ is the constant value.
        """
        return np.full_like(x, fill_value=self.value)

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"value"`.
        """
        return super()._parameters(self.value)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"value"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        if not values:
            raise ValueError("not enough values to unpack (expected 1, got 0)")
        self.value = values[0]
        self.height = 1.0


class Cosine(Term):
    """The Cosine class is an extended Term that represents the cosine
    membership function.
    @image html cosine.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 5.0.
    """

    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param center is the center of the cosine
        @param width is the width of the cosine
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        0h & \mbox{if $x < c - 0.5w \vee x > c + 0.5w$} \cr
        0.5h \times ( 1 + \cos(2.0 / w\pi(x-c))) & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $c$ is the center of the Cosine,
              $w$ is the width of the Cosine.
        """
        x = scalar(x)
        increasing = x >= self.center - 0.5 * self.width
        decreasing = x <= self.center + 0.5 * self.width
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                np.isfinite(x) & (increasing | decreasing),
                0.5 * (1.0 + np.cos(2.0 / self.width * np.pi * (x - self.center))),
                0.0,
            )
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width [height]"`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Discrete(Term):
    """The Discrete class is a basic Term that represents a discrete membership
    function. The pairs of values in any Discrete term **must** be sorted
    ascendently because the membership function is computed using binary search
    to find the lower and upper bounds of $x$.
    @image html discrete.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    Floatable = TypeVar("Floatable", SupportsFloat, str, bytes, np.generic)

    def __init__(
        self,
        name: str = "",
        xy: Array[np.floating[Any]] | None = None,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param xy is the list of pairs #TODO update
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.xy = xy if xy is not None else np.asarray([])

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$ by using binary
        search to find the lower and upper bounds of $x$ and then linearly
        interpolating the membership function between the bounds.
        @param x
        @return $ \dfrac{h (y_{\max} - y_{\min})}{(x_{\max}- x_{\min})}  (x - x_{\min}) + y_{\min}$
        where $h$ is the height of the Term,
              $x_{\min}$ and $x_{\max}$is are the lower and upper limits
                   of $x$ in `xy` (respectively),
              $y_{\min}$ and $y_{\max}$is are the membership functions
                   of $\mu(x_{\min})$ and $\mu(x_{\max})$ (respectively).
        """
        if not np.any(self.xy):
            raise ValueError(
                "expected xy to contain discrete coordinates, but it is empty"
            )
        return self.height * np.interp(scalar(x), self.xy[:, 0], self.xy[:, 1])

    def tsukamoto(
        self, activation_degree: Scalar, minimum: float, maximum: float
    ) -> Scalar:
        """Not implemented."""
        # todo: approximate tsukamoto
        raise NotImplementedError()

    def parameters(self) -> str:
        """Returns the parameters of the term as `x1 y1 xn yn [height]`
        @return `x1 y1 xn yn [height]`.
        """
        return super()._parameters(self.xy)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters given as `x1 y1 xn yn [height]`
        @param parameters as `x1 y1 xn yn [height]`.
        """
        as_list = parameters.split()
        if len(as_list) % 2 == 0:
            self.height = 1.0
        else:
            self.height = float(as_list[-1])
            del as_list[-1]
        self.xy = Discrete.to_xy(as_list[0::2], as_list[1::2])

    def x(self) -> Array[np.floating[Any]]:
        """An iterable containing the $x$ values
        @return an iterable containing the $x$ values.
        """
        return self.xy[:, 0]

    def y(self) -> Array[np.floating[Any]]:
        """An iterable containing the $y$ values
        @return an iterable containing the $y$ values.
        """
        return self.xy[:, 1]

    def sort(self) -> None:
        """Ascendantly sorts the pairs of values in this Discrete term by the
        $x$-coordinate.
        """
        self.xy[:] = self.xy[np.argsort(self.x())]

    def to_dict(self) -> dict[float, float]:
        """Returns a dictionary of values {x: y}."""
        return dict(zip(self.x(), self.y()))

    def to_list(self) -> list[float]:
        """Returns a list of values [x1, y1, x2, y2, ...]."""
        return self.xy.flatten().tolist()  # type: ignore

    @staticmethod
    def create(
        name: str,
        xy: str
        | Sequence[Floatable]
        | tuple[Sequence[Floatable], tuple[Sequence[Floatable]]]
        | dict[Floatable, Floatable],
        height: float = 1.0,
    ) -> Discrete:
        """Creates a discrete term from a flexible set of parameters
        @param name is the name of the term
        @param xy is a flexible set of parameters
        @param height is the height of the term
        @returns a Discrete term
        .
        """
        # TODO
        raise NotImplementedError()

    @staticmethod
    def to_xy(
        x: Scalar | Sequence[Floatable],
        y: Scalar | Sequence[Floatable],
    ) -> Array[np.floating[Any]]:
        """Creates a list of values from the given values.
        @param values is a string or a flat list of (x, y)-pairs or a dictionary of values {x: y}.
        """
        x = scalars(x)
        y = scalars(y)
        if x.shape != y.shape:
            raise ValueError(
                f"expected same shape from x and y, but found x={x.shape} and y={y.shape}"
            )
        return scalars([x, y]).T

    # TODO: Maybe remove this method
    @staticmethod
    def values_from(pairs: Array[np.floating[Any]]) -> list[float]:
        """Flatten the list of discrete pairs.
        @param pairs is the list of discrete pairs
        @returns a flat list of values.
        """
        return list(pairs.flatten())

    # TODO: maybe find way to do: dict(self)
    @staticmethod
    def dict_from(pairs: Array[np.floating[Any]]) -> dict[float, float]:
        """Create a dictionary from the list of discrete pairs.
        @param pairs is a list of discrete pairs
        @returns a dictionary of pairs {x: y}.
        """
        return {x: y for x, y in zip(pairs[:, 0], pairs[:, 1])}


class Gaussian(Term):
    """The Gaussian class is an extended Term that represents the %Gaussian
    curve membership function.
    @image html gaussian.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        mean: float = nan,
        standard_deviation: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param mean is the mean of the Gaussian curve
        @param standardDeviation is the standard deviation of the Gaussian curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $ h \times \exp(-(x-\mu)^2/(2\sigma^2))$
        where $h$ is the height of the Term,
              $\mu$ is the mean of the Gaussian,
              $\sigma$ is the standard deviation of the Gaussian.
        """
        x = scalar(x)
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.exp(
                (-(x - self.mean) * (x - self.mean))
                / (2.0 * self.standard_deviation * self.standard_deviation)
            )
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"mean standardDeviation [height]"`.
        """
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"mean standardDeviation [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.mean, self.standard_deviation = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class GaussianProduct(Term):
    """The GaussianProduct class is an extended Term that represents the
    two-sided %Gaussian membership function.
    @image html gaussianProduct.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        mean_a: float = nan,
        standard_deviation_a: float = nan,
        mean_b: float = nan,
        standard_deviation_b: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param meanA is the mean of the first %Gaussian curve
        @param standardDeviationA is the standard deviation of the first %Gaussian curve
        @param meanB is the mean of the second %Gaussian curve
        @param standardDeviationB is the standard deviation of the second %Gaussian curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.mean_a = mean_a
        self.standard_deviation_a = standard_deviation_a
        self.mean_b = mean_b
        self.standard_deviation_b = standard_deviation_b

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $ h \left((1 - i) + i \times \exp(-(x - \mu_a)^2 /
        (2\sigma_a^2))\right)
        \left((1 - j) + j \times \exp(-(x - \mu_b)^2 / (2 \sigma_b)^2)\right)
        $
        where $h$ is the height of the Term,
              $\mu_a$ is the mean of the first GaussianProduct,
              $\sigma_a$ is the standard deviation of the first
              GaussianProduct,
              $\mu_b$ is the mean of the second GaussianProduct,
              $\sigma_b$ is the standard deviation of the second
              GaussianProduct,
              $i=\begin{cases}1 & \mbox{if $x \leq \mu_a$} \cr 0
              &\mbox{otherwise}\end{cases}$,
              $j=\begin{cases}1 & \mbox{if $x \geq \mu_b$} \cr 0
              &\mbox{otherwise}\end{cases}$.
        """
        x = scalar(x)
        a = np.where(
            x < self.mean_a,
            np.exp(
                (-(x - self.mean_a) * (x - self.mean_a))
                / (2.0 * self.standard_deviation_a * self.standard_deviation_a)
            ),
            1.0,
        )
        b = np.where(
            x > self.mean_b,
            np.exp(
                (-(x - self.mean_b) * (x - self.mean_b))
                / (2.0 * self.standard_deviation_b * self.standard_deviation_b)
            ),
            1.0,
        )
        return self.height * np.where(np.isnan(x), np.nan, 1.0) * a * b  # type: ignore

    def parameters(self) -> str:
        """Provides the parameters of the term
        @return `"meanA standardDeviationA meanB standardDeviationB [height]"`.
        """
        return super()._parameters(
            self.mean_a,
            self.standard_deviation_a,
            self.mean_b,
            self.standard_deviation_b,
        )

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"meanA standardDeviationA meanB
        standardDeviationB [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        (
            self.mean_a,
            self.standard_deviation_a,
            self.mean_b,
            self.standard_deviation_b,
        ) = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Linear(Term):
    r"""The Linear class is a linear polynomial Term expressed as $f(x)=
    \mathbf{c}\mathbf{v}+k = \sum_i c_iv_i + k$, where variable $x$ is
    not utilized, $\mathbf{v}$ is a vector of values from the input
    variables, $\mathbf{c}$ is a vector of coefficients, and $k$ is a
    constant. Hereinafter, the vector $\mathbf{c}^\star=\{c_1, \ldots, c_i,
    \ldots, c_n, k\}$ refers to a vector containing the coefficients of
    $\mathbf{c}$ and the constant $k$.
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        coefficients: Sequence[float] | None = None,
        engine: Engine | None = None,
    ) -> None:
        r"""Create the term.
        @param name is the name of the term
        @param coefficients is the vector $\mathbf{c}^\star$
        @param height is the height of the term.
        """
        super().__init__(name)
        self.coefficients = coefficients or []
        self.engine = engine

    def membership(self, _: Scalar) -> Scalar:
        r"""Computes the linear function $f(x)=\sum_i c_iv_i +k$,
        where $v_i$ is the value of the input variable $i$ registered
        in the Linear::getEngine()
        @param x is not utilized
        @return $\sum_i c_ix_i +k$.
        """
        if not self.engine:
            raise ValueError("expected reference to an engine, but found none")

        if len(self.coefficients) not in {
            len(self.engine.input_variables),
            len(self.engine.input_variables) + 1,
        }:
            raise ValueError(
                f"expected {len(self.engine.input_variables)} (+1) coefficients (one for each input variable plus an optional constant), "
                f"but found {len(self.coefficients)} coefficients: {self.coefficients}"
            )
        coefficients = scalars([self.coefficients[: len(self.engine.input_variables)]])
        constant = (
            self.coefficients[-1]
            if len(self.coefficients) > len(self.engine.input_variables)
            else 0.0
        )
        inputs = np.array([iv.value for iv in self.engine.input_variables], ndmin=2).T
        result = (coefficients * inputs).sum(axis=1) + constant
        return result  # type:ignore

    def configure(self, parameters: str) -> None:
        r"""Configures the term with the values of $\mathbf{c}^\star$
        @param parameters as `"c1 ... ci ... cn k"`.
        """
        self.coefficients = [Op.scalar(p) for p in parameters.split()]

    def parameters(self) -> str:
        r"""Returns the vector $\mathbf{c}^\star$
        @return `"c1 ... ci ... cn k"`.
        """
        return self._parameters(*self.coefficients)

    def update_reference(self, engine: Engine | None) -> None:
        """Updates the reference to the engine."""
        self.engine = engine


class PiShape(Term):
    """The PiShape class is an extended Term that represents the Pi-shaped curve
    membership function.
    @image html piShape.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        bottom_left: float = nan,
        top_left: float = nan,
        top_right: float = nan,
        bottom_right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param bottomLeft is the bottom-left value of the curve
        @param topLeft is the top-left value of the curve
        @param topRight is the top-right value of the curve
        @param bottomRight is the bottom-right value of the curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        0h & \mbox{if $x \leq b_l$}\cr
        2h \left((x - b_l) / (t_l-b_l)\right)^2 & \mbox{if $x \leq 0.5(a+b)$}\cr
        h (1 - 2 \left((x - t_l) / (t_l-b_l)\right)^2) & \mbox{if $ x < t_l$}\cr
        h & \mbox{if $x \leq t_r$}\cr
        h (1 - 2\left((x - t_r) / (b_r - t_r)\right)^2) & \mbox{if $x \leq 0.5(t_r + b_r)$}\cr
        2h \left((x - b_r) / (b_r-t_r)\right)^2 & \mbox{if $x < b_r$} \cr
        0h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $b_l$ is the bottom left of the PiShape,
              $t_l$ is the top left of the PiShape,
              $t_r$ is the top right of the PiShape
              $b_r$ is the bottom right of the PiShape,.
        """
        x = scalar(x)
        s_shape = np.where(
            x <= self.bottom_left,
            0.0,
            np.where(
                x <= 0.5 * (self.bottom_left + self.top_left),
                2.0
                * ((x - self.bottom_left) / (self.top_left - self.bottom_left)) ** 2,
                np.where(
                    x < self.top_left,
                    1.0
                    - 2.0
                    * ((x - self.top_left) / (self.top_left - self.bottom_left)) ** 2,
                    1.0,
                ),
            ),
        )
        z_shape = np.where(
            x <= self.top_right,
            1.0,
            np.where(
                x <= 0.5 * (self.top_right + self.bottom_right),
                1.0
                - 2.0
                * ((x - self.top_right) / (self.bottom_right - self.top_right)) ** 2,
                np.where(
                    x < self.bottom_right,
                    2.0
                    * ((x - self.bottom_right) / (self.bottom_right - self.top_right))
                    ** 2,
                    0.0,
                ),
            ),
        )
        return self.height * np.where(np.isnan(x), np.nan, 1.0) * s_shape * z_shape  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"bottomLeft topLeft topRight bottomRight [height]"`.
        """
        return super()._parameters(
            self.bottom_left, self.top_left, self.top_right, self.bottom_right
        )

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"bottomLeft topLeft topRight bottomRight
        [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.bottom_left, self.top_left, self.top_right, self.bottom_right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Ramp(Term):
    """The Ramp class is an edge Term that represents the ramp membership
    function.
    @image html ramp.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param start is the start of the ramp
        @param end is the end of the ramp
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return
        $\begin{cases}
        0h & \mbox{if $x = e$}\cr
        \begin{cases}
        0h & \mbox{if $x \leq s$}\cr
        1h & \mbox{if $x \geq e$}\cr
        h (x - s) / (e - s) & \mbox{otherwise}\cr
        \end{cases} & \mbox{if $s < e$}\cr
        \begin{cases}
        0h & \mbox{if $x \geq s$}\cr
        1h & \mbox{if $x \leq e$}\cr
        h (s - x) / (s - e) & \mbox{otherwise}
        \end{cases} & \mbox{if $s > e$}\cr
        \end{cases}$
        where $h$ is the height of the Term,
              $s$ is the start of the Ramp,
              $e$ is the end of the Ramp.
        """
        x = scalar(x)
        increasing = self.start < self.end
        decreasing = self.start > self.end
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                increasing & (self.end > x) & (x > self.start),
                (x - self.start) / (self.end - self.start),
                np.where(
                    decreasing & (self.end < x) & (x < self.start),
                    (self.start - x) / (self.start - self.end),
                    (increasing & (x >= self.end)) | (decreasing & (x <= self.end)),
                ),
            )
        )

    def is_monotonic(self) -> bool:
        """Returns True as this term is monotonic."""
        return True

    def tsukamoto(
        self, activation_degree: Scalar, minimum: float, maximum: float
    ) -> Scalar:
        """Returns the Tsukamoto value of the term."""
        right = inf if self.start < self.end else -inf
        left = -inf if self.end > self.start else inf

        return np.interp(
            activation_degree,
            [0, self.height],
            [self.start, self.end],
            right=right,
            left=left,
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Rectangle(Term):
    """The Rectangle class is a basic Term that represents the rectangle
    membership function.
    @image html rectangle.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param start is the start of the rectangle
        @param end is the end of the rectangle
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        1h & \mbox{if $x \in [s, e]$} \cr
        0h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $s$ is the start of the Rectangle,
              $e$ is the end of the Rectangle.
        """
        x = scalar(x)
        return (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * ((self.start <= x) & (x <= self.end))
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class Sigmoid(Term):
    """The Sigmoid class is an edge Term that represents the sigmoid membership
    function.
    @image html sigmoid.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        inflection: float = nan,
        slope: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param inflection is the inflection of the sigmoid
        @param slope is the slope of the sigmoid
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.inflection = inflection
        self.slope = slope

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $ h / (1 + \exp(-s(x-i)))$
        where $h$ is the height of the Term,
              $s$ is the slope of the Sigmoid,
              $i$ is the inflection of the Sigmoid.
        """
        x = scalar(x)
        return (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * 1.0
            / (1.0 + np.exp(-self.slope * (x - self.inflection)))
        )

    def is_monotonic(self) -> bool:
        """Returns True as this term is monotonic."""
        return True

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"inflection slope [height]"`.
        """
        return super()._parameters(self.inflection, self.slope)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"inflection slope [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.inflection, self.slope = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class SigmoidDifference(Term):
    """The SigmoidDifference class is an extended Term that represents the
    difference between two sigmoidal membership functions.
    @image html sigmoidDifference.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        left: float = nan,
        rising: float = nan,
        falling: float = nan,
        right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param left is the inflection of the left sigmoidal curve
        @param rising is the slope of the left sigmoidal curve
        @param falling is the slope of the right sigmoidal curve
        @param right is the inflection of the right sigmoidal curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $ h (a-b)$
        where $h$ is the height of the Term,
              $a= 1 / (1 + \exp(-s_l \times (x - i_l))) $,
              $b = 1 / (1 + \exp(-s_r \times (x - i_r)))$,
              $i_l$ is the left inflection of the SigmoidDifference,
              $s_l$ is the left slope of the SigmoidDifference,
              $i_r$ is the right inflection of the SigmoidDifference,
              $s_r$ is the right slope of the SigmoidDifference.
        """
        x = scalar(x)
        a = 1.0 / (1.0 + np.exp(-self.rising * (x - self.left)))
        b = 1.0 / (1.0 + np.exp(-self.falling * (x - self.right)))
        return self.height * np.where(np.isnan(x), np.nan, 1.0) * np.abs(a - b)  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left rising falling right [height]"`.

        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left rising falling right [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class SigmoidProduct(Term):
    """The SigmoidProduct class is an extended Term that represents the product
    of two sigmoidal membership functions.
    @image html sigmoidProduct.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        left: float = nan,
        rising: float = nan,
        falling: float = nan,
        right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param left is the inflection of the left sigmoidal curve
        @param rising is the slope of the left sigmoidal curve
        @param falling is the slope of the right sigmoidal curve
        @param right is the inflection of the right sigmoidal curve
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $ h (a \times b)$
        where $h$ is the height,
              $a= 1 / (1 + \exp(-s_l *\times (x - i_l))) $,
              $b = 1 / (1 + \exp(-s_r \times (x - i_r)))$,
              $i_l$ is the left inflection of the SigmoidProduct,
              $s_l$ is the left slope of the SigmoidProduct,
              $i_r$ is the right inflection of the SigmoidProduct,
              $s_r$ is the right slope of the SigmoidProduct.
        """
        x = scalar(x)
        a = 1.0 + np.exp(-self.rising * (x - self.left))
        b = 1.0 + np.exp(-self.falling * (x - self.right))
        return self.height * np.where(np.isnan(x), np.nan, 1.0) / (a * b)

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left rising falling right [height]"`.
        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left rising falling right [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Spike(Term):
    """The Spike class is an extended Term that represents the spike membership
    function.
    @image html spike.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 5.0.
    """

    # TODO: Properly rename the parameters.
    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param center is the center of the spike
        @param width is the width of the spike
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $h \times \exp(-|10 / w (x - c)|)$
        where $h$ is the height of the Term,
              $w$ is the width of the Spike,
              $c$ is the center of the Spike.
        """
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.exp(-np.abs(10.0 / self.width * (x - self.center)))
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width [height]"`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class SShape(Term):
    """The SShape class is an edge Term that represents the S-shaped membership
    function.
    @image html sShape.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param start is the start of the edge
        @param end is the end of the edge
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        0h & \mbox{if $x \leq s$} \cr
        h(2 \left((x - s) / (e-s)\right)^2) & \mbox{if $x \leq 0.5(s+e)$}\cr
        h(1 - 2\left((x - e) / (e-s)\right)^2) & \mbox{if $x < e$}\cr
        1h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $s$ is the start of the SShape,
              $e$ is the end of the SShape.
        """
        x = scalar(x)
        s_shape = np.where(
            x <= self.start,
            0.0,
            np.where(
                x <= 0.5 * (self.start + self.end),
                2.0 * ((x - self.start) / (self.end - self.start)) ** 2,
                np.where(
                    x < self.end,
                    1.0 - 2.0 * ((x - self.end) / (self.end - self.start)) ** 2,
                    1.0,
                ),
            ),
        )
        return self.height * np.where(np.isnan(x), np.nan, 1.0) * s_shape  # type: ignore

    def is_monotonic(self) -> bool:
        """Returns True as this term is monotonic."""
        return True

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Trapezoid(Term):
    """The Trapezoid class is a basic Term that represents the trapezoidal
    membership function.
    @image html trapezoid.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    # TODO: properly rename the parameters.
    def __init__(
        self,
        name: str = "",
        bottom_left: float = nan,
        top_left: float = nan,
        top_right: float = nan,
        bottom_right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param a is the first vertex of the trapezoid
        @param b is the second vertex of the trapezoid
        @param c is the third vertex of the trapezoid
        @param d is the fourth vertex of the trapezoid
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        if isnan(top_right) and isnan(bottom_right):
            self.bottom_right = top_left
            range_ = self.bottom_right - self.bottom_left
            self.top_left = self.bottom_left + range_ * 1.0 / 5.0
            self.top_right = self.bottom_left + range_ * 4.0 / 5.0

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        0h & \mbox{if $x \not\in[a,d]$}\cr
        h \times \min(1, (x - a) / (b - a)) & \mbox{if $x < b$}\cr
        1h & \mbox{if $x \leq c$}\cr
        h (d - x) / (d - c) & \mbox{if $x < d$}\cr
        0h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $a$ is the first vertex of the Trapezoid,
              $b$ is the second vertex of the Trapezoid,
              $c$ is the third vertex of the Trapezoid,
              $d$ is the fourth vertex of the Trapezoid.
        """
        x = scalar(x)
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                ((x < self.bottom_left) | (x > self.bottom_right)),
                0.0,
                np.where(
                    ((self.top_left <= x) & (x <= self.top_right))
                    | ((self.bottom_left == -inf) & (x < self.top_left))
                    | ((self.bottom_right == inf) & (x > self.top_right)),
                    1.0,
                    np.where(
                        x < self.top_left,
                        (x - self.bottom_left) / (self.top_left - self.bottom_left),
                        np.where(
                            x > self.top_right,
                            (self.bottom_right - x)
                            / (self.bottom_right - self.top_right),
                            nan,
                        ),
                    ),
                ),
            )
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"bottom_left top_left top_right bottom_right [height]"`.
        """
        return super()._parameters(
            self.bottom_left, self.top_left, self.top_right, self.bottom_right
        )

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"bottom_left top_left top_right bottom_right [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.bottom_left, self.top_left, self.top_right, self.bottom_right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Triangle(Term):
    """The Triangle class is a basic Term that represents the triangular
    membership function.
    @image html triangle.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        left: float = nan,
        top: float = nan,
        right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param a is the first vertex of the triangle
        @param b is the second vertex of the triangle
        @param c is the third vertex of the triangle
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.left = left
        self.top = top
        self.right = right
        if isnan(right):
            self.top = 0.5 * (left + top)
            self.right = top

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $\begin{cases}
        0h & \mbox{if $x \not\in [a,c]$}\cr
        1h & \mbox{if $x = b$}\cr
        h (x - a) / (b - a) & \mbox{if $x < b$} \cr
        h (c - x) / (c - b) & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $a$ is the first vertex of the Triangle,
              $b$ is the second vertex of the Triangle,
              $c$ is the third vertex of the Triangle.
        """
        x = scalar(x)
        return (  # type: ignore
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                (x < self.left) | (x > self.right),
                0.0,
                np.where(
                    (x == self.top)
                    | ((self.left == -inf) & (x < self.top))
                    | ((self.right == inf) & (x > self.top)),
                    1.0,
                    np.where(
                        x < self.top,
                        (x - self.left) / (self.top - self.left),
                        np.where(
                            x > self.top,
                            (self.right - x) / (self.right - self.top),
                            nan,
                        ),
                    ),
                ),
            )
        )

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left top right [height]"`.
        """
        return super()._parameters(self.left, self.top, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left top right [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.left, self.top, self.right = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


# TODO: Tsukamoto
class ZShape(Term):
    """The ZShape class is an edge Term that represents the Z-shaped membership
    function.
    @image html zShape.svg
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param start is the start of the edge
        @param end is the end of the edge
        @param height is the height of the term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$
        @param x
        @return $  \begin{cases}
        1h & \mbox{if $x \leq s$} \cr
        h(1 - 2\left((x - s) / (e-s)\right)^2) & \mbox{if $x \leq 0.5(s+e)$}\cr
        h(2 \left((x - e) / (e-s)\right)^2) & \mbox{if $x < e$}\cr
        0h & \mbox{otherwise}
        \end{cases}$
        where $h$ is the height of the Term,
              $s$ is the start of the ZShape,
              $e$ is the end of the ZShape.
        """
        x = scalar(x)
        z_shape = np.where(
            x <= self.start,
            1.0,
            np.where(
                x <= 0.5 * (self.start + self.end),
                1.0 - 2.0 * ((x - self.start) / (self.end - self.start)) ** 2,
                np.where(
                    x < self.end,
                    2.0 * ((x - self.end) / (self.end - self.start)) ** 2,
                    0.0,
                ),
            ),
        )
        return self.height * np.where(np.isnan(x), np.nan, 1.0) * z_shape  # type: ignore

    def is_monotonic(self) -> bool:
        """Returns True as this term is monotonic."""
        return True

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Function(Term):
    r"""The Function class is a polynomial Term that represents a generic
    function $ f : x \mapsto f(x) $. Every Function object has a public
    key-value map, namely Function::variables, that links variable names to
    fl::scalar values, which are utilized to replace the variable names for
    their respective values in the given formula whenever the function
    $f$ is evaluated. Specifically, when the method
    Function::membership() is called, the name and value of the variable
    $x$ are automatically loaded into the map. Furthermore, if an Engine
    is given, the names of its InputVariable%s and OutputVariable%s will also
    be automatically loaded into the map linking to their respective input
    values and (previously defuzzified) output values. The
    Function::variables need to be manually loaded whenever variables other
    than $x$, input variables, and output variables, are expressed in the
    given formula, always having in mind that (a) the map replaces existing
    keys, and (b) the variable $x$, and input variables and output
    variables of an engine will automatically be replaced and will also take
    precedence over previously loaded variables.
    Besides the use of Function as a linguistic Term, it is also utilized to
    convert the text of the Antecedent of a Rule, expressed in infix
    notation, into postfix notation.
    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see Variable
    @see FunctionFactory
    @see Antecedent::load()
    @since 4.0.
    """

    class Element:
        """The Element class represents a single element in a formula, be that
        either a function or an operator. If the Element represents a
        function, the function can be Unary or Binary, that is, the function
        take one or two parameters (respectively). Else, if the Element
        represents an operator, the parameters to be defined are its `arity`,
        its `precedence`, and its `associativity`.
        """

        @enum.unique
        class Type(enum.Enum):
            """Determines the type of the element."""

            Operator, Function = range(2)

        def __init__(
            self,
            name: str,
            description: str,
            type: Function.Element.Type,
            method: Callable[..., Scalar],
            arity: int = 0,
            precedence: int = 0,
            associativity: int = -1,
        ) -> None:
            """Create the element.
            @param name is the name of the element
            @param description is the description of the element
            @param type is the type of the element
            @param method is a reference to the n-anary function
            @param arity is the number of operands required
            @param precedence clarifies which procedures should be
              performed first in a given mathematical expression
              (https://en.wikipedia.org/wiki/Order_of_operations)
            @param associativity determines how operators of the
              same precedence are grouped in the absence of parentheses
              (https://en.wikipedia.org/wiki/Operator_associativity).
            """
            self.name = name
            self.description = description
            self.type = type
            self.method = method
            self.arity = Op.arity_of(method) if arity < 0 else arity
            self.precedence = precedence
            self.associativity = associativity

        def __str__(self) -> str:
            """Returns a description of the element and its members
            @return a description of the element and its members.
            """
            result = [
                f"name='{self.name}'",
                f"description='{self.description}'",
                f"element_type='{str(self.type)}'",
                f"method='{str(self.method)}'",
                f"arity={self.arity}",
                f"precedence={self.precedence}",
                f"associativity={self.associativity}",
            ]
            return f"{self.__class__.__name__}: {', '.join(result)}"

        def is_function(self) -> bool:
            """Indicates whether the element is a Type::Function
            @return whether the element is a Type::Function.
            """
            return self.type == Function.Element.Type.Function

        def is_operator(self) -> bool:
            """Indicates whether the element is a Type::Operator
            @return whether the element is a Type::Operator.
            """
            return self.type == Function.Element.Type.Operator

    class Node:
        """The Node class structures a binary tree by storing pointers to a left
        Node and a right Node, and storing its content as a
        Function::Element, the name of an InputVariable or OutputVariable, or
        a constant value.
        """

        def __init__(
            self,
            element: Function.Element | None = None,
            variable: str = "",
            constant: float = nan,
            right: Function.Node | None = None,
            left: Function.Node | None = None,
        ) -> None:
            """Create the node.
            @param element the node takes an operation or a function
            @param variable the node can refer to a variable by name
            @param constant the node can take an arbitrary floating-point value
            @param right the node can have an expression tree on the right
            @param left the node can have an expression tree on the left.
            """
            self.element = element
            self.variable = variable
            self.constant = constant
            self.right = right
            self.left = left

        def __str__(self) -> str:
            """Gets a representation of the node in postfix notation."""
            return self.postfix()

        def value(self) -> str:
            """Gets the value in the following priority order:
            (1) operation or function name if there is an element
            (2) variable name if it is not empty
            (3) constant value.
            """
            if self.element:
                result = self.element.name
            elif self.variable:
                result = self.variable
            else:
                result = Op.str(self.constant)
            return result

        def evaluate(self, local_variables: dict[str, Scalar] | None = None) -> Scalar:
            """Evaluates the node and substitutes the variables therein for the
            values passed in the map. The expression tree is evaluated
            recursively.
            @param variables is a map of substitutions of variable names for
            fl::scalar%s
            @return a fl::scalar indicating the result of the evaluation of
            the node.
            """
            result = scalar(nan)
            if self.element:
                if self.element.method is None:
                    raise ValueError("expected a method reference, but found none")
                arity = self.element.arity
                if arity == 0:
                    result = self.element.method()
                elif arity == 1:
                    if not self.right:
                        raise ValueError("expected a right node, but found none")
                    result = self.element.method(self.right.evaluate(local_variables))
                elif arity == 2:
                    if not self.right:
                        raise ValueError("expected a right node, but found none")
                    if not self.left:
                        raise ValueError("expected a left node, but found none")
                    result = self.element.method(
                        self.left.evaluate(local_variables),
                        self.right.evaluate(local_variables),
                    )
            elif self.variable:
                if not local_variables or self.variable not in local_variables:
                    raise ValueError(
                        f"expected a map of variables containing the value for '{self.variable}', "
                        f"but the map contains: {local_variables}"
                    )
                result = local_variables[self.variable]

            else:
                result = self.constant

            return result

        def prefix(self, node: Function.Node | None = None) -> str:
            """Returns a prefix representation of the expression tree under the
            given node
            @param node is the node to start the prefix representation from.
            If the node is `fl::null`, then the starting point is `this` node
            @return a prefix representation of the expression tree under the
            given node.
            """
            if not node:
                return self.prefix(self)

            if not isnan(node.constant):
                return Op.str(node.constant)
            if node.variable:
                return node.variable

            result = [node.value()]
            if node.left:
                result.append(self.prefix(node.left))
            if node.right:
                result.append(self.prefix(node.right))
            return " ".join(result)

        def infix(self, node: Function.Node | None = None) -> str:
            """Returns an infix representation of the expression tree under the
            given node
            @param node is the node to start the infix representation from.
            If the node is `fl::null`, then the starting point is `this` node
            @return an infix representation of the expression tree under the
            given node.
            """
            if not node:
                return self.infix(self)

            if not isnan(node.constant):
                return Op.str(node.constant)
            if node.variable:
                return node.variable

            children = []
            if node.left:
                children.append(self.infix(node.left))
            if node.right:
                children.append(self.infix(node.right))

            is_function = (
                node.element and node.element.type == Function.Element.Type.Function
            )

            if is_function:
                result = node.value() + f" ( {' '.join(children)} )"
            else:  # is operator
                result = f" {node.value()} ".join(children)

            return result

        def postfix(self, node: Function.Node | None = None) -> str:
            """Returns a postfix representation of the expression tree under the
            given node
            @param node is the node to start the postfix representation from.
            If the node is `fl::null`, then the starting point is `this` node
            @return a postfix representation of the expression tree under the
            given node.
            """
            if not node:
                return self.postfix(self)

            if not isnan(node.constant):
                return Op.str(node.constant)
            if node.variable:
                return node.variable

            result = []
            if node.left:
                result.append(self.postfix(node.left))
            if node.right:
                result.append(self.postfix(node.right))
            result.append(node.value())
            return " ".join(result)

    def __init__(
        self,
        name: str = "",
        formula: str = "",
        engine: Engine | None = None,
        variables: dict[str, Scalar] | None = None,
        load: bool = False,
    ) -> None:
        """Create the function.
        @param name is the name of the term
        @param formula is the formula defining the membership function
        @param engine is the engine to which the Function can have access
        @param variables is a map of substitution variables
        @param load whether to load the function on creation.
        """
        super().__init__(name)
        self.root: Function.Node | None = None
        self.formula = formula
        self.engine = engine
        self.variables: dict[str, Scalar] = {}
        if variables:
            self.variables.update(variables)
        if load:
            self.load()

    def parameters(self) -> str:
        """Returns the parameters of the term as `formula`
        @return `formula`.
        """
        return self.formula

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters given as `formula`
        @param parameters as `formula`.
        """
        self.formula = parameters
        self.load()

    def update_reference(self, engine: Engine | None) -> None:
        """Updates the reference to the engine and loads it."""
        self.engine = engine
        if self.is_loaded():
            self.load()

    @staticmethod
    def create(name: str, formula: str, engine: Engine | None = None) -> Function:
        """Creates a Function term with the given parameters
        @param name is the name of the term
        @param formula is the formula defining the membership function
        @param engine is the engine to which the Function can have access
        @return a Function term configured with the given parameters
        @throws fl::Exception if the formula has a syntax error.
        """
        result = Function(name, formula, engine)
        result.load()
        return result

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function value of $x$ at the root node.
        If the engine has been set, the current values of the input variables
        and output variables are added to the map of Function::variables. In
        addition, the variable $x$ will also be added to the map.
        @param x
        @return the membership function value of $x$ at the root node.
        """
        if "x" in self.variables:
            raise ValueError(
                "variable 'x' is reserved for internal use of Function term, please "
                f"remove it from the map of variables: {self.variables}"
            )

        engine_variables: dict[str, Scalar] = {}
        if self.engine:
            for variable in self.engine.variables:
                engine_variables[variable.name] = variable.value

            if "x" in engine_variables:
                raise ValueError(
                    "variable 'x' is reserved for internal use of Function term, "
                    f"please rename the engine variable: {self.engine.variable('x')}"
                )
        engine_variables["x"] = x

        overrides = self.variables.keys() & engine_variables.keys()
        if overrides:
            raise ValueError(
                "function variables cannot override engine variables, please "
                f"resolve the name ambiguity of the following variables: {overrides}"
            )
        engine_variables.update(self.variables)
        return self.evaluate(engine_variables)

    def evaluate(self, variables: dict[str, Scalar] | None = None) -> Scalar:
        """Computes the function value of this term using the given map of
        variable substitutions.
        @param variables is a map of substitution variables
        @return the function value of this term using the given map of
        variable substitutions.
        """
        if not self.root:
            raise RuntimeError(f"function '{self.formula}' is not loaded")
        return self.root.evaluate(variables)

    def is_loaded(self) -> bool:
        """Indicates whether the formula is loaded
        @return whether the formula is loaded.
        """
        return bool(self.root)

    def unload(self) -> None:
        """Unloads the formula and resets the map of substitution variables."""
        self.root = None
        self.variables.clear()

    def load(self) -> None:
        """Loads the current formula expressed in infix notation."""
        self.root = self.parse(self.formula)

    @classmethod
    def format_infix(cls, formula: str) -> str:
        """Formats the infix formula.
        @param formula is the infix formula
        @returns the formula formatted.

        """
        from . import lib
        from .factory import FunctionFactory
        from .rule import Rule

        factory: FunctionFactory = lib.factory_manager.function
        operators: set[str] = set(factory.operators().keys()).union({"(", ")", ","})
        operators -= {Rule.AND, Rule.OR}

        # sorted to have multi-char operators separated first (eg., ** and *)
        regex = "|".join(re.escape(o) for o in sorted(operators, reverse=True))
        spaced = re.sub(rf"({regex})", r" \1 ", formula)
        result = re.sub(r"\s+", " ", spaced).strip()
        return result

    @classmethod
    def infix_to_postfix(cls, formula: str) -> str:
        """Translates the given formula to postfix notation
        @param formula is the right-hand side of a mathematical equation
        expressed in infix notation
        @return the formula represented in postfix notation
        @throws fl::Exception if the formula has syntax errors.
        """
        # TODO: support for unary and binary (+,-)
        from . import lib
        from .factory import FunctionFactory

        formula = cls.format_infix(formula)
        factory: FunctionFactory = lib.factory_manager.function

        from collections import deque

        queue: deque[str] = deque()
        stack: list[str] = []

        for token in formula.split():
            if lib.debugging:
                lib.logger.debug("=" * 20)
                lib.logger.debug(f"formula: {formula}")
                lib.logger.debug(f"queue: {queue}")
                lib.logger.debug(f"stack: {stack}")

            element: Function.Element | None = (
                factory.objects[token] if token in factory.objects else None
            )
            is_operand = not element and token not in {"(", ")", ","}

            if is_operand:
                queue.append(token)

            elif element and element.is_function():
                stack.append(token)

            elif token == ",":
                while stack and stack[-1] != "(":
                    queue.append(stack.pop())
                if not stack or stack[-1] != "(":
                    raise SyntaxError(f"mismatching parentheses in: {formula}")

            elif element and element.is_operator():
                while stack and stack[-1] in factory.objects:
                    top = factory.objects[stack[-1]]
                    if (
                        element.associativity < 0
                        and element.precedence <= top.precedence
                    ) or (
                        element.associativity > 0
                        and element.precedence < top.precedence
                    ):
                        queue.append(stack.pop())
                    else:
                        break

                stack.append(token)

            elif token == "(":
                stack.append(token)

            elif token == ")":
                while stack and stack[-1] != "(":
                    queue.append(stack.pop())

                if not stack or stack[-1] != "(":
                    raise SyntaxError(f"mismatching parentheses in: {formula}")

                stack.pop()  # get rid of "("

                if stack and stack[-1] in factory.objects:
                    if factory.objects[stack[-1]].is_function():
                        queue.append(stack.pop())
            else:
                raise RuntimeError(f"unexpected error with token: {token}")

        while stack:
            if stack[-1] in {"(", ")"}:
                raise SyntaxError(f"mismatching parentheses in: {formula}")
            queue.append(stack.pop())

        postfix = " ".join(queue)
        if lib.debugging:
            lib.logger.debug(f"formula={formula}")
            lib.logger.debug(f"postfix={postfix}")
        return postfix

    @classmethod
    def parse(cls, formula: str) -> Function.Node:
        """Creates a node representing a binary expression tree from the given formula
        @param formula is the right-hand side of a mathematical equation
        expressed in infix notation
        @return a node representing a binary expression tree from the given formula
        @throws fl::Exception if the formula has syntax errors.
        """
        from . import lib

        postfix = cls.infix_to_postfix(formula)
        stack: list[Function.Node] = []
        factory = lib.factory_manager.function

        for token in postfix.split():
            element: Function.Element | None = (
                factory.objects[token] if token in factory.objects else None
            )
            is_operand = not element and token not in {"(", ")", ","}

            if element:
                if element.arity > len(stack):
                    raise SyntaxError(
                        f"function element {element.name} has arity {element.arity}, "
                        f"but the size of the stack is {len(stack)}"
                    )
                node = Function.Node(factory.copy(token))
                if element.arity >= 1:
                    node.right = stack.pop()
                if element.arity == 2:
                    node.left = stack.pop()
                stack.append(node)
            elif is_operand:
                try:
                    node = Function.Node(constant=Op.scalar(token))
                except ValueError:
                    node = Function.Node(variable=token)
                stack.append(node)

        if len(stack) != 1:
            raise SyntaxError(f"invalid formula: '{formula}'")

        if lib.debugging:
            lib.logger.debug("-" * 20)
            lib.logger.debug(f"postfix={postfix}")
            lib.logger.debug(
                "\n  ".join(Op.describe(node, class_hierarchy=False) for node in stack)
            )
        return stack[-1]
