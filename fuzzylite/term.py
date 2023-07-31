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
    "Arc",
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
    "SemiEllipse",
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
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Callable, SupportsFloat, Union

import numpy as np

from .exporter import FllExporter
from .library import array, inf, nan, representation, scalar, settings, to_float
from .norm import SNorm, TNorm, UnboundedSum
from .operation import Op
from .types import Scalar, ScalarArray

if typing.TYPE_CHECKING:
    from .engine import Engine


class Term(ABC):
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
        """@return term in the FuzzyLite Language."""
        return representation.fll.term(self)

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = vars(self).copy()
        if Op.is_close(self.height, 1.0):
            fields.pop("height")
        return representation.as_constructor(self, fields, positional=True)

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
        if not Op.is_close(self.height, 1.0):
            result.append(Op.str(self.height))
        return " ".join(result)

    def configure(  # noqa: B027  empty method in an abstract base class
        self, parameters: str
    ) -> None:
        """Configures the term with the given takes_parameters. The takes_parameters are
        separated by spaces. If there is one additional parameter, the
        parameter will be considered as the height of the term; otherwise,
        the height will be set to $1.0$
        :param parameters is the takes_parameters to configure the term.
        """
        pass

    def _parse(
        self, required: int, parameters: str, *, height: bool = True
    ) -> list[float]:
        """Parses the required values from the parameters
        @param required is the number of values to parse
        @param parameters is the text containing the values
        @param height whether the parameters contain an extra value for the height of the term
        @return a list of values parsed from the parameters.
        """
        values = [to_float(x) for x in parameters.split()]
        if height and len(values) == required:
            values.append(1.0)
        if len(values) == required + height:
            return values
        height_message = f" (or {required + 1} including height)" if height else ""
        raise ValueError(
            f"expected {required} parameters"
            f"{height_message}"
            f", but got {len(values)}: '{parameters}'",
        )

    @abstractmethod
    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the has_membership function value at $x$
        :param x
        :return the has_membership function value $\mu(x)$.
        """
        raise NotImplementedError()

    def update_reference(  # noqa: B027  empty method in an abstract base class
        self, engine: Engine | None
    ) -> None:
        """Updates the references (if any) to point to the current engine (useful
        when cloning engines or creating terms within Importer objects
        :param engine: is the engine to which this term belongs to.
        """
        pass

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        r"""For monotonic terms, computes the tsukamoto value of the term for the
        given activation degree $\alpha$, that is,
        $ g_j(\alpha) = \{ z \in\mathbb{R} : \mu_j(z) = \alpha \} $. If
        the term is not monotonic (or does not override this method) the
        method computes the has_membership function $\mu(\alpha)$.
        :param y: is the activation degree
        :return the tsukamoto value of the term for the given activation degree
                if the term is monotonic (or overrides this method)
        :raises NotImplementedError if the term is not monotonic (or does not override this method).
        """
        raise NotImplementedError(
            f"expected term to implement tsukamoto, but it did not: {str(self)}"
        )

    def is_monotonic(self) -> bool:
        """Indicates whether the term is monotonic.
        :return whether the term is monotonic.
        """
        return False

    def discretize(
        self, start: float, end: float, resolution: int = 10, midpoints: bool = True
    ) -> Discrete:
        """Discretise the term.
        @param start is the start of the range
        @param end is the end of the range
        @param resolution is the number of points to discretize
        @param midpoints if true, discretize using midpoints at the given resolution,
                         otherwise discretize from start to end at the given resolution + 1.
        """
        if midpoints:
            x = Op.midpoints(start, end, resolution)
        else:
            x = np.linspace(start, end, resolution + 1, endpoint=True)
        y = self.membership(x)
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

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = {
            "term": self.term,
            "degree": self.degree,
            "implication": self.implication,
        }
        return representation.as_constructor(self, fields)

    @property
    def degree(self) -> Scalar:
        """Gets the activation degree of the term."""
        return self._degree

    @degree.setter
    def degree(self, value: Scalar) -> None:
        """Sets the activation degree of the term, replacing {nan: 0.0, -inf: 0.0, inf: 1.0}.
        Note: nan input values may result in nan activations, which would result in nan defuzzifications,
        thus we replace nan with 0.0 to avoid activations, and sensibly replace infinity values if they were present.
        """
        # TODO: Reconsider this.
        self._degree = np.nan_to_num(value, nan=0.0, neginf=0.0, posinf=1.0)

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"degree implication term"`.
        """
        name = self.term.name if self.term else "none"
        if self.implication:
            result = f"{self.implication}({Op.str(self.degree)},{name})"
        else:
            result = f"({Op.str(self.degree)}*{name})"
        return result

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the implication of the activation degree and the membership
        function value of $x$
        @param x is a value
        @return $d \otimes \mu(x)$, where $d$ is the activation degree.
        """
        if not self.implication:
            raise ValueError("expected an implication operator, but none found")
        y = self.implication.compute(
            np.atleast_2d(self.degree).T,
            self.term.membership(x),
        )
        return y.squeeze()  # type:ignore


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
        self.terms = list(terms) if terms else []

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = vars(self).copy()
        fields.pop("height")
        return representation.as_constructor(self, fields)

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

        y = scalar(0.0)
        for term in self.terms:
            y = self.aggregation.compute(y, term.membership(x))  # type: ignore
        return y

    def grouped_terms(self) -> list[Activated]:
        """@return list of Activated terms grouped by term and aggregated their degrees with the Aggregation operator
        (or UnboundedSum, if none).

        Used by `WeightedDefuzzifier`s to aggregate multiple activations of the same term before defuzzification
        """
        aggregation = self.aggregation or UnboundedSum()
        groups: dict[str, Activated] = {}
        for activated in self.terms:
            if activated.term.name not in groups:
                groups[activated.term.name] = Activated(
                    activated.term, activated.degree, implication=None
                )
                continue
            aggregated_term = groups[activated.term.name]
            aggregated_term.degree = aggregation.compute(
                aggregated_term.degree, activated.degree
            )
        return list(groups.values())

    def activation_degree(self, term: Term) -> Scalar:
        """Computes the aggregated activation degree for the given term.

        @param term is the term for which to compute the aggregated
        activation degree
        @return the aggregated activation degree for the given term.
        """
        for activated in self.grouped_terms():
            if activated.term == term:
                return activated.degree
        return scalar(0.0)

    def highest_activated_term(self) -> Activated | None:
        """Find the term with the maximum activation degree from the list of grouped activated terms.

        @return the term with the maximum grouped activation degree.
        """
        highest: Activated | None = None
        for activated in self.grouped_terms():
            if highest is None or activated.degree > highest.degree:
                highest = activated
        return highest

    def clear(self) -> None:
        """Clears the list of activated terms."""
        self.terms.clear()


class Arc(Term):
    """The Arc class is an edge term that represents the arc-shaped membership function."""

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param start is the start of the term
        @param end is the end of the term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function value of $x$."""
        x = scalar(x)
        s = self.start
        e = self.end
        r = e - s
        c = s + r
        left = s > e
        right = s < e
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                (left & (c <= x) & (x <= s)) | (right & (s <= x) & (x <= c)),
                np.sqrt(r**2 - np.square(x - c)) / abs(r),
                (left & (x < e)) | (right & (x > e)),
            )
        )
        return y  # type: ignore

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        """Computes the tsukamoto function value of $y$."""
        y = scalar(y)
        h = self.height
        s = self.start
        e = self.end
        r = e - s
        c = s + r
        sign = -1 if s < e else 1
        x = c + sign * np.sqrt(r**2 - np.square(y * r / h))
        return x  # type: ignore

    def is_monotonic(self) -> bool:
        """Returns True because the term is monotonic."""
        return True

    def parameters(self) -> str:
        """Returns the parameters of the term."""
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the given parameters: start end [height]."""
        self.start, self.end, self.height = self._parse(2, parameters)


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
        c = self.center
        w = self.width
        s = self.slope
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * (1.0 / (1.0 + np.power(np.abs((x - c) / w), 2.0 * s)))
        )
        return y

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width slope [height]"`.
        """
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width slope [height]"`.
        """
        self.center, self.width, self.slope, self.height = self._parse(3, parameters)


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
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(right | left, 1.0, 0.0)
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start direction [height]"`.
        """
        return super()._parameters(self.start, self.direction)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start direction [height]"`.
        """
        self.start, self.direction, self.height = self._parse(2, parameters)


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
        i = self.inflection
        e = self.end
        increasing = (i <= e) & (x < e)
        decreasing = (i >= e) & (x > e)
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                increasing,
                (e - i) / (2.0 * e - i - x),
                np.where(
                    decreasing,
                    (i - e) / (i - 2.0 * e + x),
                    1.0,
                ),
            )
        )
        return y  # type: ignore

    def is_monotonic(self) -> bool:
        """Returns True because this term is monotonic."""
        return True

    def tsukamoto(self, y: Scalar) -> Scalar:
        """Returns the tsukamoto value of the term."""
        # The equation is the same for increasing and decreasing.
        y = scalar(y)
        h = self.height
        i = self.inflection
        e = self.end
        x = h * (i - e) / y + 2 * e - i
        return x

    def parameters(self) -> str:
        """Returns the parameters of the term as
        @return `"inflection end [height]"`.
        """
        return super()._parameters(self.inflection, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters given
        @param parameters as `"inflection end [height]"`.
        """
        self.inflection, self.end, self.height = self._parse(2, parameters)


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

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = vars(self).copy()
        fields.pop("height")
        return representation.as_constructor(self, fields, positional=True)

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function evaluated at $x$
        @param x is irrelevant
        @return $c$, where $c$ is the constant value.
        """
        y = np.full_like(x, fill_value=self.value)
        return y

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"value"`.
        """
        return super()._parameters(self.value)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"value"`.
        """
        self.value = self._parse(1, parameters, height=False)[0]


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
        c = self.center
        w = self.width
        within = (x >= c - 0.5 * w) & (x <= c + 0.5 * w)
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                np.isfinite(x) & within,
                0.5 * (1.0 + np.cos(2.0 / w * np.pi * (x - c))),
                0.0,
            )
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width [height]"`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width [height]"`.
        """
        self.center, self.width, self.height = self._parse(2, parameters)


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

    Floatable = Union[SupportsFloat, str]

    def __init__(
        self,
        name: str = "",
        values: ScalarArray | Sequence[Floatable] | None = None,
        height: float = 1.0,
    ) -> None:
        """Create the term.
        @param name is the name of the term
        @param values is an 2D array of x,y pairs
        @param height is the height of the term.
        """
        super().__init__(name, height)
        if isinstance(values, Sequence):
            x = [to_float(xi) for xi in values[0::2]]
            y = [to_float(yi) for yi in values[1::2]]
            values = scalar([x, y]).T
        elif values is None:
            values = np.atleast_2d(scalar([]))
        self.values = values

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
        if self.values.size == 0:
            raise ValueError("expected xy to contain coordinate pairs, but it is empty")
        if self.values.ndim != 2:
            raise ValueError(
                "expected xy to have with 2 columns, "
                f"but got {self.values.ndim} in shape {self.values.shape}: {self.values}"
            )
        y = self.height * np.interp(scalar(x), self.values[:, 0], self.values[:, 1])
        return y

    def tsukamoto(self, y: Scalar) -> Scalar:
        """Not implemented."""
        # todo: approximate tsukamoto if monotonic
        raise NotImplementedError()

    def parameters(self) -> str:
        """Returns the parameters of the term as `x1 y1 xn yn [height]`
        @return `x1 y1 xn yn [height]`.
        """
        return super()._parameters(self.to_list())

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters given as `x1 y1 xn yn [height]`
        @param parameters as `x1 y1 xn yn [height]`.
        """
        as_list = parameters.split()
        if len(as_list) % 2 == 0:
            self.height = 1.0
        else:
            self.height = to_float(as_list[-1])
            del as_list[-1]
        self.values = Discrete.to_xy(as_list[0::2], as_list[1::2])

    def x(self) -> ScalarArray:
        """An iterable containing the $x$ values
        @return an iterable containing the $x$ values.
        """
        return self.values[:, 0]

    def y(self) -> ScalarArray:
        """An iterable containing the $y$ values
        @return an iterable containing the $y$ values.
        """
        return self.values[:, 1]

    def sort(self) -> None:
        """Ascendantly sorts the pairs of values in this Discrete term by the
        $x$-coordinate.
        """
        self.values[:] = self.values[np.argsort(self.x())]

    def to_dict(self) -> dict[float, float]:
        """Returns a dictionary of values {x: y}."""
        return dict(zip(self.x(), self.y()))

    def to_list(self) -> list[float]:
        """Returns a list of values [x1, y1, x2, y2, ...]."""
        return self.values.flatten().tolist()  # type: ignore

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
        @returns a Discrete term.
        """
        x: Scalar = scalar(0)
        y: Scalar = scalar(0)
        if isinstance(xy, str):
            xy = xy.split()
        if isinstance(xy, Sequence):
            x = scalar(xy[0::2])
            y = scalar(xy[1::2])
        if isinstance(xy, tuple):
            x, y = map(scalar, xy)
        if isinstance(xy, dict):
            x = scalar([xi for xi in xy])
            y = scalar([yi for yi in xy.values()])
        return Discrete(name, Discrete.to_xy(x, y), height=height)

    @staticmethod
    def to_xy(x: Any, y: Any) -> ScalarArray:
        """Creates a list of values from the given parameters.
        @param x is the x-coordinate(s) that can be converted into scalar(s)
        @param y is the y-coordinate(s) that can be converted into scalar(s)
        @return an array of n-rows and 2-columns (n,2).
        """
        x = array(x, dtype=settings.float_type)
        y = array(y, dtype=settings.float_type)
        if x.shape != y.shape:
            raise ValueError(
                f"expected same shape from x and y, but found x={x.shape} and y={y.shape}"
            )
        return array([x, y]).T


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
        m = self.mean
        std = self.standard_deviation
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.exp(-np.square(x - m) / (2.0 * std**2))
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"mean standardDeviation [height]"`.
        """
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"mean standardDeviation [height]"`.
        """
        self.mean, self.standard_deviation, self.height = self._parse(2, parameters)


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
        ma = self.mean_a
        stda = self.standard_deviation_a
        mb = self.mean_b
        stdb = self.standard_deviation_b
        a = np.where(
            x < ma,
            np.exp((-(np.square(x - ma))) / (2.0 * stda**2)),
            1.0,
        )
        b = np.where(
            x > mb,
            np.exp((-(np.square(x - mb))) / (2.0 * stdb**2)),
            1.0,
        )
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * a * b
        return y  # type: ignore

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
        (
            self.mean_a,
            self.standard_deviation_a,
            self.mean_b,
            self.standard_deviation_b,
            self.height,
        ) = self._parse(4, parameters)


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

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = vars(self).copy()
        fields.pop("height")
        fields.pop("engine")
        return representation.as_constructor(self, fields, positional=True)

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the linear function $f(x)=\sum_i c_iv_i +k$,
        where $v_i$ is the value of the input variable $i$ registered
        in the Linear::getEngine()
        @param x is not used
        @return $\sum_i c_ix_i +k$.
        """
        if not self.engine:
            raise ValueError("expected reference to an engine, but found none")

        if len(self.coefficients) not in {
            len(self.engine.input_variables),
            len(self.engine.input_variables) + 1,
        }:
            raise ValueError(
                f"expected {len(self.engine.input_variables)} (+1) coefficients "
                "(one for each input variable plus an optional constant), "
                f"but found {len(self.coefficients)} coefficients: {self.coefficients}"
            )
        coefficients = scalar([self.coefficients[: len(self.engine.input_variables)]])
        constant = (
            self.coefficients[-1]
            if len(self.coefficients) > len(self.engine.input_variables)
            else 0.0
        )
        inputs = self.engine.input_values
        y = (coefficients * inputs).sum(axis=1, keepdims=False) + constant
        return y  # type:ignore

    def configure(self, parameters: str) -> None:
        r"""Configures the term with the values of $\mathbf{c}^\star$
        @param parameters as `"c1 ... ci ... cn k"`.
        """
        self.coefficients = [to_float(p) for p in parameters.split()]

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
        bl = self.bottom_left
        tl = self.top_left
        br = self.bottom_right
        tr = self.top_right
        s_shape = np.where(
            x <= bl,
            0.0,
            np.where(
                x <= 0.5 * (bl + tl),
                2.0 * np.square((x - bl) / (tl - bl)),
                np.where(
                    x < tl,
                    1.0 - 2.0 * np.square((x - tl) / (tl - bl)),
                    1.0,
                ),
            ),
        )
        z_shape = np.where(
            x <= tr,
            1.0,
            np.where(
                x <= 0.5 * (tr + br),
                1.0 - 2.0 * np.square((x - tr) / (br - tr)),
                np.where(
                    x < br,
                    2.0 * np.square((x - br) / (br - tr)),
                    0.0,
                ),
            ),
        )
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * s_shape * z_shape
        return y  # type: ignore

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
        (
            self.bottom_left,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.height,
        ) = self._parse(4, parameters)


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
        s = self.start
        e = self.end
        increasing = s < e
        decreasing = s > e
        y = (
            self.height
            * np.where(np.isnan(x) | (increasing == decreasing), np.nan, 1.0)
            * np.where(
                increasing & (e > x) & (x > s),
                (x - s) / (e - s),
                np.where(
                    decreasing & (e < x) & (x < s),
                    (s - x) / (s - e),
                    (increasing & (x >= e)) | (decreasing & (x <= e)),
                ),
            )
        )
        return y  # type: ignore

    def is_monotonic(self) -> bool:
        """Returns True as this term is monotonic."""
        return True

    def tsukamoto(self, y: Scalar) -> Scalar:
        """Returns the Tsukamoto value of the term."""
        y = scalar(y)
        h = self.height
        s = self.start
        e = self.end
        x = s + (e - s) * y / h
        return x

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


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
        s = min(self.start, self.end)
        e = max(self.start, self.end)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * ((s <= x) & (x <= e))
        return y

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"start end [height]"`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"start end [height]"`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class SemiEllipse(Term):
    """The SemiEllipse class is a basic Term that represents the semi-ellipse membership function."""

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Create the term."""
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function evaluated at $x$."""
        x = scalar(x)
        s = min(self.start, self.end)
        e = max(self.start, self.end)
        r = (e - s) / 2
        c = s + r
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                (x >= s) & (x <= e),
                np.sqrt(r**2 - np.square(x - c)) / r,
                0,
            )
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term."""
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters: start end [height]."""
        self.start, self.end, self.height = self._parse(2, parameters)


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
        i = self.inflection
        s = self.slope
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            / (1.0 + np.exp(-s * (x - i)))
        )
        return y

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        """Returns the Tsukamoto value of the term."""
        y = scalar(y)
        h = self.height
        i = self.inflection
        s = self.slope
        x = i + np.log(h / y - 1.0) / -s
        return x

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
        self.inflection, self.slope, self.height = self._parse(2, parameters)


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
        left = self.left
        right = self.right
        rise = self.rising
        fall = self.falling
        a = 1.0 / (1.0 + np.exp(-rise * (x - left)))
        b = 1.0 / (1.0 + np.exp(-fall * (x - right)))
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * np.abs(a - b)
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left rising falling right [height]"`.
        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left rising falling right [height]"`.
        """
        (
            self.left,
            self.rising,
            self.falling,
            self.right,
            self.height,
        ) = self._parse(4, parameters)


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
        left = self.left
        right = self.right
        rise = self.rising
        fall = self.falling
        a = 1.0 + np.exp(-rise * (x - left))
        b = 1.0 + np.exp(-fall * (x - right))
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) / (a * b)
        return y

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left rising falling right [height]"`.
        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left rising falling right [height]"`.
        """
        (
            self.left,
            self.rising,
            self.falling,
            self.right,
            self.height,
        ) = self._parse(4, parameters)


class Spike(Term):
    """The Spike class is an extended Term that represents the spike membership
    function.
    @image html spike.svg
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
        x = scalar(x)
        c = self.center
        w = self.width
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.exp(-np.abs(10.0 / w * (x - c)))
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"center width [height]"`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"center width [height]"`.
        """
        self.center, self.width, self.height = self._parse(2, parameters)


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
        s = self.start
        e = self.end
        s_shape = np.where(
            x <= self.start,
            0.0,
            np.where(
                x <= 0.5 * (s + e),
                2.0 * np.square((x - s) / (e - s)),
                np.where(
                    x < e,
                    1.0 - 2.0 * np.square((x - e) / (e - s)),
                    1.0,
                ),
            ),
        )
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * s_shape
        return y  # type: ignore

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        """Computes the Tsukamoto activation degree of the term
        @param y is the activation degree of the term
        @return the Tsukamoto activation degree of the term.
        """
        y = scalar(y)
        h = self.height
        s = self.start
        e = self.end
        x = np.where(
            y <= h / 2.0,
            s + (e - s) * np.sqrt(y / (2 * h)),
            e - (e - s) * np.sqrt((h - y) / (2 * h)),
        )
        return x

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
        self.start, self.end, self.height = self._parse(2, parameters)


class Trapezoid(Term):
    """The Trapezoid class is a basic Term that represents the trapezoidal
    membership function.
    @image html trapezoid.svg
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
        if Op.isnan(top_right) and Op.isnan(bottom_right):
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
        bl = self.bottom_left
        br = self.bottom_right
        tl = self.top_left
        tr = self.top_right
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                ((x < bl) | (x > br)),
                0.0,
                np.where(
                    ((tl <= x) & (x <= tr))
                    | ((bl == -inf) & (x < tl))
                    | ((br == inf) & (x > tr)),
                    1.0,
                    np.where(
                        x < tl,
                        (x - bl) / (tl - bl),
                        np.where(
                            x > tr,
                            (br - x) / (br - tr),
                            nan,
                        ),
                    ),
                ),
            )
        )
        return y  # type: ignore

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
        (
            self.bottom_left,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.height,
        ) = self._parse(4, parameters)


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
        if Op.isnan(right):
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
        a = self.left
        b = self.top
        c = self.right
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                (x < a) | (x > c),
                0.0,
                np.where(
                    (x == b) | ((a == -inf) & (x < b)) | ((c == inf) & (x > b)),
                    1.0,
                    np.where(
                        x < b,
                        (x - a) / (b - a),
                        np.where(
                            x > b,
                            (c - x) / (c - b),
                            nan,
                        ),
                    ),
                ),
            )
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Returns the parameters of the term
        @return `"left top right [height]"`.
        """
        return super()._parameters(self.left, self.top, self.right)

    def configure(self, parameters: str) -> None:
        """Configures the term with the parameters
        @param parameters as `"left top right [height]"`.
        """
        self.left, self.top, self.right, self.height = self._parse(3, parameters)


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
        s = self.start
        e = self.end
        z_shape = np.where(
            x <= s,
            1.0,
            np.where(
                x < 0.5 * (s + e),
                1.0 - 2.0 * np.square((x - s) / (e - s)),
                np.where(
                    x < e,
                    2.0 * np.square((x - e) / (e - s)),
                    0.0,
                ),
            ),
        )
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * z_shape
        return y  # type: ignore

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        """Computes the Tsukamoto inference of the term with the given."""
        y = scalar(y)
        h = self.height
        s = self.start
        e = self.end
        x = np.where(
            y <= h / 2.0,
            e - (e - s) * np.sqrt(y / (2 * h)),
            s + (e - s) * np.sqrt((h - y) / (2 * h)),
        )
        return x

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
        self.start, self.end, self.height = self._parse(2, parameters)


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

            Operator = enum.auto()
            Function = enum.auto()

            def __repr__(self) -> str:
                """@return Python code to construct the type."""
                return f"'{self.name}'"

        def __init__(
            self,
            name: str,
            description: str,
            type: Function.Element.Type | str,
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
            self.type = (
                type
                if isinstance(type, Function.Element.Type)
                else Function.Element.Type[type]
            )
            self.method = method
            self.arity = arity
            self.precedence = precedence
            self.associativity = associativity

        def __repr__(self) -> str:
            """@return Python code to construct the element."""
            return representation.as_constructor(self)

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

        def __repr__(self) -> str:
            """@return Python code to construct the node."""
            return representation.as_constructor(self)

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
            """Evaluates the node and substitutes the variables therein for the values in the local variables (if any).
            The expression tree is evaluated recursively.
            @param local_variables is a map of substitutions of variable names for scalars
            @return a scalar corresponding to the result of the evaluation.
            """
            result = scalar(nan)
            if self.element:
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
                        f"but the map contains: {representation.repr(local_variables)}"
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

            if not Op.isnan(node.constant):
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

            if not Op.isnan(node.constant):
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

            if not Op.isnan(node.constant):
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
        self.variables: dict[str, Scalar] = variables.copy() if variables else {}
        if load:
            self.load()

    def __repr__(self) -> str:
        """@return Python code to construct the term."""
        fields = vars(self).copy()
        fields.pop("height")
        fields.pop("root")
        fields.pop("engine")
        if not self.variables:
            fields.pop("variables")
        return representation.as_constructor(self, fields, positional=True)

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
        if not self.is_loaded():
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
        y = self.evaluate(engine_variables)
        return y

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
        from .factory import FunctionFactory
        from .rule import Rule

        factory: FunctionFactory = settings.factory_manager.function
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
        from .factory import FunctionFactory

        formula = cls.format_infix(formula)
        factory: FunctionFactory = settings.factory_manager.function

        from collections import deque

        queue: deque[str] = deque()
        stack: list[str] = []

        for token in formula.split():
            if settings.debugging:
                settings.logger.debug("=" * 20)
                settings.logger.debug(f"formula: {formula}")
                settings.logger.debug(f"queue: {queue}")
                settings.logger.debug(f"stack: {stack}")

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
        if settings.debugging:
            settings.logger.debug(f"formula={formula}")
            settings.logger.debug(f"postfix={postfix}")
        return postfix

    @classmethod
    def parse(cls, formula: str) -> Function.Node:
        """Creates a node representing a binary expression tree from the given formula
        @param formula is the right-hand side of a mathematical equation
        expressed in infix notation
        @return a node representing a binary expression tree from the given formula
        @throws fl::Exception if the formula has syntax errors.
        """
        postfix = cls.infix_to_postfix(formula)
        stack: list[Function.Node] = []
        factory = settings.factory_manager.function

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
                    node = Function.Node(constant=to_float(token))
                except ValueError:
                    node = Function.Node(variable=token)
                stack.append(node)

        if len(stack) != 1:
            raise SyntaxError(f"invalid formula: '{formula}'")

        if settings.debugging:
            settings.logger.debug("-" * 20)
            settings.logger.debug(f"postfix={postfix}")
            settings.logger.debug(
                "\n  ".join(Op.describe(node, class_hierarchy=False) for node in stack)
            )
        return stack[-1]
