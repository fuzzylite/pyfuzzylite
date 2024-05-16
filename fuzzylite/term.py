"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
from .types import Array, Scalar, ScalarArray

if typing.TYPE_CHECKING:
    from .engine import Engine


class Term(ABC):
    """Abstract class for linguistic terms.

    The linguistic terms in this library can be divided into four groups, namely `basic`, `extended`, `edge`, and `function`.

    | `basic/function`                       	| `extended`                                  	| `edge`                                               	|
    |----------------------------------------	|---------------------------------------------	|------------------------------------------------------	|
    | [fuzzylite.term.Discrete][]            	| [fuzzylite.term.Bell][]                     	| [fuzzylite.term.Arc][]                               	|
    | ![](../../image/term/Discrete.svg )    	| ![](../../image/term/Bell.svg)              	| ![](../../image/term/Arc.svg)                        	|
    | [fuzzylite.term.Rectangle][]           	| [fuzzylite.term.Cosine][]                   	| [fuzzylite.term.Binary][]                            	|
    | ![](../../image/term/Rectangle.svg)    	| ![](../../image/term/Cosine.svg)            	| ![](../../image/term/Binary.svg)                     	|
    | [fuzzylite.term.SemiEllipse][]         	| [fuzzylite.term.Gaussian][]                 	| [fuzzylite.term.Concave][]                           	|
    | ![](../../image/term/SemiEllipse.svg ) 	| ![](../../image/term/Gaussian.svg)          	| ![](../../image/term/Concave.svg)                    	|
    | [fuzzylite.term.Triangle][]            	| [fuzzylite.term.GaussianProduct][]          	| [fuzzylite.term.Ramp][]                              	|
    | ![](../../image/term/Triangle.svg )    	| ![](../../image/term/GaussianProduct.svg)   	| ![](../../image/term/Ramp.svg)                       	|
    | [fuzzylite.term.Trapezoid][]           	| [fuzzylite.term.PiShape][]                  	| [fuzzylite.term.Sigmoid][]                           	|
    | ![](../../image/term/Trapezoid.svg)    	| ![](../../image/term/PiShape.svg)           	| ![](../../image/term/Sigmoid.svg)                    	|
    | [fuzzylite.term.Constant][]            	| [fuzzylite.term.SigmoidDifference][]        	| [fuzzylite.term.SShape][]	- [fuzzylite.term.ZShape][] 	|
    | ![](../../image/term/Constant.svg)     	| ![](../../image/term/SigmoidDifference.svg) 	| ![](../../image/term/ZShape - SShape.svg)            	|
    | [fuzzylite.term.Linear][]              	| [fuzzylite.term.SigmoidProduct][]           	| [fuzzylite.term.SShape][]                            	|
    | ![](../../image/term/Linear.svg)       	| ![](../../image/term/SigmoidProduct.svg)    	| ![](../../image/term/SShape.svg)                     	|
    | [fuzzylite.term.Function][]            	| [fuzzylite.term.Spike][]                    	| [fuzzylite.term.ZShape][]                            	|
    | ![](../../image/term/Function.svg)     	| ![](../../image/term/Spike.svg)             	| ![](../../image/term/ZShape.svg)                     	|
    |                                        	|                                             	|                                                      	|


    info: related
        - [fuzzylite.variable.Variable][]
        - [fuzzylite.variable.InputVariable][]
        - [fuzzylite.variable.OutputVariable][]
    """

    def __init__(self, name: str = "", height: float = 1.0) -> None:
        """Constructor.

        Args:
            name: name of the term
            height: height of the term.
        """
        self.name = name
        self.height = height

    def __str__(self) -> str:
        """Return the code to construct the term in the FuzzyLite Language.

        Returns:
            code to construct the term in the FuzzyLite Language.
        """
        return representation.fll.term(self)

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = vars(self).copy()
        if Op.is_close(self.height, 1.0):
            fields.pop("height")
        return representation.as_constructor(self, fields, positional=True)

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            list of space-separated parameters of the term.
        """
        return self._parameters()

    def _parameters(self, *args: object) -> str:
        """Concatenate the arguments and the height.

        Args:
            *args: arguments to configure the term

        Returns:
             parameters concatenated and an optional `height` at the end.
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
        """Configure the term with the parameters.

        The `parameters` is a list of space-separated values,
        with an optional value at the end to set the `height` (defaults to `1.0` if absent)

        Args:
             parameters: space-separated parameter values to configure the term.
        """
        pass

    def _parse(self, required: int, parameters: str, *, height: bool = True) -> list[float]:
        """Parse the required values from the parameters.

        Args:
            required: number of values to parse
            parameters: text containing the values
            height: whether `parameters` contains an extra value for the height of the term

        Returns:
             list of floating-point values parsed from the parameters.
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
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
             membership function value $\mu(x)$.
        """

    def update_reference(  # noqa: B027  empty method in an abstract base class
        self, engine: Engine | None
    ) -> None:
        """Update the reference (if any) to the engine the term belongs to.

        Args:
             engine: engine to which the term belongs to.
        """
        pass

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $g(y) = \{ z \in\mathbb{R} : \mu(z) = y \}$

        Warning:
            Raises `RuntimeError` because the term does not support Tsukamoto

        Args:
            y: activation degree

        Raises:
            RuntimeError: because the term does not support Tsukamoto
        """
        raise RuntimeError(f"term does not support tsukamoto: {str(self)}")

    def is_monotonic(self) -> bool:
        """Return whether the term is monotonic.

        Returns:
             False.
        """
        return False

    def discretize(
        self, start: float, end: float, resolution: int = 10, midpoints: bool = True
    ) -> Discrete:
        """Discretize the term.

        Args:
            start: start of the range
            end: end of the range
            resolution: number of points to discretize
            midpoints: use midpoints method or include start and end.

        info: related
            - [fuzzylite.operation.Operation.midpoints][]
            - [numpy.linspace][]
        """
        if midpoints:
            x = Op.midpoints(start, end, resolution)
        else:
            x = np.linspace(start, end, resolution + 1, endpoint=True)
        y = self.membership(x)
        xy = Discrete.to_xy(x, y)
        return Discrete(self.name, xy)


class Activated(Term):
    r"""Special term that represents the activation of terms when processing the antecedent of a rule.

    Note: Equation
        $\mu(x) = \alpha_a \otimes \mu_a(x)$

        where

        - $\alpha_a$: activation degree of term $a$
        - $\otimes$: implication operator
        - $\mu_a$: activated term $a$

    info: related
        - [fuzzylite.term.Term][]
        - [fuzzylite.term.Aggregated][]
        - [fuzzylite.variable.OutputVariable][]
        - [fuzzylite.defuzzifier.WeightedDefuzzifier][]
    """

    def __init__(self, term: Term, degree: Scalar = 1.0, implication: TNorm | None = None) -> None:
        """Constructor.

        Args:
            term: activated term
            degree: activation degree of the term
            implication: implication operator
        """
        super().__init__("_")
        self.term = term
        self.degree = degree
        self.implication = implication

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = {
            "term": self.term,
            "degree": self.degree,
            "implication": self.implication,
        }
        return representation.as_constructor(self, fields)

    def fuzzy_value(self, padding: bool = False) -> Array[np.str_]:
        """Return fuzzy value in the form `{degree}/{name}`.

        Args:
            padding: whether to pad the degree sign (eg, `" - "` when `True` and `"-"` otherwise)

        Returns:
            fuzzy value in the form `{degree}/{name}`
        """
        y = np.atleast_1d(self.degree)
        sign = np.nan_to_num(np.sign(y), nan=0.0)
        pad = " " if padding else ""
        y_str = np.char.add(
            np.where(sign >= 0, f"{pad}+{pad}" if pad else "", f"{pad}-{pad}"),
            array(list(map(Op.str, np.absolute(y))), dtype=np.str_),
        )
        value = np.char.add(y_str, f"/{self.term.name}")
        return value.squeeze()

    @property
    def degree(self) -> Scalar:
        """Get/Set the activation degree of the term.

        # Getter

        Returns:
            activation degree of the term.

        # Setter

        Args:
            value (Scalar): activation degree of the term, with replacements of `{nan: 0.0, -inf: 0.0, inf: 1.0}`

        Note:
            replacements of `{nan: 0.0, -inf: 0.0, inf: 1.0}` are made to sensibly deal with non-finite activations (eg, `NaN` input values)
        """
        return self._degree

    @degree.setter
    def degree(self, value: Scalar) -> None:
        """Set the activation degree of the term, with replacements of `{nan: 0.0, -inf: 0.0, inf: 1.0}`.

        Args:
              value: activation degree of the term, with replacements of `{nan: 0.0, -inf: 0.0, inf: 1.0}`

        Warning:
            nan input values may result in nan activations, which result in nan defuzzifications,
            so we replace nan with 0.0 to avoid activations, and sensibly replace infinity values if they are present.
        """
        self._degree = np.nan_to_num(value, nan=0.0, neginf=0.0, posinf=1.0)

    def parameters(self) -> str:
        """Return the space-separated parameters of the term.

        Returns:
            `degree * term` if not implication else `implication(degree, term)`
        """
        degree = (
            f"[{Op.str(self.degree, delimiter=', ')}]"
            if np.size(self.degree) > 1
            else Op.str(self.degree)
        )
        if self.implication:
            result = f"{self.implication}({degree},{self.term.name})"
        else:
            result = f"({degree}*{self.term.name})"
        return result

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the implication of the activation degree and the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \alpha_a \otimes \mu_a(x)$
        """
        if not self.implication:
            raise ValueError("expected an implication operator, but found none")
        y = self.implication.compute(
            np.atleast_2d(self.degree).T,
            self.term.membership(x),
        )
        return y.squeeze()  # type:ignore


class Aggregated(Term):
    r"""Special term that represents a fuzzy set of activated terms to mainly serve as the fuzzy output value of output variables.

    Note: Equation
        $\mu(x)=\bigoplus_i^n\alpha_i\otimes\mu_i(x) = \alpha_1\otimes\mu_1(x) \oplus \ldots \oplus \alpha_n\otimes\mu_n(x)$

        where

        - $\alpha_i$: activation degree of term $i$
        - $\mu_i$: membership function of term $i$
        - $\otimes$: implication operator
        - $\oplus$: aggregation operator

    info: related
        - [fuzzylite.term.Activated][]
        - [fuzzylite.variable.OutputVariable][]
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.term.Term][]
    """

    def __init__(
        self,
        name: str = "",
        minimum: float = nan,
        maximum: float = nan,
        aggregation: SNorm | None = None,
        terms: Iterable[Activated] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: name of the aggregated term
            minimum: minimum value of the range of the fuzzy set
            maximum: maximum value of the range of the fuzzy set
            aggregation: aggregation operator
            terms: list of activated terms
        """
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum
        self.aggregation = aggregation
        self.terms = list(terms or [])

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = vars(self).copy()
        fields.pop("height")
        return representation.as_constructor(self, fields)

    def parameters(self) -> str:
        """Return the space-separated parameters of the term.

        Returns:
            `aggregation minimum maximum terms`
        """
        result = []
        activated = [term.parameters() for term in self.terms]
        if self.aggregation:
            result.append(f"{FllExporter().norm(self.aggregation)}[{','.join(activated)}]")
        else:
            result.append(f"[{'+'.join(activated)}]")

        return " ".join(result)

    def range(self) -> float:
        """Return the magnitude of the range of the fuzzy set.

        Returns:
             `maximum - minimum`
        """
        return self.maximum - self.minimum

    def membership(self, x: Scalar) -> Scalar:
        r"""Aggregate the activated terms' membership function values of $x$ using the aggregation operator.

        Args:
            x: scalar

        Returns:
            $\mu(x)=\bigoplus_i^n\alpha_i\otimes\mu_i(x) = \alpha_1\otimes\mu_1(x) \oplus \ldots \oplus \alpha_n\otimes\mu_n(x)$
        """
        if self.terms and not self.aggregation:
            raise ValueError("expected an aggregation operator, but found none")

        y = scalar(0.0)
        for term in self.terms:
            y = self.aggregation.compute(y, term.membership(x))  # type: ignore
        return y

    def grouped_terms(self) -> dict[str, Activated]:
        """Group the activated terms and aggregate their activation degrees.

        Returns:
             grouped activated terms by name with aggregated activation degrees.

        info: related
            - [fuzzylite.defuzzifier.WeightedSum][]
            - [fuzzylite.defuzzifier.WeightedAverage][]
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
            aggregated_term.degree = aggregation.compute(aggregated_term.degree, activated.degree)
        return groups

    def activation_degree(self, term: Term) -> Scalar:
        """Compute the aggregated activation degree of the term.

        Args:
            term: term to compute the aggregated activation degree

        Returns:
             aggregated activation degree for the term.
        """
        activated = self.grouped_terms().get(term.name)
        return activated.degree if activated else scalar(0.0)

    def highest_activated_term(self) -> Activated | None:
        """Find the term with the maximum aggregated activation degree.

        Returns:
            term with the maximum aggregated activation degree.

        Raises:
            ValueError: when working with vectorization (eg, size(activation_degree) > 1)
        """
        highest: Activated | None = None
        for activated in self.grouped_terms().values():
            if (size := np.size(activated.degree)) > 1:
                raise ValueError(
                    f"expected a unit scalar, but got vector of size {size}: {activated.degree=}"
                )
            if (highest is None and activated.degree > 0.0) or (
                highest and activated.degree > highest.degree
            ):
                highest = activated
        return highest

    def clear(self) -> None:
        """Clear the list of activated terms."""
        self.terms.clear()


class Arc(Term):
    r"""Edge term that represents the arc-shaped membership function.

    ![](../../image/term/Arc.svg)

    Note: Equation
        $\mu(x)=\dfrac{h\sqrt{r^2 - (x-c)^2}}{|r|}$

        where

        - $h$: height of the Term
        - $r$: radius of the Arc
        - $c$: center of the Arc
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the Arc
            end: end of the Arc
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x)=\dfrac{h\sqrt{r^2 - (x-c)^2}}{|r|}$, clipped accordingly
        """
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
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y=\dfrac{h\sqrt{r^2 - (x-c)^2}}{|r|}$

            $x=c\pm \sqrt{r^2-\dfrac{yr}{h}}$

        Args:
            y: activation degree

        Returns:
            $x=c\pm \sqrt{r^2-\dfrac{yr}{h}}$
        """
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
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `start end`
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class Bell(Term):
    r"""Extended term that represents the generalized bell curve membership function.

    ![](../../image/term/Bell.svg)

    Note: Equation
        $\mu(x)=\dfrac{h}{1 + \left(\dfrac{|x-c|}{w}\right)^{2s}}$

        where

        - $h$: height of the Term
        - $c$: center of the Bell
        - $w$: width of the Bell
        - $s$: slope of the Bell
    """

    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        slope: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            center: center of the Bell
            width: width of the Bell
            slope: slope of the Bell
            height: height of the Term
        """
        super().__init__(name, height)
        self.center = center
        self.width = width
        self.slope = slope

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x)=\dfrac{h}{1 + \left(\dfrac{|x-c|}{w}\right)^{2s}}$
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
        """Return the space-separated parameters of the term.

        Returns:
            `center width slope [height]`.
        """
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `center width slope [height]`.
        """
        self.center, self.width, self.slope, self.height = self._parse(3, parameters)


class Binary(Term):
    r"""Edge Term that represents the binary membership function.

    ![](../../image/term/Binary.svg)

    Note: Equation
        $\mu(x) = \begin{cases}
            h & \mbox{if } (d=\infty \wedge x \ge s) \vee (d=-\infty \wedge x \le s) \cr
            0 & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $s$: start of the Binary
        - $d$: direction of the Binary
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        direction: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the Binary
            direction: direction of the Binary (-inf, inf)
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.direction = direction

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} h & \mbox{if } (d=\infty \wedge x \ge s) \vee (d=-\infty \wedge x \le s) \cr 0 & \mbox{otherwise} \end{cases}$
        """
        x = scalar(x)
        right = (self.direction > self.start) & (x >= self.start)
        left = (self.direction < self.start) & (x <= self.start)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * np.where(right | left, 1.0, 0.0)
        return y  # type: ignore

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `start direction [height]`.
        """
        return super()._parameters(self.start, self.direction)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start direction [height]`.
        """
        self.start, self.direction, self.height = self._parse(2, parameters)


class Concave(Term):
    r"""Edge Term that represents the concave membership function.

    ![](../../image/term/Concave.svg)

    Note: Equation
        $\mu(x) = \begin{cases}
            h  \dfrac{e - i} {2e - i - x} & \mbox{if } i \leq e \wedge x < e \mbox{ (increasing concave)} \cr
            h  \dfrac{i - e} {-2e + i + x} & \mbox{if } i > e \wedge x > e \mbox{ (decreasing concave)} \cr
            h & \mbox{otherwise} \cr
        \end{cases}$

        where

        - $h$: height of the Term
        - $i$: inflection of the Concave
        - $e$: end of the Concave
    """

    def __init__(
        self,
        name: str = "",
        inflection: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            inflection: inflection of the Concave
            end: end of the Concave
            height: height of the Term
        """
        super().__init__(name, height)
        self.inflection = inflection
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} h  \dfrac{e - i} {2e - i - x} & \mbox{if } i \leq e \wedge x < e \mbox{ (increasing concave)} \cr h  \dfrac{i - e} {-2e + i + x} & \mbox{if } i > e \wedge x > e \mbox{ (decreasing concave)} \cr h & \mbox{otherwise} \cr \end{cases}$
        """
        # TODO: Check when e=i and compare with QtFuzzyLite
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
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def tsukamoto(self, y: Scalar) -> Scalar:
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y = h \dfrac{e - i} {2e - i - x}$

            $x = h \dfrac{e-i}{y} + 2e -i$

        Args:
            y: activation degree

        Returns:
            $x = h \dfrac{e-i}{y} + 2e -i$
        """
        # The equation is the same for increasing and decreasing.
        y = scalar(y)
        h = self.height
        i = self.inflection
        e = self.end
        x = h * (i - e) / y + 2 * e - i
        return x

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `inflection end [height]`.
        """
        return super()._parameters(self.inflection, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `inflection end [height]`.
        """
        self.inflection, self.end, self.height = self._parse(2, parameters)


class Constant(Term):
    r"""Zero polynomial term $k$ that represents a constant value.

    ![](../../image/term/Constant.svg)

    Note: Equation
        $\mu(x) = k$

        where

        - $k$: value of the Constant
    """

    def __init__(self, name: str = "", value: float = nan) -> None:
        """Constructor.

        Args:
            name: name of the Term
            value: value of the Constant
        """
        super().__init__(name)
        self.value = value

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = vars(self).copy()
        fields.pop("height")
        return representation.as_constructor(self, fields, positional=True)

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: irrelevant

        Returns:
             $\mu(x) = k$
        """
        y = np.full_like(x, fill_value=self.value)
        return y

    def parameters(self) -> str:
        """Return the space-separated parameters of the term.

        Returns:
            `k`
        """
        return super()._parameters(self.value)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `k`
        """
        self.value = self._parse(1, parameters, height=False)[0]


class Cosine(Term):
    r"""Extended term that represents the cosine membership function.

    ![](../../image/term/Cosine.svg)

    Note: Equation
        $\mu(x) = \begin{cases}
            \dfrac{h}{2} \left(1 + \cos\left(\dfrac{2.0}{w}\pi(x-c)\right)\right) & \mbox{if } c - \dfrac{w}{2} \le x \le c + \dfrac{w}{2} \cr
            0 & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $c$: center of the Cosine
        - $w$: width of the Cosine

    """

    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            center: center of the Cosine
            width: width of the Cosine
            height: height of the Term
        """
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} \dfrac{h}{2} \left(1 + \cos\left(\dfrac{2.0}{w}\pi(x-c)\right)\right) & \mbox{if } c - \dfrac{w}{2} \le x \le c + \dfrac{w}{2} \cr 0 & \mbox{otherwise} \end{cases}$
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
        """Return the space-separated parameters of the term.

        Returns:
            `center width [height]`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `center width [height]`.
        """
        self.center, self.width, self.height = self._parse(2, parameters)


class Discrete(Term):
    r"""Basic term that represents a discrete membership function.

    ![](../../image/term/Discrete.svg)

    Note: Equation
        $\mu(x) = h\dfrac{(y_\max - y_\min)}{(x_\max - x_\min)}  (x - x_\min) + y_\min$

        where

        - $h$: height of the Term
        - $x_{\min}, x_{\max}$: lower and upper bounds of $x$, respectively
        - $y_{\min}, y_{\max}$: membership function values $\mu(x_{\min})$ and $\mu(x_{\max})$, respectively

    info: related
        - [numpy.interp][]

    Warning:
        The pairs of values in any Discrete term must be sorted in ascending order by the $x$ coordinate
        because the membership function is computed using binary search to find the lower and upper bounds of $x$.
    """

    Floatable = Union[SupportsFloat, str]

    def __init__(
        self,
        name: str = "",
        values: ScalarArray | Sequence[Floatable] | None = None,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the term
            values: 2D array of $(x,y)$ pairs
            height: height of the term.
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
        r"""Compute the membership function value of $x$.

        The function uses binary search to find the lower and upper bounds of $x$ and then linearly
        interpolates the membership function value between the bounds.

        Warning:
            The pairs of values in any Discrete term must be sorted in ascending order by the $x$ coordinate
            because the membership function is computed using binary search to find the lower and upper bounds of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h\dfrac{(y_\max - y_\min)}{(x_\max - x_\min)}  (x - x_\min) + y_\min$
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

    # def tsukamoto(self, y: Scalar) -> Scalar:
    #     todo: approximate tsukamoto if monotonic

    def parameters(self) -> str:
        """Return the space-separated parameters of the term.

        Returns:
            `x1 y1 ... xn yn [height]`.
        """
        return super()._parameters(self.to_list())

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `x1 y1 ... xn yn [height]`.
        """
        as_list = parameters.split()
        if len(as_list) % 2 == 0:
            self.height = 1.0
        else:
            self.height = to_float(as_list[-1])
            del as_list[-1]
        self.values = Discrete.to_xy(as_list[0::2], as_list[1::2])

    def x(self) -> ScalarArray:
        """Return $x$ coordinates.

        Returns:
            $x$ coordinates.
        """
        return self.values[:, 0]

    def y(self) -> ScalarArray:
        """Return $y$ coordinates.

        Returns:
            $y$ coordinates.
        """
        return self.values[:, 1]

    def sort(self) -> None:
        """Sort in ascending order the pairs of values by the $x$-coordinate."""
        self.values[:] = self.values[np.argsort(self.x())]

    def to_dict(self) -> dict[float, float]:
        """Return a dictionary of values in the form `{x: y}`.

        Returns:
            dictionary of values in the form `{x: y}`.
        """
        return dict(zip(self.x(), self.y()))

    def to_list(self) -> list[float]:
        """Return a list of values in the form `[x1,y1, ..., xn, yn]`.

        Returns:
            list of values in the form `[x1,y1, ..., xn, yn]`.
        """
        return self.values.flatten().tolist()  # type: ignore

    @staticmethod
    def create(
        name: str,
        xy: (
            str
            | Sequence[Floatable]
            | tuple[Sequence[Floatable], Sequence[Floatable]]
            | dict[Floatable, Floatable]
        ),
        height: float = 1.0,
    ) -> Discrete:
        """Create a discrete term from the parameters.

        Args:
            name: name of the term
            xy: coordinates
            height: height of the term

        Returns:
             Discrete term.
        """
        x: Scalar = scalar(0)
        y: Scalar = scalar(0)
        if isinstance(xy, str):
            xy = xy.split()
        if isinstance(xy, Sequence):
            x = scalar(xy[0::2])
            y = scalar(xy[1::2])
        if isinstance(xy, tuple):
            x = scalar(xy[0])
            y = scalar(xy[1])
        if isinstance(xy, dict):
            x = scalar([xi for xi in xy.keys()])  # noqa: SIM118
            y = scalar([yi for yi in xy.values()])
        return Discrete(name, Discrete.to_xy(x, y), height=height)

    @staticmethod
    def to_xy(x: Any, y: Any) -> ScalarArray:
        """Create list of values from the parameters.

        Args:
            x: $x$-coordinate(s) that can be converted into scalar(s)
            y: $y$-coordinate(s) that can be converted into scalar(s)

        Returns:
            array of $n$-rows and $2$-columns $(n,2)$.

        Raises:
            ValueError: when the shapes of $x$ and $y$ are different.
        """
        x = array(x, dtype=settings.float_type)
        y = array(y, dtype=settings.float_type)
        if x.shape != y.shape:
            raise ValueError(
                f"expected same shape from x and y, but found x={x.shape} and y={y.shape}"
            )
        return array([x, y]).T


class Gaussian(Term):
    r"""Extended term that represents the gaussian curve membership function.

    ![](../../image/term/Gaussian.svg)

    Note: Equation
        $\mu(x) = h \exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$

        where

        - $h$: height of the Term
        - $\mu$: mean of the Gaussian
        - $\sigma$: standard deviation of the Gaussian
    """

    def __init__(
        self,
        name: str = "",
        mean: float = nan,
        standard_deviation: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            mean: mean of the Gaussian
            standard_deviation: standard deviation of the Gaussian
            height: height of the Term.
        """
        super().__init__(name, height)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h \exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$
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
        """Return the space-separated parameters of the term.

        Returns:
            `mean standard_deviation [height]`.
        """
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `mean standard_deviation [height]`.
        """
        self.mean, self.standard_deviation, self.height = self._parse(2, parameters)


class GaussianProduct(Term):
    r"""Extended term that represents the two-sided gaussian membership function.

    ![](../../image/term/GaussianProduct.svg)

    Note: Equation
        $a = \begin{cases}
        \mbox{Gaussian}^{\mu_a}_{\sigma_a}(x) & \mbox{if } x < \mu_a \cr
        1.0 & \mbox{otherwise} \cr
        \end{cases}$

        $b = \begin{cases}
        \mbox{Gaussian}^{\mu_b}_{\sigma_b}(x) & \mbox{if } x > \mu_b \cr
        1.0 & \mbox{otherwise} \cr
        \end{cases}$

        $\mu(x) = h (a \times b)$

    where

    - $h$: height of the Term
    - $\mu_a, \sigma_a$: mean and standard deviation of the first Gaussian
    - $\mu_b, \sigma_b$: mean and standard deviation of the second Gaussian
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
        """Constructor.

        Args:
            name: name of the Term
            mean_a: mean of the first Gaussian
            standard_deviation_a: standard deviation of the first Gaussian
            mean_b: mean of the second Gaussian
            standard_deviation_b: standard deviation of the second Gaussian
            height: height of the Term
        """
        super().__init__(name, height)
        self.mean_a = mean_a
        self.standard_deviation_a = standard_deviation_a
        self.mean_b = mean_b
        self.standard_deviation_b = standard_deviation_b

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h (a \times b)$
        """
        x = scalar(x)
        a = np.where(
            x < self.mean_a,
            Gaussian(mean=self.mean_a, standard_deviation=self.standard_deviation_a).membership(x),
            1.0,
        )
        b = np.where(
            x > self.mean_b,
            Gaussian(mean=self.mean_b, standard_deviation=self.standard_deviation_b).membership(x),
            1.0,
        )
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * a * b
        return y  # type: ignore

    def parameters(self) -> str:
        """Return the space-separated parameters of the term.

        Returns:
             `mean_a standard_deviation_a mean_b standard_deviation_b [height]`.
        """
        return super()._parameters(
            self.mean_a,
            self.standard_deviation_a,
            self.mean_b,
            self.standard_deviation_b,
        )

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `mean_a standard_deviation_a mean_b standard_deviation_b [height]`.
        """
        (
            self.mean_a,
            self.standard_deviation_a,
            self.mean_b,
            self.standard_deviation_b,
            self.height,
        ) = self._parse(4, parameters)


class Linear(Term):
    r"""Linear polynomial term.

    ![](../../image/term/Linear.svg)

    Note: Equation
        $\mu(x)= \mathbf{c}\mathbf{v}+k = \sum_i c_iv_i + k$

        where

        - $x$: irrelevant
        - $\mathbf{v}$: vector of values from the input variables
        - $\mathbf{c}$ vector of coefficients for the input variables
        - $k$ is a constant
    """

    def __init__(
        self,
        name: str = "",
        coefficients: Sequence[float] | None = None,
        engine: Engine | None = None,
    ) -> None:
        r"""Constructor.

        Args:
            name: name of the term
            coefficients: coefficients for the input variables (plus constant $k$, optionally)
            engine: engine with the input variables
        """
        super().__init__(name)
        self.coefficients = list(coefficients or [])
        self.engine = engine

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = vars(self).copy()
        fields.pop("height")
        fields.pop("engine")
        return representation.as_constructor(self, fields, positional=True)

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x)=\sum_i c_iv_i + k$

        Raises:
            ValueError: when the number of coefficients (+1) is different from the number of input variables
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
        """Configure the term with the parameters.

        Args:
            parameters: coefficients `c1 ... cn k`.
        """
        self.coefficients = [to_float(p) for p in parameters.split()]

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `c1 ... cn k`.
        """
        return self._parameters(*self.coefficients)

    def update_reference(self, engine: Engine | None) -> None:
        """Set the reference to the engine.

        Args:
            engine: engine with the input variables
        """
        self.engine = engine


class PiShape(Term):
    r"""Extended term that represents the Pi-shaped membership function.

    ![](../../image/term/PiShape.svg)

    Note: Equation
        $\mu(x) = h \left(\mbox{SShape}_{a}^{b}(x) \times \mbox{ZShape}_{c}^{d}(x)\right)$

        where

        - $h$: height of the Term
        - $a, b$: bottom left and top left parameters of the PiShape
        - $c, d$: top right and bottom right parameters of the PiShape

    info: related
        - [fuzzylite.term.SShape][]
        - [fuzzylite.term.ZShape][]
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
        """Constructor.

        Args:
            name: name of the Term
            bottom_left: bottom-left value of the PiShape
            top_left: top-left value of the PiShape
            top_right: top-right value of the PiShape
            bottom_right: bottom-right value of the PiShape
            height: height of the Term.
        """
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h \left(\mbox{SShape}_{a}^{b}(x) \times \mbox{ZShape}_{c}^{d}(x)\right)$
        """
        s_shape = SShape(start=self.bottom_left, end=self.top_left).membership(x)
        z_shape = ZShape(start=self.top_right, end=self.bottom_right).membership(x)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * s_shape * z_shape
        return y

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `bottom_left top_left top_right bottom_right [height]`.
        """
        return super()._parameters(
            self.bottom_left, self.top_left, self.top_right, self.bottom_right
        )

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `bottom_left top_left top_right bottom_right [height]`.
        """
        (
            self.bottom_left,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.height,
        ) = self._parse(4, parameters)


class Ramp(Term):
    r"""Edge term that represents the ramp membership function.

    ![](../../image/term/Ramp.svg)

    Note: Equation
        $\mu(x) =  \begin{cases}
            h \dfrac{x - s} {e - s} & \mbox{if } s < x < e \cr
            h \dfrac{s - x} {s - e} & \mbox{if } e < x < s \cr
            h & \mbox{if } s < e \wedge x \ge e \cr
            h & \mbox{if } s > e \wedge x \le e \cr
            0 & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $s$: start of the Ramp
        - $e$: end of the Ramp
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the Ramp
            end: end of the Ramp
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) =  \begin{cases} h \dfrac{x - s} {e - s} & \mbox{if } s < x < e \cr h \dfrac{s - x} {s - e} & \mbox{if } e < x < s \cr h & \mbox{if } s < e \wedge x \ge e \cr h & \mbox{if } s > e \wedge x \le e \cr 0 & \mbox{otherwise} \end{cases}$
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
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def tsukamoto(self, y: Scalar) -> Scalar:
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y = h \dfrac{x - s} {e - s}$

            $x = s + (e-s) \dfrac{y}{h}$

        Args:
            y: activation degree

        Returns:
            $x = s + (e-s) \dfrac{y}{h}$
        """
        y = scalar(y)
        h = self.height
        s = self.start
        e = self.end
        x = s + (e - s) * y / h
        return x

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `start end [height]`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class Rectangle(Term):
    r"""Basic term that represents the rectangle membership function.

    ![](../../image/term/Rectangle.svg)

    Note: Equation
        $\mu(x) = \begin{cases}
            h & \mbox{if } s \le x \le e \cr
            0 & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $s$: start of the Rectangle
        - $e$: end of the Rectangle
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the Rectangle
            end: end of the Rectangle
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} h & \mbox{if } s \le x \le e \cr 0 & \mbox{otherwise} \end{cases}$
        """
        x = scalar(x)
        s = min(self.start, self.end)
        e = max(self.start, self.end)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * ((s <= x) & (x <= e))
        return y

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `start end [height]`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class SemiEllipse(Term):
    r"""Basic term that represents the semi-ellipse membership function.

    ![](../../image/term/SemiEllipse.svg)

    Note: Equation
        $\mu(x) = h \dfrac{\sqrt{r^2- (x-c)^2}}{r}$

        where

        - $h$: height of the Term
        - $r$: radius of the SemiEllipse
        - $c$: center of the SemiEllipse
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the SemiEllipse
            end: end of the SemiEllipse
            height: height of the Term.
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
           $\mu(x) = h \dfrac{\sqrt{r^2- (x-c)^2}}{r}$
        """
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
        """Return the parameters of the term.

        Returns:
             `start end [height]`
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class Sigmoid(Term):
    r"""Edge Term that represents the sigmoid membership function.

    ![](../../image/term/Sigmoid.svg)

    Note: Equation
        $\mu(x) = \dfrac{h}{1 + \exp(-s(x-i))}$

        where

        - $h$: height of the Term
        - $s$: slope of the Sigmoid
        - $i$: inflection of the Sigmoid
    """

    def __init__(
        self,
        name: str = "",
        inflection: float = nan,
        slope: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            inflection: inflection of the Sigmoid
            slope: slope of the Sigmoid
            height: height of the Term
        """
        super().__init__(name, height)
        self.inflection = inflection
        self.slope = slope

    def membership(self, x: Scalar) -> Scalar:
        r"""Compute the membership function value of $x$.

        Args:
            x: scalar

        Returns:
             $\mu(x) = \dfrac{h}{1 + \exp(-s(x-i))}$
        """
        x = scalar(x)
        i = self.inflection
        s = self.slope
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) / (1.0 + np.exp(-s * (x - i)))
        return y

    def tsukamoto(
        self,
        y: Scalar,
    ) -> Scalar:
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y=\dfrac{h}{1 + \exp(-s(x-i))}$

            $x=i\dfrac{\log{\left(\dfrac{h}{y}-1\right)}}{-s}$

        Args:
            y: activation degree

        Returns:
            $x=i\dfrac{\log{\left(\dfrac{h}{y}-1\right)}}{-s}$
        """
        y = scalar(y)
        h = self.height
        i = self.inflection
        s = self.slope
        x = i + np.log(h / y - 1.0) / -s
        return x

    def is_monotonic(self) -> bool:
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `inflection slope [height]`.
        """
        return super()._parameters(self.inflection, self.slope)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `inflection slope [height]`.
        """
        self.inflection, self.slope, self.height = self._parse(2, parameters)


class SigmoidDifference(Term):
    r"""Extended Term that represents the difference between two sigmoid membership functions.

    ![](../../image/term/SigmoidDifference.svg)

    Note: Equation
        $a = \mbox{Sigmoid}_\mbox{left}^\mbox{rise}(x)$

        $b = \mbox{Sigmoid}_\mbox{right}^\mbox{fall}(x)$

        $\mu(x) = h (a-b)$

        where

        - $h$: height of the Term
        - $\mbox{left}, \mbox{rise}$: inflection and slope of left Sigmoid
        - $\mbox{right}, \mbox{fall}$: inflection and slope of right Sigmoid
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
        """Constructor.

        Args:
            name: name of the Term
            left: inflection of the left Sigmoid
            rising: slope of the left Sigmoid
            falling: slope of the right Sigmoid
            right: inflection of the right Sigmoid
            height: height of the Term.
        """
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h (a-b)$
        """
        a = Sigmoid(inflection=self.left, slope=self.rising).membership(x)
        b = Sigmoid(inflection=self.right, slope=self.falling).membership(x)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * np.abs(a - b)
        return y  # type: ignore

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `left rising falling right [height]`.
        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `left rising falling right [height]`.
        """
        (
            self.left,
            self.rising,
            self.falling,
            self.right,
            self.height,
        ) = self._parse(4, parameters)


class SigmoidProduct(Term):
    r"""Extended Term that represents the product of two sigmoid membership functions.

    ![](../../image/term/SigmoidProduct.svg)

    Note: Equation
        $a = \mbox{Sigmoid}_\mbox{left}^\mbox{rise}(x)$

        $b = \mbox{Sigmoid}_\mbox{right}^\mbox{fall}(x)$

        $\mu(x) = h (a \times b)$

        where

        - $h$: height of the Term
        - $\mbox{left}, \mbox{rise}$: inflection and slope of left Sigmoid
        - $\mbox{right}, \mbox{fall}$: inflection and slope of right Sigmoid
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
        """Constructor.

        Args:
            name: name of the Term
            left: inflection of the left Sigmoid
            rising: slope of the left Sigmoid
            falling: slope of the right Sigmoid
            right: inflection of the right Sigmoid
            height: height of the Term.
        """
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = h (a \times b)$
        """
        a = Sigmoid(inflection=self.left, slope=self.rising).membership(x)
        b = Sigmoid(inflection=self.right, slope=self.falling).membership(x)
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * (a * b)
        return y

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `left rising falling right [height]`.
        """
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `left rising falling right [height]`.
        """
        (
            self.left,
            self.rising,
            self.falling,
            self.right,
            self.height,
        ) = self._parse(4, parameters)


class Spike(Term):
    r"""Extended Term that represents the spike membership function.

    ![](../../image/term/Spike.svg)

    Note: Equation
        $\mu(x)=h \exp\left(-\left|\dfrac{10}{w} (x - c)\right|\right)$

        where

        - $h$: height of the Term
        - $w$: width of the Spike
        - $c$: center of the Spike
    """

    def __init__(
        self,
        name: str = "",
        center: float = nan,
        width: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            center: center of the Spike
            width: width of the Spike
            height: height of the Term
        """
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
             $\mu(x)=h \exp\left(-\left|\dfrac{10}{w} (x - c)\right|\right)$
        """
        x = scalar(x)
        c = self.center
        w = self.width
        y = self.height * np.where(np.isnan(x), np.nan, 1.0) * np.exp(-np.abs(10.0 / w * (x - c)))
        return y  # type: ignore

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `center width [height]`.
        """
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `center width [height]`.
        """
        self.center, self.width, self.height = self._parse(2, parameters)


class SShape(Term):
    r"""Edge Term that represents the S-shaped membership function.

    ![](../../image/term/SShape.svg)

    Note: Equation:
        $\mu(x) = \begin{cases}
            0 & \mbox{if } x \leq s \cr
            2h \left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if } s < x \leq \dfrac{s+e}{2}\cr
            h - 2h\left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} < x < e\cr
            h & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $s$: start of the SShape
        - $e$: end of the SShape
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the SShape
            end: end of the SShape
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} 0 & \mbox{if } x \leq s \cr 2h \left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if } s < x \leq \dfrac{s+e}{2}\cr h - 2h\left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} < x < e\cr h & \mbox{otherwise} \end{cases}$
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
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y = \begin{cases} 0 & \mbox{if } x \leq s \cr 2h \left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if } s < x \leq \dfrac{s+e}{2}\cr h - 2h\left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} < x < e\cr h & \mbox{otherwise} \end{cases}$

            $x = \begin{cases}
                s + (e-s) \sqrt{\dfrac{y}{2h}} & \mbox{if } y \le \dfrac{h}{2} \cr
                e - (e-s) \sqrt{\dfrac{h-y}{2h}} & \mbox{otherwise}
            \end{cases}$

        Args:
            y: activation degree

        Returns:
            $x = \begin{cases} s + (e-s) \sqrt{\dfrac{y}{2h}} & \mbox{if } y \le \dfrac{h}{2} \cr e - (e-s) \sqrt{\dfrac{h-y}{2h}} & \mbox{otherwise} \end{cases}$
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
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
            `start end [height]`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class Trapezoid(Term):
    r"""Basic Term that represents the trapezoid membership function.

    ![](../../image/term/Trapezoid.svg)

    Note: Equation
        $\mu(x)= \begin{cases}
            0 & \mbox{if } x < a \vee x > d\cr
            h \dfrac{x - a}{b - a}  & \mbox{if } a \le x < b\cr
            h & \mbox{if } (b \le x \le c) \vee (a=-\infty \wedge x < b) \vee (d=\infty \wedge x > c) \cr
            h \dfrac{d - x}{d - c} & \mbox{if } c < x \le d\cr
            \text{NaN} & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $a$: bottom left vertex of the Trapezoid
        - $b$: top left vertex of the Trapezoid
        - $c$: top right vertex of the Trapezoid
        - $d$: bottom right vertex of the trapezoid
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
        """Constructor.

        Args:
            name: name of the Term
            bottom_left: first vertex of the Trapezoid
            top_left: second vertex of the Trapezoid
            top_right: third vertex of the Trapezoid
            bottom_right: fourth vertex of the Trapezoid
            height: height of the Term
        """
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        if np.isnan(top_right) and np.isnan(bottom_right):
            self.bottom_right = top_left
            range_ = self.bottom_right - self.bottom_left
            self.top_left = self.bottom_left + range_ * 1.0 / 5.0
            self.top_right = self.bottom_left + range_ * 4.0 / 5.0

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x)= \begin{cases} 0 & \mbox{if } x < a \vee x > d\cr h \dfrac{x - a}{b - a}  & \mbox{if } a \le x < b\cr h & \mbox{if } (b \le x \le c) \vee (a=-\infty \wedge x < b) \vee (d=\infty \wedge x > c) \cr h \dfrac{d - x}{d - c} & \mbox{if } c < x \le d\cr \text{NaN} & \mbox{otherwise} \end{cases}$
        """
        x = scalar(x)
        a = self.bottom_left
        b = self.top_left
        c = self.top_right
        d = self.bottom_right
        y = (
            self.height
            * np.where(np.isnan(x), np.nan, 1.0)
            * np.where(
                ((x < a) | (x > d)),
                0.0,
                np.where(
                    ((b <= x) & (x <= c)) | ((a == -inf) & (x < b)) | ((d == inf) & (x > c)),
                    1.0,
                    np.where(
                        x < b,
                        (x - a) / (b - a),
                        np.where(
                            x > c,
                            (d - x) / (d - c),
                            nan,
                        ),
                    ),
                ),
            )
        )
        return y  # type: ignore

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `bottom_left top_left top_right bottom_right [height]`.
        """
        return super()._parameters(
            self.bottom_left, self.top_left, self.top_right, self.bottom_right
        )

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `bottom_left top_left top_right bottom_right [height]`.
        """
        (
            self.bottom_left,
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.height,
        ) = self._parse(4, parameters)


class Triangle(Term):
    r"""Basic Term that represents the triangle membership function.

    ![](../../image/term/Triangle.svg)

    Note: Equation
        $\mu(x)= \begin{cases}
            0 & \mbox{if } x < a \vee x > c \cr
            h & \mbox{if } (x = b) \vee (a=-\infty \wedge x < b) \vee (c=\infty \wedge x > b) \cr
            h \dfrac{x - a}{b - a} & \mbox{if } a \le x < b \cr
            h \dfrac{c - x}{c - b} & \mbox{if } b < x \le c
        \end{cases}$

        where

        - $h$: height of the Term
        - $a$: left vertex of the Triangle
        - $b$: top vertex of the Triangle
        - $c$: right vertex of the Triangle
    """

    def __init__(
        self,
        name: str = "",
        left: float = nan,
        top: float = nan,
        right: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            left: first vertex of the Triangle
            top: second vertex of the Triangle
            right: third vertex of the Triangle
            height: height of the Term
        """
        super().__init__(name, height)
        self.left = left
        self.top = top
        self.right = right
        if np.isnan(right):
            self.top = 0.5 * (left + top)
            self.right = top

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x)= \begin{cases} 0 & \mbox{if } x < a \vee x > c \cr h & \mbox{if } (x = b) \vee (a=-\infty \wedge x < b) \vee (c=\infty \wedge x > b) \cr h \dfrac{x - a}{b - a} & \mbox{if } a \le x < b \cr h \dfrac{c - x}{c - b} & \mbox{if } b < x \le c \end{cases}$
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
        """Return the parameters of the term.

        Returns:
            `left top right [height]`.
        """
        return super()._parameters(self.left, self.top, self.right)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `left top right [height]`.
        """
        self.left, self.top, self.right, self.height = self._parse(3, parameters)


class ZShape(Term):
    r"""Edge Term that represents the ZShape membership function.

    ![](../../image/term/ZShape.svg)

    Note: Equation
        $\mu(x) = \begin{cases}
            1 & \mbox{if } x \leq s \cr
            h - 2h\left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if }  s < x < \dfrac{s+e}{2} \cr
            2h \left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} \le  x < e\cr
            0 & \mbox{otherwise}
        \end{cases}$

        where

        - $h$: height of the Term
        - $s$: start of the ZShape
        - $e$: end of the ZShape
    """

    def __init__(
        self,
        name: str = "",
        start: float = nan,
        end: float = nan,
        height: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            name: name of the Term
            start: start of the ZShape
            end: end of the ZShape
            height: height of the Term
        """
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: Scalar) -> Scalar:
        r"""Computes the membership function evaluated at $x$.

        Args:
            x: scalar

        Returns:
            $\mu(x) = \begin{cases} 1 & \mbox{if } x \leq s \cr h - 2h\left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if }  s < x < \dfrac{s+e}{2} \cr 2h \left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} \le  x < e\cr 0 & \mbox{otherwise} \end{cases}$
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
        r"""Compute the tsukamoto value of the monotonic term for activation degree $y$.

        Note: Equation
            $y = \begin{cases} 1 & \mbox{if } x \leq s \cr h - 2h\left(\dfrac{x - s}{e-s}\right)^2 & \mbox{if }  s < x < \dfrac{s+e}{2} \cr 2h \left(\dfrac{x - e}{e-s}\right)^2 & \mbox{if } \dfrac{s+e}{2} \le  x < e\cr 0 & \mbox{otherwise} \end{cases}$

            $x = \begin{cases}
                e + (e-s) \sqrt{\dfrac{y}{2h}} & \mbox{if } y \le \dfrac{h}{2} \cr
                s + (e-s) \sqrt{\dfrac{h-y}{2h}} & \mbox{otherwise}
            \end{cases}$

        Args:
            y: activation degree

        Returns:
            $x = \begin{cases} e + (e-s) \sqrt{\dfrac{y}{2h}} & \mbox{if } y \le \dfrac{h}{2} \cr s + (e-s) \sqrt{\dfrac{h-y}{2h}} & \mbox{otherwise} \end{cases}$
        """
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
        """Return `True` because the term is monotonic.

        Returns:
            `True`
        """
        return True

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `start end [height]`.
        """
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `start end [height]`.
        """
        self.start, self.end, self.height = self._parse(2, parameters)


class Function(Term):
    r"""Polynomial term that represents a generic function.

    ![](../../image/term/Function.svg)

    Note: Equation
        $f : x \mapsto f(x)$

    The function term is also used to convert the text of the antecedent of a rule from infix to postfix notation.

    info: related
        - [fuzzylite.rule.Antecedent][]
    """

    class Element:
        """Representation of a single element in a formula: either a function or an operator.

        If the Element represents a function, its parameter is the arity of the function (only unary or binary supported)
        If the Element represents an operator, its parameters are: `arity`, `precedence`, and `associativity`.
        """

        @enum.unique
        class Type(enum.Enum):
            """Type of function element."""

            Operator = enum.auto()
            Function = enum.auto()

            def __repr__(self) -> str:
                """Return the code to construct the name of the element in Python.

                Returns:
                    code to construct the name of the element in Python.
                """
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
            """Constructor.

            Args:
                name: name of the element
                description: description of the element
                type: type of the element
                method: reference to the function (only supports unary or binary)
                arity: number of operands required
                precedence: precedence of operators, where higher precedence comes first (see [Order of operations](https://en.wikipedia.org/wiki/Order_of_operations))
                associativity: precedence of grouping operators in the absence of parentheses (see [Operator associativity](https://en.wikipedia.org/wiki/Operator_associativity))
            """
            self.name = name
            self.description = description
            self.type = (
                type if isinstance(type, Function.Element.Type) else Function.Element.Type[type]
            )
            self.method = method
            self.arity = arity
            self.precedence = precedence
            self.associativity = associativity

        def __repr__(self) -> str:
            """Return the code to construct the element in Python.

            Returns:
                code to construct the element in Python.
            """
            return representation.as_constructor(self)

        def is_function(self) -> bool:
            """Return whether the element is a [fuzzylite.term.Function.Element.Type.Function][].

            Returns:
                element is [fuzzylite.term.Function.Element.Type.Function][]
            """
            return self.type == Function.Element.Type.Function

        def is_operator(self) -> bool:
            """Return whether the element is a [fuzzylite.term.Function.Element.Type.Operator][].

            Returns:
                element is [fuzzylite.term.Function.Element.Type.Operator][]
            """
            return self.type == Function.Element.Type.Operator

    class Node:
        """Basic binary tree structure.

        A node can point to left and right nodes to build a binary tree.

        A node can represent:

        - an element (Function or Operator),
        - an input or output variable by name,
        - a constant floating-point value
        """

        def __init__(
            self,
            element: Function.Element | None = None,
            variable: str = "",
            constant: float = nan,
            left: Function.Node | None = None,
            right: Function.Node | None = None,
        ) -> None:
            """Constructor.

            Args:
                element: node refers to a function or an operator
                variable: node refers to a variable by name
                constant: node refers to an arbitrary floating-point value
                right: node has an expression tree on the right
                left: node has an expression tree on the left.
            """
            self.element = element
            self.variable = variable
            self.constant = constant
            self.left = left
            self.right = right

        def __repr__(self) -> str:
            """Return the code to construct the node in Python.

            Returns:
                code to construct the node in Python.
            """
            return representation.as_constructor(self)

        def value(self) -> str:
            """Return the value of the node based on its contents.

            The value of the node is the first of:

            1. operation or function name if there is an element
            2. variable name if it is not empty
            3. constant value.

            Returns:
                value of the node based on its contents.
            """
            if self.element:
                return self.element.name
            if self.variable:
                return self.variable
            return Op.str(self.constant)

        def evaluate(self, local_variables: dict[str, Scalar] | None = None) -> Scalar:
            """Recursively evaluate the node and substitute the variables with the given values (if any).

            Args:
                local_variables: map of substitutions of variable names for scalars

            Returns:
                 scalar as the result of the evaluation.

            Raises:
                ValueError: when function arity is not met with left and right nodes
                ValueError: when the node represents a variable but the map does not contain its substitution value
            """
            result = scalar(nan)
            if self.element:
                arity = self.element.arity
                if arity == 0:
                    result = self.element.method()
                elif arity == 1:
                    if node := (self.left or self.right):
                        result = self.element.method(node.evaluate(local_variables))
                    else:
                        raise ValueError("expected a node, but found none")
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
            """Return the prefix notation of the node.

            Args:
                 node: node in the expression tree (defaults to `self` if `None`)

            Returns:
                 prefix notation of the node.
            """
            if not node:
                return self.prefix(self)

            if not np.isnan(node.constant):
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
            """Return the infix notation of the node.

            Args:
                 node: node in the expression tree (defaults to `self` if `None`)

            Returns:
                 infix notation of the node.
            """
            if not node:
                return self.infix(self)

            if not np.isnan(node.constant):
                return Op.str(node.constant)
            if node.variable:
                return node.variable

            children = []
            if node.left:
                children.append(self.infix(node.left))
            if node.right:
                children.append(self.infix(node.right))

            is_function = node.element and node.element.type == Function.Element.Type.Function

            if is_function:
                result = node.value() + f" ( {' '.join(children)} )"
            else:  # is operator
                if len(children) == 1:
                    result = f"{node.value()} {children[0]}"
                else:
                    result = f" {node.value()} ".join(children)

            return result

        def postfix(self, node: Function.Node | None = None) -> str:
            """Return the postfix notation of the node.

            Args:
                 node: node in the expression tree (defaults to `self` if `None`)

            Returns:
                 postfix notation of the node.
            """
            if not node:
                return self.postfix(self)

            if not np.isnan(node.constant):
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
        """Constructor.

        Args:
            name: name of the term
            formula: formula defining the membership function
            engine: engine to which the Function can have access
            variables: map of substitution variables
            load: load the function on creation.
        """
        super().__init__(name)
        self.root: Function.Node | None = None
        self.formula = formula
        self.engine = engine
        self.variables: dict[str, Scalar] = variables.copy() if variables else {}
        if load:
            self.load()

    def __repr__(self) -> str:
        """Return the code to construct the term in Python.

        Returns:
            code to construct the term in Python.
        """
        fields = vars(self).copy()
        fields.pop("height")
        fields.pop("root")
        fields.pop("engine")
        if not self.variables:
            fields.pop("variables")
        return representation.as_constructor(self, fields, positional=True)

    def parameters(self) -> str:
        """Return the parameters of the term.

        Returns:
             `formula`.
        """
        return self.formula

    def configure(self, parameters: str) -> None:
        """Configure the term with the parameters.

        Args:
            parameters: `formula`.
        """
        self.formula = parameters
        self.load()

    def update_reference(self, engine: Engine | None) -> None:
        """Update the reference to the engine (if any) and load the function if it is not loaded.

        Args:
             engine: engine to which the term belongs to.
        """
        self.engine = engine
        if not self.is_loaded():
            self.load()

    @staticmethod
    def create(name: str, formula: str, engine: Engine | None = None) -> Function:
        """Create and configure a function term.

        Args:
            name: name of the term
            formula: formula defining the membership function
            engine: engine to which the Function can have access

        Returns:
            configured function term

        Raises:
             SyntaxError: when the formula has a syntax error
        """
        result = Function(name, formula, engine)
        result.load()
        return result

    def membership(self, x: Scalar) -> Scalar:
        """Computes the membership function evaluated at $x$.

        The value of variable `x` will be added to the `variables` map, and so will the current values of the input
        variables and output variables if the engine has been set.

        Args:
            x: scalar

        Returns:
            $f(x)$

        Raises:
            ValueError: when an input or output variable is named `x`
            ValueError: when the map of variables contain names of input or output variables
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
        """Evaluate the function value using the map of variable substitutions (if any).

        Args:
            variables: map containing substitutions of variable names for values

        Returns:
            function value using the map of variable substitutions (if any).

        Raises:
            RuntimeError: when the function is not loaded
        """
        if not self.root:
            raise RuntimeError(f"function '{self.formula}' is not loaded")
        return self.root.evaluate(variables)

    def is_loaded(self) -> bool:
        """Return whether the function is loaded.

        Returns:
             function is loaded.
        """
        return bool(self.root)

    def unload(self) -> None:
        """Unload the function and reset the map of substitution variables."""
        self.root = None
        self.variables.clear()

    def load(self) -> None:
        """Load the function using the formula expressed in infix notation."""
        self.root = self.parse(self.formula)

    @classmethod
    def format_infix(cls, formula: str) -> str:
        """Format the formula expressed in infix notation.

        Args:
            formula: formula expressed in infix notation.

        Returns:
               formatted formula expressed in infix notation.
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
        """Convert the formula to postfix notation.

        Args:
            formula: right-hand side of an equation expressed in infix notation

        Returns:
             formula in postfix notation

        Raises:
             SyntaxError: when the formula has syntax errors.
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

            element: Function.Element | None = factory.objects.get(token)

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
                    if (element.associativity < 0 and element.precedence <= top.precedence) or (
                        element.associativity > 0 and element.precedence < top.precedence
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
        """Create a node as a binary expression tree of the formula.

        Args:
            formula: right-hand side of an equation expressed in infix notation

        Returns:
            node as a binary expression tree of the formula

        Raises:
            SyntaxError: when the formula has syntax errors.
        """
        postfix = cls.infix_to_postfix(formula)
        stack: list[Function.Node] = []
        factory = settings.factory_manager.function

        for token in postfix.split():
            element: Function.Element | None = factory.objects.get(token)
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
