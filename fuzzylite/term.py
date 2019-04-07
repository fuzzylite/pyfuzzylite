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

__all__ = ["Activated", "Aggregated", "Bell", "Binary", "Concave", "Constant", "Cosine",
           "Discrete", "Function", "Gaussian", "GaussianProduct", "Linear", "PiShape", "Ramp",
           "Rectangle", "SShape", "Sigmoid", "SigmoidDifference", "SigmoidProduct", "Spike",
           "Term", "Trapezoid", "Triangle", "ZShape"]

import bisect
import enum
import logging
import re
import typing
from math import cos, exp, fabs, inf, isnan, nan, pi
from typing import (Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set,
                    SupportsFloat, Tuple, TypeVar, Union)

from .exporter import FllExporter
from .norm import SNorm, TNorm
from .operation import Op

if typing.TYPE_CHECKING:
    from .engine import Engine  # noqa F401


class Term:
    """
      The Term class is the abstract class for linguistic terms. The linguistic
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
        self.name = name
        self.height = height

    def __str__(self) -> str:
        """
         Returns the representation of the term in the FuzzyLite Language
          :return the representation of the term in FuzzyLite Language
          @see FllExporter
        """
        return FllExporter().term(self)

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> str:
        """
          Returns the parameters to configure the term. The parameters are
          separated by spaces. If there is one additional parameter, the
          parameter will be considered as the height of the term; otherwise,
          the height will be set to @f$1.0@f$
          :return the parameters to configure the term (@see Term::configure())
         """
        return self._parameters()

    def _parameters(self, *args: object) -> str:
        """
        Concatenates the parameters given, and the height if it is different from 1.0 or None
        :param args: is the parameters to configure the term
        :return: the parameters concatenated and an optional height at the end
        """
        result: List[str] = []
        if args:
            result.extend(map(Op.str, args))
        if self.height and self.height != 1.0:
            result.append(Op.str(self.height))
        return " ".join(result)

    def configure(self, parameters: str) -> None:
        """
          Configures the term with the given takes_parameters. The takes_parameters are
          separated by spaces. If there is one additional parameter, the
          parameter will be considered as the height of the term; otherwise,
          the height will be set to @f$1.0@f$
          :param parameters is the takes_parameters to configure the term
        """
        pass

    def membership(self, x: float) -> float:
        r"""
          Computes the has_membership function value at @f$x@f$
          :param x
          :return the has_membership function value @f$\mu(x)@f$
        """
        raise NotImplementedError()

    def update_reference(self, engine: Optional['Engine']) -> None:
        """
          Updates the references (if any) to point to the current engine (useful
          when cloning engines or creating terms within Importer objects
          :param engine: is the engine to which this term belongs to
        """
        pass

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        r"""
          For monotonic terms, computes the tsukamoto value of the term for the
          given activation degree @f$\alpha@f$, that is,
          @f$ g_j(\alpha) = \{ z \in\mathbb{R} : \mu_j(z) = \alpha \} $@f. If
          the term is not monotonic (or does not override this method) the
          method computes the has_membership function @f$\mu(\alpha)@f$.
          :param activation_degree: is the activationDegree
          :param minimum is the minimum value of the range of the term
          :param maximum is the maximum value of the range of the term
          :return the tsukamoto value of the term for the given activation degree
                  if the term is monotonic (or overrides this method), or
                  the has_membership function for the activation degree otherwise.
        """
        return self.membership(activation_degree)

    def is_monotonic(self) -> bool:
        """
        Indicates whether the term is monotonic.
          :return whether the term is monotonic.
        """
        return False

    def discretize(self, start: float, end: float, resolution: int = 100,
                   bounded_mf: bool = True) -> 'Discrete':
        result = Discrete(self.name)
        dx = (end - start) / resolution
        for i in range(0, resolution + 1):
            x = start + i * dx
            y = self.membership(x)
            if bounded_mf:
                y = Op.bound(y, 0.0, 1.0)
            result.xy.append(Discrete.Pair(x, y))
        return result


class Activated(Term):

    def __init__(self, term: Term, degree: float = 1.0,
                 implication: Optional[TNorm] = None) -> None:
        super().__init__("_")
        self.term = term
        self.degree = degree
        self.implication = implication

    def parameters(self) -> str:
        name = self.term.name if self.term else "none"
        if self.implication:
            result = "{0}({1},{2})".format(FllExporter().norm(self.implication),
                                           Op.str(self.degree), name)
        else:
            result = "({0}*{1})".format(Op.str(self.degree), name)
        return result

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if not self.term:
            raise ValueError("expected a term to activate, but none found")
        if not self.implication:
            raise ValueError("expected an implication operator, but none found")
        result = self.implication.compute(self.term.membership(x), self.degree)
        logging.debug(f"{Op.str(result)}: {str(self)}")
        return result


class Aggregated(Term):

    def __init__(self, name: str = "", minimum: float = nan, maximum: float = nan,
                 aggregation: Optional[SNorm] = None,
                 terms: Optional[Iterable[Activated]] = None) -> None:
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum
        self.aggregation = aggregation
        self.terms: List[Activated] = []
        if terms:
            self.terms.extend(terms)

    def parameters(self) -> str:
        result = []
        activated = [term.parameters() for term in self.terms]
        if self.aggregation:
            result.append("{0}[{1}]".format(FllExporter().norm(self.aggregation),
                                            ",".join(activated)))
        else:
            result.append("[{0}]".format("+".join(activated)))

        return " ".join(result)

    def range(self) -> float:
        return self.maximum - self.minimum

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        if self.terms and not self.aggregation:
            raise ValueError("expected an aggregation operator, but none found")

        result = 0.0
        for term in self.terms:
            result = self.aggregation.compute(result, term.membership(x))  # type: ignore
        logging.debug(f"{Op.str(result)}: {str(self)}")
        return result

    def activation_degree(self, term: Term) -> float:
        result = 0.0

        for activation in self.terms:
            if activation.term == term:
                if self.aggregation:
                    result = self.aggregation.compute(result, activation.degree)
                else:
                    result += activation.degree

        return result

    def highest_activated_term(self) -> Optional[Activated]:
        result = None
        maximum_activation = -inf
        for activated in self.terms:
            if activated.degree > maximum_activation:
                maximum_activation = activated.degree
                result = activated
        return result

    def clear(self) -> None:
        self.terms.clear()


class Bell(Term):

    def __init__(self, name: str = "", center: float = nan, width: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.center = center
        self.width = width
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * (1.0 / (1.0 + (fabs((x - self.center) / self.width)
                                            ** (2.0 * self.slope))))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width, self.slope = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


class Binary(Term):

    def __init__(self, name: str = "", start: float = nan, direction: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.start = start
        self.direction = direction

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if self.direction > self.start and x >= self.start:
            return self.height * 1.0

        if self.direction < self.start and x <= self.start:
            return self.height * 1.0

        return self.height * 0.0

    def parameters(self) -> str:
        return super()._parameters(self.start, self.direction)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.direction = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Concave(Term):

    def __init__(self, name: str = "", inflection: float = nan, end: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.inflection = inflection
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if self.inflection <= self.end:  # Concave increasing
            if x < self.end:
                return (self.height * (self.end - self.inflection)
                        / (2.0 * self.end - self.inflection - x))

        else:  # Concave decreasing
            if x > self.end:
                return (self.height * (self.inflection - self.end)
                        / (self.inflection - 2.0 * self.end + x))

        return self.height * 1.0

    def is_monotonic(self) -> bool:
        return True

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        i = self.inflection
        e = self.end
        return (i - e) / self.membership(activation_degree) + 2 * e - i

    def parameters(self) -> str:
        return super()._parameters(self.inflection, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.inflection, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Constant(Term):

    def __init__(self, name: str = "", value: float = nan) -> None:
        super().__init__(name)
        self.value = value

    def membership(self, x: float) -> float:
        return self.value

    def parameters(self) -> str:
        return super()._parameters(self.value)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        if not values:
            raise ValueError("not enough values to unpack (expected 1, got 0)")
        self.value = values[0]
        self.height = 1.0


class Cosine(Term):

    def __init__(self, name: str = "", center: float = nan, width: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x < self.center - 0.5 * self.width or x > self.center + 0.5 * self.width:
            return self.height * 0.0

        return self.height * 0.5 * (1.0 + cos(2.0 / self.width * pi * (x - self.center)))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Discrete(Term):
    Floatable = TypeVar("Floatable", SupportsFloat, str, bytes)

    class Pair:

        def __init__(self, x: float = nan, y: float = nan) -> None:
            self.x = x
            self.y = y

        def __str__(self) -> str:
            return f"({self.x}, {self.y})"

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.values == other.values
            return self.values == other

        def __ne__(self, other: object) -> bool:
            if isinstance(other, Discrete.Pair):
                return not (self.values == other.values)
            return self.values != other

        def __lt__(self, other: Union[Tuple[float, float], 'Discrete.Pair']) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.values < other.values
            if isinstance(other, tuple):
                return self.values < other
            raise ValueError("expected Union[Tuple[float, float], 'Discrete.Pair'], "
                             f"but found {type(other)}")

        def __le__(self, other: Union[Tuple[float, float], 'Discrete.Pair']) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.values <= other.values
            if isinstance(other, tuple):
                return self.values <= other
            raise ValueError("expected Union[Tuple[float, float], 'Discrete.Pair'], "
                             f"but found {type(other)}")

        def __gt__(self, other: Union[Tuple[float, float], 'Discrete.Pair']) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.values > other.values
            if isinstance(other, tuple):
                return self.values >= other
            raise ValueError("expected Union[Tuple[float, float], 'Discrete.Pair'], "
                             f"but found {type(other)}")

        def __ge__(self, other: Union[Tuple[float, float], 'Discrete.Pair']) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.values >= other.values
            if isinstance(other, tuple):
                return self.values >= other
            raise ValueError("expected Union[Tuple[float, float], 'Discrete.Pair'], "
                             f"but found {type(other)}")

        @property
        def values(self) -> Tuple[float, float]:
            return self.x, self.y

        @values.setter
        def values(self, xy: Tuple[float, float]) -> None:
            self.x, self.y = xy

    def __init__(self,
                 name: str = "",
                 xy: Optional[Sequence[Floatable]] = None,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.xy: List[Discrete.Pair] = []
        if xy:
            self.xy = Discrete.pairs_from(xy)

    def __iter__(self) -> Iterator['Discrete.Pair']:
        return iter(self.xy)

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if not self.xy:
            raise ValueError("expected a list of (x,y)-pairs, but found none")

        if x <= self.xy[0].x:
            return self.height * self.xy[0].y

        if x >= self.xy[-1].x:
            return self.height * self.xy[-1].y

        index = bisect.bisect(self.xy, (x, -inf))

        upper_bound = self.xy[index]
        if Op.eq(x, upper_bound.x):
            return self.height * upper_bound.y

        lower_bound = self.xy[index - 1]

        return self.height * Op.scale(x, lower_bound.x, upper_bound.x, lower_bound.y, upper_bound.y)

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        # todo: approximate tsukamoto
        pass

    def parameters(self) -> str:
        return super()._parameters(*Discrete.values_from(self.xy))

    def configure(self, parameters: str) -> None:
        values = [Op.scalar(x) for x in parameters.split()]
        if len(values) % 2 == 0:
            self.height = 1.0
        else:
            self.height = values[-1]
            del values[-1]

        self.xy = Discrete.pairs_from(values)

    def x(self) -> Iterable[float]:
        return (pair.x for pair in self.xy)

    def y(self) -> Iterable[float]:
        return (pair.y for pair in self.xy)

    def sort(self) -> None:
        self.xy.sort()

    @staticmethod
    def pairs_from(values: Union[Sequence[Floatable],
                                 Dict[Floatable, Floatable]]) -> List['Discrete.Pair']:
        if isinstance(values, dict):
            return [Discrete.Pair(Op.scalar(x), Op.scalar(y)) for x, y in values.items()]

        if len(values) % 2 != 0:
            raise ValueError("not enough values to unpack (expected an even number, "
                             f"but got {len(values)}) in {values}")

        result = [Discrete.Pair(Op.scalar(values[i]), Op.scalar(values[i + 1]))
                  for i in range(0, len(values) - 1, 2)]
        return result

    # TODO: More pythonic?
    @staticmethod
    def values_from(pairs: List['Discrete.Pair']) -> List[float]:
        result: List[float] = []
        for xy in pairs:
            result.extend([xy.x, xy.y])
        return result

    @staticmethod
    def dict_from(pairs: List['Discrete.Pair']) -> Dict[float, float]:
        return {pair.x: pair.y for pair in pairs}


class Gaussian(Term):

    def __init__(self, name: str = "", mean: float = nan, standard_deviation: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * exp((-(x - self.mean) * (x - self.mean))
                                 / (2.0 * self.standard_deviation * self.standard_deviation))

    def parameters(self) -> str:
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.mean, self.standard_deviation = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class GaussianProduct(Term):

    def __init__(self, name: str = "", mean_a: float = nan, standard_deviation_a: float = nan,
                 mean_b: float = nan, standard_deviation_b: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.mean_a = mean_a
        self.standard_deviation_a = standard_deviation_a
        self.mean_b = mean_b
        self.standard_deviation_b = standard_deviation_b

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        a = b = 1.0

        if x < self.mean_a:
            a = exp((-(x - self.mean_a) * (x - self.mean_a))
                    / (2.0 * self.standard_deviation_a * self.standard_deviation_a))

        if x > self.mean_b:
            b = exp((-(x - self.mean_b) * (x - self.mean_b))
                    / (2.0 * self.standard_deviation_b * self.standard_deviation_b))

        return self.height * a * b

    def parameters(self) -> str:
        return super()._parameters(self.mean_a, self.standard_deviation_a,
                                   self.mean_b, self.standard_deviation_b)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.mean_a, self.standard_deviation_a, self.mean_b, self.standard_deviation_b = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Linear(Term):

    def __init__(self, name: str = "", coefficients: Optional[Iterable[float]] = None,
                 engine: Optional['Engine'] = None) -> None:
        super().__init__(name)
        self.coefficients: List[float] = []
        if coefficients:
            self.coefficients.extend(coefficients)
        self.engine = engine

    def membership(self, _: float) -> float:
        if not self.engine:
            raise ValueError("expected the reference to an engine, but found none")

        result = 0.0
        number_of_coefficients = len(self.coefficients)
        input_variables = self.engine.input_variables
        for i, input_variable in enumerate(input_variables):
            if i < number_of_coefficients:
                result += self.coefficients[i] * input_variable.value
        if number_of_coefficients > len(input_variables):
            result += self.coefficients[len(input_variables)]

        return result

    def configure(self, parameters: str) -> None:
        self.coefficients = [Op.scalar(p) for p in parameters.split()]

    def parameters(self) -> str:
        return self._parameters(*self.coefficients)

    def update_reference(self, engine: Optional['Engine']) -> None:
        self.engine = engine


class PiShape(Term):

    def __init__(self, name: str = "", bottom_left: float = nan, top_left: float = nan,
                 top_right: float = nan, bottom_right: float = nan, height: float = 1.0) -> None:
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x <= self.bottom_left:
            s_shape = 0.0
        elif x <= 0.5 * (self.bottom_left + self.top_left):
            s_shape = 2.0 * ((x - self.bottom_left) / (self.top_left - self.bottom_left)) ** 2
        elif x < self.top_left:
            s_shape = 1.0 - 2.0 * ((x - self.top_left) / (self.top_left - self.bottom_left)) ** 2
        else:
            s_shape = 1.0

        if x <= self.top_right:
            z_shape = 1.0
        elif x <= 0.5 * (self.top_right + self.bottom_right):
            z_shape = 1.0 - 2.0 * ((x - self.top_right) / (self.bottom_right - self.top_right)) ** 2
        elif x < self.bottom_right:
            z_shape = 2.0 * ((x - self.bottom_right) / (self.bottom_right - self.top_right)) ** 2
        else:
            z_shape = 0.0

        return self.height * s_shape * z_shape

    def parameters(self) -> str:
        return super()._parameters(self.bottom_left, self.top_left,
                                   self.top_right, self.bottom_right)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.bottom_left, self.top_left, self.top_right, self.bottom_right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Ramp(Term):

    def __init__(self, name: str = "", start: float = nan, end: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if self.start == self.end:
            return self.height * 0.0

        if self.start < self.end:
            if x <= self.start:
                return self.height * 0.0
            if x >= self.end:
                return self.height * 1.0
            return self.height * (x - self.start) / (self.end - self.start)

        else:
            if x >= self.start:
                return self.height * 0.0
            if x <= self.end:
                return self.height * 1.0
            return self.height * (self.start - x) / (self.start - self.end)

    def is_monotonic(self) -> bool:
        return True

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        if isnan(activation_degree):
            return nan
        return Op.scale(activation_degree, 0.0, self.height * 1.0, self.start, self.end)

    def parameters(self) -> str:
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Rectangle(Term):

    def __init__(self, name: str = "", start: float = nan, end: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if self.start <= x <= self.end:
            return self.height * 1.0

        return self.height * 0.0

    def parameters(self) -> str:
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class Sigmoid(Term):

    def __init__(self, name: str = "", inflection: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.inflection = inflection
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * 1.0 / (1.0 + exp(-self.slope * (x - self.inflection)))

    def is_monotonic(self) -> bool:
        return True

    def parameters(self) -> str:
        return super()._parameters(self.inflection, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.inflection, self.slope = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class SigmoidDifference(Term):

    def __init__(self, name: str = "", left: float = nan, rising: float = nan,
                 falling: float = nan, right: float = nan, height: float = 1.0) -> None:
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        a = 1.0 / (1.0 + exp(-self.rising * (x - self.left)))
        b = 1.0 / (1.0 + exp(-self.falling * (x - self.right)))

        return self.height * fabs(a - b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class SigmoidProduct(Term):

    def __init__(self, name: str = "", left: float = nan, rising: float = nan,
                 falling: float = nan, right: float = nan, height: float = 1.0) -> None:
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        a = 1.0 + exp(-self.rising * (x - self.left))
        b = 1.0 + exp(-self.falling * (x - self.right))

        return self.height * 1.0 / (a * b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Spike(Term):

    def __init__(self, name: str = "", inflection: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.center = inflection
        self.width = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * exp(-fabs(10.0 / self.width * (x - self.center)))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class SShape(Term):

    def __init__(self, name: str = "", start: float = nan, end: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x <= self.start:
            return self.height * 0.0

        if x <= 0.5 * (self.start + self.end):
            return self.height * 2.0 * ((x - self.start) / (self.end - self.start)) ** 2

        if x < self.end:
            return self.height * (1.0 - 2.0 * ((x - self.end) / (self.end - self.start)) ** 2)

        return self.height * 1.0

    def is_monotonic(self) -> bool:
        return True

    def parameters(self) -> str:
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Trapezoid(Term):

    def __init__(self, name: str = "", vertex_a: float = nan, vertex_b: float = nan,
                 vertex_c: float = nan, vertex_d: float = nan, height: float = 1.0) -> None:
        super().__init__(name, height)
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.vertex_c = vertex_c
        self.vertex_d = vertex_d
        if isnan(vertex_c) and isnan(vertex_d):
            self.vertex_d = vertex_b
            range_ = self.vertex_d - self.vertex_a
            self.vertex_b = self.vertex_a + range_ * 1.0 / 5.0
            self.vertex_c = self.vertex_a + range_ * 4.0 / 5.0

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x < self.vertex_a or x > self.vertex_d:
            return self.height * 0.0

        if x < self.vertex_b:
            if self.vertex_a == -inf:
                return self.height * 1.0
            return self.height * (x - self.vertex_a) / (self.vertex_b - self.vertex_a)

        if self.vertex_b <= x <= self.vertex_c:
            return self.height * 1.0

        if x > self.vertex_c:
            if self.vertex_d == inf:
                return self.height * 1.0
            return self.height * (self.vertex_d - x) / (self.vertex_d - self.vertex_c)

        return self.height * 0.0

    def parameters(self) -> str:
        return super()._parameters(self.vertex_a, self.vertex_b, self.vertex_c, self.vertex_d)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.vertex_a, self.vertex_b, self.vertex_c, self.vertex_d = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Triangle(Term):

    def __init__(self, name: str = "", vertex_a: float = nan, vertex_b: float = nan,
                 vertex_c: float = nan, height: float = 1.0) -> None:
        super().__init__(name, height)
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.vertex_c = vertex_c
        if isnan(vertex_c):
            self.vertex_b = 0.5 * (vertex_a + vertex_b)
            self.vertex_c = vertex_b

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x < self.vertex_a or x > self.vertex_c:
            return self.height * 0.0

        if x < self.vertex_b:
            if self.vertex_a == -inf:
                return self.height * 1.0
            return self.height * (x - self.vertex_a) / (self.vertex_b - self.vertex_a)

        if x == self.vertex_b:
            return self.height * 1.0

        if x > self.vertex_b:
            if self.vertex_c == inf:
                return self.height * 1.0
            return self.height * (self.vertex_c - x) / (self.vertex_c - self.vertex_b)

        return self.height * 0.0

    def parameters(self) -> str:
        return super()._parameters(self.vertex_a, self.vertex_b, self.vertex_c)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.vertex_a, self.vertex_b, self.vertex_c = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


# TODO: Tsukamoto
class ZShape(Term):

    def __init__(self, name: str = "", start: float = nan, end: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan

        if x <= self.start:
            return self.height * 1.0

        if x <= 0.5 * (self.start + self.end):
            return self.height * (1.0 - 2.0 * ((x - self.start) / (self.end - self.start)) ** 2)

        if x < self.end:
            return self.height * 2.0 * ((x - self.end) / (self.end - self.start)) ** 2

        return self.height * 0.0

    def is_monotonic(self) -> bool:
        return True

    def parameters(self) -> str:
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(Op.scalar(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Function(Term):
    class Element:
        @enum.unique
        class Type(enum.Enum):
            Operator, Function = range(2)

        def __init__(self, name: str, description: str, type: 'Function.Element.Type',
                     method: Callable[..., float],
                     arity: int = 0, precedence: int = 0,
                     associativity: int = -1) -> None:
            self.name = name
            self.description = description
            self.type = type
            self.method = method
            self.arity = Op.arity_of(method) if arity < 0 else arity
            self.precedence = precedence
            self.associativity = associativity

        def __str__(self) -> str:
            result = [f"name='{self.name}'",
                      f"description='{self.description}'",
                      f"element_type='{str(self.type)}'",
                      f"method='{str(self.method)}'",
                      f"arity={self.arity}",
                      f"precedence={self.precedence}",
                      f"associativity={self.associativity}"]
            return "{0}: {1}".format(self.__class__.__name__, ", ".join(result))

        def is_function(self) -> bool:
            return self.type == Function.Element.Type.Function

        def is_operator(self) -> bool:
            return self.type == Function.Element.Type.Operator

    class Node(object):

        def __init__(self, element: Optional['Function.Element'] = None,
                     variable: str = "", constant: float = nan,
                     right: Optional['Function.Node'] = None,
                     left: Optional['Function.Node'] = None) -> None:
            self.element = element
            self.variable = variable
            self.constant = constant
            self.left = left
            self.right = right

        def __str__(self) -> str:
            return self.postfix()

        def value(self) -> str:
            if self.element:
                result = self.element.name
            elif self.variable:
                result = self.variable
            else:
                result = Op.str(self.constant)
            return result

        def evaluate(self,
                     local_variables: Optional[Dict[str, float]] = None) -> float:
            result = nan
            if self.element:
                if not self.element.method:
                    raise ValueError("expected a method reference, but found none")
                arity = self.element.arity
                if arity == 0:
                    result = self.element.method()
                elif arity == 1:
                    if not self.right:
                        raise ValueError("expected a right node, but found none")
                    result = self.element.method(
                        self.right.evaluate(local_variables))
                elif arity == 2:
                    if not self.right:
                        raise ValueError("expected a right node, but found none")
                    if not self.left:
                        raise ValueError("expected a left node, but found none")
                    result = self.element.method(
                        self.left.evaluate(local_variables),
                        self.right.evaluate(local_variables))
            elif self.variable:
                if not local_variables or self.variable not in local_variables:
                    raise ValueError(
                        f"expected a map of variables containing the value for '{self.variable}', "
                        f"but the map contains: {local_variables}")
                result = local_variables[self.variable]

            else:
                result = self.constant

            return result

        def prefix(self, node: Optional['Function.Node'] = None) -> str:
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

        def infix(self, node: Optional['Function.Node'] = None) -> str:
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

            is_function = (node.element
                           and node.element.type == Function.Element.Type.Function)

            if is_function:
                result = node.value() + f" ( {' '.join(children)} )"
            else:  # is operator
                result = f" {node.value()} ".join(children)

            return result

        def postfix(self, node: Optional['Function.Node'] = None) -> str:
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

    def __init__(self, name: str = "", formula: str = "", engine: Optional['Engine'] = None,
                 variables: Optional[Dict[str, float]] = None, load: bool = False) -> None:
        super().__init__(name)
        self.root: Optional[Function.Node] = None
        self.formula = formula
        self.engine = engine
        self.variables: Dict[str, float] = {}
        if variables:
            self.variables.update(variables)
        if load:
            self.load()

    def parameters(self) -> str:
        return self.formula

    def configure(self, parameters: str) -> None:
        self.formula = parameters
        self.load()

    def update_reference(self, engine: Optional['Engine']) -> None:
        self.engine = engine
        if self.is_loaded():
            self.load()

    @staticmethod
    def create(name: str, formula: str, engine: Optional['Engine'] = None) -> 'Function':
        result = Function(name, formula, engine)
        result.load()
        return result

    def membership(self, x: float) -> float:
        if 'x' in self.variables:
            raise ValueError("variable 'x' is reserved for internal use of Function term, please "
                             f"remove it from the map of variables: {self.variables}")

        engine_variables: Dict[str, float] = {}
        if self.engine:
            for variable in self.engine.variables:
                engine_variables[variable.name] = variable.value

            if 'x' in engine_variables:
                raise ValueError("variable 'x' is reserved for internal use of Function term, "
                                 f"please rename the engine variable: {self.engine.variable('x')}")
        engine_variables['x'] = x

        overrides = self.variables.keys() & engine_variables.keys()
        if overrides:
            raise ValueError("function variables cannot override engine variables, please "
                             f"resolve the name ambiguity of the following variables: {overrides}")
        engine_variables.update(self.variables)
        return self.evaluate(engine_variables)

    def evaluate(self, variables: Optional[Dict[str, float]] = None) -> float:
        if not self.root:
            raise RuntimeError(f"function '{self.formula}' is not loaded")
        return self.root.evaluate(variables)

    def is_loaded(self) -> bool:
        return bool(self.root)

    def unload(self) -> None:
        self.root = None
        self.variables.clear()

    def load(self) -> None:
        self.root = self.parse(self.formula)

    @classmethod
    def format_infix(cls, formula: str) -> str:
        from . import lib
        from .factory import FunctionFactory
        from .rule import Rule

        factory: FunctionFactory = lib.factory_manager.function
        operators: Set[str] = set(factory.operators().keys()).union({'(', ')', ','})
        operators -= {Rule.AND, Rule.OR}

        # sorted to have multi-char operators separated first (eg., ** and *)
        regex = "|".join(re.escape(o) for o in sorted(operators, reverse=True))
        spaced = re.sub(fr"({regex})", r' \1 ', formula)
        result = re.sub(r"\s+", " ", spaced).strip()
        return result

    @classmethod  # noqa: C901 mccabe complexity=19
    def infix_to_postfix(cls, formula: str) -> str:
        # TODO: support for unary and binary (+,-)
        from . import lib
        from .factory import FunctionFactory

        formula = cls.format_infix(formula)
        factory: FunctionFactory = lib.factory_manager.function

        from collections import deque
        queue: Deque[str] = deque()
        stack: List[str] = []

        for token in formula.split():
            if lib.debugging:
                lib.logger.debug("=" * 20)
                lib.logger.debug(f"formula: {formula}")
                lib.logger.debug(f"queue: {queue}")
                lib.logger.debug(f"stack: {stack}")

            element: Optional[Function.Element] = (factory.objects[token]
                                                   if token in factory.objects else None)
            is_operand = not element and token not in {"(", ")", ","}

            if is_operand:
                queue.append(token)

            elif element and element.is_function():
                stack.append(token)

            elif token == ',':
                while stack and stack[-1] != '(':
                    queue.append(stack.pop())
                if not stack or stack[-1] != '(':
                    raise SyntaxError(f"mismatching parentheses in: {formula}")

            elif element and element.is_operator():
                while stack and stack[-1] in factory.objects:
                    top = factory.objects[stack[-1]]
                    if ((element.associativity < 0 and element.precedence <= top.precedence)
                            or (element.associativity > 0 and element.precedence < top.precedence)):
                        queue.append(stack.pop())
                    else:
                        break

                stack.append(token)

            elif token == '(':
                stack.append(token)

            elif token == ')':
                while stack and stack[-1] != '(':
                    queue.append(stack.pop())

                if not stack or stack[-1] != '(':
                    raise SyntaxError(f"mismatching parentheses in: {formula}")

                stack.pop()  # get rid of "("

                if stack and stack[-1] in factory.objects:
                    if factory.objects[stack[-1]].is_function():
                        queue.append(stack.pop())
            else:
                raise RuntimeError(f"unexpected error with token: {token}")

        while stack:
            if stack[-1] in {'(', ')'}:
                raise SyntaxError(f"mismatching parentheses in: {formula}")
            queue.append(stack.pop())

        postfix = " ".join(queue)
        if lib.debugging:
            lib.logger.debug(f"formula={formula}")
            lib.logger.debug(f"postfix={postfix}")
        return postfix

    @classmethod
    def parse(cls, formula: str) -> 'Function.Node':
        from . import lib

        postfix = cls.infix_to_postfix(formula)
        stack: List[Function.Node] = []
        factory = lib.factory_manager.function

        for token in postfix.split():
            element: Optional[Function.Element] = (factory.objects[token]
                                                   if token in factory.objects else None)
            is_operand = not element and token not in {"(", ")", ","}

            if element:
                if element.arity > len(stack):
                    raise SyntaxError(f"function element {element.name} has arity {element.arity}, "
                                      f"but the size of the stack is {len(stack)}")
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
            lib.logger.debug("\n  ".join(Op.describe(node, class_hierarchy=False)
                                         for node in stack))
        return stack[-1]
