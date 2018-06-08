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

import bisect
import enum
import logging
import math
import typing
from math import inf, isnan, nan

import fuzzylite
from .exporter import FllExporter
from .norm import SNorm, TNorm
from .operation import Op

if typing.TYPE_CHECKING:
    from .engine import Engine


class Term(object):
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
    __slots__ = ("name", "height")

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
        result: typing.List[str] = []
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

    def update_reference(self, engine: 'Engine') -> None:
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
    __slots__ = ("term", "degree", "implication")

    def __init__(self, term: Term, degree: float = 1.0,
                 implication: typing.Optional[TNorm] = None) -> None:
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
    __slots__ = ("terms", "minimum", "maximum", "aggregation")

    def __init__(self, name: str = "", minimum: float = nan, maximum: float = nan,
                 aggregation: typing.Optional[SNorm] = None,
                 terms: typing.Iterable[Activated] = None) -> None:
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum
        self.aggregation = aggregation
        self.terms: typing.List[Activated] = []
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
            result = self.aggregation.compute(result, term.membership(x))
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

    def highest_activated_term(self) -> typing.Optional[Activated]:
        result = None
        maximum_activation = -inf
        for activation in self.terms:
            if activation.degree > maximum_activation:
                maximum_activation = activation.degree
                result = activation
        return result

    def clear(self) -> None:
        self.terms.clear()


class Bell(Term):
    __slots__ = ("center", "width", "slope")

    def __init__(self, name: str = "", center: float = nan, width: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.center = center
        self.width = width
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * (1.0 /
                              (1.0 + (math.fabs((x - self.center) / self.width) ** (
                                      2.0 * self.slope))))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width, self.slope = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


class Binary(Term):
    __slots__ = ("start", "direction")

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
        values = tuple(float(x) for x in parameters.split())
        self.start, self.direction = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Concave(Term):
    __slots__ = ("inflection", "end")

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
                return (self.height * (self.end - self.inflection) /
                        (2.0 * self.end - self.inflection - x))

        else:  # Concave decreasing
            if x > self.end:
                return (self.height * (self.inflection - self.end) /
                        (self.inflection - 2.0 * self.end + x))

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
        values = tuple(float(x) for x in parameters.split())
        self.inflection, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Constant(Term):
    __slots__ = ("value",)

    def __init__(self, name: str = "", value: float = nan) -> None:
        super().__init__(name)
        self.value = value

    def membership(self, x: float) -> float:
        return self.value

    def parameters(self) -> str:
        return super()._parameters(self.value)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        if not values:
            raise ValueError("not enough values to unpack (expected 1, got 0)")
        self.value = values[0]
        self.height = 1.0


class Cosine(Term):
    __slots__ = ("center", "width")

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

        return self.height * (
                0.5 * (1.0 + math.cos(2.0 / self.width * math.pi * (x - self.center))))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Discrete(Term):
    class Pair(object):
        __slots__ = ("x", "y")

        def __init__(self, x: float = nan, y: float = nan) -> None:
            self.x = x
            self.y = y

        def __str__(self) -> str:
            return f"({self.x},{self.y})"

        # todo: consider y
        def __eq__(self, other: object) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.x == other.x
            return self.x == other

        def __ne__(self, other: object) -> bool:
            if isinstance(other, Discrete.Pair):
                return self.x != other.x
            return self.x != other

        def __lt__(self, other: typing.Union[float, 'Discrete.Pair']) -> bool:
            if isinstance(other, float):
                return self.x < other
            return self.x < other.x

        def __le__(self, other: typing.Union[float, 'Discrete.Pair']) -> bool:
            if isinstance(other, float):
                return self.x <= other
            return self.x <= other.x

        def __gt__(self, other: typing.Union[float, 'Discrete.Pair']) -> bool:
            if isinstance(other, float):
                return self.x > other
            return self.x > other.x

        def __ge__(self, other: typing.Union[float, 'Discrete.Pair']) -> bool:
            if isinstance(other, float):
                return self.x >= other
            return self.x >= other.x

    __slots__ = ("xy",)

    def __init__(self, name: str = "", xy: typing.Iterable[Pair] = None,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.xy: typing.List[Discrete.Pair] = []
        if xy:
            self.xy.extend(xy)

    def __iter__(self) -> typing.Iterable['Discrete.Pair']:
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

        index = bisect.bisect(self.xy, x)

        lower_bound = self.xy[index - 1]
        if x == lower_bound.x:
            return self.height * lower_bound.y

        upper_bound = self.xy[index]

        return self.height * Op.scale(x, lower_bound.x, upper_bound.x, lower_bound.y, upper_bound.y)

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        # todo: approximate tsukamoto
        pass

    def parameters(self) -> str:
        return super()._parameters(*Discrete.values_from(self.xy))

    def configure(self, parameters: str) -> None:
        values = [float(x) for x in parameters.split()]
        if len(values) % 2 == 0:
            self.height = 1.0
        else:
            self.height = values[-1]
            del values[-1]

        self.xy = Discrete.pairs_from(values)

    def x(self) -> typing.Iterable[float]:
        return (pair.x for pair in self.xy)

    def y(self) -> typing.Iterable[float]:
        return (pair.y for pair in self.xy)

    def sort(self) -> None:
        Discrete.sort_pairs(self.xy)

    @staticmethod
    def sort_pairs(xy: typing.List['Discrete.Pair']) -> None:
        xy.sort(key=lambda pair: pair.x)

    Floatable = typing.TypeVar("Floatable", typing.SupportsFloat, str, bytes)

    @staticmethod
    def pairs_from(
            values: typing.Union[typing.Sequence[Floatable], typing.Dict[Floatable, Floatable]]) -> \
            typing.List['Discrete.Pair']:
        if isinstance(values, dict):
            return [Discrete.Pair(float(x), float(y)) for x, y in values.items()]

        if len(values) % 2 != 0:
            raise ValueError("not enough values to unpack (expected an even number, "
                             f"but got {len(values)}) in {values}")

        result = [Discrete.Pair(float(values[i]), float(values[i + 1]))
                  for i in range(0, len(values) - 1, 2)]
        return result

    # TODO: More pythonic?
    @staticmethod
    def values_from(pairs: typing.List['Discrete.Pair']) -> typing.List[float]:
        result: typing.List[float] = []
        for xy in pairs:
            result.extend([xy.x, xy.y])
        return result

    @staticmethod
    def dict_from(pairs: typing.List['Discrete.Pair']) -> typing.Dict[float, float]:
        return {pair.x: pair.y for pair in pairs}


class Gaussian(Term):
    __slots__ = ("mean", "standard_deviation")

    def __init__(self, name: str = "", mean: float = nan, standard_deviation: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * math.exp((-(x - self.mean) * (x - self.mean)) /
                                      (2.0 * self.standard_deviation * self.standard_deviation))

    def parameters(self) -> str:
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.mean, self.standard_deviation = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class GaussianProduct(Term):
    __slots__ = ("mean_a", "standard_deviation_a", "mean_b", "standard_deviation_b")

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
            a = math.exp((-(x - self.mean_a) * (x - self.mean_a)) /
                         (2.0 * self.standard_deviation_a * self.standard_deviation_a))

        if x > self.mean_b:
            b = math.exp((-(x - self.mean_b) * (x - self.mean_b)) /
                         (2.0 * self.standard_deviation_b * self.standard_deviation_b))

        return self.height * a * b

    def parameters(self) -> str:
        return super()._parameters(self.mean_a, self.standard_deviation_a,
                                   self.mean_b, self.standard_deviation_b)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.mean_a, self.standard_deviation_a, self.mean_b, self.standard_deviation_b = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Linear(Term):
    __slots__ = ("coefficients", "engine")

    def __init__(self, name: str = "", coefficients: typing.Iterable[float] = None,
                 engine: 'Engine' = None) -> None:
        super().__init__(name)
        self.coefficients: typing.List[float] = []
        if coefficients:
            self.coefficients.extend(coefficients)
        self.engine = engine

    def membership(self, _: float) -> float:
        if not self.engine:
            raise ValueError("expected the reference to an engine, but found none")

        result = 0.0
        number_of_coefficients = len(self.coefficients)
        input_variables = self.engine.inputs
        for i, input_variable in enumerate(input_variables):
            if i < number_of_coefficients:
                result += self.coefficients[i] * input_variable.value
        if number_of_coefficients > len(input_variables):
            result += self.coefficients[len(input_variables)]

        return result

    def configure(self, parameters: str) -> None:
        self.coefficients = [float(p) for p in parameters.split()]

    def parameters(self) -> str:
        return self._parameters(*self.coefficients)

    def update_reference(self, engine: 'Engine') -> None:
        self.engine = engine


class PiShape(Term):
    __slots__ = ("bottom_left", "top_left", "top_right", "bottom_right")

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
        values = tuple(float(x) for x in parameters.split())
        self.bottom_left, self.top_left, self.top_right, self.bottom_right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Ramp(Term):
    __slots__ = ("start", "end")

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
        values = tuple(float(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Rectangle(Term):
    __slots__ = ("start", "end")

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
        values = tuple(float(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class Sigmoid(Term):
    __slots__ = ("inflection", "slope")

    def __init__(self, name: str = "", inflection: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.inflection = inflection
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * 1.0 / (1.0 + math.exp(-self.slope * (x - self.inflection)))

    def is_monotonic(self) -> bool:
        return True

    def parameters(self) -> str:
        return super()._parameters(self.inflection, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.inflection, self.slope = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class SigmoidDifference(Term):
    __slots__ = ("left", "rising", "falling", "right")

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

        a = 1.0 / (1.0 + math.exp(-self.rising * (x - self.left)))
        b = 1.0 / (1.0 + math.exp(-self.falling * (x - self.right)))

        return self.height * math.fabs(a - b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class SigmoidProduct(Term):
    __slots__ = ("left", "rising", "falling", "right")

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

        a = 1.0 + math.exp(-self.rising * (x - self.left))
        b = 1.0 + math.exp(-self.falling * (x - self.right))

        return self.height * 1.0 / (a * b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Spike(Term):
    __slots__ = ("center", "width")

    def __init__(self, name: str = "", inflection: float = nan, slope: float = nan,
                 height: float = 1.0) -> None:
        super().__init__(name, height)
        self.center = inflection
        self.width = slope

    def membership(self, x: float) -> float:
        if isnan(x):
            return nan
        return self.height * math.exp(-math.fabs(10.0 / self.width * (x - self.center)))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


# TODO: Tsukamoto
class SShape(Term):
    __slots__ = ("start", "end")

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
        values = tuple(float(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Trapezoid(Term):
    __slots__ = ("vertex_a", "vertex_b", "vertex_c", "vertex_d")

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
        values = tuple(float(x) for x in parameters.split())
        self.vertex_a, self.vertex_b, self.vertex_c, self.vertex_d = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Triangle(Term):
    __slots__ = ("vertex_a", "vertex_b", "vertex_c")

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
        values = tuple(float(x) for x in parameters.split())
        self.vertex_a, self.vertex_b, self.vertex_c = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


# TODO: Tsukamoto
class ZShape(Term):
    __slots__ = ("start", "end")

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
        values = tuple(float(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Function(Term):
    __slots__ = ("root", "formula", "engine", "variables")

    class Element(object):
        __slots__ = ("name", "description", "element_type", "method", "arity", "precedence",
                     "associativity")

        class Type(enum.Enum):
            Operator, Function = range(2)

        def __init__(self, name: str, description: str, element_type: 'Function.Element.Type',
                     method: typing.Callable[..., float] = None,
                     arity: int = 0, precedence: int = 0,
                     associativity: int = -1) -> None:
            self.name = name
            self.description = description
            self.element_type = element_type
            self.method = method
            self.arity = Op.arity_of(method) if method and arity < 0 else arity
            self.precedence = precedence
            self.associativity = associativity

        def __str__(self) -> str:
            result = [f"name='{self.name}'",
                      f"description='{self.description}'",
                      f"element_type='{str(self.element_type)}'",
                      f"method='{str(self.method)}'",
                      f"arity={self.arity}",
                      f"precedence={self.precedence}",
                      f"associativity={self.associativity}"]
            return "{0}: {1}".format(self.__class__.__name__, ", ".join(result))

    class Node(object):
        __slots__ = ("element", "variable", "value", "left", "right")

        def __init__(self, element: 'Function.Element' = None, variable: str = "",
                     value: float = nan,
                     left: typing.Optional['Function.Node'] = None,
                     right: typing.Optional['Function.Node'] = None) -> None:
            self.element = element
            self.variable = variable
            self.value = value
            self.left = left
            self.right = right

        def __str__(self) -> str:
            if self.element:
                result = self.element.name
            elif self.variable:
                result = self.variable
            else:
                result = Op.str(self.value)
            return result

        def evaluate(self, local_variables: typing.Dict[str, float] = None) -> float:
            result = nan
            if self.element:
                if not self.element.method:
                    raise ValueError("expected a method reference, but found none")
                arity = self.element.arity
                try:
                    if arity == 0:
                        result = self.element.method()
                    elif arity == 1:
                        if not self.left:
                            raise ValueError("expected a left node, but found none")
                        result = self.element.method(
                            self.left.evaluate(local_variables))
                    elif arity == 2:
                        if not self.left:
                            raise ValueError("expected a left node, but found none")
                        if not self.right:
                            raise ValueError("expected a right node, but found none")
                        result = self.element.method(
                            self.left.evaluate(local_variables),
                            self.right.evaluate(local_variables))
                except Exception as ex:
                    raise ValueError(f"error occurred during the evaluation of method "
                                     f"'{self.element.method.__name__}': {ex}") from ex
            elif self.variable:
                if not local_variables or self.variable not in local_variables:
                    raise ValueError(
                        f"expected a map of variables containing the value for '{self.variable}', "
                        f"but the map contains: {local_variables}")
                result = local_variables[self.variable]

            else:
                result = self.value

            if fuzzylite.library.debugging:
                fuzzylite.library.logger.debug(f"{self.postfix()} = {Op.str(result)}")

            return result

        def prefix(self, node: 'Function.Node' = None) -> str:
            if not node:
                return self.prefix(self)

            if not isnan(node.value):
                return Op.str(node.value)
            if node.variable:
                return node.variable

            result = [str(node)]
            if node.left:
                result.append(self.prefix(node.left))
            if node.right:
                result.append(self.prefix(node.right))
            return " ".join(result)

        def infix(self, node: 'Function.Node' = None) -> str:
            if not node:
                return self.infix(self)

            if not isnan(node.value):
                return Op.str(node.value)
            if node.variable:
                return node.variable

            children = []
            if node.left:
                children.append(self.infix(node.left))
            if node.right:
                children.append(self.infix(node.right))

            is_function = (node.element and
                           node.element.element_type == Function.Element.Type.Function)

            if is_function:
                result = str(node) + f" ( {' '.join(children)} )"
            else:  # is operator
                result = f" {str(node)} ".join(children)

            return result

        def postfix(self, node: 'Function.Node' = None) -> str:
            if not node:
                return self.postfix(self)

            if not isnan(node.value):
                return Op.str(node.value)
            if node.variable:
                return node.variable

            result = []
            if node.left:
                result.append(self.postfix(node.left))
            if node.right:
                result.append(self.postfix(node.right))
            result.append(str(node))
            return " ".join(result)

    def __init__(self, name: str = "", formula: str = "", engine: 'Engine' = None,
                 variables: typing.Dict[str, float] = None) -> None:
        super().__init__(name)
        self.root: typing.Optional[Function.Node] = None
        self.formula = formula
        self.engine = engine
        self.variables: typing.Dict[str, float] = {}
        if variables:
            self.variables.update(variables)

    def parameters(self) -> str:
        return self.formula

    def configure(self, parameters: str) -> None:
        self.formula = parameters
        self.load()

    def update_reference(self, engine: 'Engine') -> None:
        self.engine = engine
        # noinspection PyBroadException
        try:
            self.load()
        except:  # noqa: E722
            pass

    @staticmethod
    def create(name: str, formula: str, engine: 'Engine') -> 'Function':
        result = Function(name, formula, engine)
        result.load()
        return result

    def membership(self, x: float) -> float:
        if not self.is_loaded():
            raise RuntimeError(f"function '{self.formula}' is not loaded")

        if self.engine:
            self.variables.update(
                {variable.name: variable.value for variable in self.engine.inputs})
            self.variables.update(
                {variable.name: variable.value for variable in self.engine.outputs})

        self.variables['x'] = x

        return self.evaluate(self.variables)

    def evaluate(self, variables: typing.Dict[str, float] = None) -> float:
        if not self.is_loaded():
            raise RuntimeError("evaluation failed because function is not loaded")
        return self.root.evaluate(variables)

    def is_loaded(self) -> bool:
        return bool(self.root)

    def unload(self) -> None:
        self.root = None
        self.variables.clear()

    def load(self) -> None:
        pass
