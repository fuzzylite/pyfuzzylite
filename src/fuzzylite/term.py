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

import logging
from bisect import bisect
from math import cos, exp, fabs, inf, isnan, nan, pi
from typing import Dict, List, Union

from .engine import Engine
from .exporter import FllExporter
from .norm import SNorm, TNorm
from .operation import Operation as Op


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
    __slots__ = "name", "height"

    def __init__(self, name="", height=1.0):
        self.name = name
        self.height = height

    def __str__(self):
        """
         Returns the representation of the term in the FuzzyLite Language
          :return the representation of the term in FuzzyLite Language
          @see FllExporter
        """
        return FllExporter().term(self)

    def parameters(self) -> str:
        """
          Returns the parameters to configure the term. The parameters are
          separated by spaces. If there is one additional parameter, the
          parameter will be considered as the height of the term; otherwise,
          the height will be set to @f$1.0@f$
          :return the parameters to configure the term (@see Term::configure())
         """
        return self._parameters()

    def _parameters(self, *args):
        """
        Concatenates the parameters given, and the height if it is different from 1.0 or None
        :param args: is the parameters to configure the term
        :return: the parameters concatenated and an optional height at the end
        """
        result = []
        if args:
            result.extend([Op.str(x) for x in args])
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

    def membership(self, x) -> float:
        """
          Computes the has_membership function value at @f$x@f$
          :param x
          :return the has_membership function value @f$\mu(x)@f$
        """
        raise NotImplementedError()

    def update_reference(self, engine: Engine) -> None:
        """
          Updates the references (if any) to point to the current engine (useful
          when cloning engines or creating terms within Importer objects
          :param engine: is the engine to which this term belongs to
        """
        pass

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        """
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

    def is_monotonic(self):
        """
        Indicates whether the term is monotonic.
          :return whether the term is monotonic.
        """
        return False

    def discretize(self, start: float, end: float, resolution: int = 100, bounded_mf=True):
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
    __slots__ = "term", "degree", "implication"

    def __init__(self, term: Term = None, degree: float = 1.0, implication: TNorm = None):
        super().__init__()
        self.term = term
        self.degree = degree
        self.implication = implication

    def __str__(self):
        if self.implication:
            return "%s(%s,%s)" % (FllExporter().norm(self.implication), Op.str(self.degree),
                                  self.term.name if self.term else "none")
        else:
            return "(%s*%s)" % (Op.str(self.degree), self.term.name if self.term else "none")

    def membership(self, x: float):
        if isnan(x): return nan
        if not self.term: raise ValueError("expected a term to activate, but none found")
        if not self.implication: raise ValueError("expected an implication operator, but none found")
        result = self.implication.compute(self.term.membership(x), self.degree)
        logging.debug("%s: %s" % (Op.str(result), str(self)))
        return result


class Aggregated(Term):
    __slots__ = "terms", "minimum", "maximum", "aggregation"

    def __init__(self, name: str = "", minimum: float = nan, maximum: float = nan, aggregation: SNorm = None):
        super().__init__(name)
        self.terms = []
        self.minimum = minimum
        self.maximum = maximum
        self.aggregation = aggregation

    def __str__(self):
        result = [self.name + ":", self.__class__.__name__]

        activated = [str(term) for term in self.terms]
        if self.aggregation:
            result.append("%s[%s]" % (FllExporter().norm(self.aggregation), ",".join(activated)))
        else:
            result.append("[%s]" % "+".join(activated))

        return " ".join(result)

    def range(self):
        return self.maximum - self.minimum

    def membership(self, x):
        if isnan(x): return nan
        if self.terms and not self.aggregation:
            raise ValueError("expected an aggregation operator, but none found")

        result = 0.0
        for term in self.terms:
            result = self.aggregation.compute(result, term.membership(x))
        logging.debug("%s: %s" % (Op.str(result), str(self)))
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

    def highest_activated_term(self) -> Activated:
        result = None
        maximum_activation = -inf
        for activation in self.terms:
            if activation.degree > maximum_activation:
                maximum_activation = activation.degree
                result = activation
        return result


class Bell(Term):
    __slots__ = "center", "width", "slope"

    def __init__(self, name="", center=nan, width=nan, slope=nan, height=1.0):
        super().__init__(name, height)
        self.center = center
        self.width = width
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        return self.height * (1.0 / (1.0 + (fabs((x - self.center) / self.width) ** (2.0 * self.slope))))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width, self.slope = values[0:3]
        self.height = 1.0 if len(values) == 3 else values[-1]


class Binary(Term):
    __slots__ = "start", "direction"

    def __init__(self, name="", start=nan, direction=nan, height=1.0):
        super().__init__(name, height)
        self.start = start
        self.direction = direction

    def membership(self, x: float) -> float:
        if isnan(x): return nan

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
    __slots__ = "inflection", "end"

    def __init__(self, name="", inflection=nan, end=nan, height=1.0):
        super().__init__(name, height)
        self.inflection = inflection
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        if self.inflection <= self.end:  # Concave increasing
            if x < self.end:
                return self.height * (self.end - self.inflection) / (2.0 * self.end - self.inflection - x)

        else:  # Concave decreasing
            if x > self.end:
                return self.height * (self.inflection - self.end) / (self.inflection - 2.0 * self.end + x)

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
    __slots__ = "value"

    def __init__(self, name="", value=nan):
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
    __slots__ = "center", "width"

    def __init__(self, name="", center=nan, width=nan, height=1.0):
        super().__init__(name, height)
        self.center = center
        self.width = width

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        if x < self.center - 0.5 * self.width or x > self.center + 0.5 * self.width:
            return self.height * 0.0

        return self.height * (0.5 * (1.0 + cos(2.0 / self.width * pi * (x - self.center))))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Discrete(Term):
    class Pair(object):
        __slots__ = "x", "y"

        def __init__(self, x=nan, y=nan):
            self.x = x
            self.y = y

        def __iter__(self):
            return iter((self.x, self.y))

        def __str__(self):
            return "(%s,%s)" % (self.x, self.y)

        def __eq__(self, other):
            if isinstance(other, float): return self.x == other
            return self.x == other.x

        def __ne__(self, other):
            if isinstance(other, float): return self.x != other
            return self.x != other.x

        def __lt__(self, other):
            if isinstance(other, float): return self.x < other
            return self.x < other.x

        def __le__(self, other):
            if isinstance(other, float): return self.x <= other
            return self.x <= other.x

        def __gt__(self, other):
            if isinstance(other, float): return self.x > other
            return self.x > other.x

        def __ge__(self, other):
            if isinstance(other, float): return self.x >= other
            return self.x >= other.x

    __slots__ = "xy"

    def __init__(self, name="", xy: [Pair] = None, height=1.0):
        super().__init__(name, height)
        self.xy = xy if xy else []

    def __iter__(self):
        return iter(self.xy)

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        if not self.xy:
            raise ValueError()

        if x <= self.xy[0].x:
            return self.height * self.xy[0].y

        if x >= self.xy[-1].x:
            return self.height * self.xy[-1].y

        index = bisect(self.xy, x)

        lower_bound = self.xy[index - 1]
        if x == lower_bound.x:
            return self.height * lower_bound.y

        upper_bound = self.xy[index]

        return self.height * Op.scale(x, lower_bound.x, upper_bound.x, lower_bound.y, upper_bound.y)

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float):
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

    def x(self):
        return (pair.x for pair in self.xy)

    def y(self):
        return (pair.y for pair in self.xy)

    def sort(self):
        Discrete.sort_pairs(self.xy)

    @staticmethod
    def sort_pairs(xy: [Pair]):
        xy.sort(key=lambda pair: pair.x)

    @staticmethod
    def pairs_from(values: Union[List[float], Dict[float, float]]) -> [Pair]:
        if isinstance(values, dict):
            return [Discrete.Pair(x, y) for x, y in values.items()]

        if len(values) % 2 != 0:
            raise ValueError("not enough values to unpack (expected even number, got %i) in %s" % (len(values), values))

        return [Discrete.Pair(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]

    @staticmethod
    def values_from(pairs: [Pair]) -> [float]:
        result = []
        for xy in pairs:
            result.extend(xy)
        return result

    @staticmethod
    def dict_from(pairs: [Pair]) -> Dict[float, float]:
        return {pair.x: pair.y for pair in pairs}


class Gaussian(Term):
    __slots__ = "mean", "standard_deviation"

    def __init__(self, name="", mean=nan, standard_deviation=nan, height=1.0):
        super().__init__(name, height)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        return self.height * exp((-(x - self.mean) * (x - self.mean)) /
                                 (2.0 * self.standard_deviation * self.standard_deviation))

    def parameters(self) -> str:
        return super()._parameters(self.mean, self.standard_deviation)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.mean, self.standard_deviation = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class GaussianProduct(Term):
    __slots__ = "mean_a", "standard_deviation_a", "mean_b", "standard_deviation_b"

    def __init__(self, name="", mean_a=nan, standard_deviation_a=nan,
                 mean_b=nan, standard_deviation_b=nan, height=1.0):
        super().__init__(name, height)
        self.mean_a = mean_a
        self.standard_deviation_a = standard_deviation_a
        self.mean_b = mean_b
        self.standard_deviation_b = standard_deviation_b

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        a = b = 1.0

        if x < self.mean_a:
            a = exp((-(x - self.mean_a) * (x - self.mean_a)) /
                    (2.0 * self.standard_deviation_a * self.standard_deviation_a))

        if x > self.mean_b:
            b = exp((-(x - self.mean_b) * (x - self.mean_b)) /
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
    __slots__ = "coefficients", "engine"

    def __init__(self, name: str = "", coefficients: List[float] = None, engine: Engine = None):
        super().__init__(name)
        self.coefficients = [c for c in coefficients] if coefficients else []
        self.engine = engine

    def membership(self, _):
        if not self.engine:
            raise ValueError("expected the reference to an engine, but found none")

        result = 0
        number_of_coefficients = len(self.coefficients)
        input_variables = self.engine.input
        for i, input_variable in enumerate(input_variables.values()):
            if i < number_of_coefficients:
                result += self.coefficients[i] * input_variable.value
        if number_of_coefficients > len(input_variables):
            result += self.coefficients[len(input_variables)]

        return result

    def configure(self, parameters: str):
        self.coefficients = [float(p) for p in parameters.split()]

    def parameters(self):
        return self._parameters(*self.coefficients)

    def update_reference(self, engine: Engine):
        self.engine = engine


class PiShape(Term):
    __slots__ = "bottom_left", "top_left", "top_right", "bottom_right"

    def __init__(self, name="", bottom_left=nan, top_left=nan,
                 top_right=nan, bottom_right=nan, height=1.0):
        super().__init__(name, height)
        self.bottom_left = bottom_left
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right

    def membership(self, x: float) -> float:
        if isnan(x): return nan

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
    __slots__ = "start", "end"

    def __init__(self, name="", start=nan, end=nan, height=1.0):
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x): return nan

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
    __slots__ = "start", "end"

    def __init__(self, name="", start=nan, end=nan, height=1.0):
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x): return nan

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
class SShape(Term):
    __slots__ = "start", "end"

    def __init__(self, name="", start=nan, end=nan, height=1.0):
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x): return nan

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


# TODO: Tsukamoto
class Sigmoid(Term):
    __slots__ = "inflection", "slope"

    def __init__(self, name="", inflection=nan, slope=nan, height=1.0):
        super().__init__(name, height)
        self.inflection = inflection
        self.slope = slope

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        return self.height * 1.0 / (1.0 + exp(-self.slope * (x - self.inflection)))

    def is_monotonic(self) -> bool:
        return True

    def parameters(self) -> str:
        return super()._parameters(self.inflection, self.slope)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.inflection, self.slope = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class SigmoidDifference(Term):
    __slots__ = "left", "rising", "falling", "right"

    def __init__(self, name="", left=nan, rising=nan,
                 falling=nan, right=nan, height=1.0):
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        a = 1.0 / (1.0 + exp(-self.rising * (x - self.left)))
        b = 1.0 / (1.0 + exp(-self.falling * (x - self.right)))

        return self.height * fabs(a - b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class SigmoidProduct(Term):
    __slots__ = "left", "rising", "falling", "right"

    def __init__(self, name="", left=nan, rising=nan,
                 falling=nan, right=nan, height=1.0):
        super().__init__(name, height)
        self.left = left
        self.rising = rising
        self.falling = falling
        self.right = right

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        a = 1.0 + exp(-self.rising * (x - self.left))
        b = 1.0 + exp(-self.falling * (x - self.right))

        return self.height * 1.0 / (a * b)

    def parameters(self) -> str:
        return super()._parameters(self.left, self.rising, self.falling, self.right)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.left, self.rising, self.falling, self.right = values[0:4]
        self.height = 1.0 if len(values) == 4 else values[-1]


class Spike(Term):
    __slots__ = "center", "width"

    def __init__(self, name="", inflection=nan, slope=nan, height=1.0):
        super().__init__(name, height)
        self.center = inflection
        self.width = slope

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        return self.height * exp(-fabs(10.0 / self.width * (x - self.center)))

    def parameters(self) -> str:
        return super()._parameters(self.center, self.width)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.center, self.width = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Trapezoid(Term):
    __slots__ = "vertex_a", "vertex_b", "vertex_c", "vertex_d"

    def __init__(self, name="", vertex_a=nan, vertex_b=nan,
                 vertex_c=nan, vertex_d=nan, height=1.0):
        super().__init__(name, height)
        # todo: constructor with two takes_parameters
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.vertex_c = vertex_c
        self.vertex_d = vertex_d

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        # TODO: check in fuzzylite and jfuzzylite

        if x <= self.vertex_a or x >= self.vertex_d:
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
    __slots__ = "vertex_a", "vertex_b", "vertex_c"

    def __init__(self, name="", vertex_a=nan, vertex_b=nan,
                 vertex_c=nan, height=1.0):
        super().__init__(name, height)
        # todo: constructor with two takes_parameters
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.vertex_c = vertex_c

    def membership(self, x: float) -> float:
        if isnan(x): return nan
        # TODO: check in fuzzylite and jfuzzylite

        if x <= self.vertex_a or x >= self.vertex_c:
            return self.height * 0.0

        if x < self.vertex_b:
            return self.height * (x - self.vertex_a) / (self.vertex_b - self.vertex_a)

        if x == self.vertex_b:
            return self.height * 1.0

        if x > self.vertex_b:
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
    __slots__ = "start", "end"

    def __init__(self, name="", start=nan, end=nan, height=1.0):
        super().__init__(name, height)
        self.start = start
        self.end = end

    def membership(self, x: float) -> float:
        if isnan(x): return nan

        if x <= self.start:
            return self.height * 1.0

        if x <= 0.5 * (self.start + self.end):
            return self.height * (1.0 - 2.0 * ((x - self.start) / (self.end - self.start)) ** 2)

        if x < self.end:
            return self.height * 2.0 * ((x - self.end) / (self.end - self.start)) ** 2

        return self.height * 0.0

    def is_monotonic(self):
        return True

    def parameters(self) -> str:
        return super()._parameters(self.start, self.end)

    def configure(self, parameters: str) -> None:
        values = tuple(float(x) for x in parameters.split())
        self.start, self.end = values[0:2]
        self.height = 1.0 if len(values) == 2 else values[-1]


class Function(Term):
    __slots__ = "variables",

    def load(self, formula: str) -> None:
        raise NotImplemented()

    def evaluate(self) -> float:
        raise NotImplemented()
