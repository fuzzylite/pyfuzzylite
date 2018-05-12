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

from math import fabs, inf, isfinite, isnan, nan
from typing import Iterable, Tuple

from .exporter import FllExporter
from .norm import SNorm
from .operation import Operation as Op
from .term import Aggregated


class Variable(object):
    __slots__ = ["name", "description", "minimum", "maximum", "enabled", "lock_range", "_value", "terms"]

    def __init__(self, name: str = "", description: str = "", minimum: float = -inf, maximum: float = inf,
                 terms: Iterable['Term'] = None):
        self.name = name
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.enabled = True
        self.lock_range = False
        self._value = nan
        self.terms = []
        if terms:
            self.terms.extend(terms)

    def __str__(self):
        return FllExporter().variable(self)

    @property
    def range(self) -> Tuple[float, float]:
        return self.minimum, self.maximum

    @range.setter
    def range(self, min_max: Tuple[float, float]):
        self.minimum, self.maximum = min_max

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = Op.bound(value, self.minimum, self.maximum) if self.lock_range else value

    def fuzzify(self, x: float) -> str:
        result = []
        for term in self.terms:
            fx = nan
            try:
                fx = term.membership(x)
            except ValueError:
                pass
            if not result:
                result.append("%s/%s" % (Op.str(fx), term.name))
            else:
                result.append(" %s %s/%s" % ("+" if isnan(fx) or fx >= 0.0 else "-",
                                             Op.str(fx), term.name))
        return "".join(result)

    def highest_membership(self, x: float) -> Tuple[float, 'Term']:
        result = (0.0, None)
        for term in self.terms:
            y = nan
            try:
                y = term.membership(x)
            except ValueError:
                pass
            if y > result[0]:
                result = (y, term)
        return result


class InputVariable(Variable):
    def __init__(self, name: str = "", description: str = "", minimum: float = -inf, maximum: float = inf,
                 terms: Iterable['Term'] = None):
        super().__init__(name, description, minimum, maximum, terms)

    def __str__(self):
        return FllExporter().input_variable(self)

    def fuzzy_value(self):
        return super().fuzzify(self.value)


class OutputVariable(Variable):
    __slots__ = ["fuzzy", "defuzzifier", "previous_value", "default_value", "lock_previous_value"]

    def __init__(self, name: str = "", description: str = "", minimum: float = -inf, maximum: float = inf,
                 terms: Iterable['Term'] = None):
        # name, minimum, and maximum are properties in this class, replacing the inherited members to point to
        # the Aggregated object named fuzzy. Thus, first we need to set up the fuzzy object such that initializing
        # the parent object will use the respective replacements.
        self.fuzzy = Aggregated()
        # initialize parent members
        super().__init__(name, description, minimum, maximum, terms)

        # set values of output variable
        self.defuzzifier = None
        self.previous_value = nan
        self.default_value = nan
        self.lock_previous_value = False

    def __str__(self):
        return FllExporter().output_variable(self)

    @property
    def name(self) -> str:
        return self.fuzzy.name

    @name.setter
    def name(self, value: str):
        self.fuzzy.name = value

    @property
    def minimum(self) -> float:
        return self.fuzzy.minimum

    @minimum.setter
    def minimum(self, value: float):
        self.fuzzy.minimum = value

    @property
    def maximum(self) -> float:
        return self.fuzzy.maximum

    @maximum.setter
    def maximum(self, value: float):
        self.fuzzy.maximum = value

    @property
    def aggregation(self) -> SNorm:
        return self.fuzzy.aggregation

    @aggregation.setter
    def aggregation(self, value: SNorm):
        self.fuzzy.aggregation = value

    def defuzzify(self) -> None:
        if not self.enabled: return
        if isfinite(self.value):
            self.previous_value = self.value

        result = nan
        exception = None
        is_valid = bool(self.fuzzy.terms)
        if is_valid:
            # Checks whether the variable can be defuzzified without exceptions.
            # If it cannot be defuzzified, be that due to a missing defuzzifier
            # or aggregation operator, the expected behaviour is to leave the
            # variable in a state that reflects an invalid defuzzification,
            # that is, apply logic of default values and previous values.
            is_valid = False
            if self.defuzzifier:
                try:
                    result = self.defuzzifier.defuzzify(self.fuzzy, self.minimum, self.maximum)
                    is_valid = True
                except ValueError as ex:
                    exception = ex
            else:
                exception = ValueError(f"expected a defuzzifier in output variable {self.name}, but found none")

        if not is_valid:
            # if a previous defuzzification was successfully performed and
            # and the output value is supposed not to change when the output is empty
            if self.lock_previous_value and not isnan(self.previous_value):
                result = self.previous_value
            else:
                result = self.default_value

        self.value = result

        if exception:
            raise exception

    def clear(self):
        self.fuzzy.clear()
        self.value = nan
        self.previous_value = nan

    def fuzzy_value(self):
        result = []
        for term in self.terms:
            degree = self.fuzzy.activation_degree(term)

            if not result:
                result.append("%s/%s" % (Op.str(degree), term.name))
            else:
                result.append(" %s %s/%s" % ("+" if isnan(degree) or degree >= 0 else "-",
                                             Op.str(fabs(degree)), term.name))
        return "".join(result)
