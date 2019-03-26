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

__all__ = ["Variable", "InputVariable", "OutputVariable"]

import math
import typing
from math import inf, isnan, nan
from typing import Iterable, List, Optional, Tuple

from .exporter import FllExporter
from .norm import SNorm
from .operation import Op
from .term import Aggregated

if typing.TYPE_CHECKING:
    from .term import Term
    from .defuzzifier import Defuzzifier  # noqa: F401


class Variable:

    def __init__(self,
                 name: str = "",
                 description: str = "",
                 enabled: bool = True,
                 minimum: float = -inf,
                 maximum: float = inf,
                 lock_range: bool = False,
                 terms: Optional[Iterable['Term']] = None) -> None:
        self.name = name
        self.description = description
        self.enabled = enabled
        self.minimum = minimum
        self.maximum = maximum
        self.lock_range = lock_range
        self.terms: List[Term] = []
        if terms:
            self.terms.extend(terms)
        self._value = nan

    def __str__(self) -> str:
        return FllExporter().variable(self)

    def term(self, name: str) -> 'Term':
        for term in self.terms:
            if term.name == name:
                return term
        raise ValueError(f"term '{name}' not found in {t.name for t in self.terms}")

    @property
    def drange(self) -> float:
        return self.maximum - self.minimum

    @property
    def range(self) -> Tuple[float, float]:
        return self.minimum, self.maximum

    @range.setter
    def range(self, min_max: Tuple[float, float]) -> None:
        self.minimum, self.maximum = min_max

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = Op.bound(value, self.minimum, self.maximum) if self.lock_range else value

    def fuzzify(self, x: float) -> str:
        result: List[str] = []
        for term in self.terms:
            fx = nan
            try:
                fx = term.membership(x)
            except ValueError:
                pass
            if not result:
                result.append(f"{Op.str(fx)}/{term.name}")
            else:
                pm = '+' if Op.ge(fx, 0.0) or isnan(fx) else '-'
                result.append(f" {pm} {Op.str(fx)}/{term.name}")

        return "".join(result)

    def highest_membership(self, x: float) -> Tuple[float, Optional['Term']]:
        result: Tuple[float, Optional[Term]] = (0.0, None)
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

    def __init__(self,
                 name: str = "",
                 description: str = "",
                 enabled: bool = True,
                 minimum: float = -inf,
                 maximum: float = inf,
                 lock_range: bool = False,
                 terms: Optional[Iterable['Term']] = None) -> None:
        super().__init__(name=name,
                         description=description,
                         enabled=enabled,
                         minimum=minimum,
                         maximum=maximum,
                         lock_range=lock_range,
                         terms=terms)

    def __str__(self) -> str:
        return FllExporter().input_variable(self)

    def fuzzy_value(self) -> str:
        return super().fuzzify(self.value)


class OutputVariable(Variable):

    def __init__(self,
                 name: str = "",
                 description: str = "",
                 enabled: bool = True,
                 minimum: float = -inf,
                 maximum: float = inf,
                 lock_range: bool = False,
                 lock_previous: bool = False,
                 default_value: float = nan,
                 aggregation: Optional[SNorm] = None,
                 defuzzifier: Optional['Defuzzifier'] = None,
                 terms: Optional[Iterable['Term']] = None) -> None:
        # name, minimum, and maximum are properties in this class, replacing the inherited members
        # to point to the Aggregated object named fuzzy. Thus, first we need to set up the fuzzy
        # object such that initializing the parent object will use the respective replacements.
        self.fuzzy = Aggregated(aggregation=aggregation)
        # initialize parent members
        super().__init__(name=name,
                         description=description,
                         enabled=enabled,
                         minimum=minimum,
                         maximum=maximum,
                         lock_range=lock_range,
                         terms=terms)
        # set values of output variable
        self.defuzzifier = defuzzifier
        self.lock_previous = lock_previous
        self.default_value = default_value
        self.previous_value = nan

    def __str__(self) -> str:
        return FllExporter().output_variable(self)

    @property  # type: ignore
    def name(self) -> str:  # type: ignore
        return self.fuzzy.name

    @name.setter
    def name(self, value: str) -> None:
        self.fuzzy.name = value

    @property  # type: ignore
    def minimum(self) -> float:  # type: ignore
        return self.fuzzy.minimum

    @minimum.setter
    def minimum(self, value: float) -> None:
        self.fuzzy.minimum = value

    @property  # type: ignore
    def maximum(self) -> float:  # type: ignore
        return self.fuzzy.maximum

    @maximum.setter
    def maximum(self, value: float) -> None:
        self.fuzzy.maximum = value

    @property
    def aggregation(self) -> Optional[SNorm]:
        return self.fuzzy.aggregation

    @aggregation.setter
    def aggregation(self, value: SNorm) -> None:
        self.fuzzy.aggregation = value

    def defuzzify(self) -> None:
        if not self.enabled:
            return
        if math.isfinite(self.value):
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
                exception = ValueError(f"expected a defuzzifier in output variable {self.name}, "
                                       "but found none")

        if not is_valid:
            # if a previous defuzzification was successfully performed and
            # and the output value is supposed not to change when the output is empty
            if self.lock_previous and not isnan(self.previous_value):
                result = self.previous_value
            else:
                result = self.default_value

        self.value = result

        if exception:
            raise exception

    def clear(self) -> None:
        self.fuzzy.clear()
        self.previous_value = nan
        self._value = nan

    def fuzzy_value(self) -> str:
        result: List[str] = []
        for term in self.terms:
            degree = self.fuzzy.activation_degree(term)

            if not result:
                result.append("{0}/{1}".format(Op.str(degree), term.name))
            else:
                result.append(" {0} {1}/{2}".format("+" if isnan(degree) or degree >= 0 else "-",
                                                    Op.str(math.fabs(degree)), term.name))
        return "".join(result)
