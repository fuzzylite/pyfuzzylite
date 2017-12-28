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

from math import inf, nan
from typing import List

from .exporter import FllExporter
from .term import Term


class Variable(object):
    __slots__ = "name", "description", "terms", "value", "minimum", "maximum", "enabled", "lock_range"

    def __init__(self, name: str = "", minimum: float = -inf, maximum: float = inf, terms: List[Term] = None):
        self.name = name
        self.description = ""
        self.terms = [term for term in terms] if terms else []
        self.value = nan
        self.minimum = minimum
        self.maximum = maximum
        self.enabled = True
        self.lock_range = False

    def range(self) -> float:
        return self.maximum - self.minimum

    def __str__(self):
        return FllExporter().variable(self)


class InputVariable(Variable):
    pass


class OutputVariable(Variable):
    pass
