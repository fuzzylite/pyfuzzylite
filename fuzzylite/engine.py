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

import enum
from typing import Iterable, List, Optional, Tuple

from .activation import Activation
from .defuzzifier import Defuzzifier
from .exporter import FllExporter
from .norm import SNorm, TNorm
from .rule import RuleBlock
from .variable import InputVariable, OutputVariable, Variable


class Engine(object):
    __slots__ = ["name", "description", "input_variables", "output_variables", "rule_blocks"]

    def __init__(self, name: str = "",
                 description: str = "",
                 input_variables: Optional[Iterable[InputVariable]] = None,
                 output_variables: Optional[Iterable[OutputVariable]] = None,
                 rule_blocks: Optional[Iterable[RuleBlock]] = None) -> None:
        self.name = name
        self.description = description
        self.input_variables: List[InputVariable] = []
        self.output_variables: List[OutputVariable] = []
        self.rule_blocks: List[RuleBlock] = []
        if input_variables:
            self.input_variables.extend(input_variables)
        if output_variables:
            self.output_variables.extend(output_variables)
        if rule_blocks:
            self.rule_blocks.extend(rule_blocks)

    def __str__(self) -> str:
        return FllExporter().engine(self)

    def configure(self, conjunction: Optional[TNorm] = None,
                  disjunction: Optional[SNorm] = None,
                  implication: Optional[TNorm] = None,
                  aggregation: Optional[SNorm] = None,
                  defuzzifier: Optional[Defuzzifier] = None,
                  activation: Optional[Activation] = None) -> None:

        pass

    def is_ready(self) -> Tuple[bool, str]:
        pass

    def process(self) -> None:
        pass

    def restart(self) -> None:
        pass

    class Type(enum.Enum):
        Unknown, Mamdani, Larsen, TakagiSugeno, Tsukamoto, InverseTsukamoto, Hybrid = range(7)

    def infer_type(self) -> Tuple[Type, str]:
        pass

    @property
    def variables(self) -> List[Variable]:
        return [*self.input_variables, *self.output_variables]

    def variable(self, name: str) -> Optional[Variable]:
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def input_variable(self, name: str) -> Optional[InputVariable]:
        for variable in self.input_variables:
            if variable.name == name:
                return variable
        return None

    def output_variable(self, name: str) -> Optional[OutputVariable]:
        for variable in self.output_variables:
            if variable.name == name:
                return variable
        return None
