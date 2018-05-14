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

from enum import Enum
from typing import Iterable, Tuple

from .activation import Activation
from .defuzzifier import Defuzzifier
from .exporter import FllExporter
from .norm import SNorm, TNorm
from .rule import RuleBlock
from .variable import InputVariable, OutputVariable


class Engine(object):
    __slots__ = ["name", "description", "inputs", "outputs", "blocks"]

    def __init__(self, name: str = "",
                 description: str = "",
                 inputs: Iterable[InputVariable] = None,
                 outputs: Iterable[OutputVariable] = None,
                 blocks: Iterable[RuleBlock] = None):
        self.name = name
        self.description = description
        self.inputs = []
        self.outputs = []
        self.blocks = []
        if inputs:
            self.inputs.extend(inputs)
        if outputs:
            self.outputs.extend(outputs)
        if blocks:
            self.blocks.extend(blocks)

    def __str__(self):
        return FllExporter().engine(self)

    def configure(self, conjunction: TNorm = None, disjunction: SNorm = None,
                  implication: TNorm = None, aggregation: SNorm = None,
                  defuzzifier: Defuzzifier = None, activation: Activation = None):

        pass

    def is_ready(self) -> Tuple[bool, str]:
        pass

    def process(self) -> None:
        pass

    def restart(self) -> None:
        pass

    class Type(Enum):
        Unknown, Mamdani, Larsen, TakagiSugeno, Tsukamoto, InverseTsukamoto, Hybrid = range(7)

    def infer_type(self) -> Tuple[Type, 'reason']:
        pass

    def variables(self):
        return [*self.inputs, *self.outputs]
