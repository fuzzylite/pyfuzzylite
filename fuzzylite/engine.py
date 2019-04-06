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

__all__ = ["Engine"]

import enum
from math import nan
from typing import Iterable, List, Optional, Tuple, Union

from .activation import Activation
from .defuzzifier import Defuzzifier
from .exporter import FllExporter
from .norm import SNorm, TNorm
from .rule import RuleBlock
from .variable import InputVariable, OutputVariable, Variable


class Engine:
    @enum.unique
    class Type(enum.Enum):
        Unknown, Mamdani, Larsen, TakagiSugeno, Tsukamoto, InverseTsukamoto, Hybrid = range(7)

    def __init__(self, name: str = "",
                 description: str = "",
                 input_variables: Optional[Iterable[InputVariable]] = None,
                 output_variables: Optional[Iterable[OutputVariable]] = None,
                 rule_blocks: Optional[Iterable[RuleBlock]] = None,
                 load_rules: bool = False) -> None:
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
        if load_rules:
            for rb in self.rule_blocks:
                rb.load_rules(self)

    def __str__(self) -> str:
        return FllExporter().engine(self)

    def configure(self,
                  conjunction: Optional[Union[TNorm, str]] = None,
                  disjunction: Optional[Union[SNorm, str]] = None,
                  implication: Optional[Union[TNorm, str]] = None,
                  aggregation: Optional[Union[SNorm, str]] = None,
                  defuzzifier: Optional[Union[Defuzzifier, str]] = None,
                  activation: Optional[Union[Activation, str]] = None) -> None:
        from . import lib
        factory = lib.factory_manager
        if isinstance(conjunction, str):
            conjunction = factory.tnorm.construct(conjunction)
        if isinstance(disjunction, str):
            disjunction = factory.snorm.construct(disjunction)
        if isinstance(implication, str):
            implication = factory.tnorm.construct(implication)
        if isinstance(aggregation, str):
            aggregation = factory.snorm.construct(aggregation)
        if isinstance(defuzzifier, str):
            defuzzifier = factory.defuzzifier.construct(defuzzifier)
        if isinstance(activation, str):
            activation = factory.activation.construct(activation)

        for block in self.rule_blocks:
            block.conjunction = conjunction
            block.disjunction = disjunction
            block.implication = implication
            block.activation = activation

        for variable in self.output_variables:
            variable.aggregation = aggregation
            variable.defuzzifier = defuzzifier

    @property
    def variables(self) -> List[Variable]:
        return [*self.input_variables, *self.output_variables]

    def variable(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise ValueError(f"variable '{name}' not found in {v.name for v in self.variables}")

    def input_variable(self, name: str) -> InputVariable:
        for variable in self.input_variables:
            if variable.name == name:
                return variable
        raise ValueError(f"input variable '{name}' not found in "
                         f"{v.name for v in self.input_variables}")

    def output_variable(self, name: str) -> OutputVariable:
        for variable in self.output_variables:
            if variable.name == name:
                return variable
        raise ValueError(f"output variable '{name}' not found in "
                         f"{v.name for v in self.output_variables}")

    def rule_block(self, name: str) -> RuleBlock:
        for block in self.rule_blocks:
            if block.name == name:
                return block
        raise ValueError(f"rule block '{name}' not found in {r.name for r in self.rule_blocks}")

    def restart(self) -> None:
        for input_variable in self.input_variables:
            input_variable.value = nan

        for output_variable in self.output_variables:
            output_variable.clear()

    def process(self) -> None:
        from . import lib

        # Clear output values
        for variable in self.output_variables:
            variable.fuzzy.clear()

        if lib.debugging:
            pass

        # Activate rule blocks
        for block in self.rule_blocks:
            if block.enabled:
                block.activate()

        if lib.debugging:
            pass

        # Defuzzify output variables
        for variable in self.output_variables:
            variable.defuzzify()

        if lib.debugging:
            pass

    def is_ready(self) -> Tuple[bool, str]:
        raise NotImplementedError()

    def infer_type(self) -> Tuple['Engine.Type', str]:
        raise NotImplementedError()
