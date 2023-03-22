"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

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
    """The Engine class is the core class of the library as it groups the
    necessary components of a fuzzy logic controller.

    @author Juan Rada-Vilela, Ph.D.
    @see InputVariable
    @see OutputVariable
    @see RuleBlock
    @since 4.0
    """

    @enum.unique
    class Type(enum.Enum):
        """Determine type of engine."""

        (
            # /**Unknown: When output variables have no defuzzifiers*/
            Unknown,
            # /**Mamdani: When the output variables have IntegralDefuzzifier%s*/
            Mamdani,
            # /**Larsen: When Mamdani and AlgebraicProduct is the implication operator
            # of the rule blocks */
            Larsen,
            # /**TakagiSugeno: When output variables have WeightedDefuzzifier%s of type
            # TakagiSugeno and the output variables have Constant, Linear, or
            # Function terms*/
            TakagiSugeno,
            # /**Tsukamoto: When output variables have WeightedDefuzzifier%s of type
            # Tsukamoto and the output variables only have monotonic terms
            # (Concave, Ramp, Sigmoid, SShape, and ZShape)*/
            Tsukamoto,
            # /**InverseTsukamoto: When output variables have WeightedDefuzzifier%s of type
            # TakagiSugeno and the output variables do not only have Constant,
            # Linear or Function terms*/
            InverseTsukamoto,
            # /**Hybrid: When output variables have different defuzzifiers*/
            Hybrid,
        ) = range(7)

    def __init__(
        self,
        name: str = "",
        description: str = "",
        input_variables: Optional[Iterable[InputVariable]] = None,
        output_variables: Optional[Iterable[OutputVariable]] = None,
        rule_blocks: Optional[Iterable[RuleBlock]] = None,
        load_rules: bool = False,
    ) -> None:
        """Creates an Engine with the parameters given.

        @param name is the name of the engine
        @param description is the description of the engine
        @param input_variables a list of input variables
        @param output_variables a list of output variables
        @param rule_blocks a list of rule blocks
        @param load_rules whether to automatically load the rules in the rule blocks
        """
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
        """Returns the FLL code for the engine
        @return the FLL code for the engine.
        """
        return FllExporter().engine(self)

    def configure(
        self,
        conjunction: Optional[Union[TNorm, str]] = None,
        disjunction: Optional[Union[SNorm, str]] = None,
        implication: Optional[Union[TNorm, str]] = None,
        aggregation: Optional[Union[SNorm, str]] = None,
        defuzzifier: Optional[Union[Defuzzifier, str]] = None,
        activation: Optional[Union[Activation, str]] = None,
    ) -> None:
        """Configures the engine with the given operators.

        @param conjunction is a TNorm registered in the TNormFactory
        @param disjunction is an SNorm registered in the SNormFactory
        @param implication is an TNorm registered in the TNormFactory
        @param aggregation is an SNorm registered in the SNormFactory
        @param defuzzifier is a defuzzifier registered in the DefuzzifierFactory
        @param activation is an activation method registered in the ActivationFactory
        """
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
        """Returns a list that contains the input variables followed by the
        output variables in the order of insertion.

        @return a list that contains the input variables followed by the
        output variables in the order of insertion
        """
        return [*self.input_variables, *self.output_variables]

    def variable(self, name: str) -> Variable:
        """Gets the first variable of the given name after iterating the input
        variables and the output variables. The cost of this method is O(n),
        where n is the number of variables in the engine. For performance,
        please get the variables by index.
        @param name is the name of the input variable
        @return variable of the given name
        @throws ValueError if there is no variable with the given name.
        """
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise ValueError(
            f"variable '{name}' not found in {[v.name for v in self.variables]}"
        )

    def input_variable(self, name: str) -> InputVariable:
        """Gets the input variable of the given name after iterating the input
        variables. The cost of this method is O(n), where n is the number of
        input variables in the engine. For performance, please get the
        variables by index.
        @param name is the name of the input variable
        @return input variable of the given name
        @throws ValueError if there is no variable with the given name.
        """
        for variable in self.input_variables:
            if variable.name == name:
                return variable
        raise ValueError(
            f"input variable '{name}' not found in "
            f"{[v.name for v in self.input_variables]}"
        )

    def output_variable(self, name: str) -> OutputVariable:
        """Gets the output variable of the given name after iterating the output
        variables. The cost of this method is O(n), where n is the number of
        output variables in the engine. For performance, please get the
        variables by index.
        @param name is the name of the output variable
        @return output variable of the given name
        @throws ValueError if there is no variable with the given name.
        """
        for variable in self.output_variables:
            if variable.name == name:
                return variable
        raise ValueError(
            f"output variable '{name}' not found in "
            f"{[v.name for v in self.output_variables]}"
        )

    def rule_block(self, name: str) -> RuleBlock:
        """Gets the rule block of the given name after iterating the rule blocks.
        The cost of this method is O(n), where n is the number of
        rule blocks in the engine. For performance, please get the rule blocks
        by index.
        @param name is the name of the rule block
        @return rule block of the given name
        @throws ValueError if there is no block with the given name.
        """
        for block in self.rule_blocks:
            if block.name == name:
                return block
        raise ValueError(
            f"rule block '{name}' not found in {[r.name for r in self.rule_blocks]}"
        )

    def restart(self) -> None:
        """Restarts the engine by setting the values of the input variables to
        fl::nan and clearing the output variables
        @see Variable::setValue()
        @see OutputVariable::clear().
        """
        for input_variable in self.input_variables:
            input_variable.value = nan

        for output_variable in self.output_variables:
            output_variable.clear()

    def process(self) -> None:
        """Processes the engine in its current state as follows: (a) Clears the
        aggregated fuzzy output variables, (b) Activates the rule blocks, and
        (c) Defuzzifies the output variables
        @see Aggregated::clear()
        @see RuleBlock::activate()
        @see OutputVariable::defuzzify().
        """
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
        """Indicates whether the engine has been configured correctly and is
        ready for operation. In more advanced engines, the result of this
        method should be taken as a suggestion and not as a prerequisite to
        operate the engine.

        @return a Tuple[bool,str] that indicates whether the engine is ready
        to operate and any related messages if it is not.
        """
        # TODO: Implement
        raise NotImplementedError()

    def infer_type(self) -> Tuple["Engine.Type", str]:
        """Infers the type of the engine based on its current configuration.

        @return a Tuple[Engine.Type, str] indicating the inferred type of the
        engine based on its current configuration, and a string explaining
        the reasons for the inferred type
        """
        # TODO: Implement
        raise NotImplementedError()
