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
from __future__ import annotations

__all__ = ["Engine"]

import enum
from collections.abc import Iterable

import numpy as np

from .activation import Activation
from .defuzzifier import Defuzzifier
from .library import nan, representation, settings
from .norm import SNorm, TNorm
from .rule import RuleBlock
from .types import ScalarArray
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

        # /**Unknown: When output variables have no defuzzifiers*/
        Unknown = enum.auto()
        # /**Mamdani: When the output variables have IntegralDefuzzifier%s*/
        Mamdani = enum.auto()
        # /**Larsen: When Mamdani and AlgebraicProduct is the implication operator
        # of the rule blocks */
        Larsen = enum.auto()
        # /**TakagiSugeno: When output variables have WeightedDefuzzifier%s of type
        # TakagiSugeno and the output variables have Constant, Linear, or
        # Function terms*/
        TakagiSugeno = enum.auto()
        # /**Tsukamoto: When output variables have WeightedDefuzzifier%s of type
        # Tsukamoto and the output variables only have monotonic terms
        # (Concave, Ramp, Sigmoid, SShape, and ZShape)*/
        Tsukamoto = enum.auto()
        # /**InverseTsukamoto: When output variables have WeightedDefuzzifier%s of type
        # TakagiSugeno and the output variables do not only have Constant,
        # Linear or Function terms*/
        InverseTsukamoto = enum.auto()
        # /**Hybrid: When output variables have different defuzzifiers*/
        Hybrid = enum.auto()

    def __init__(
        self,
        name: str = "",
        description: str = "",
        input_variables: Iterable[InputVariable] | None = None,
        output_variables: Iterable[OutputVariable] | None = None,
        rule_blocks: Iterable[RuleBlock] | None = None,
        load: bool = True,
    ) -> None:
        """Creates an Engine with the parameters given.

        @param name is the name of the engine
        @param description is the description of the engine
        @param input_variables a list of input variables
        @param output_variables a list of output variables
        @param rule_blocks a list of rule blocks
        @param load whether to automatically update references to this engine and load the rules in the rule blocks
        """
        self.name = name
        self.description = description
        self.input_variables = list(input_variables) if input_variables else []
        self.output_variables = list(output_variables) if output_variables else []
        self.rule_blocks = list(rule_blocks) if rule_blocks else []
        if load:
            for variable in self.variables:
                for term in variable.terms:
                    term.update_reference(self)
            for rb in self.rule_blocks:
                rb.load_rules(self)

    # TODO: implement properly. Currently, RecursionError when using self[item]
    # def __getattr__(self, item: str) -> InputVariable | OutputVariable | RuleBlock:
    #     """@return the component with the given name in input variables, output variables, or rule blocks,
    #     so it can be used like `engine.power.value`.
    #     """
    #     try:
    #         return self[item]
    #     except ValueError:
    #         raise AttributeError(
    #             f"'{self.__class__.__name__}' object has no attribute '{item}'"
    #         ) from None

    def __getitem__(self, item: str) -> InputVariable | OutputVariable | RuleBlock:
        """@return the component with the given name in input variables, output variables, or rule blocks,
        so it can be used like `engine["power"].value`.
        """
        components = [self.input_variable, self.output_variable, self.rule_block]
        for component in components:
            try:
                return component(item)  # type: ignore
            except:  # noqa
                pass
        raise ValueError(
            f"engine '{self.name}' does not have a component named '{item}'"
        )

    def __str__(self) -> str:
        """@return engine in the FuzzyLite Language."""
        return representation.fll.engine(self)

    def __repr__(self) -> str:
        """@return Python code to construct the engine."""
        fields = vars(self).copy()
        if not self.description:
            fields.pop("description")
        return representation.as_constructor(self, fields)

    def configure(
        self,
        conjunction: TNorm | str | None = None,
        disjunction: SNorm | str | None = None,
        implication: TNorm | str | None = None,
        aggregation: SNorm | str | None = None,
        defuzzifier: Defuzzifier | str | None = None,
        activation: Activation | str | None = None,
    ) -> None:
        """Configures the engine with the given operators.

        @param conjunction is a TNorm registered in the TNormFactory
        @param disjunction is an SNorm registered in the SNormFactory
        @param implication is an TNorm registered in the TNormFactory
        @param aggregation is an SNorm registered in the SNormFactory
        @param defuzzifier is a defuzzifier registered in the DefuzzifierFactory
        @param activation is an activation method registered in the ActivationFactory
        """
        factory = settings.factory_manager
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
    def variables(self) -> list[InputVariable | OutputVariable]:
        """Returns a list that contains the input variables followed by the
        output variables in the order of insertion.

        @return a list that contains the input variables followed by the
        output variables in the order of insertion
        """
        return self.input_variables + self.output_variables

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

    def input_variable(self, name_or_index: str | int, /) -> InputVariable:
        """Gets the input variable of the given name after iterating the input
        variables. The cost of this method is O(n), where n is the number of
        input variables in the engine. For performance, please get the
        variables by index.
        @param name is the name of the input variable
        @return input variable of the given name
        @throws ValueError if there is no variable with the given name.
        """
        if isinstance(name_or_index, int):
            return self.input_variables[name_or_index]
        for variable in self.input_variables:
            if variable.name == name_or_index:
                return variable
        raise ValueError(
            f"input variable '{name_or_index}' not found in "
            f"{[v.name for v in self.input_variables]}"
        )

    def output_variable(self, name_or_index: str | int, /) -> OutputVariable:
        """Gets the output variable of the given name after iterating the output
        variables. The cost of this method is O(n), where n is the number of
        output variables in the engine. For performance, please get the
        variables by index.
        @param name is the name of the output variable
        @return output variable of the given name
        @throws ValueError if there is no variable with the given name.
        """
        if isinstance(name_or_index, int):
            return self.output_variables[name_or_index]
        for variable in self.output_variables:
            if variable.name == name_or_index:
                return variable
        raise ValueError(
            f"output variable '{name_or_index}' not found in "
            f"{[v.name for v in self.output_variables]}"
        )

    def rule_block(self, name_or_index: str | int, /) -> RuleBlock:
        """Gets the rule block of the given name after iterating the rule blocks.
        The cost of this method is O(n), where n is the number of
        rule blocks in the engine. For performance, please get the rule blocks
        by index.
        @param name is the name of the rule block
        @return rule block of the given name
        @throws ValueError if there is no block with the given name.
        """
        if isinstance(name_or_index, int):
            return self.rule_blocks[name_or_index]
        for block in self.rule_blocks:
            if block.name == name_or_index:
                return block
        raise ValueError(
            f"rule block '{name_or_index}' not found in {[r.name for r in self.rule_blocks]}"
        )

    @property
    def input_values(self) -> ScalarArray:
        """Returns a matrix containing the input values of the engine,
        where columns represent variables and rows represent input values
        @return the input values of the engine.
        """
        return np.atleast_2d(  # type:ignore
            np.array([v.value for v in self.input_variables])
        ).T

    @input_values.setter
    def input_values(self, values: ScalarArray) -> None:
        """Sets the input values of the engine
        @param values the input values of the engine.
        """
        if not self.input_variables:
            raise RuntimeError(
                "can't set input values to an engine without input variables"
            )
        if values.ndim == 0:
            # all variables set their value to the same
            values = np.full((1, len(self.input_variables)), fill_value=values.item())
        elif values.ndim == 1:
            values = np.atleast_2d(values)
            if len(self.input_variables) == 1:
                values = values.T
        elif values.ndim == 2:
            pass
        else:
            raise ValueError(
                "expected a 0d-array (single value), 1d-array (vector), or 2d-array (matrix), "
                f"but got a {values.ndim}d-array: {values}"
            )
        if values.shape[1] != len(self.input_variables):
            raise ValueError(
                f"expected a value with {len(self.input_variables)} columns (one for each input variable), "
                f"but got {values.shape[1]} columns: {values}"
            )
        for i, v in enumerate(self.input_variables):
            v.value = values[:, i]

    @property
    def output_values(self) -> ScalarArray:
        """Returns a matrix containing the output values of the engine,
        where columns represent variables and rows represent output values
        @return the output values of the engine.
        """
        return np.atleast_2d(  # type:ignore
            np.array([v.value for v in self.output_variables])
        ).T

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
        # Clear output values
        for variable in self.output_variables:
            variable.fuzzy.clear()

        if settings.debugging:
            pass

        # Activate rule blocks
        for block in self.rule_blocks:
            if block.enabled:
                block.activate()

        if settings.debugging:
            pass

        # Defuzzify output variables
        for variable in self.output_variables:
            variable.defuzzify()

        if settings.debugging:
            pass

    def is_ready(self) -> tuple[bool, str]:
        """Indicates whether the engine has been configured correctly and is
        ready for operation. In more advanced engines, the result of this
        method should be taken as a suggestion and not as a prerequisite to
        operate the engine.

        @return a Tuple[bool,str] that indicates whether the engine is ready
        to operate and any related messages if it is not.
        """
        # TODO: Implement
        raise NotImplementedError()

    def infer_type(self) -> tuple[Engine.Type, str]:
        """Infers the type of the engine based on its current configuration.

        @return a Tuple[Engine.Type, str] indicating the inferred type of the
        engine based on its current configuration, and a string explaining
        the reasons for the inferred type
        """
        # TODO: Implement
        raise NotImplementedError()

    # def copy(self) -> Engine:
    #     # TODO: Revisit deep copies and deal with engines in Function and Linear
    #     """Creates a copy of the engine, including all variables, rule blocks,
    #     and rules. The copy is a deep copy, meaning that all objects are
    #     duplicated such that the copy can be modified without affecting the
    #     original.
    #
    #     @return a deep copy of the engine
    #     """
    #     return copy.deepcopy(self)
