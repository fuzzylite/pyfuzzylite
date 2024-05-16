"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = ["Engine"]

import enum
from collections.abc import Iterable

import numpy as np

from .activation import Activation
from .defuzzifier import Defuzzifier, IntegralDefuzzifier, WeightedDefuzzifier
from .library import nan, representation, settings
from .norm import AlgebraicProduct, SNorm, TNorm
from .rule import Rule, RuleBlock
from .types import ScalarArray
from .variable import InputVariable, OutputVariable, Variable


class Engine:
    """Core class of the library that groups the necessary components of a fuzzy logic controller.

    info: related
        - [fuzzylite.variable.InputVariable][]
        - [fuzzylite.variable.OutputVariable][]
        - [fuzzylite.rule.RuleBlock][]
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
        """Constructor.

        Args:
            name: name of the engine
            description: description of the engine
            input_variables: list of input variables
            output_variables: list of output variables
            rule_blocks: list of rule blocks
            load: whether to automatically update references to this engine and load the rules in the rule blocks.
        """
        self.name = name
        self.description = description
        self.input_variables = list(input_variables or [])
        self.output_variables = list(output_variables or [])
        self.rule_blocks = list(rule_blocks or [])
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
        """Allow operation of engines as `engine["power"].value`.

        Args:
            item: name of the component to find in input variables, output variables, or rule blocks

        Returns:
            first component found with the name
        """
        components = [self.input_variable, self.output_variable, self.rule_block]
        for component in components:
            try:
                return component(item)  # type: ignore
            except:  # noqa
                pass
        raise ValueError(f"engine '{self.name}' does not have a component named '{item}'")

    def __str__(self) -> str:
        """Return the code to construct the engine in the FuzzyLite Language.

        Returns:
            code to construct the engine in the FuzzyLite Language.
        """
        return representation.fll.engine(self)

    def __repr__(self) -> str:
        """Return the code to construct the engine in Python.

        Returns:
            code to construct the engine in Python.
        """
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
        """Configure the engine with the given operators.

        Args:
            conjunction: object or name of TNorm registered in the TNormFactory
            disjunction: object or name of SNorm registered in the SNormFactory
            implication: object or name of TNorm registered in the TNormFactory
            aggregation: object or name of SNorm registered in the SNormFactory
            defuzzifier: object or name of defuzzifier registered in the DefuzzifierFactory
            activation: object or name of activation method registered in the ActivationFactory
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
        """Return the list of input and output variables.

        Returns:
            list of input and output variables.
        """
        return self.input_variables + self.output_variables

    def variable(self, name: str) -> Variable:
        """Find the variable by the name, iterating first over the input variables and then over the output variables.

        The cost of this method is $O(n)$, where $n$ is the number of variables in the engine.
        For better performance, get the variables by index.

        Args:
            name: name of the input or output variable

        Returns:
             variable of the given name

        Raises:
            ValueError: when there is no variable by the given name.
        """
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise ValueError(f"variable '{name}' not found in {[v.name for v in self.variables]}")

    def input_variable(self, name_or_index: str | int, /) -> InputVariable:
        """Find the input variable by the name or index.

        The best performance is $O(1)$ when using indices,
        and the worst performance is $O(n)$ when using names, where $n$ is the number of input variables.

        Args:
            name_or_index: name or index of the input variable

        Returns:
            input variable by the name or index

        Raises:
             ValueError: when there is no variable by the given name.
             IndexError: when the index is out of range
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
        """Find the output variable of the given name or at the given index.

        The best performance is $O(1)$ when using indices,
        and the worst performance is $O(n)$ when using names, where $n$ is the number of output variables.

        Args:
            name_or_index: name or index of the output variable

        Returns:
            output variable by the given name or at the given index

        Raises:
             ValueError: when there is no variable with the given name.
             IndexError: when the index is out of range
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
        """Find the rule block of the given name or at the given index.

        The best performance is $O(1)$ when using indices,
        and the worst performance is $O(n)$ when using names, where $n$ is the number of rule blocks.

        Args:
            name_or_index: name or index of the rule block

        Returns:
            rule block by the given name or at the given index

        Raises:
             ValueError: when there is no variable with the given name.
             IndexError: when the index is out of range
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
        """Get/Set 2D array where columns represent input variables and rows their input values.

        # Getter

        Returns:
            2D array of input values (rows) for each input variable (columns)

        # Setter

        Args:
            values (ScalarArray): input values of the engine.

        Tip:
            | when `values` is a: | the result: |
            |---------------------|-------------|
            | single scalar value  | sets the values of all input variables |
            | 1D array on an engine with a single variable | sets the values for the input variable |
            | 1D array on an engine with multiple variables | sets each value to each input variable |
            | 2D array | sets each column of values to each input variable |

        Raises:
            RuntimeError: when there are no input variables
            ValueError: when the dimensionality of values is greater than 2
            ValueError: when the number of columns in the values is different from the number of input variables
        """
        values = tuple(input_variable.value for input_variable in self.input_variables)
        result = np.column_stack(values) if values else np.array(values)
        return result

    @input_values.setter
    def input_values(self, values: ScalarArray) -> None:
        """Sets the input values of the engine.

        Args:
            values: input values of the engine.

        Tip:
            | when `values` is a: | the result: |
            |---------------------|-------------|
            | single scalar value  | sets the values of all input variables |
            | 1D array on an engine with a single variable | sets the values for the input variable |
            | 1D array on an engine with multiple variables | sets each value to each input variable |
            | 2D array | sets each column of values to each input variable |

        Raises:
            RuntimeError: when there are no input variables
            ValueError: when the dimensionality of values is greater than 2
            ValueError: when the number of columns in the values is different from the number of input variables
        """
        if not self.input_variables:
            raise RuntimeError("can't set input values to an engine without input variables")
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
                f"expected an array with {len(self.input_variables)} columns (one for each input variable), "
                f"but got {values.shape[1]} columns: {values}"
            )
        for i, v in enumerate(self.input_variables):
            v.value = values[:, i]

    @property
    def output_values(self) -> ScalarArray:
        """Return a 2D array of output values (rows) for each output variable (columns).

        Returns:
            2D array of output values (rows) for each output variable (columns).
        """
        # TODO: Maybe a property setter like input_values.
        values = tuple(output_variable.value for output_variable in self.output_variables)
        result = np.column_stack(values) if values else np.array(values)
        return result

    @property
    def values(self) -> ScalarArray:
        """Return a 2D array of current input and output values.

        Returns:
            2D array of current input and output values.
        """
        return np.hstack((self.input_values, self.output_values))

    def restart(self) -> None:
        """Restart the engine as follows.

        1. setting the values of the input variables to nan,
        2. reloading the rules of the rule blocks, and
        3. clearing the output variables

        info: related
            - [fuzzylite.variable.Variable.value][]
            - [fuzzylite.variable.OutputVariable.clear][]
        """
        for input_variable in self.input_variables:
            input_variable.value = nan

        for rule_block in self.rule_blocks:
            rule_block.reload_rules(self)

        for output_variable in self.output_variables:
            output_variable.clear()

    def process(self) -> None:
        """Process the engine in its current state as follows.

        1. clear the aggregated fuzzy output variables,
        2. activate the rule blocks, and
        3. defuzzify the output variables

        info: related
            - [fuzzylite.term.Aggregated.clear][]
            - [fuzzylite.rule.RuleBlock.activate][]
            - [fuzzylite.variable.OutputVariable.defuzzify][]
        """
        for variable in self.output_variables:
            variable.fuzzy.clear()

        for block in self.rule_blocks:
            if block.enabled:
                block.activate()

        for variable in self.output_variables:
            variable.defuzzify()

    def is_ready(self, errors: list[str] | None = None) -> bool:
        """Determine whether the engine is configured correctly and ready for operation.

        Note:
            In advanced engines, the result of this method should be taken as a suggestion and not as a prerequisite to
            operate the engine.

        Args:
            errors: optional output list that stores the errors found if the engine is not ready

        Returns:
            whether the engine is ready.
        """
        if errors is None:
            errors = []

        # Input Variables
        if not self.input_variables:
            errors.append(f"Engine '{self.name}' does not have any input variables")

        # ignore because sometimes inputs can be empty: takagi-sugeno/matlab/slcpp1.fis
        # for variable in self.input_variables:
        #     if not variable.terms:
        #         missing.append(
        #             f"Variable '{variable.name}' does not have any input terms"
        #         )

        # Output Variables
        if not self.output_variables:
            errors.append(f"Engine '{self.name}' does not have any output variables")
        for variable in self.output_variables:
            if not variable.terms:
                errors.append(f"Output variable '{variable.name}' does not have any terms")
            if not variable.defuzzifier:
                errors.append(f"Output variable '{variable.name}' does not have any defuzzifier")
            if not variable.aggregation and isinstance(variable.defuzzifier, IntegralDefuzzifier):
                errors.append(
                    f"Output variable '{variable.name}' does not have any aggregation operator"
                )

        # Rule Blocks
        if not self.rule_blocks:
            errors.append(f"Engine '{self.name}' does not have any rule blocks")
        for index, rule_block in enumerate(self.rule_blocks):
            name_or_index = f"'{rule_block.name}'" if rule_block.name else f"[{index}]"
            if not rule_block.rules:
                errors.append(f"Rule block {name_or_index} does not have any rules")

            # Operators needed for rules
            conjunction_needed, disjunction_needed, implication_needed = (0, 0, 0)
            for rule in rule_block.rules:
                conjunction_needed += f" {Rule.AND} " in rule.antecedent.text
                disjunction_needed += f" {Rule.OR} " in rule.antecedent.text
                if rule.is_loaded():
                    mamdani_consequents = 0
                    for consequent in rule.consequent.conclusions:
                        mamdani_consequents += isinstance(
                            consequent.variable, OutputVariable
                        ) and isinstance(consequent.variable.defuzzifier, IntegralDefuzzifier)
                    implication_needed += mamdani_consequents > 0

            if conjunction_needed and not rule_block.conjunction:
                errors.append(
                    f"Rule block {name_or_index} does not have any conjunction operator "
                    f"and is needed by {conjunction_needed} rule{'s'[:conjunction_needed ^ 1]}"
                )

                if disjunction_needed and not rule_block.disjunction:
                    errors.append(
                        f"Rule block {name_or_index} does not have any disjunction operator "
                        f"and is needed by {disjunction_needed} rule{'s'[:disjunction_needed ^ 1]}"
                    )

            if implication_needed and not rule_block.implication:
                errors.append(
                    f"Rule block {name_or_index} does not have any implication operator "
                    f"and is needed by {implication_needed} rule{'s'[:implication_needed ^ 1]}"
                )

        return not errors

    def infer_type(self, reasons: list[str] | None = None) -> Engine.Type:
        """Infer the type of the engine based on its configuration.

        Args:
            reasons: optional output list explaining the reasons for the inferred type

        Returns:
             type of engine inferred from its configuration.
        """
        if reasons is None:
            reasons = []

        # Unknown
        if not self.output_variables:
            reasons.append(f"Engine '{self.name}' does not have any output variables")
            return Engine.Type.Unknown

        # Mamdani
        mamdani = all(
            isinstance(variable.defuzzifier, IntegralDefuzzifier)
            for variable in self.output_variables
        )

        # Larsen
        larsen = (
            mamdani
            and self.rule_blocks
            and all(
                isinstance(rule_block.implication, AlgebraicProduct)
                for rule_block in self.rule_blocks
            )
        )
        if larsen:
            reasons.append("Output variables have integral defuzzifiers")
            reasons.append("Implication in rule blocks is the AlgebraicProduct")
            return Engine.Type.Larsen

        if mamdani:
            reasons.append("Output variables have integral defuzzifiers")
            return Engine.Type.Mamdani

        # Takagi-Sugeno
        takagi_sugeno = all(
            isinstance(variable.defuzzifier, WeightedDefuzzifier)
            and (variable.defuzzifier.infer_type(variable) == WeightedDefuzzifier.Type.TakagiSugeno)
            for variable in self.output_variables
        )
        if takagi_sugeno:
            reasons.append("Output variables have weighted defuzzifiers")
            reasons.append("Output variables only have Constant, Linear, or Function terms")
            return Engine.Type.TakagiSugeno

        # Tsukamoto
        tsukamoto = all(
            isinstance(variable.defuzzifier, WeightedDefuzzifier)
            and (variable.defuzzifier.infer_type(variable) == WeightedDefuzzifier.Type.Tsukamoto)
            for variable in self.output_variables
        )
        if tsukamoto:
            reasons.append("Output variables have weighted defuzzifiers")
            reasons.append("Output variables only have monotonic terms")
            return Engine.Type.Tsukamoto

        # Inverse Tsukamoto
        inverse_tsukamoto = all(
            isinstance(variable.defuzzifier, WeightedDefuzzifier)
            and (variable.defuzzifier.infer_type(variable) == WeightedDefuzzifier.Type.Automatic)
            for variable in self.output_variables
        )
        if inverse_tsukamoto:
            reasons.append("Output variables have weighted defuzzifiers")
            reasons.append("Output variables have non-monotonic terms")
            reasons.append(
                "Output variables have terms different from Constant, Linear, or Function terms"
            )
            return Engine.Type.InverseTsukamoto

        # Hybrids
        hybrid = all(variable.defuzzifier for variable in self.output_variables)
        if hybrid:
            reasons.append("Output variables have different types of defuzzifiers")
            return Engine.Type.Hybrid

        # Unknown
        reasons.append("One or more output variables do not have a defuzzifier")
        return Engine.Type.Unknown

    def copy(self) -> Engine:
        """Create a deep copy of the engine.

        Returns:
            deep copy of the engine
        """
        import copy

        engine = copy.deepcopy(self)
        return engine
