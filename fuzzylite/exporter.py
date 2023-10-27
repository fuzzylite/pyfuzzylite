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

__all__ = ["Exporter", "FllExporter", "PythonExporter", "FldExporter"]

import enum
import io
import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any

import numpy as np

from .library import settings, to_float
from .operation import Op
from .types import Scalar

if typing.TYPE_CHECKING:
    from .activation import Activation
    from .defuzzifier import Defuzzifier
    from .engine import Engine
    from .norm import Norm
    from .rule import Rule, RuleBlock
    from .term import Term
    from .variable import InputVariable, OutputVariable, Variable


class Exporter(ABC):
    """The Exporter class is the abstract class for exporters to translate an
    Engine into different formats.
    @author Juan Rada-Vilela, Ph.D.
    @see Importer
    @since 4.0.
    """

    def __str__(self) -> str:
        """@return class name of the exporter."""
        return Op.class_name(self)

    def __repr__(self) -> str:
        """@return Python code to construct the exporter."""
        from .library import representation

        return representation.as_constructor(self)

    @abstractmethod
    def to_string(self, instance: object) -> str:
        """Returns a string representation of the FuzzyLite component
        @param instance is the FuzzyLite component
        @return a string representation of the FuzzyLite component.
        """
        raise NotImplementedError()

    def to_file(self, path: str | Path, instance: object) -> None:
        """Stores the string representation of the FuzzyLite component into the specified file
        @param path is the full path of the file to export the component to
        @param instance is the component to export.

        TODO: change instance: object to engine: Engine
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open(mode="w", encoding="UTF8") as fll:
            fll.write(self.to_string(instance))


class FllExporter(Exporter):
    """The FllExporter class is an Exporter that translates an Engine and its
    components to the FuzzyLite Language (FLL), see
    [https://fuzzylite.com/fll-fld](https://fuzzylite.com/fll-fld) for
    more information.

    @author Juan Rada-Vilela, Ph.D.
    @see FllImporter
    @see Exporter
    @since 4.0
    """

    def __init__(self, indent: str = "  ", separator: str = "\n") -> None:
        """@param indent is the indent string of the FuzzyLite Language
        @param separator of the FuzzyLite Language.
        """
        self.indent = indent
        self.separator = separator

    def format(self, key: str | None, value: Any) -> str:
        """Formats the arguments to produce a valid FLL from them."""
        result = []
        if key:
            result.append(f"{key}:")
        if value == "":
            pass
        elif value is None:
            result.append("none")
        elif isinstance(value, bool):
            result.append(str(value).lower())
        elif isinstance(value, float):
            result.append(Op.str(value))
        elif isinstance(value, (tuple, list, set)):
            for v_i in value:
                f_value = self.format(key=None, value=v_i)
                if f_value:
                    result.append(f_value)
        else:
            result.append(str(value))
        return " ".join(result)

    def to_string(self, obj: Any, /) -> str:
        """Returns a string representation of the FuzzyLite component
        @param obj is the FuzzyLite component
        @return a string representation of the FuzzyLite component.
        """
        from . import (
            Activation,
            Defuzzifier,
            Engine,
            InputVariable,
            Norm,
            OutputVariable,
            Rule,
            RuleBlock,
            Term,
            Variable,
        )

        if isinstance(obj, Engine):
            return self.engine(obj)

        if isinstance(obj, InputVariable):
            return self.input_variable(obj)
        if isinstance(obj, OutputVariable):
            return self.output_variable(obj)
        if isinstance(obj, Variable):
            return self.variable(obj)

        if isinstance(obj, Term):
            return self.term(obj)

        if isinstance(obj, Activation):
            return self.activation(obj)
        if isinstance(obj, Defuzzifier):
            return self.defuzzifier(obj)
        if isinstance(obj, Norm):
            return self.norm(obj)

        if isinstance(obj, RuleBlock):
            return self.rule_block(obj)
        if isinstance(obj, Rule):
            return self.rule(obj)

        raise TypeError(f"expected a fuzzylite object, but got {type(obj)}")

    def engine(self, engine: Engine, /) -> str:
        """Returns a string representation of the engine
        @param engine is the engine
        @return a string representation of the engine.
        """
        result = [self.format(Op.class_name(engine), engine.name)]
        if engine.description:
            result += [self.indent + self.format("description", engine.description)]
        result += [self.input_variable(iv) for iv in engine.input_variables]
        result += [self.output_variable(ov) for ov in engine.output_variables]
        result += [self.rule_block(rb) for rb in engine.rule_blocks]
        result += [""]
        return self.separator.join(result)

    def variable(self, variable: Variable, /, terms: bool = True) -> str:
        """Returns a string representation of the variable
        @param variable is the variable
        @param terms whether to export the terms
        @return a string representation of the variable.
        """
        result = [self.format(Op.class_name(variable), variable.name)]
        if variable.description:
            result += [self.indent + self.format("description", variable.description)]
        result += [
            self.indent + self.format("enabled", variable.enabled),
            self.indent + self.format("range", (variable.minimum, variable.maximum)),
            self.indent + self.format("lock-range", variable.lock_range),
        ]
        if terms and variable.terms:
            result += [(self.indent + self.term(term)) for term in variable.terms]
        return self.separator.join(result)

    def input_variable(self, variable: InputVariable, /) -> str:
        """Returns a string representation of the input variable
        @param variable is the input variable
        @return a string representation of the input variable.
        """
        return self.variable(variable)

    def output_variable(self, variable: OutputVariable, /) -> str:
        """Returns a string representation of the output variable
        @param variable is the variable
        @return a string representation of the output variable.
        """
        result = [self.variable(variable, terms=False)]
        result += [
            self.indent + self.format("aggregation", self.norm(variable.aggregation)),
            self.indent
            + self.format("defuzzifier", self.defuzzifier(variable.defuzzifier)),
            self.indent + self.format("default", variable.default_value),
            self.indent + self.format("lock-previous", variable.lock_previous),
        ]
        if variable.terms:
            result += [(self.indent + self.term(term)) for term in variable.terms]
        return self.separator.join(result)

    def rule_block(self, rule_block: RuleBlock, /) -> str:
        """Returns a string representation of the rule block
        @param rule_block is the rule block
        @return a string representation of the rule block.
        """
        result = [self.format(Op.class_name(rule_block), rule_block.name)]
        if rule_block.description:
            result += [self.indent + self.format("description", rule_block.description)]
        result += [
            self.indent + self.format("enabled", rule_block.enabled),
            self.indent + self.format("conjunction", self.norm(rule_block.conjunction)),
            self.indent + self.format("disjunction", self.norm(rule_block.disjunction)),
            self.indent + self.format("implication", self.norm(rule_block.implication)),
            self.indent
            + self.format("activation", self.activation(rule_block.activation)),
        ]
        if rule_block.rules:
            result += [self.indent + self.rule(rule) for rule in rule_block.rules]
        return self.separator.join(result)

    def term(self, term: Term, /) -> str:
        """Returns a string representation of the linguistic term
        @param term is the linguistic term
        @return a string representation of the linguistic term.
        """
        return self.format(
            "term",
            (Op.as_identifier(term.name), Op.class_name(term), term.parameters()),
        )

    def norm(self, norm: Norm | None, /) -> str:
        """Returns a string representation of the norm
        @param norm is the norm
        @return a string representation of the norm.
        """
        return Op.class_name(norm) if norm else "none"

    def activation(self, activation: Activation | None, /) -> str:
        """Returns a string representation of the activation method
        @param activation is the activation method
        @return a string representation of the activation method.
        """
        if not activation:
            return "none"
        return self.format(
            key=None, value=(Op.class_name(activation), activation.parameters())
        )

    def defuzzifier(self, defuzzifier: Defuzzifier | None, /) -> str:
        """Returns a string representation of the defuzzifier
        @param defuzzifier is the defuzzifier
        @return a string representation of the defuzzifier.
        """
        if not defuzzifier:
            return "none"
        return self.format(
            key=None, value=(Op.class_name(defuzzifier), defuzzifier.parameters())
        )

    def rule(self, rule: Rule, /) -> str:
        """Returns a string representation of the rule
        @param rule is the rule
        @return a string representation of the rule.
        """
        return self.format("rule", rule.text)


class PythonExporter(Exporter):
    """The PythonExporter class is an Exporter that translates an Engine and its
    components to the `Python` programming language using the `pyfuzzylite`
    library.

    @author Juan Rada-Vilela, Ph.D.
    @see CppExporter
    @see JavaExporter
    @see Exporter
    @since 7.0
    """

    def __init__(self, formatted: bool = True, encapsulated: bool = False) -> None:
        """@param formatted: whether to attempt to format the code using `black` if it is installed."""
        self.formatted = formatted
        self.encapsulated = encapsulated

    def format(self, code: str) -> str:
        """Try to format the code using the `black` formatter if it is installed, otherwise returns the code as is.
        @param code: code to format.
        """
        try:
            import black

            return black.format_str(code, mode=black.Mode())
        except ModuleNotFoundError:
            settings.logger.error(
                "expected `black` module to be installed, but could not be found"
            )
        except ValueError:  # black.parsing.InvalidInput
            raise
        return code

    def encapsulate(self, instance: Any) -> str:
        """Encapsulate the instance in a new class if it is an engine, or in a create method otherwise."""
        from .engine import Engine
        from .library import representation

        code = f"{representation.import_statement()}\n\n"
        if isinstance(instance, Engine):
            code += f"""\
class {Op.pascal_case(instance.name)}:

    def __init__(self) -> None:
        self.engine = {repr(instance)}
"""
        else:
            code += f"""\
def create() -> {Op.class_name(instance, qualname=True)}:
    return {repr(instance)}
"""
        return code

    def to_string(self, instance: Any, /) -> str:
        """@return Python code to construct the given instance."""
        code = self.encapsulate(instance) if self.encapsulated else repr(instance)

        if self.formatted:
            code = self.format(code)
        return code

    def engine(self, engine: Engine, /) -> str:
        """@return Python code to construct the engine."""
        return self.to_string(engine)

    def input_variable(self, input_variable: InputVariable, /) -> str:
        """@return Python code to construct the variable."""
        return self.to_string(input_variable)

    def output_variable(self, output_variable: OutputVariable, /) -> str:
        """@return Python code to construct the variable."""
        return self.to_string(output_variable)

    def rule_block(self, rule_block: RuleBlock, /) -> str:
        """@return Python code to construct the rule block."""
        return self.to_string(rule_block)

    def term(self, term: Term, /) -> str:
        """@return Python code to construct the term."""
        return self.to_string(term)

    def norm(self, norm: Norm | None, /) -> str:
        """@return Python code to construct the norm."""
        return self.to_string(norm) if norm else "None"

    def activation(self, activation: Activation | None, /) -> str:
        """@return Python code to construct the activation method."""
        return self.to_string(activation) if activation else "None"

    def defuzzifier(self, defuzzifier: Defuzzifier | None, /) -> str:
        """@return Python code to construct the defuzzifier."""
        return self.to_string(defuzzifier) if defuzzifier else "None"

    def rule(self, rule: Rule, /) -> str:
        """@return Python code to construct the rule."""
        return self.to_string(rule)


class FldExporter(Exporter):
    """The FldExporter class is an Exporter that evaluates an Engine and exports
    its input values and output values to the FuzzyLite Dataset (FLD) format,
    see [https://fuzzylite.com/fll-fld](https://fuzzylite.com/fll-fld)
    for more information.

    @author Juan Rada-Vilela, Ph.D.
    @see FllExporter
    @see Exporter
    @since 4.0
    """

    @enum.unique
    class ScopeOfValues(enum.Enum):
        """The ScopeOfValues refers to the scope of the equally-distributed values
        to generate.
        """

        # /**Generates $n$ values for each variable*/
        EachVariable = enum.auto()
        # /**Generates $n$ values for all variables*/
        AllVariables = enum.auto()

    def __init__(
        self,
        separator: str = " ",
        headers: bool = True,
        input_values: bool = True,
        output_values: bool = True,
    ) -> None:
        """Creates a FuzzyLite Dataset exporter
        @param separator is the separator of the dataset columns
        @param headers indicates whether the header of the dataset is to be exported
        @param input_values indicates whether the values of the input variables are to be exported
        @param output_values indicates whether the values of the output variables are to be exported.
        """
        self.separator = separator
        self.headers = headers
        self.input_values = input_values
        self.output_values = output_values

    def header(self, engine: Engine) -> str:
        """Gets the header of the dataset for the given engine
        @param engine is the engine to be exported
        @return the header of the dataset for the given engine.
        """
        result: list[str] = []
        if self.input_values:
            result += [iv.name for iv in engine.input_variables]
        if self.output_values:
            result += [ov.name for ov in engine.output_variables]
        return self.separator.join(result)

    def to_string(self, instance: object) -> str:
        """Returns a FuzzyLite Dataset from the engine.
        @param instance is the engine to export
        @return a FuzzyLite Dataset from the engine
        @throws ValueError if the instance is not an Engine.
        """
        from .engine import Engine

        if isinstance(instance, Engine):
            return self.to_string_from_scope(instance)
        raise ValueError(f"expected an Engine, but got {type(instance).__name__}")

    def to_string_from_scope(
        self,
        engine: Engine,
        values: int = 1024,
        scope: ScopeOfValues = ScopeOfValues.AllVariables,
        active_variables: set[InputVariable] | None = None,
    ) -> str:
        """Returns a FuzzyLite Dataset from the engine.
        @param engine is the engine to export
        @param values is the number of values to export
        @param scope indicates the scope of the values
        @param active_variables is the set of input variables to set values for
        @return a FuzzyLite Dataset from the engine.

        # TODO: Remove active_variables?
        """
        writer = io.StringIO()
        self.write_from_scope(engine, writer, values, scope, active_variables)
        return writer.getvalue()

    def to_file_from_scope(
        self,
        path: Path,
        engine: Engine,
        values: int = 1024,
        scope: ScopeOfValues = ScopeOfValues.AllVariables,
        active_variables: set[InputVariable] | None = None,
    ) -> None:
        """Saves the engine as a FuzzyLite Dataset into the specified file
        @param path is the full path of the file
        @param engine is the engine to export
        @param values is the number of values to export
        @param scope indicates the scope of the values
        @param active_variables is the set of input variables to set values for.
        """
        with path.open("w") as writer:
            self.write_from_scope(engine, writer, values, scope, active_variables)

    def write_from_scope(
        self,
        engine: Engine,
        writer: IO[str],
        values: int,
        scope: ScopeOfValues,
        active_variables: set[InputVariable] | None = None,
    ) -> None:
        """Writes the engine into the given writer
        @param engine is the engine to export
        @param writer is the output where the engine will be written to
        @param values is the number of values to export
        @param scope indicates the scope of the values
        @param active_variables contains the input variables to generate values for.

        # TODO: improve handling of active_variables?
        """
        if active_variables is None:
            active_variables = set(engine.input_variables)

        if scope == FldExporter.ScopeOfValues.AllVariables:
            if len(engine.input_variables) == 0:
                raise ValueError("expected input variables in engine, but got none")
            resolution = -1 + max(
                1, int(pow(values, (1.0 / len(engine.input_variables))))
            )
        else:
            resolution = values - 1

        sample_values = [0] * len(engine.input_variables)
        min_values = [0] * len(engine.input_variables)
        max_values = [
            resolution if iv in active_variables else 0 for iv in engine.input_variables
        ]

        input_values: list[list[float]] = []
        incremented = True
        while incremented:
            row = []
            for index, variable in enumerate(engine.input_variables):
                if variable in active_variables:
                    dx = variable.drange / max(1.0, resolution)
                    value = variable.minimum + sample_values[index] * dx
                    row.append(value)
                else:
                    row.append(np.take(variable.value, -1).astype(float))
            input_values.append(row)
            incremented = Op.increment(sample_values, min_values, max_values)
        self.write(engine, writer, np.array(input_values))

    def to_string_from_reader(
        self, engine: Engine, reader: IO[str], skip_lines: int = 0
    ) -> str:
        """Returns a FuzzyLite Dataset from the engine.
        @param engine is the engine to export
        @param reader is the reader of a set of lines containing space-separated
        input values
        @param skip_lines is the number of lines to initially skip
        @return a FuzzyLite Dataset from the engine.
        """
        writer = io.StringIO()
        self.write_from_reader(engine, writer, reader, skip_lines)
        return writer.getvalue()

    def to_file_from_reader(
        self, path: Path, engine: Engine, reader: IO[str], skip_lines: int = 0
    ) -> None:
        """Saves the engine as a FuzzyLite Dataset into the specified file
        @param path is the path to the output file
        @param engine is the engine to export
        @param reader is the reader of a set of lines containing space-separated
        input values
        @param skip_lines is the number of lines to initially skip
        @return a FuzzyLite Dataset from the engine.
        """
        with path.open("w") as writer:
            self.write_from_reader(engine, writer, reader, skip_lines)

    def write_from_reader(
        self, engine: Engine, writer: IO[str], reader: IO[str], skip_lines: int = 0
    ) -> None:
        """Writes the engine into the given writer
        @param engine is the engine to export
        @param writer is the output where the engine will be written to
        @param reader is the reader of a set of lines containing space-separated input values
        @param skip_lines is the number of lines to initially skip.
        """
        input_values: list[list[float]] = []
        for i, line in enumerate(reader.readlines()):
            if i < skip_lines:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            input_values.append([to_float(x) for x in line.split()])

        self.write(engine, writer, np.asarray(input_values))

    def write(
        self,
        engine: Engine,
        writer: IO[str],
        input_values: Scalar,
    ) -> None:
        """Writes the engine into the given writer
        @param engine is the engine to export
        @param writer is the output where the engine will be written to
        @param input_values is a matrix of input values.
        """
        input_values = np.atleast_2d(input_values)
        if input_values.shape[1] < len(engine.input_variables):
            raise ValueError(
                f"expected {len(engine.input_variables)} input values (one per input variable), "
                f"but got {input_values.shape[1]} instead"
            )

        engine.restart()

        for index, variable in enumerate(engine.input_variables):
            variable.value = input_values[:, index]

        engine.process()

        values: list[Any] = []
        if self.input_values:
            values.append(engine.input_values)
        if self.output_values:
            values.append(engine.output_values)
        if not values:
            # TODO: Fix this. It's a hack to use hstack without blowing up.
            values = [[]]

        np.savetxt(
            writer,
            np.hstack(values),
            fmt=f"%0.{settings.decimals}f",
            delimiter=self.separator,
            header=self.header(engine) if self.headers else "",
            comments="",
        )
