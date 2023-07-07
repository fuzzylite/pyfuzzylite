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

    def __init__(self, indent: str = "    ") -> None:
        """Creates a PythonExporter with the given indentation
        @param indent is the indentation.
        """
        self.indent = indent

    def engine(self, engine: Engine, level: int = 0) -> str:
        """Returns a string representation of the engine
        @param engine is the engine to export
        @param level refers to the number of indentation to add
        @return a string representation of the engine.

        TODO: level should be a local variable
        """
        result = [f"""{level * self.indent}import fuzzylite as fl""", ""]
        result += [
            f"{level * self.indent}engine = fl.Engine(",
            f"{(level + 1) * self.indent}name={self.format(engine.name)},",
            f"{(level + 1) * self.indent}description={self.format(engine.description)}",
            ")",
        ]

        input_variables: list[str] = []
        for iv in engine.input_variables:
            input_variables += [
                f"{(level + 1) * self.indent}{self.input_variable(iv, level + 2)}"
            ]
        result += [self.key_values("engine.input_variables", input_variables)]

        output_variables: list[str] = []
        for ov in engine.output_variables:
            output_variables += [
                f"{(level + 1) * self.indent}{self.output_variable(ov, level + 2)}"
            ]
        result += [self.key_values("engine.output_variables", output_variables)]

        rule_blocks: list[str] = []
        for rb in engine.rule_blocks:
            rule_blocks += [
                f"{(level + 1) * self.indent}{self.rule_block(rb, level + 2)}"
            ]
        result += [self.key_values("engine.rule_blocks", rule_blocks)]
        result += [""]
        return "\n".join(result)

    def to_string(self, instance: object) -> str:
        """Returns a string representation of the FuzzyLite component in Python
        @param instance is the FuzzyLite component
        @return a string representation of the FuzzyLite component.
        """
        from .engine import Engine

        if isinstance(instance, Engine):
            return self.engine(instance)

        from .variable import InputVariable, OutputVariable

        if isinstance(instance, InputVariable):
            return self.input_variable(instance)
        if isinstance(instance, OutputVariable):
            return self.output_variable(instance)

        from .term import Term

        if isinstance(instance, Term):
            return self.term(instance)

        from .defuzzifier import Defuzzifier

        if isinstance(instance, Defuzzifier):
            return self.defuzzifier(instance)

        from .rule import Rule, RuleBlock

        if isinstance(instance, RuleBlock):
            return self.rule_block(instance)
        if isinstance(instance, Rule):
            return self.rule(instance)

        from .norm import Norm

        if isinstance(instance, Norm):
            return self.norm(instance)

        from .activation import Activation

        if isinstance(instance, Activation):
            return self.activation(instance)

        raise ValueError(
            f"expected a fuzzylite object, but found '{type(instance).__name__}'"
        )

    def format(self, x: Any) -> str:
        """Formats any value as a string
        @param x is the value
        @return the value formatted.
        """
        if isinstance(x, str):
            return f'"{x}"'
        if isinstance(x, float):
            if Op.isinf(x):
                return ("" if x > 0 else "-") + "fl.inf"
            if Op.isnan(x):
                return "fl.nan"
            return Op.str(x)
        if isinstance(x, bool):
            return str(x)
        return str(x)

    def key_values(self, name: str, values: list[Any], level: int = 0) -> str:
        """Formats a key-value pair at the given level
        @param name is the name of the Python variable
        @param values is the list of values to assign to the variable as a list
        @param level is the number of indentations to add
        @return a Python assignment of the variable to the values as a list.
        """
        result = [f"{level * self.indent}{name} = "]
        if not values:
            result[0] += "[]"
        else:
            result[0] += "["
            result += [",\n".join(values), f"{level * self.indent}]"]
        return "\n".join(result)

    def input_variable(self, iv: InputVariable, level: int = 1) -> str:
        """Returns a string representation of the input variable in the Python
        programming language
        @param iv is the input variable
        @return a string representation of the input variable in the Python
        programming language.

        TODO: Rename iv to input_variable
        """
        result = [
            f"{level * self.indent}name={self.format(iv.name)}",
            f"{level * self.indent}description={self.format(iv.description)}",
            f"{level * self.indent}enabled={self.format(iv.enabled)}",
            f"{level * self.indent}minimum={self.format(iv.minimum)}",
            f"{level * self.indent}maximum={self.format(iv.maximum)}",
            f"{level * self.indent}lock_range={self.format(iv.lock_range)}",
        ]
        if iv.terms:
            if len(iv.terms) == 1:
                terms = f"{level * self.indent}terms=[{self.term(iv.terms[0])}]"
            else:
                terms = (
                    f"{level * self.indent}terms=[\n"
                    + ",\n".join(
                        f"{(level + 1) * self.indent}{self.term(term)}"
                        for term in iv.terms
                    )
                    + f"\n{level * self.indent}]"
                )
            result += [terms]

        input_variable = ["fl.InputVariable("]
        input_variable += [",\n".join(result)]
        input_variable += [f"{max(0, level - 1) * self.indent})"]

        return "\n".join(input_variable)

    def output_variable(self, ov: OutputVariable, level: int = 1) -> str:
        """Returns a string representation of the output variable in the Python
        programming language
        @param ov is the output variable
        @return a string representation of the output variable in the Python
        programming language.

        TODO: Rename ov to output_variable
        """
        result = [
            f"{level * self.indent}name={self.format(ov.name)}",
            f"{level * self.indent}description={self.format(ov.description)}",
            f"{level * self.indent}enabled={self.format(ov.enabled)}",
            f"{level * self.indent}minimum={self.format(ov.minimum)}",
            f"{level * self.indent}maximum={self.format(ov.maximum)}",
            f"{level * self.indent}lock_range={self.format(ov.lock_range)}",
            f"{level * self.indent}aggregation={self.norm(ov.aggregation)}",
            f"{level * self.indent}defuzzifier={self.defuzzifier(ov.defuzzifier)}",
            f"{level * self.indent}lock_previous={self.format(ov.lock_previous)}",
        ]
        if ov.terms:
            if len(ov.terms) == 1:
                terms = f"{level * self.indent}terms=[{self.term(ov.terms[0])}]"
            else:
                terms = (
                    f"{level * self.indent}terms=[\n"
                    + ",\n".join(
                        f"{(level + 1) * self.indent}{self.term(term)}"
                        for term in ov.terms
                    )
                    + f"\n{level * self.indent}]"
                )
            result += [terms]

        output_variable = ["fl.OutputVariable("]
        output_variable += [",\n".join(result)]
        output_variable += [f"{max(0, level - 1) * self.indent})"]

        return "\n".join(output_variable)

    def rule_block(self, rb: RuleBlock, level: int = 1) -> str:
        """Returns a string representation of the rule block in the Python
        programming language
        @param rb is the rule block
        @return a string representation of the rule block in the Python
        programming language.

        TODO: Rename rb to rule_block
        """
        result = [
            f"{level * self.indent}name={self.format(rb.name)}",
            f"{level * self.indent}description={self.format(rb.description)}",
            f"{level * self.indent}enabled={self.format(rb.enabled)}",
            f"{level * self.indent}conjunction={self.norm(rb.conjunction)}",
            f"{level * self.indent}disjunction={self.norm(rb.disjunction)}",
            f"{level * self.indent}implication={self.norm(rb.implication)}",
            f"{level * self.indent}activation={self.activation(rb.activation)}",
        ]
        if rb.rules:
            if len(rb.rules) == 1:
                rules = f"{level * self.indent}rules=[{self.rule(rb.rules[0])}]"
            else:
                rules = (
                    f"{level * self.indent}rules=[\n{(level + 1) * self.indent}"
                    + f",\n{(level + 1) * self.indent}".join(
                        self.rule(rule) for rule in rb.rules
                    )
                    + f"\n{level * self.indent}]"
                )
            result += [rules]

        rule_block = ["fl.RuleBlock("]
        rule_block += [",\n".join(result)]
        rule_block += [f"{max(0, level - 1) * self.indent})"]

        return "\n".join(rule_block)

    def term(self, term: Term) -> str:
        """Returns a string representation of the term in the Python
        programming language
        @param term is the linguistic term
        @return a string representation of the term in the Python
        programming language.
        """
        from .term import Discrete, Function, Linear

        result = ["fl."]
        if isinstance(term, Discrete):
            result += [
                f"{Discrete.create.__qualname__}(",
                f"{self.format(term.name)}, ",
                repr(
                    {self.format(x): self.format(y) for x, y in term.to_dict().items()}
                ).replace("'", ""),
                ")",
            ]
        elif isinstance(term, Function):
            result += [
                f"{Function.create.__qualname__}(",
                f"{self.format(term.name)}, ",
                self.format(term.formula),
                ", engine",
                ")",
            ]
        elif isinstance(term, Linear):
            result += [
                f"{Op.class_name(term)}(",
                f"{self.format(term.name)}, ",
                "[",
                ", ".join(self.format(c) for c in term.coefficients),
                "]",
                ", engine",
                ")",
            ]
        else:
            result += [
                f"{Op.class_name(term)}(",
                f"{self.format(term.name)}, ",
                ", ".join(self.format(to_float(p)) for p in term.parameters().split()),
                ")",
            ]
        return "".join(result)

    def norm(self, norm: Norm | None) -> str:
        """Returns a string representation of the norm in the Python
        programming language
        @param norm is the norm
        @return a string representation of the norm in the Python
        programming language.
        """
        return f"fl.{Op.class_name(norm)}()" if norm else str(None)

    def activation(self, activation: Activation | None) -> str:
        """Returns a string representation of the activation method in the Python
        programming language
        @param activation is the activation method
        @return a string representation of the activation method in the Python
        programming language.
        """
        return f"fl.{Op.class_name(activation)}()" if activation else str(None)

    def defuzzifier(self, defuzzifier: Defuzzifier | None) -> str:
        """Returns a string representation of the defuzzifier in the Python
        programming language
        @param defuzzifier is the defuzzifier
        @return a string representation of the defuzzifier in the Python
        programming language.
        """
        if not defuzzifier:
            return str(None)

        from .defuzzifier import IntegralDefuzzifier, WeightedDefuzzifier

        if isinstance(defuzzifier, IntegralDefuzzifier):
            parameters = f"{defuzzifier.resolution}"
        elif isinstance(defuzzifier, WeightedDefuzzifier):
            parameters = f'"{defuzzifier.type.name}"'
        else:
            parameters = ""
        return f"fl.{Op.class_name(defuzzifier)}({parameters})"

    def rule(self, rule: Rule) -> str:
        """Returns a string representation of the rule in the Python
        programming language
        @param rule is the rule
        @return a string representation of the rule in the Python
        programming language.
        """
        return f"fl.{rule.create.__qualname__}({self.format(rule.text)}, engine)"


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

        input_values = []
        incremented = True
        while incremented:
            row = []
            for index, variable in enumerate(engine.input_variables):
                if variable in active_variables:
                    row.append(
                        variable.minimum
                        + sample_values[index] * variable.drange / max(1.0, resolution)
                    )
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
        input_values = []
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
