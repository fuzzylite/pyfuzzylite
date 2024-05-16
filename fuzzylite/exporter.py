"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
    """Abstract class to export engines and its components to different formats.

    info: related
        - [fuzzylite.exporter.FldExporter][]
        - [fuzzylite.exporter.FllExporter][]
        - [fuzzylite.exporter.PythonExporter][]
        - [fuzzylite.engine.Engine][]
    """

    def __str__(self) -> str:
        """Return the class name of the exporter.

        Returns:
            class name of the exporter
        """
        return Op.class_name(self)

    def __repr__(self) -> str:
        """Return code to construct the exporter in Python.

        Returns:
            code to construct the exporter in Python
        """
        from .library import representation

        return representation.as_constructor(self)

    @abstractmethod
    def to_string(self, instance: Any, /) -> str:
        """Return string representation of the instance.

        Args:
             instance: a fuzzylite object

        Returns:
            string representation of the object
        """

    def to_file(self, path: str | Path, instance: Any) -> None:
        """Write the string representation of the instance into the file.

        Args:
            path: file path to export the instance.
            instance: a fuzzylite object.
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open(mode="w", encoding="utf-8") as fll:
            fll.write(self.to_string(instance))


class FllExporter(Exporter):
    """Export an engine and its components to the FuzzyLite Language.

    info: related
        - [fuzzylite.exporter.Exporter][]
        - [fuzzylite.importer.FllImporter][]
        - [FuzzyLite Language (FLL)](https://fuzzylite.com/fll-fld/)
    """

    def __init__(self, indent: str = "  ", separator: str = "\n") -> None:
        """Constructor.

        Args:
            indent: indentation of the FuzzyLite Language.
            separator: separation between components of the FuzzyLite Language.
        """
        self.indent = indent
        self.separator = separator

    def format(self, key: str | None, value: Any) -> str:
        """Format the arguments according to the FuzzyLite Language.

        info: formatting table
            | value | formatted |
            |-------|-----------|
            | `None`  | `none`    |
            | `bool` | `true`, `false` |
            | `float` | `0.999` using [fuzzylite.library.Settings.decimals][] |
            | `list|set|tuple` | space-separated values, each formatted with this method |
            | object | anything else uses the object's `__str__()` method |

        Args:
            key: name of the property
            value: value to format

        Returns:
            formatted (key and) value according to the FuzzyLite Language
        """
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

    def to_string(self, instance: Any, /) -> str:
        """Return the object in the FuzzyLite Language.

        Args:
            instance: fuzzylite object.

        Returns:
            object in the FuzzyLite Language.
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

        if isinstance(instance, Engine):
            return self.engine(instance)

        if isinstance(instance, InputVariable):
            return self.input_variable(instance)
        if isinstance(instance, OutputVariable):
            return self.output_variable(instance)
        if isinstance(instance, Variable):
            return self.variable(instance)

        if isinstance(instance, Term):
            return self.term(instance)

        if isinstance(instance, Activation):
            return self.activation(instance)
        if isinstance(instance, Defuzzifier):
            return self.defuzzifier(instance)
        if isinstance(instance, Norm):
            return self.norm(instance)

        if isinstance(instance, RuleBlock):
            return self.rule_block(instance)
        if isinstance(instance, Rule):
            return self.rule(instance)

        raise TypeError(f"expected a fuzzylite object, but got {type(instance)}")

    def engine(self, engine: Engine, /) -> str:
        """Return the engine in the FuzzyLite Language.

        Args:
             engine: engine to export

        Returns:
             engine in the FuzzyLite Language
        """
        result = [self.format(Op.class_name(engine), engine.name)]
        if engine.description:
            result += [self.indent + self.format("description", engine.description)]
        result += [self.input_variable(iv) for iv in engine.input_variables]
        result += [self.output_variable(ov) for ov in engine.output_variables]
        result += [self.rule_block(rb) for rb in engine.rule_blocks]
        result += [""]
        return self.separator.join(result)

    def variable(  # noqa: D417 # Missing argument description in the docstring: `self`
        self, variable: Variable, /, terms: bool = True
    ) -> str:
        """Return the variable in the FuzzyLite Language.

        Args:
            variable: variable to export
            terms: whether to export the terms

        Returns:
             variable in the FuzzyLite Language
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
        """Return the input variable in the FuzzyLite Language.

        Args:
            variable: input variable to export

        Returns:
        input variable in the FuzzyLite Language
        """
        return self.variable(variable)

    def output_variable(self, variable: OutputVariable, /) -> str:
        """Return the output variable in the FuzzyLite Language.

        Args:
            variable: output variable to export

        Returns:
             output variable in the FuzzyLite Language
        """
        result = [self.variable(variable, terms=False)]
        result += [
            self.indent + self.format("aggregation", self.norm(variable.aggregation)),
            self.indent + self.format("defuzzifier", self.defuzzifier(variable.defuzzifier)),
            self.indent + self.format("default", variable.default_value),
            self.indent + self.format("lock-previous", variable.lock_previous),
        ]
        if variable.terms:
            result += [(self.indent + self.term(term)) for term in variable.terms]
        return self.separator.join(result)

    def rule_block(self, rule_block: RuleBlock, /) -> str:
        """Return the rule block in the FuzzyLite Language.

        Args:
            rule_block: rule block to export

        Returns:
             rule block in the FuzzyLite Language
        """
        result = [self.format(Op.class_name(rule_block), rule_block.name)]
        if rule_block.description:
            result += [self.indent + self.format("description", rule_block.description)]
        result += [
            self.indent + self.format("enabled", rule_block.enabled),
            self.indent + self.format("conjunction", self.norm(rule_block.conjunction)),
            self.indent + self.format("disjunction", self.norm(rule_block.disjunction)),
            self.indent + self.format("implication", self.norm(rule_block.implication)),
            self.indent + self.format("activation", self.activation(rule_block.activation)),
        ]
        if rule_block.rules:
            result += [self.indent + self.rule(rule) for rule in rule_block.rules]
        return self.separator.join(result)

    def term(self, term: Term, /) -> str:
        """Return the term in the FuzzyLite Language.

        Args:
            term: term to export

        Returns:
             term in the FuzzyLite Language
        """
        return self.format(
            "term",
            (Op.as_identifier(term.name), Op.class_name(term), term.parameters()),
        )

    def norm(self, norm: Norm | None, /) -> str:
        """Return the norm in the FuzzyLite Language.

        Args:
            norm: norm to export

        Returns:
             norm in the FuzzyLite Language
        """
        return Op.class_name(norm) if norm else "none"

    def activation(self, activation: Activation | None, /) -> str:
        """Return the activation method in the FuzzyLite Language.

        Args:
            activation: activation method to export

        Returns:
             activation method in the FuzzyLite Language
        """
        if not activation:
            return "none"
        return self.format(key=None, value=(Op.class_name(activation), activation.parameters()))

    def defuzzifier(self, defuzzifier: Defuzzifier | None, /) -> str:
        """Return the defuzzifier in the FuzzyLite Language.

        Args:
            defuzzifier: defuzzifier to export

        Returns:
             defuzzifier in the FuzzyLite Language
        """
        if not defuzzifier:
            return "none"
        return self.format(key=None, value=(Op.class_name(defuzzifier), defuzzifier.parameters()))

    def rule(self, rule: Rule, /) -> str:
        """Return the rule in the FuzzyLite Language.

        Args:
            rule: rule to export

        Returns:
             rule in the FuzzyLite Language
        """
        return self.format("rule", rule.text)


class PythonExporter(Exporter):
    """Export an engine and its components to Python.

    info: related
        - [fuzzylite.exporter.Exporter][]
    """

    def __init__(self, formatted: bool = True, encapsulated: bool = False) -> None:
        """Constructor.

        Args:
            formatted: try to format the code using `black` if it is installed
            encapsulated: whether to encapsulate the code (using classes for engines and methods for other components).
        """
        self.formatted = formatted
        self.encapsulated = encapsulated

    def format(self, code: str, **kwargs: Any) -> str:
        """Format the code using the `black` formatter if it is installed, otherwise no effects on the code.

        Args:
            code: code to format.
            **kwargs: keyword arguments to pass to `black.Mode`
        Returns:
            code formatted if `black` is installed, otherwise the code without format
        """
        try:
            import black

            kwargs = dict(line_length=100) | kwargs
            formatted = black.format_str(code, mode=black.Mode(**kwargs))
            return formatted
        except ModuleNotFoundError:
            settings.logger.error("expected `black` module to be installed, but could not be found")
        except ValueError:  # black.parsing.InvalidInput
            raise
        return code

    def encapsulate(self, instance: Any) -> str:
        """Encapsulate the instance in a new class if it is an engine, or in a create method otherwise.

        Args:
            instance: object to encapsulate

        Returns:
            if the instance is an engine, then the class constructing the engine during initialization;
            otherwise a method constructing the object
        """
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
        """Return the code to construct the instance in Python.

        Args:
            instance: fuzzylite object

        Returns:
             code to construct the instance in Python
        """
        code = self.encapsulate(instance) if self.encapsulated else repr(instance)

        if self.formatted:
            code = self.format(code)
        return code

    def engine(self, engine: Engine, /) -> str:
        """Return the code to construct the engine in Python.

        Args:
            engine: engine to export

        Returns:
             code to construct the engine in Python
        """
        return self.to_string(engine)

    def input_variable(self, input_variable: InputVariable, /) -> str:
        """Return the code to construct the input variable in Python.

        Args:
            input_variable: input variable to export

        Returns:
             code to construct the input variable in Python
        """
        return self.to_string(input_variable)

    def output_variable(self, output_variable: OutputVariable, /) -> str:
        """Return the code to construct the output variable in Python.

        Args:
            output_variable: output variable to export

        Returns:
             code to construct the output variable in Python
        """
        return self.to_string(output_variable)

    def rule_block(self, rule_block: RuleBlock, /) -> str:
        """Return the code to construct the rule block in Python.

        Args:
            rule_block: rule block variable to export

        Returns:
             code to construct the rule block in Python
        """
        return self.to_string(rule_block)

    def term(self, term: Term, /) -> str:
        """Return the code to construct the term in Python.

        Args:
            term: term to export

        Returns:
             code to construct the term in Python
        """
        return self.to_string(term)

    def norm(self, norm: Norm | None, /) -> str:
        """Return the code to construct the norm in Python.

        Args:
            norm: norm to export

        Returns:
             code to construct the norm in Python
        """
        return self.to_string(norm) if norm else "None"

    def activation(self, activation: Activation | None, /) -> str:
        """Return the code to construct the activation method in Python.

        Args:
            activation: activation method to export

        Returns:
             code to construct the activation method in Python
        """
        return self.to_string(activation) if activation else "None"

    def defuzzifier(self, defuzzifier: Defuzzifier | None, /) -> str:
        """Return the code to construct the defuzzifier in Python.

        Args:
            defuzzifier: defuzzifier to export

        Returns:
             code to construct the defuzzifier in Python
        """
        return self.to_string(defuzzifier) if defuzzifier else "None"

    def rule(self, rule: Rule, /) -> str:
        """Return the code to construct the rule in Python.

        Args:
            rule: rule to export

        Returns:
             code to construct the rule in Python
        """
        return self.to_string(rule)


class FldExporter(Exporter):
    """Export the input values and output values of an engine to the FuzzyLite Dataset (FLD) format.

    info: related
        - [fuzzylite.exporter.Exporter][]

    warning: warning
        FldExporter uses vectorization so it only works with the [fuzzylite.activation.General][] activation method

    todo: todo
        include option for non-vectorized export so other activation methods can be used
    """

    @enum.unique
    class ScopeOfValues(enum.Enum):
        r"""Scope of the equally-distributed values to generate.

        - `EachVariable`: Generates $v$ values for each variable, resulting in a total resolution of
        $-1 + \max(1, v^{\frac{1}{|I|})$ from all combinations, where $I$ refers to the input variables.
        - `AllVariables`: Generates values for each variable such that the total resolution is $v$.
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
        """Constructor.

        Args:
            separator: separator of the dataset columns
            headers: whether to export the header of the dataset
            input_values: whether to export the input values
            output_values: whether to export the output values.
        """
        self.separator = separator
        self.headers = headers
        self.input_values = input_values
        self.output_values = output_values

    def header(self, engine: Engine) -> str:
        """Return the header of the dataset for the engine.

        Args:
            engine: engine to export

        Returns:
            header of the dataset for the engine.
        """
        result: list[str] = []
        if self.input_values:
            result += [iv.name for iv in engine.input_variables]
        if self.output_values:
            result += [ov.name for ov in engine.output_variables]
        return self.separator.join(result)

    def to_string(self, instance: object) -> str:
        """Return a FuzzyLite Dataset from the engine using 1024 input values for all variables.

        Args:
            instance: engine to export

        Returns:
             FuzzyLite Dataset from the engine

        Raises:
            ValueError: if the instance is not an Engine.
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
        """Return a FuzzyLite Dataset from the engine using the input values and their scope.

        Args:
            engine: engine to export
            values: number of values to export
            scope: scope of the values
            active_variables: input variables to set values for

        Returns:
             FuzzyLite Dataset from the engine.
        """
        # TODO: Remove active_variables?
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
        """Write the FuzzyLite Dataset from the engine to the file using the input values and their scope.

        Args:
            path: file path
            engine: engine to export
            values: number of values to export
            scope: scope of the values
            active_variables: set of input variables to set values for
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
        """Write a FuzzyLite dataset from the engine to the writer.

        Args:
            engine: engine to export
            writer: output to write the engine
            values: number of values to export
            scope:  scope of the values
            active_variables: input variables to generate values for.
        """
        # TODO: improve handling of active_variables?
        if active_variables is None:
            active_variables = set(engine.input_variables)

        if scope == FldExporter.ScopeOfValues.AllVariables:
            if len(engine.input_variables) == 0:
                raise ValueError("expected input variables in engine, but got none")
            resolution = -1 + max(1, int(pow(values, (1.0 / len(engine.input_variables)))))
        else:
            resolution = values - 1

        sample_values = [0] * len(engine.input_variables)
        min_values = [0] * len(engine.input_variables)
        max_values = [resolution if iv in active_variables else 0 for iv in engine.input_variables]

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

    def to_string_from_reader(self, engine: Engine, reader: IO[str], skip_lines: int = 0) -> str:
        """Return a FuzzyLite Dataset from the engine using the input values from the reader.

        Args:
            engine: engine to export
            reader: reader of a set of lines containing space-separated input values
            skip_lines: number of lines to skip from the beginning

        Returns:
             FuzzyLite Dataset from the engine.
        """
        writer = io.StringIO()
        self.write_from_reader(engine, writer, reader, skip_lines)
        return writer.getvalue()

    def to_file_from_reader(
        self, path: Path, engine: Engine, reader: IO[str], skip_lines: int = 0
    ) -> None:
        """Write the FuzzyLite Dataset to the file using the input values from the engine.

        Args:
            path: path to the output file
            engine: engine to export
            reader: reader of a set of lines containing space-separated input values
            skip_lines: number of lines to skip from the beginning.

        Returns:
            FuzzyLite Dataset from the engine.
        """
        with path.open("w") as writer:
            self.write_from_reader(engine, writer, reader, skip_lines)

    def write_from_reader(
        self, engine: Engine, writer: IO[str], reader: IO[str], skip_lines: int = 0
    ) -> None:
        """Write the FuzzyLite Dataset from the engine to the writer using the input values from the reader.

        Args:
            engine: engine to export
            writer: output to write the engine
            reader: reader of a set of lines containing space-separated input values
            skip_lines: number of lines to skip from the beginning.
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
        """Write a FuzzyLite Dataset line from the engine to the writer using the input values.

        Args:
            engine: engine to export
            writer: output where the engine will be written to
            input_values: matrix of input values.
        """
        input_values = np.atleast_2d(input_values)
        if input_values.shape[1] < len(engine.input_variables):
            raise ValueError(
                f"expected {len(engine.input_variables)} input values (one per input variable), "
                f"but got {input_values.shape[1]} instead"
            )

        engine.restart()

        # TODO: Vectorization here will not work for activation methods other than General
        for index, variable in enumerate(engine.input_variables):
            variable.value = input_values[:, index]

        engine.process()

        values: list[Any] = []
        if self.input_values:
            values.append(engine.input_values)
        if self.output_values:
            values.append(engine.output_values)
        if not values:
            values.append([])

        np.savetxt(
            writer,
            np.hstack(values),
            fmt=f"%0.{settings.decimals}f",
            delimiter=self.separator,
            header=self.header(engine) if self.headers else "",
            comments="",
        )
