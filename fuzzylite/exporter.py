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

__all__ = ["Exporter", "FllExporter", "PythonExporter", "FldExporter"]

import enum
import io
import math
import typing
from pathlib import Path
from typing import IO, Any, List, Optional, Set, Union

from .operation import Op

if typing.TYPE_CHECKING:
    from .activation import Activation
    from .defuzzifier import Defuzzifier
    from .engine import Engine
    from .norm import Norm
    from .rule import Rule, RuleBlock
    from .term import Term
    from .variable import InputVariable, OutputVariable, Variable


class Exporter:
    """The Exporter class is the abstract class for exporters to translate an
    Engine into different formats.

    @todo declare methods for exporting other components (e.g., Variable)

    @author Juan Rada-Vilela, Ph.D.
    @see Importer
    @since 4.0
    """

    @property
    def class_name(self) -> str:
        """Gets the name of the exporter."""
        return self.__class__.__name__

    def to_string(self, instance: object) -> str:
        """Returns a string representation of the FuzzyLite component
        @param instance is the FuzzyLite component
        @return a string representation of the FuzzyLite component.
        """
        raise NotImplementedError()

    def to_file(self, path: Union[str, Path], instance: object) -> None:
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
    [http://www.fuzzylite.com/fll-fld](http://www.fuzzylite.com/fll-fld) for
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

    def to_string(self, instance: object) -> str:
        """Returns a string representation of the FuzzyLite component
        @param instance is the FuzzyLite component
        @return a string representation of the FuzzyLite component.
        """
        from .engine import Engine

        if isinstance(instance, Engine):
            return self.engine(instance)

        from .variable import InputVariable, OutputVariable, Variable

        if isinstance(instance, InputVariable):
            return self.input_variable(instance)
        if isinstance(instance, OutputVariable):
            return self.output_variable(instance)
        if isinstance(instance, Variable):
            return self.variable(instance)

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

    def engine(self, engine: "Engine") -> str:
        """Returns a string representation of the engine
        @param engine is the engine
        @return a string representation of the engine.
        """
        result = [f"Engine: {engine.name}"]
        if engine.description:
            result += [f"{self.indent}description: {engine.description}"]
        for input_variable in engine.input_variables:
            result += [self.input_variable(input_variable)]
        for output_variable in engine.output_variables:
            result += [self.output_variable(output_variable)]
        for rule_block in engine.rule_blocks:
            result += [self.rule_block(rule_block)]
        result += [""]
        return self.separator.join(result)

    def variable(self, v: "Variable") -> str:
        """Returns a string representation of the variable
        @param v is the variable
        @return a string representation of the variable.

        TODO: rename v to variable
        """
        result = [f"Variable: {v.name}"]
        if v.description:
            result += [f"{self.indent}description: {v.description}"]
        result += [
            f"{self.indent}enabled: {str(v.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(v.minimum), Op.str(v.maximum)])}",
            f"{self.indent}lock-range: {str(v.lock_range).lower()}",
        ]
        if v.terms:
            result += [f"{self.indent}{self.term(term)}" for term in v.terms]
        return self.separator.join(result)

    def input_variable(self, iv: "InputVariable") -> str:
        """Returns a string representation of the input variable
        @param iv is the input variable
        @return a string representation of the input variable.

        TODO: rename iv to input_variable
        """
        result = [f"InputVariable: {iv.name}"]
        if iv.description:
            result += [f"{self.indent}description: {iv.description}"]
        result += [
            f"{self.indent}enabled: {str(iv.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(iv.minimum), Op.str(iv.maximum)])}",
            f"{self.indent}lock-range: {str(iv.lock_range).lower()}",
        ]
        if iv.terms:
            result += [f"{self.indent}{self.term(term)}" for term in iv.terms]
        return self.separator.join(result)

    def output_variable(self, ov: "OutputVariable") -> str:
        """Returns a string representation of the output variable
        @param ov is the variable
        @return a string representation of the output variable.

        TODO: rename ov to output_variable
        """
        result = [f"OutputVariable: {ov.name}"]
        if ov.description:
            result += [f"{self.indent}description: {ov.description}"]
        result += [
            f"{self.indent}enabled: {str(ov.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(ov.minimum), Op.str(ov.maximum)])}",
            f"{self.indent}lock-range: {str(ov.lock_range).lower()}",
            f"{self.indent}aggregation: {self.norm(ov.aggregation)}",
            f"{self.indent}defuzzifier: {self.defuzzifier(ov.defuzzifier)}",
            f"{self.indent}default: {Op.str(ov.default_value)}",
            f"{self.indent}lock-previous: {str(ov.lock_previous).lower()}",
        ]
        if ov.terms:
            result += [f"{self.indent}{self.term(term)}" for term in ov.terms]
        return self.separator.join(result)

    def rule_block(self, rb: "RuleBlock") -> str:
        """Returns a string representation of the rule block
        @param rb is the rule block
        @return a string representation of the rule block.

        TODO: rename rb to rule_block
        """
        result = [f"RuleBlock: {rb.name}"]
        if rb.description:
            result += [f"{self.indent}description: {rb.description}"]
        result += [
            f"{self.indent}enabled: {str(rb.enabled).lower()}",
            f"{self.indent}conjunction: {self.norm(rb.conjunction)}",
            f"{self.indent}disjunction: {self.norm(rb.disjunction)}",
            f"{self.indent}implication: {self.norm(rb.implication)}",
            f"{self.indent}activation: {self.activation(rb.activation)}",
        ]
        if rb.rules:
            result += [f"{self.indent}{self.rule(rule)}" for rule in rb.rules]
        return self.separator.join(result)

    def term(self, term: "Term") -> str:
        """Returns a string representation of the linguistic term
        @param term is the linguistic term
        @return a string representation of the linguistic term.
        """
        result = ["term:", Op.as_identifier(term.name), term.class_name]
        parameters = term.parameters()
        if parameters:
            result += [parameters]
        return " ".join(result)

    def norm(self, norm: Optional["Norm"]) -> str:
        """Returns a string representation of the norm
        @param norm is the norm
        @return a string representation of the norm.
        """
        return norm.class_name if norm else "none"

    def activation(self, activation: Optional["Activation"]) -> str:
        """Returns a string representation of the activation method
        @param activation is the activation method
        @return a string representation of the activation method.
        """
        return activation.class_name if activation else "none"

    def defuzzifier(self, defuzzifier: Optional["Defuzzifier"]) -> str:
        """Returns a string representation of the defuzzifier
        @param defuzzifier is the defuzzifier
        @return a string representation of the defuzzifier.
        """
        if not defuzzifier:
            return "none"
        from .defuzzifier import IntegralDefuzzifier, WeightedDefuzzifier

        result = [defuzzifier.class_name]
        if isinstance(defuzzifier, IntegralDefuzzifier):
            result += [str(defuzzifier.resolution)]
        elif isinstance(defuzzifier, WeightedDefuzzifier):
            result += [defuzzifier.type.name]
        return " ".join(result)

    def rule(self, rule: "Rule") -> str:
        """Returns a string representation of the rule
        @param rule is the rule
        @return a string representation of the rule.
        """
        return f"rule: {rule.text}"


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

    def engine(self, engine: "Engine", level: int = 0) -> str:
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

        input_variables: List[str] = []
        for iv in engine.input_variables:
            input_variables += [
                f"{(level + 1) * self.indent}{self.input_variable(iv, level + 2)}"
            ]
        result += [self.key_values("engine.input_variables", input_variables)]

        output_variables: List[str] = []
        for ov in engine.output_variables:
            output_variables += [
                f"{(level + 1) * self.indent}{self.output_variable(ov, level + 2)}"
            ]
        result += [self.key_values("engine.output_variables", output_variables)]

        rule_blocks: List[str] = []
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
            if math.isinf(x):
                return ("fl." if x > 0 else "-fl.") + str(math.fabs(x))
            return Op.str(x)
        if isinstance(x, bool):
            return str(x)
        return str(x)

    def key_values(self, name: str, values: List[Any], level: int = 0) -> str:
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

    def input_variable(self, iv: "InputVariable", level: int = 1) -> str:
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

    def output_variable(self, ov: "OutputVariable", level: int = 1) -> str:
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

    def rule_block(self, rb: "RuleBlock", level: int = 1) -> str:
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

    def term(self, term: "Term") -> str:
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
                f"{term.class_name}({self.format(term.name)}, ",
                "[",
                ", ".join(
                    self.format(value) for value in Discrete.values_from(term.xy)
                ),
                "])",
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
                f"{term.class_name}(",
                f"{self.format(term.name)}, ",
                "[",
                ", ".join(self.format(c) for c in term.coefficients),
                "]",
                ", engine",
                ")",
            ]
        else:
            result += [
                f"{term.class_name}(",
                f"{self.format(term.name)}, ",
                ", ".join(self.format(Op.scalar(p)) for p in term.parameters().split()),
                ")",
            ]
        return "".join(result)

    def norm(self, norm: Optional["Norm"]) -> str:
        """Returns a string representation of the norm in the Python
        programming language
        @param norm is the norm
        @return a string representation of the norm in the Python
        programming language.
        """
        return f"fl.{norm.class_name}()" if norm else str(None)

    def activation(self, activation: Optional["Activation"]) -> str:
        """Returns a string representation of the activation method in the Python
        programming language
        @param activation is the activation method
        @return a string representation of the activation method in the Python
        programming language.
        """
        return f"fl.{activation.class_name}()" if activation else str(None)

    def defuzzifier(self, defuzzifier: Optional["Defuzzifier"]) -> str:
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
        return f"fl.{defuzzifier.class_name}({parameters})"

    def rule(self, rule: "Rule") -> str:
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
    see [http://www.fuzzylite.com/fll-fld](http://www.fuzzylite.com/fll-fld)
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

        (
            # /**Generates $n$ values for each variable*/
            EachVariable,
            # /**Generates $n$ values for all variables*/
            AllVariables,
        ) = range(2)

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

    def header(self, engine: "Engine") -> str:
        """Gets the header of the dataset for the given engine
        @param engine is the engine to be exported
        @return the header of the dataset for the given engine.
        """
        result: List[str] = []
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
        engine: "Engine",
        values: int = 1024,
        scope: ScopeOfValues = ScopeOfValues.AllVariables,
        active_variables: Optional[Set["InputVariable"]] = None,
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
        engine: "Engine",
        values: int = 1024,
        scope: ScopeOfValues = ScopeOfValues.AllVariables,
        active_variables: Optional[Set["InputVariable"]] = None,
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
        engine: "Engine",
        writer: IO[str],
        values: int,
        scope: ScopeOfValues,
        active_variables: Optional[Set["InputVariable"]] = None,
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

        if self.headers:
            writer.writelines(self.header(engine) + "\n")

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

        input_values = [Op.scalar("nan")] * len(engine.input_variables)
        incremented = True
        while incremented:
            for i, iv in enumerate(engine.input_variables):
                if iv in active_variables:
                    input_values[i] = iv.minimum + sample_values[i] * iv.drange / max(
                        1.0, resolution
                    )
                else:
                    input_values[i] = iv.value
            self.write(engine, writer, input_values, active_variables)

            incremented = Op.increment(sample_values, min_values, max_values)

    def to_string_from_reader(
        self, engine: "Engine", reader: IO[str], skip_lines: int = 0
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
        self, path: Path, engine: "Engine", reader: IO[str], skip_lines: int = 0
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
        self, engine: "Engine", writer: IO[str], reader: IO[str], skip_lines: int = 0
    ) -> None:
        """Writes the engine into the given writer
        @param engine is the engine to export
        @param writer is the output where the engine will be written to
        @param reader is the reader of a set of lines containing space-separated input values
        @param skip_lines is the number of lines to initially skip.
        """
        if self.headers:
            writer.writelines(self.header(engine) + "\n")
        active_variables = set(engine.input_variables)
        for i, line in enumerate(reader.readlines()):
            if i < skip_lines:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            input_values = [Op.scalar(x) for x in line.split()]
            self.write(engine, writer, input_values, active_variables)

    def write(
        self,
        engine: "Engine",
        writer: IO[str],
        input_values: List[float],
        active_variables: Set["InputVariable"],
    ) -> None:
        """Writes the engine into the given writer
        @param engine is the engine to export
        @param writer is the output where the engine will be written to
        @param input_values is the vector of input values
        @param active_variables contains the input variables to generate values for.
        """
        # if not input_values:
        #     writer.writelines("\n")
        if len(input_values) < len(engine.input_variables):
            raise ValueError(
                f"not enough input values ({len(input_values)}) "
                f"for the input variables ({len(engine.input_variables)})"
            )

        values: List[str] = []
        for i, iv in enumerate(engine.input_variables):
            if iv in active_variables:
                iv.value = input_values[i]
            if self.input_values:
                values.append(Op.str(iv.value))

        engine.process()

        if self.output_values:
            values.extend(Op.str(ov.value) for ov in engine.output_variables)

        writer.writelines(self.separator.join(values) + "\n")
