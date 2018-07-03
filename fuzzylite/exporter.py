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
import io
import typing
from pathlib import Path
from typing import IO, List, Optional, Set

from .operation import Op

if typing.TYPE_CHECKING:
    from .activation import Activation  # noqa: F401
    from .defuzzifier import Defuzzifier  # noqa: F401
    from .engine import Engine
    from .norm import Norm  # noqa: F401
    from .rule import Rule, RuleBlock
    from .term import Term
    from .variable import InputVariable, OutputVariable, Variable


class Exporter(object):
    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def to_string(self, instance: object) -> str:
        raise NotImplementedError()

    def to_file(self, path: Path, instance: object) -> None:
        with path.open(mode='w') as fll:
            fll.write(self.to_string(instance))


class FllExporter(Exporter):
    __slots__ = ["indent", "separator"]

    def __init__(self, indent: str = "  ", separator: str = "\n") -> None:
        self.indent = indent
        self.separator = separator

    def to_string(self, instance: object) -> str:
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

        from .defuzzifier import Defuzzifier  # noqa: F811
        if isinstance(instance, Defuzzifier):
            return self.defuzzifier(instance)

        from .rule import RuleBlock, Rule
        if isinstance(instance, RuleBlock):
            return self.rule_block(instance)
        if isinstance(instance, Rule):
            return self.rule(instance)

        from .norm import Norm  # noqa: F811
        if isinstance(instance, Norm):
            return self.norm(instance)

        from .activation import Activation  # noqa: F811
        if isinstance(instance, Activation):
            return self.activation(instance)

        raise ValueError(f"expected a fuzzylite object, but found '{type(instance).__name__}'")

    def engine(self, engine: 'Engine') -> str:
        result = [f"Engine: {engine.name}"]
        if engine.description:
            result.append(f"description: {engine.description}")
        for input_variable in engine.input_variables:
            result.append(self.input_variable(input_variable))
        for output_variable in engine.output_variables:
            result.append(self.output_variable(output_variable))
        for rule_block in engine.rule_blocks:
            result.append(self.rule_block(rule_block))
        return self.separator.join(result)

    def variable(self, v: 'Variable') -> str:
        result = [f"Variable: {v.name}"]
        if v.description:
            result.append(f"{self.indent}description: {v.description}")
        result.extend([
            f"{self.indent}enabled: {str(v.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(v.minimum), Op.str(v.maximum)])}",
            f"{self.indent}lock-range: {str(v.enabled).lower()}",
            *[f"{self.indent}{self.term(term)}" for term in v.terms]
        ])
        return self.separator.join(result)

    def input_variable(self, iv: 'InputVariable') -> str:
        result = [f"InputVariable: {iv.name}"]
        if iv.description:
            result.append(f"{self.indent}description: {iv.description}")
        result.extend([
            f"{self.indent}enabled: {str(iv.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(iv.minimum), Op.str(iv.maximum)])}",
            f"{self.indent}lock-range: {str(iv.lock_range).lower()}",
            *[f"{self.indent}{self.term(term)}" for term in iv.terms]
        ])
        return self.separator.join(result)

    def output_variable(self, ov: 'OutputVariable') -> str:
        result = [f"OutputVariable: {ov.name}"]
        if ov.description:
            result.append(f"{self.indent}description: {ov.description}")
        result.extend([
            f"{self.indent}enabled: {str(ov.enabled).lower()}",
            f"{self.indent}range: {' '.join([Op.str(ov.minimum), Op.str(ov.maximum)])}",
            f"{self.indent}lock-range: {str(ov.lock_range).lower()}",
            f"{self.indent}aggregation: {self.norm(ov.aggregation)}",
            f"{self.indent}defuzzifier: {self.defuzzifier(ov.defuzzifier)}",
            f"{self.indent}default: {Op.str(ov.default_value)}",
            f"{self.indent}lock-previous: {str(ov.lock_previous).lower()}",
            *[f"{self.indent}{self.term(term)}" for term in ov.terms]
        ])
        return self.separator.join(result)

    def rule_block(self, rb: 'RuleBlock') -> str:
        result = [f"RuleBlock: {rb.name}"]
        if rb.description:
            result.append(f"{self.indent}description: {rb.description}")
        result.extend([
            f"{self.indent}enabled: {str(rb.enabled).lower()}",
            f"{self.indent}conjunction: {self.norm(rb.conjunction)}",
            f"{self.indent}disjunction: {self.norm(rb.disjunction)}",
            f"{self.indent}implication: {self.norm(rb.implication)}",
            f"{self.indent}activation: {self.activation(rb.activation)}",
            *[f"{self.indent}{self.rule(rule)}" for rule in rb.rules]
        ])
        return self.separator.join(result)

    def term(self, term: 'Term') -> str:
        result = ["term:", Op.as_identifier(term.name), term.class_name]
        parameters = term.parameters()
        if parameters:
            result.append(parameters)
        return " ".join(result)

    def norm(self, norm: Optional['Norm']) -> str:
        return norm.class_name if norm else "none"

    def activation(self, activation: Optional['Activation']) -> str:
        return activation.class_name if activation else "none"

    def defuzzifier(self, defuzzifier: Optional['Defuzzifier']) -> str:
        if not defuzzifier:
            return "none"
        from .defuzzifier import IntegralDefuzzifier, WeightedDefuzzifier
        result = [defuzzifier.class_name]
        if isinstance(defuzzifier, IntegralDefuzzifier):
            result.append(str(defuzzifier.resolution))
        elif isinstance(defuzzifier, WeightedDefuzzifier):
            result.append(defuzzifier.type.name)
        return " ".join(result)

    def rule(self, rule: 'Rule') -> str:
        return f"rule: {rule.text}"


class FldExporter(Exporter):
    __slots__ = ["separator", "headers", "input_values", "output_values"]

    @enum.unique
    class ScopeOfValues(enum.Enum):
        EachVariable, AllVariables = range(2)

    def __init__(self,
                 separator: str = ' ',
                 headers: bool = True,
                 input_values: bool = True,
                 output_values: bool = True) -> None:
        self.separator = separator
        self.headers = headers
        self.input_values = input_values
        self.output_values = output_values

    def header(self, engine: 'Engine') -> str:
        result: List[str] = []
        if self.input_values:
            result.extend(iv.name for iv in engine.input_variables)
        if self.output_values:
            result.extend(ov.name for ov in engine.output_variables)
        return self.separator.join(result)

    def to_string(self, instance: object) -> str:
        from .engine import Engine
        if isinstance(instance, Engine):
            return self.to_string_from_scope(instance)
        raise ValueError(f"expected an Engine, but got {type(instance)}")

    def to_string_from_scope(self, engine: 'Engine', values: int = 1024,
                             scope: ScopeOfValues = ScopeOfValues.AllVariables,
                             active_variables: Optional[Set['InputVariable']] = None) -> str:
        if not active_variables:
            active_variables = set(engine.input_variables)

        writer = io.StringIO()
        self.write_from_scope(engine, writer, values, scope, active_variables)
        return writer.getvalue()

    def to_file_from_scope(self, path: Path, engine: 'Engine', values: int = 1024,
                           scope: ScopeOfValues = ScopeOfValues.AllVariables,
                           active_variables: Optional[Set['InputVariable']] = None) -> None:
        if not active_variables:
            active_variables = set(engine.input_variables)

        with path.open('w') as writer:
            self.write_from_scope(engine, writer, values, scope, active_variables)

    def write_from_scope(self, engine: 'Engine', writer: IO[str], values: int,
                         scope: ScopeOfValues, active_variables: Set['InputVariable']) -> None:
        if self.headers:
            writer.writelines(self.header(engine) + "\n")

        if scope == FldExporter.ScopeOfValues.AllVariables:
            resolution = max(1, int(pow(values, (1.0 / len(engine.input_variables)))))
        else:
            resolution = values

        sample_values = [0] * len(engine.input_variables)
        min_values = [0] * len(engine.input_variables)
        max_values = [resolution if iv in active_variables else 0
                      for iv in engine.input_variables]

        input_values = [Op.scalar('nan')] * len(engine.input_variables)
        incremented = True
        while incremented:
            for i, iv in enumerate(engine.input_variables):
                if iv in active_variables:
                    input_values[i] = (iv.minimum +
                                       sample_values[i] * iv.drange / max(1.0, resolution))
                else:
                    input_values[i] = iv.value
            self.write(engine, writer, input_values, active_variables)

            incremented = Op.increment(sample_values, min_values, max_values)

    def to_string_from_reader(self, engine: 'Engine', reader: IO[str]) -> str:
        writer = io.StringIO()
        self.write_from_reader(engine, writer, reader)
        return writer.getvalue()

    def to_file_from_reader(self, path: Path, engine: 'Engine', reader: IO[str]) -> None:
        with path.open('w') as writer:
            self.write_from_reader(engine, writer, reader)

    def write_from_reader(self, engine: 'Engine', writer: IO[str], reader: IO[str]) -> None:
        if self.headers:
            writer.writelines(self.header(engine) + "\n")
        active_variables = set(engine.input_variables)
        for i, line in enumerate(reader.readlines()):
            line = line.strip()
            if not line or line[0] == '#':
                continue
            try:
                input_values = [Op.scalar(x) for x in line.split()]
            except ValueError:
                if line == 0:  # ignore headers
                    continue
                raise
            self.write(engine, writer, input_values, active_variables)

    def write(self, engine: 'Engine', writer: IO[str], input_values: List[float],
              active_variables: Set['InputVariable']) -> None:
        if not input_values:
            writer.writelines("\n")
        if len(input_values) != len(engine.input_variables):
            raise ValueError()

        values: List[str] = []
        for i, iv in enumerate(engine.input_variables):
            if iv in active_variables:
                iv.value = input_values[i]
            if self.input_values:
                values.append(Op.str(iv.value))

        engine.process()

        if self.output_values:
            for ov in engine.output_variables:
                values.append(Op.str(ov.value))

        writer.writelines(self.separator.join(values) + "\n")
