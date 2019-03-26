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

__all__ = ["Importer", "FllImporter"]

from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union

from .activation import Activation
from .defuzzifier import Defuzzifier
from .engine import Engine
from .factory import ConstructionFactory
from .norm import SNorm, TNorm
from .operation import Op
from .rule import Rule, RuleBlock
from .term import Term
from .variable import InputVariable, OutputVariable


class Importer:

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def from_string(self, fll: str) -> 'Engine':
        raise NotImplementedError()

    def from_file(self, path: Union[Path, str]) -> 'Engine':
        if isinstance(path, str):
            path = Path(path)
        with path.open() as fll:
            return self.from_string(fll.read())


class FllImporter(Importer):

    def __init__(self, separator: str = '\n') -> None:
        self.separator = separator

    def _process(self, component: str, block: List[str], engine: 'Engine') -> None:
        if component == "Engine":
            for line in block:
                line = Op.strip_comments(line)

                key, value = self.extract_key_value(line)
                if key == "Engine":
                    engine.name = value
                elif key == "description":
                    engine.description = value
                else:
                    raise SyntaxError(f"'{key}' is not a valid component of '{component}'")
        elif component == "InputVariable":
            input_variable = self.input_variable(self.separator.join(block), engine)
            engine.input_variables.append(input_variable)
        elif component == "OutputVariable":
            output_variable = self.output_variable(self.separator.join(block), engine)
            engine.output_variables.append(output_variable)
        elif component == "RuleBlock":
            rule_block = self.rule_block(self.separator.join(block), engine)
            engine.rule_blocks.append(rule_block)

    def from_string(self, fll: str) -> 'Engine':
        return self.engine(fll)

    def engine(self, fll: str) -> 'Engine':
        engine = Engine()
        component = ""
        block: List[str] = []

        for line in fll.split(self.separator):
            line = Op.strip_comments(line)
            if not line:
                continue
            key, value = self.extract_key_value(line)
            if key in {"Engine", "InputVariable", "OutputVariable", "RuleBlock"}:
                if component:
                    # Process previous block
                    self._process(component, block, engine)
                component = key
                block = [line]
            else:
                block.append(line)
        if component and block:
            self._process(component, block, engine)
        return engine

    def input_variable(self, fll: str, engine: Optional['Engine'] = None) -> 'InputVariable':

        iv = InputVariable()
        for line in fll.split(self.separator):
            line = Op.strip_comments(line)
            if not line:
                continue
            key, value = self.extract_key_value(line)
            if key == "InputVariable":
                iv.name = value
            elif key == "description":
                iv.description = value
            elif key == "enabled":
                iv.enabled = self.boolean(value)
            elif key == "range":
                iv.range = self.range(value)
            elif key == "lock-range":
                iv.lock_range = self.boolean(value)
            elif key == "term":
                iv.terms.append(self.term(line, engine))
            else:
                raise SyntaxError(f"'{key}' is not a valid component of '{iv.__class__.__name__}'")
        iv.name = Op.as_identifier(iv.name)
        return iv

    def output_variable(self, fll: str, engine: Optional['Engine'] = None) -> 'OutputVariable':
        ov = OutputVariable()
        for line in fll.split(self.separator):
            line = Op.strip_comments(line)
            if not line:
                continue
            key, value = self.extract_key_value(line)
            if key == "OutputVariable":
                ov.name = value
            elif key == "description":
                ov.description = value
            elif key == "enabled":
                ov.enabled = self.boolean(value)
            elif key == "range":
                ov.range = self.range(value)
            elif key == "default":
                ov.default_value = Op.scalar(value)
            elif key == "lock-previous":
                ov.lock_previous = self.boolean(value)
            elif key == "lock-range":
                ov.lock_range = self.boolean(value)
            elif key == "defuzzifier":
                ov.defuzzifier = self.defuzzifier(value)
            elif key == "aggregation":
                ov.aggregation = self.snorm(value)
            elif key == "term":
                ov.terms.append(self.term(line, engine))
            else:
                raise SyntaxError(f"'{key}' is not a valid component of '{ov.__class__.__name__}'")
        ov.name = Op.as_identifier(ov.name)
        return ov

    def rule_block(self, fll: str, engine: Optional['Engine'] = None) -> 'RuleBlock':
        rb = RuleBlock()
        for line in fll.split(self.separator):
            line = Op.strip_comments(line)
            if not line:
                continue
            key, value = self.extract_key_value(line)
            if key == "RuleBlock":
                rb.name = value
            elif key == "description":
                rb.description = value
            elif key == "enabled":
                rb.enabled = self.boolean(value)
            elif key == "conjunction":
                rb.conjunction = self.tnorm(value)
            elif key == "disjunction":
                rb.disjunction = self.snorm(value)
            elif key == "implication":
                rb.implication = self.tnorm(value)
            elif key == "activation":
                rb.activation = self.activation(value)
            elif key == "rule":
                rule = self.rule(line, engine)
                if rule:
                    rb.rules.append(rule)
            else:
                raise SyntaxError(f"'{key}' is not a valid component of '{rb.__class__.__name__}'")
        return rb

    def term(self, fll: str, engine: Optional['Engine'] = None) -> 'Term':
        from . import lib

        values = self.extract_value(fll, "term").split(maxsplit=2)
        if len(values) < 2:
            raise SyntaxError(f"expected format 'term: name Term [parameters]', but got '{fll}'")

        term = lib.factory_manager.term.construct(values[1])
        term.name = Op.as_identifier(values[0])
        term.update_reference(engine)
        if len(values) > 2:
            term.configure(values[2])
        return term

    def rule(self, fll: str, engine: Optional['Engine'] = None) -> Optional['Rule']:
        return Rule.create(self.extract_value(fll, "rule"), engine)

    def tnorm(self, fll: str) -> Optional['TNorm']:
        return self.component(TNorm, fll)

    def snorm(self, fll: str) -> Optional['SNorm']:
        return self.component(SNorm, fll)

    def activation(self, fll: str) -> Optional['Activation']:
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        return self.component(Activation, name, parameters)

    def defuzzifier(self, fll: str) -> Optional['Defuzzifier']:
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        return self.component(Defuzzifier, name, parameters)

    T = TypeVar('T', 'Activation', 'Defuzzifier', 'SNorm', 'TNorm')

    def component(self,
                  cls: Type['FllImporter.T'],
                  fll: str, parameters: Optional[str] = None
                  ) -> Optional['FllImporter.T']:
        from . import lib

        fll = Op.strip_comments(fll)
        if not fll or fll == "none":
            return None

        factory_attr = cls.__name__.lower()
        if not hasattr(lib.factory_manager, factory_attr):
            raise SyntaxError(f"factory manager does not contain a factory named '{factory_attr}' "
                              f"to construct objects of type '{cls}'")

        factory: ConstructionFactory[FllImporter.T] = getattr(lib.factory_manager, factory_attr)
        result = factory.construct(fll)
        if parameters and hasattr(result, "configure"):
            getattr(result, "configure")(parameters)
        return result

    def range(self, fll: str) -> Tuple[float, float]:
        values = fll.split()
        if len(values) != 2:
            raise SyntaxError(f"expected range of two values, but got {values}")
        return Op.scalar(values[0]), Op.scalar(values[1])

    def boolean(self, fll: str) -> bool:
        if fll.strip() == "true":
            return True
        if fll.strip() == "false":
            return False
        raise SyntaxError(f"expected boolean in {['true', 'false']}, but got '{fll}'")

    def extract_key_value(self, fll: str, component: Optional[str] = None) -> Tuple[str, str]:
        parts = Op.strip_comments(fll).split(":", maxsplit=1)
        if len(parts) != 2 or (component and parts[0] != component):
            key = component if component else 'key'
            raise SyntaxError(f"expected '{key}: value' definition, but found '{fll}'")
        return parts[0].strip(), parts[1].strip()

    def extract_value(self, fll: str, component: Optional[str] = None) -> str:
        return self.extract_key_value(fll, component)[1]
