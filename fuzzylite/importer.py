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
    """The Importer class is the abstract class for importers to configure an
    Engine and its components from different text formats.
    @todo declare methods to import specific components
    @author Juan Rada-Vilela, Ph.D.
    @see Exporter
    @since 4.0.
    """

    @property
    def class_name(self) -> str:
        """Returns the name of the importer
        @return the name of the importer.
        """
        return self.__class__.__name__

    def from_string(self, fll: str) -> Engine:
        """Imports the engine from the given text
        @param text is the string representation of the engine to import from
        @return the engine represented by the text.
        """
        raise NotImplementedError()

    def from_file(self, path: Union[Path, str]) -> Engine:
        """Imports the engine from the given file
        @param path is the full path of the file containing the engine to import from
        @return the engine represented by the file.
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open(encoding="UTF8") as fll:
            return self.from_string(fll.read())


class FllImporter(Importer):
    """The FllImporter class is an Importer that configures an Engine and its
    components utilizing the FuzzyLite Language (FLL), see
    [http://www.fuzzylite.com/fll-fld](http://www.fuzzylite.com/fll-fld) for
    more information.
    @author Juan Rada-Vilela, Ph.D.
    @see FllExporter
    @see Importer
    @since 4.0
    @todo parse methods returning respective instances from blocks of text.
    """

    T = TypeVar("T", Activation, Defuzzifier, SNorm, TNorm)

    def __init__(self, separator: str = "\n") -> None:
        """Creates an importer with the specific separator
        @param separator is the separator of the language.
        """
        self.separator = separator

    def _process(self, component: str, block: List[str], engine: Engine) -> None:
        if component == "Engine":
            for line in block:
                line = Op.strip_comments(line)

                key, value = self.extract_key_value(line)
                if key == "Engine":
                    engine.name = value
                elif key == "description":
                    engine.description = value
                else:
                    raise SyntaxError(
                        f"'{key}' is not a valid component of '{component}'"
                    )
        elif component == "InputVariable":
            input_variable = self.input_variable(self.separator.join(block), engine)
            engine.input_variables.append(input_variable)
        elif component == "OutputVariable":
            output_variable = self.output_variable(self.separator.join(block), engine)
            engine.output_variables.append(output_variable)
        elif component == "RuleBlock":
            rule_block = self.rule_block(self.separator.join(block), engine)
            engine.rule_blocks.append(rule_block)

    def from_string(self, fll: str) -> Engine:
        """Creates an engine from the FuzzyLite Language.
        @param fll is the engine in the FuzzyLite Language
        @returns the engine.
        """
        return self.engine(fll)

    def engine(self, fll: str) -> Engine:
        """Creates an engine from the FuzzyLite Language.
        @param fll is the engine in the FuzzyLite Language
        @returns the engine.
        """
        engine = Engine()
        component = ""
        block: List[str] = []

        for line in fll.split(self.separator):
            line = Op.strip_comments(line)
            if not line:
                continue
            key, _ = self.extract_key_value(line)
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

    def input_variable(
        self, fll: str, engine: Optional[Engine] = None
    ) -> InputVariable:
        """Creates an input variable from the FuzzyLite Language.
        @param fll is the input variable in the FuzzyLite Language
        @param engine is the reference engine for the input variable
        @returns the input variable.
        """
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
                raise SyntaxError(
                    f"'{key}' is not a valid component of '{iv.__class__.__name__}'"
                )
        iv.name = Op.as_identifier(iv.name)
        return iv

    def output_variable(
        self, fll: str, engine: Optional[Engine] = None
    ) -> OutputVariable:
        """Creates an output variable from the FuzzyLite Language.
        @param fll is the output variable in the FuzzyLite Language
        @param engine is the reference engine for the output variable
        @returns the output variable.
        """
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
                raise SyntaxError(
                    f"'{key}' is not a valid component of '{ov.__class__.__name__}'"
                )
        ov.name = Op.as_identifier(ov.name)
        return ov

    def rule_block(self, fll: str, engine: Optional[Engine] = None) -> RuleBlock:
        """Creates a rule block from the FuzzyLite Language.
        @param fll is the rule block in the FuzzyLite Language
        @param engine is the reference engine for the rule block
        @returns the rule block.
        """
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
                raise SyntaxError(
                    f"'{key}' is not a valid component of '{rb.__class__.__name__}'"
                )
        return rb

    def term(self, fll: str, engine: Optional[Engine] = None) -> Term:
        """Creates a term from the FuzzyLite Language.
        @param fll is the term in the FuzzyLite Language
        @param engine is the reference engine for the term
        @returns the term.
        """
        from . import lib

        values = self.extract_value(fll, "term").split(maxsplit=2)
        if len(values) < 2:
            raise SyntaxError(
                f"expected format 'term: name Term [parameters]', but got '{fll}'"
            )

        term = lib.factory_manager.term.construct(values[1])
        term.name = Op.as_identifier(values[0])
        term.update_reference(engine)
        if len(values) > 2:
            term.configure(values[2])
        return term

    def rule(self, fll: str, engine: Optional[Engine] = None) -> Optional[Rule]:
        """Creates a rule from the FuzzyLite Language.
        @param fll is the rule in the FuzzyLite Language
        @param engine is the reference engine for the rule
        @returns the rule.
        """
        return Rule.create(self.extract_value(fll, "rule"), engine)

    def tnorm(self, fll: str) -> Optional[TNorm]:
        """Creates a T-Norm from the FuzzyLite Language.
        @param fll is the T-Norm in the FuzzyLite Language
        @returns the T-Norm.
        """
        return self.component(TNorm, fll)

    def snorm(self, fll: str) -> Optional[SNorm]:
        """Creates a S-Norm from the FuzzyLite Language.
        @param fll is the S-Norm in the FuzzyLite Language
        @returns the S-Norm.
        """
        return self.component(SNorm, fll)

    def activation(self, fll: str) -> Optional[Activation]:
        """Creates an activation method from the FuzzyLite Language.
        @param fll is the activation method in the FuzzyLite Language
        @returns the activation method.
        """
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        return self.component(Activation, name, parameters)

    def defuzzifier(self, fll: str) -> Optional[Defuzzifier]:
        """Creates a defuzzifier from the FuzzyLite Language.
        @param fll is the defuzzifier in the FuzzyLite Language
        @returns the defuzzifier.
        """
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        return self.component(Defuzzifier, name, parameters)

    def component(
        self, cls: Type["FllImporter.T"], fll: str, parameters: Optional[str] = None
    ) -> Optional["FllImporter.T"]:
        """Create component from the factory.
        @param cls is the component class to create
        @param fll is the component in the FuzzyLite Language
        @param parameters is the component parameters in the FuzzyLite Language
        @returns the component or None.
        """
        from . import lib

        fll = Op.strip_comments(fll)
        if not fll or fll == "none":
            return None

        factory_attr = cls.__name__.lower()
        if not hasattr(lib.factory_manager, factory_attr):
            raise SyntaxError(
                f"factory manager does not contain a factory named '{factory_attr}' "
                f"to construct objects of type '{cls}'"
            )

        factory: ConstructionFactory[FllImporter.T] = getattr(
            lib.factory_manager, factory_attr
        )
        result = factory.construct(fll)
        if parameters and hasattr(result, "configure"):
            result.configure(parameters)
        return result

    def range(self, fll: str) -> Tuple[float, float]:
        """Gets the range from a value in the FuzzyLite Language."""
        values = fll.split()
        if len(values) != 2:
            raise SyntaxError(f"expected range of two values, but got {values}")
        return Op.scalar(values[0]), Op.scalar(values[1])

    def boolean(self, fll: str) -> bool:
        """Gets a boolean from a value in the FuzzyLite Language."""
        if fll.strip() == "true":
            return True
        if fll.strip() == "false":
            return False
        raise SyntaxError(f"expected boolean in {['true', 'false']}, but got '{fll}'")

    def extract_key_value(
        self, fll: str, component: Optional[str] = None
    ) -> Tuple[str, str]:
        """Extract 'key: value' pair from the line of text in the FuzzyLite Language
        @param fll is the line of text in the FuzzyLite Language
        @param component is the name of the specific key to extract
        @returns tuple of (key, value).
        """
        parts = Op.strip_comments(fll).split(":", maxsplit=1)
        if len(parts) != 2 or (component and parts[0] != component):
            key = component if component else "key"
            raise SyntaxError(f"expected '{key}: value' definition, but found '{fll}'")
        return parts[0].strip(), parts[1].strip()

    def extract_value(self, fll: str, component: Optional[str] = None) -> str:
        """Extract the value from the line of text in the FuzzyLite Language
        @param fll is the line of text in the FuzzyLite Language
        @param component is the name of the specific key to extract the value from
        @returns the value.
        """
        return self.extract_key_value(fll, component)[1]
