"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = ["Importer", "FllImporter"]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload

from .activation import Activation
from .defuzzifier import Defuzzifier
from .engine import Engine
from .library import representation, settings, to_float
from .norm import SNorm, TNorm
from .operation import Op
from .rule import Rule, RuleBlock
from .term import Term
from .variable import InputVariable, OutputVariable


class Importer(ABC):
    """Abstract class for importers to configure an engine and its components from different text formats.

    info: related
        - [fuzzylite.exporter.Exporter][]
    """

    # TODO: declare methods to import specific components

    def __str__(self) -> str:
        """Return the class name of the importer.

        Returns:
            class name of the importer.
        """
        return Op.class_name(self)

    def __repr__(self) -> str:
        """Return the Python code to construct the importer.

        Returns:
            Python code to construct the importer.
        """
        return representation.as_constructor(self)

    @abstractmethod
    def from_string(self, text: str, /) -> Engine:
        """Return the engine described in the text representation.

        Args:
            text: representation of the engine to import

        Returns:
            engine described in the text representation
        """

    def from_file(self, path: Path | str, /) -> Engine:
        """Read from the file the text representation of an engine.

        Args:
            path: file path to import engine

        Returns:
            engine represented in the file
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open(encoding="utf-8") as fll:
            return self.from_string(fll.read())


class FllImporter(Importer):
    """Import an engine and its components described using the FuzzyLite Language.

    info: related
        - [fuzzylite.importer.Importer][]
        - [fuzzylite.exporter.FllExporter][]
        - [FuzzyLite Language (FLL)](https://fuzzylite.com/fll-fld/)
    """

    # todo: parse methods returning respective instances from blocks of text.

    def __init__(self, separator: str = "\n") -> None:
        """Constructor.

        Args:
            separator: separation between components of the FuzzyLite Language.
        """
        self.separator = separator

    def _process(self, component: str, block: list[str], engine: Engine) -> None:
        """Process the main components of the FuzzyLite Language, namely Engine, InputVariable, OutputVariable and RuleBlock.

        Args:
            component: one of `Engine`, `InputVariable`, `OutputVariable` and `RuleBlock`
            block: list of lines that make up the component
            engine: engine to add the component to
        """
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

    def from_string(self, text: str, /) -> Engine:
        """Return the engine describe using the FuzzyLite Language.

        Args:
            text: engine described using the FuzzyLite Language

        Returns:
            engine described using the FuzzyLite Language
        """
        return self.engine(text)

    def engine(self, fll: str) -> Engine:
        """Return the engine describe using the FuzzyLite Language.

        Args:
            fll: engine described using the FuzzyLite Language

        Returns:
            engine described using the FuzzyLite Language
        """
        engine = Engine()
        component = ""
        block: list[str] = []

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

    def input_variable(self, fll: str, engine: Engine | None = None) -> InputVariable:
        """Return the input variable described using the FuzzyLite Language.

        Args:
            fll: input variable described using the FuzzyLite Language
            engine: engine to update the reference of the terms in the variable

        Returns:
            input variable described using the FuzzyLite Language
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
                raise SyntaxError(f"'{key}' is not a valid component of '{iv.__class__.__name__}'")
        iv.name = Op.as_identifier(iv.name)
        return iv

    def output_variable(self, fll: str, engine: Engine | None = None) -> OutputVariable:
        """Return the output variable described using the FuzzyLite Language.

        Args:
            fll: output variable described using the FuzzyLite Language
            engine: engine to update the reference of the terms in the variable

        Returns:
            output variable described using the FuzzyLite Language
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
                ov.default_value = to_float(value)
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

    def rule_block(self, fll: str, engine: Engine | None = None) -> RuleBlock:
        """Return the rule block described using the FuzzyLite Language.

        Args:
            fll: rule block described using the FuzzyLite Language
            engine: engine to use for loading the rules

        Returns:
            rule block described using the FuzzyLite Language
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
                raise SyntaxError(f"'{key}' is not a valid component of '{rb.__class__.__name__}'")
        return rb

    def term(self, fll: str, engine: Engine | None = None) -> Term:
        """Return the term described using the FuzzyLite Language.

        Args:
            fll: term described using the FuzzyLite Language
            engine: engine to update the reference of the term

        Returns:
            term described using the FuzzyLite Language
        """
        values = self.extract_value(fll, "term").split(maxsplit=2)
        if len(values) < 2:
            raise SyntaxError(f"expected format 'term: name Term [parameters]', but got '{fll}'")

        term = settings.factory_manager.term.construct(values[1], name=Op.as_identifier(values[0]))
        if len(values) > 2:
            term.configure(values[2])
        term.update_reference(engine)
        return term

    def rule(self, fll: str, engine: Engine | None = None) -> Rule | None:
        """Return the rule described using the FuzzyLite Language.

        Args:
            fll: rule described using the FuzzyLite Language
            engine: engine to load the rule

        Returns:
            rule described using the FuzzyLite Language
        """
        return Rule.create(self.extract_value(fll, "rule"), engine)

    def tnorm(self, fll: str) -> TNorm | None:
        """Return the TNorm described using the FuzzyLite Language.

        Args:
            fll: TNorm described using the FuzzyLite Language

        Returns:
            TNorm described using the FuzzyLite Language
        """
        if not fll or fll == "none":
            return None
        return settings.factory_manager.tnorm.construct(fll)

    def snorm(self, fll: str) -> SNorm | None:
        """Return the SNorm described using the FuzzyLite Language.

        Args:
            fll: SNorm described using the FuzzyLite Language

        Returns:
            SNorm described using the FuzzyLite Language
        """
        if not fll or fll == "none":
            return None
        return settings.factory_manager.snorm.construct(fll)

    def activation(self, fll: str) -> Activation | None:
        """Return the activation method described using the FuzzyLite Language.

        Args:
            fll: activation method described using the FuzzyLite Language

        Returns:
            activation method described using the FuzzyLite Language
        """
        if not fll or fll == "none":
            return None
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        result = settings.factory_manager.activation.construct(name)
        if parameters:
            result.configure(parameters)
        return result

    def defuzzifier(self, fll: str) -> Defuzzifier | None:
        """Return the defuzzifier described using the FuzzyLite Language.

        Args:
            fll: defuzzifier described using the FuzzyLite Language

        Returns:
            defuzzifier described using the FuzzyLite Language
        """
        if not fll or fll == "none":
            return None
        values = fll.split(maxsplit=1)
        name = values[0]
        parameters = values[1] if len(values) > 1 else None
        result = settings.factory_manager.defuzzifier.construct(name)
        if parameters:
            result.configure(parameters)
        return result

    @overload
    def component(self, cls: type[Activation], fll: str) -> Activation | None: ...

    @overload
    def component(self, cls: type[Defuzzifier], fll: str) -> Defuzzifier | None: ...

    @overload
    def component(self, cls: type[SNorm], fll: str) -> SNorm | None: ...

    @overload
    def component(self, cls: type[TNorm], fll: str) -> TNorm | None: ...

    def component(
        self,
        cls: type[Activation | Defuzzifier | TNorm | SNorm],
        fll: str,
    ) -> Activation | Defuzzifier | TNorm | SNorm | None:
        """Return the component described using the FuzzyLite Language.

        Args:
            cls: class of the component to import
            fll: component described using the FuzzyLite Language

        Returns:
            component described using the FuzzyLite Language
        """
        if issubclass(cls, Activation):
            return self.activation(fll)
        if issubclass(cls, Defuzzifier):
            return self.defuzzifier(fll)
        if issubclass(cls, SNorm):
            return self.snorm(fll)
        if issubclass(cls, TNorm):
            return self.tnorm(fll)
        else:
            raise TypeError(
                f"expected {Activation}, {Defuzzifier}, {SNorm} or {TNorm}, but got {cls}"
            )

    def range(self, fll: str) -> tuple[float, float]:
        """Returns the values of a range described using the FuzzyLite Language.

        Args:
            fll: range of values described using the FuzzyLite Language (eg, `0.0 1.0`)

        Returns:
              range of values described using the FuzzyLite Language
        """
        values = fll.split()
        if len(values) != 2:
            raise SyntaxError(f"expected range of two values, but got {values}")
        return to_float(values[0]), to_float(values[1])

    def boolean(self, fll: str) -> bool:
        """Returns a boolean value described using the FuzzyLite Language.

        Args:
            fll: `true` or `false`

        Returns:
              boolean value described using the FuzzyLite Language.
        """
        if fll.strip() == "true":
            return True
        if fll.strip() == "false":
            return False
        raise SyntaxError(f"expected boolean in {['true', 'false']}, but got '{fll}'")

    def extract_key_value(self, fll: str, component: str | None = None) -> tuple[str, str]:
        """Return key-value pair described using the FuzzyLite Language.

        Args:
            fll: key-value pair in the form `key: value`
            component: name of the key to extract

        Returns:
            tuple of `(key, value)`
        """
        parts = Op.strip_comments(fll).split(":", maxsplit=1)
        if len(parts) != 2 or (component and parts[0] != component):
            key = component if component else "key"
            raise SyntaxError(f"expected '{key}: value' definition, but found '{fll}'")
        return parts[0].strip(), parts[1].strip()

    def extract_value(self, fll: str, component: str | None = None) -> str:
        """Return value from the `key: value` pair described using the FuzzyLite Language.

        Args:
            fll: key-value pair in the form `key: value`
            component: name of the key to extract

        Returns:
            value from the `key: value` pair described using the FuzzyLite Language
        """
        return self.extract_key_value(fll, component)[1]
