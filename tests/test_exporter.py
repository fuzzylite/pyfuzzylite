"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import io
import logging
import os
import random
import string
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import black

import fuzzylite as fl
from fuzzylite.examples import hybrid, mamdani, takagi_sugeno


class TestExporter(unittest.TestCase):
    """Test exporters."""

    def setUp(self) -> None:
        """Display the entire diff in tests."""
        self.maxDiff = None

    def test_to_file(self) -> None:
        """Test the exporter saves to file."""

        class BaseExporter(fl.Exporter):
            """Base exporter for testing."""

            def to_string(self, instance: object) -> str:
                """Content for testing."""
                return "BaseExporter.to_string(self, instance)"

        exporter = BaseExporter()
        path = tempfile.mkstemp(text=True)[1]

        exporter.to_file(path, object())

        with open(path) as exported:
            self.assertEqual("BaseExporter.to_string(self, instance)", exported.read())

        os.remove(path)


class TestFllExporter(unittest.TestCase):
    """Test the FuzzyLite Language exporter."""

    def test_single_line_indent(self) -> None:
        """Test the separator and indentation."""
        engine = fl.Engine(
            "engine",
            "single line export to FLL",
            [fl.InputVariable("A", "variable A")],
            [fl.OutputVariable("Z", "variable Z")],
            [fl.RuleBlock("R", "rule block R")],
        )
        self.assertEqual(
            fl.FllExporter(separator="; ", indent="\t").engine(engine),
            "Engine: engine; "
            "\tdescription: single line export to FLL; "
            "InputVariable: A; "
            "\tdescription: variable A; "
            "\tenabled: true; "
            "\trange: -inf inf; "
            "\tlock-range: false; "
            "OutputVariable: Z; "
            "\tdescription: variable Z; "
            "\tenabled: true; "
            "\trange: -inf inf; "
            "\tlock-range: false; "
            "\taggregation: none; "
            "\tdefuzzifier: none; "
            "\tdefault: nan; "
            "\tlock-previous: false; "
            "RuleBlock: R; "
            "\tdescription: rule block R; "
            "\tenabled: true; "
            "\tconjunction: none; "
            "\tdisjunction: none; "
            "\timplication: none; "
            "\tactivation: none; ",
        )

    def test_engine(self) -> None:
        """Test an engine is exported."""
        engine = fl.Engine(
            name="engine",
            description="an engine",
            input_variables=[
                fl.InputVariable(
                    name="input_variable",
                    description="an input variable",
                    minimum=0,
                    maximum=1,
                    terms=[fl.Triangle("A")],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="output_variable",
                    description="an output variable",
                    minimum=0,
                    maximum=1,
                    terms=[fl.Triangle("A")],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="rb",
                    description="a rule block",
                    rules=[fl.Rule.create("if a then z")],
                )
            ],
            load=False,
        )
        self.assertEqual(fl.FllExporter().to_string(engine), fl.FllExporter().engine(engine))
        self.assertEqual(
            fl.FllExporter().engine(engine),
            """\
Engine: engine
  description: an engine
InputVariable: input_variable
  description: an input variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan
OutputVariable: output_variable
  description: an output variable
  enabled: true
  range: 0 1
  lock-range: false
  aggregation: none
  defuzzifier: none
  default: nan
  lock-previous: false
  term: A Triangle nan nan nan
RuleBlock: rb
  description: a rule block
  enabled: true
  conjunction: none
  disjunction: none
  implication: none
  activation: none
  rule: if a then z
""",
        )

    def test_variable(self) -> None:
        """Test base variables are exported."""
        variable = fl.Variable(
            name="variable",
            description="a variable",
            minimum=0,
            maximum=1,
            terms=[fl.Triangle("A")],
        )
        self.assertEqual(fl.FllExporter().to_string(variable), fl.FllExporter().variable(variable))
        self.assertEqual(
            fl.FllExporter().variable(variable),
            """\
Variable: variable
  description: a variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan""",
        )

    def test_input_variable(self) -> None:
        """Test input variables are exported."""
        variable = fl.InputVariable(
            name="input_variable",
            description="an input variable",
            minimum=0,
            maximum=1,
            terms=[fl.Triangle("A")],
        )
        self.assertEqual(
            fl.FllExporter().to_string(variable),
            fl.FllExporter().input_variable(variable),
        )
        self.assertEqual(
            fl.FllExporter().input_variable(variable),
            """\
InputVariable: input_variable
  description: an input variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan""",
        )

    def test_output_variable(self) -> None:
        """Test output variables are exported."""
        variable = fl.OutputVariable(
            name="output_variable",
            description="an output variable",
            minimum=0,
            maximum=1,
            terms=[fl.Triangle("A")],
        )
        self.assertEqual(
            fl.FllExporter().to_string(variable),
            fl.FllExporter().output_variable(variable),
        )
        self.assertEqual(
            fl.FllExporter().output_variable(variable),
            """\
OutputVariable: output_variable
  description: an output variable
  enabled: true
  range: 0 1
  lock-range: false
  aggregation: none
  defuzzifier: none
  default: nan
  lock-previous: false
  term: A Triangle nan nan nan""",
        )

    def test_rule_block(self) -> None:
        """Test rule blocks are exported."""
        rb = fl.RuleBlock(
            name="rb", description="a rule block", rules=[fl.Rule.create("if a then z")]
        )
        self.assertEqual(fl.FllExporter().to_string(rb), fl.FllExporter().rule_block(rb))
        self.assertEqual(
            fl.FllExporter().rule_block(rb),
            """\
RuleBlock: rb
  description: a rule block
  enabled: true
  conjunction: none
  disjunction: none
  implication: none
  activation: none
  rule: if a then z""",
        )

    def test_term(self) -> None:
        """Test terms are exported."""
        term = fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)
        self.assertEqual(fl.FllExporter().to_string(term), fl.FllExporter().term(term))
        self.assertEqual(fl.FllExporter().term(term), "term: A Triangle 0.000 1.000 2.000 0.500")

    def test_rule(self) -> None:
        """Test rules are exported."""
        rule = fl.Rule.create("if a then z")
        self.assertEqual(fl.FllExporter().to_string(rule), fl.FllExporter().rule(rule))
        self.assertEqual(fl.FllExporter().rule(rule), "rule: if a then z")

    def test_norm(self) -> None:
        """Test norms are exported."""
        self.assertEqual(fl.FllExporter().norm(None), "none")
        norm = fl.AlgebraicProduct()
        self.assertEqual(fl.FllExporter().to_string(norm), fl.FllExporter().norm(norm))
        self.assertEqual(fl.FllExporter().norm(norm), "AlgebraicProduct")

    def test_activation(self) -> None:
        """Test activations are exported."""
        self.assertEqual(fl.FllExporter().activation(None), "none")
        activation = fl.General()
        self.assertEqual(
            fl.FllExporter().to_string(activation),
            fl.FllExporter().activation(activation),
        )
        self.assertEqual(fl.FllExporter().activation(activation), "General")

    def test_defuzzifier(self) -> None:
        """Test defuzzifiers are exported."""
        self.assertEqual(fl.FllExporter().defuzzifier(None), "none")
        defuzzifier = fl.Centroid()
        self.assertEqual(
            fl.FllExporter().to_string(defuzzifier),
            fl.FllExporter().defuzzifier(defuzzifier),
        )
        self.assertEqual(fl.FllExporter().defuzzifier(defuzzifier), "Centroid")

    def test_object(self) -> None:
        """Test a non-fuzzylite object cannot exported."""
        with self.assertRaisesRegex(
            TypeError, r"expected a fuzzylite object, but got <class 'object'>"
        ):
            fl.FllExporter().to_string(object())


class TestPythonExporter(unittest.TestCase):
    """Test Python exporter."""

    def setUp(self) -> None:
        """Display the entire diff in tests."""
        self.maxDiff = None

    def assert_that(self, instance: Any, expected: str, encapsulated: str | None = None) -> None:
        """Assert helper to compare the Python code of the instance against what is expected, plus other tests."""
        exporter = fl.PythonExporter()
        obtained = exporter.to_string(instance)
        self.assertEqual(
            black.format_str(expected, mode=black.Mode()),
            obtained,
        )

        def expected_encapsulated(return_type: str, code: str) -> str:
            return black.format_str(
                f"""\
import fuzzylite as fl
def create() -> {return_type}:
    return {code}
    """,
                mode=black.Mode(),
            )

        obtained_encapsulated = fl.PythonExporter(encapsulated=True).to_string(instance)
        if encapsulated is not None:
            self.assertEqual(
                black.format_str(encapsulated, mode=black.Mode()),
                obtained_encapsulated,
            )
        else:
            if instance is None:
                self.assertEqual(
                    expected_encapsulated(return_type="NoneType", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.Engine):
                self.assertEqual(expected, exporter.engine(instance))
                self.assertEqual(
                    expected_encapsulated(return_type="fl.Engine", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.InputVariable):
                self.assertEqual(expected, exporter.input_variable(instance))
                self.assertEqual(
                    expected_encapsulated(return_type="fl.InputVariable", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.OutputVariable):
                self.assertEqual(expected, exporter.output_variable(instance))
                self.assertEqual(
                    expected_encapsulated(return_type="fl.OutputVariable", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.RuleBlock):
                self.assertEqual(expected, exporter.rule_block(instance))
                self.assertEqual(
                    expected_encapsulated(return_type="fl.RuleBlock", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.Term):
                self.assertEqual(expected, exporter.term(instance))
                self.assertEqual(
                    expected_encapsulated(
                        return_type=f"fl.{fl.Op.class_name(instance)}", code=expected
                    ),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.Rule):
                self.assertEqual(expected, exporter.rule(instance))
                self.assertEqual(
                    expected_encapsulated(return_type="fl.Rule", code=expected),
                    obtained_encapsulated,
                )
            elif isinstance(instance, fl.Norm):
                self.assertEqual(expected, exporter.norm(instance))
                self.assertEqual(
                    expected_encapsulated(
                        return_type=f"fl.{fl.Op.class_name(instance)}", code=expected
                    ),
                    obtained_encapsulated,
                )
                self.assertEqual("None", exporter.norm(None))
            elif isinstance(instance, fl.Activation):
                self.assertEqual(expected, exporter.activation(instance))
                self.assertEqual(
                    expected_encapsulated(
                        return_type=f"fl.{fl.Op.class_name(instance)}", code=expected
                    ),
                    obtained_encapsulated,
                )
                self.assertEqual("None", exporter.activation(None))
            elif isinstance(instance, fl.Defuzzifier):
                self.assertEqual(expected, exporter.defuzzifier(instance))
                self.assertEqual(
                    expected_encapsulated(
                        return_type=f"fl.{fl.Op.class_name(instance)}", code=expected
                    ),
                    obtained_encapsulated,
                )
                self.assertEqual("None", exporter.defuzzifier(None))
            else:
                raise NotImplementedError()

    def test_missing_black_when_formatting(self) -> None:
        """Tests missing black library when formatting."""
        import sys

        black = sys.modules.get("black")
        logger = logging.getLogger("test")
        logger.error = MagicMock()  # type: ignore
        try:
            sys.modules["black"] = None  # type: ignore
            with fl.settings.context(logger=logger):
                none_formatted = fl.PythonExporter().format("None")
                self.assertEqual("None", none_formatted)
                logger.error.assert_called_with(
                    "expected `black` module to be installed, but could not be found"
                )
        finally:
            sys.modules["black"] = black  # type: ignore

    def test_black_format_options(self) -> None:
        """Test overriding black format options."""
        engine = mamdani.simple_dimmer.SimpleDimmer().engine
        obtained = fl.PythonExporter().format(repr(engine), line_length=1000)
        expected = (
            'fl.Engine(name="SimpleDimmer", '
            'input_variables=[fl.InputVariable(name="Ambient", minimum=0.0, maximum=1.0, '
            'lock_range=False, terms=[fl.Triangle("DARK", 0.0, 0.25, 0.5), '
            'fl.Triangle("MEDIUM", 0.25, 0.5, 0.75), fl.Triangle("BRIGHT", 0.5, 0.75, '
            '1.0)])], output_variables=[fl.OutputVariable(name="Power", minimum=0.0, '
            "maximum=1.0, lock_range=False, lock_previous=False, default_value=fl.nan, "
            "aggregation=fl.Maximum(), defuzzifier=fl.Centroid(), "
            'terms=[fl.Triangle("LOW", 0.0, 0.25, 0.5), fl.Triangle("MEDIUM", 0.25, 0.5, '
            '0.75), fl.Triangle("HIGH", 0.5, 0.75, 1.0)])], '
            'rule_blocks=[fl.RuleBlock(name="", conjunction=None, disjunction=None, '
            'implication=fl.Minimum(), activation=fl.General(), rules=[fl.Rule.create("if '
            'Ambient is DARK then Power is HIGH"), fl.Rule.create("if Ambient is MEDIUM '
            'then Power is MEDIUM"), fl.Rule.create("if Ambient is BRIGHT then Power is '
            'LOW")])])\n'
        )
        self.assertEqual(expected, obtained)

    def test_none_export(self) -> None:
        """Tests export of None values, like in Norm, Activation, or Defuzzifier."""
        self.assert_that(None, "None\n")

    def test_empty_engine(self) -> None:
        """Test an empty engine is exported."""
        engine = fl.Engine(name="Choo Choo", description="My Choo-Choo engine")
        self.assert_that(
            engine,
            expected="""\
fl.Engine(
    name="Choo Choo",
    description="My Choo-Choo engine",
    input_variables=[],
    output_variables=[],
    rule_blocks=[],
)
""",
            encapsulated="""\
import fuzzylite as fl

class ChooChoo:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="Choo Choo",
            description="My Choo-Choo engine",
            input_variables=[],
            output_variables=[],
            rule_blocks=[],
        )
""",
        )

    def test_engine(self) -> None:
        """Test a basic engine is exported."""
        engine = mamdani.simple_dimmer.SimpleDimmer().engine
        self.assertEqual(fl.PythonExporter().to_string(engine), fl.PythonExporter().engine(engine))
        constructor = """\
fl.Engine(
    name="SimpleDimmer",
    input_variables=[
        fl.InputVariable(
            name="Ambient",
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=[
                fl.Triangle("DARK", 0.0, 0.25, 0.5),
                fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
                fl.Triangle("BRIGHT", 0.5, 0.75, 1.0),
            ],
        )
    ],
    output_variables=[
        fl.OutputVariable(
            name="Power",
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            lock_previous=False,
            default_value=fl.nan,
            aggregation=fl.Maximum(),
            defuzzifier=fl.Centroid(),
            terms=[
                fl.Triangle("LOW", 0.0, 0.25, 0.5),
                fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
                fl.Triangle("HIGH", 0.5, 0.75, 1.0),
            ],
        )
    ],
    rule_blocks=[
        fl.RuleBlock(
            name="",
            conjunction=None,
            disjunction=None,
            implication=fl.Minimum(),
            activation=fl.General(),
            rules=[
                fl.Rule.create("if Ambient is DARK then Power is HIGH"),
                fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM"),
                fl.Rule.create("if Ambient is BRIGHT then Power is LOW"),
            ],
        )
    ],
)"""
        self.assert_that(
            engine,
            constructor,
            encapsulated=f"""\
import fuzzylite as fl

class SimpleDimmer:
    def __init__(self) -> None:
        self.engine = {constructor}
        """,
        )

    def test_input_variable(self) -> None:
        """Test input variables are exported."""
        input_variable = mamdani.simple_dimmer.SimpleDimmer().engine.input_variable(0)
        self.assert_that(
            input_variable,
            expected="""\
fl.InputVariable(
    name="Ambient",
    minimum=0.0,
    maximum=1.0,
    lock_range=False,
    terms=[
        fl.Triangle("DARK", 0.0, 0.25, 0.5),
        fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
        fl.Triangle("BRIGHT", 0.5, 0.75, 1.0),
    ],
)
""",
        )
        input_variable.enabled = False
        input_variable.description = "Description"
        self.assert_that(
            input_variable,
            expected="""\
fl.InputVariable(
    name="Ambient",
    description="Description",
    enabled=False,
    minimum=0.0,
    maximum=1.0,
    lock_range=False,
    terms=[
        fl.Triangle("DARK", 0.0, 0.25, 0.5),
        fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
        fl.Triangle("BRIGHT", 0.5, 0.75, 1.0),
    ],
)
""",
        )

    def test_output_variable(self) -> None:
        """Test output variables are exported."""
        output_variable = mamdani.simple_dimmer.SimpleDimmer().engine.output_variable(0)
        self.assert_that(
            output_variable,
            """\
fl.OutputVariable(
    name="Power",
    minimum=0.0,
    maximum=1.0,
    lock_range=False,
    lock_previous=False,
    default_value=fl.nan,
    aggregation=fl.Maximum(),
    defuzzifier=fl.Centroid(),
    terms=[
        fl.Triangle("LOW", 0.0, 0.25, 0.5),
        fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
        fl.Triangle("HIGH", 0.5, 0.75, 1.0),
    ],
)
""",
        )

    def test_rule_block(self) -> None:
        """Test rule blocks are exported."""
        rule_block = mamdani.simple_dimmer.SimpleDimmer().engine.rule_block(0)
        self.assert_that(
            rule_block,
            """\
fl.RuleBlock(
    name="",
    conjunction=None,
    disjunction=None,
    implication=fl.Minimum(),
    activation=fl.General(),
    rules=[
        fl.Rule.create("if Ambient is DARK then Power is HIGH"),
        fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM"),
        fl.Rule.create("if Ambient is BRIGHT then Power is LOW"),
    ],
)
""",
        )
        rule_block.description = "Description"
        rule_block.enabled = False
        self.assert_that(
            rule_block,
            """\
fl.RuleBlock(
    name="",
    description="Description",
    enabled=False,
    conjunction=None,
    disjunction=None,
    implication=fl.Minimum(),
    activation=fl.General(),
    rules=[
        fl.Rule.create("if Ambient is DARK then Power is HIGH"),
        fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM"),
        fl.Rule.create("if Ambient is BRIGHT then Power is LOW"),
    ],
)
""",
        )

    def test_term(self) -> None:
        """Test terms are exported."""
        self.assert_that(
            fl.Triangle("A", 0.0, 1.0, 2.0, 0.5),
            'fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)\n',
        )

    def test_rule(self) -> None:
        """Test rules are exported."""
        self.assert_that(
            fl.Rule.create("if a then z"),
            'fl.Rule.create("if a then z")\n',
        )

    def test_norm(self) -> None:
        """Test norms are exported."""
        self.assert_that(fl.AlgebraicProduct(), "fl.AlgebraicProduct()\n")
        self.assert_that(fl.AlgebraicSum(), "fl.AlgebraicSum()\n")

    def test_activation(self) -> None:
        """Test activation methods are exported."""
        self.assert_that(fl.General(), "fl.General()\n")
        self.assert_that(fl.First(), "fl.First(rules=1, threshold=0.0)\n")

    def test_defuzzifier(self) -> None:
        """Test defuzzifiers are exported."""
        self.assert_that(fl.Centroid(), "fl.Centroid()\n")
        self.assert_that(fl.Centroid(resolution=100), "fl.Centroid(resolution=100)\n")
        self.assert_that(fl.WeightedSum(), "fl.WeightedSum()\n")
        self.assert_that(fl.WeightedSum("TakagiSugeno"), 'fl.WeightedSum(type="TakagiSugeno")\n')

    def test_object(self) -> None:
        """Test objects are exported with `repr`."""
        from black.parsing import InvalidInput

        with self.assertRaises(ValueError) as error:
            fl.PythonExporter().to_string(object())
        self.assertEqual(error.exception.__class__, InvalidInput)
        self.assertTrue(str(error.exception).startswith("Cannot parse: 1:0: <object object at "))


class TestFldExporter(unittest.TestCase):
    """Test FuzzyLite Dataset exporter."""

    def test_default_constructor(self) -> None:
        """Test the default constructor."""
        exporter = fl.FldExporter()
        self.assertEqual(" ", exporter.separator)
        self.assertTrue(exporter.headers)
        self.assertTrue(exporter.input_values)
        self.assertTrue(exporter.output_values)

    def test_header(self) -> None:
        """Test output header."""
        engine = fl.Engine(
            input_variables=[fl.InputVariable("A"), fl.InputVariable("B")],
            output_variables=[
                fl.OutputVariable("X"),
                fl.OutputVariable("Y"),
                fl.OutputVariable("Z"),
            ],
        )
        self.assertEqual("A B X Y Z", fl.FldExporter().header(engine))
        self.assertEqual("A\tB\tX\tY\tZ", fl.FldExporter(separator="\t").header(engine))

    def test_write(self) -> None:
        """Test writing lines."""
        # Empty write
        writer = io.StringIO()
        fl.FldExporter().write(fl.Engine(), writer, fl.array([]))
        self.assertEqual("", writer.getvalue())

        # No values to write
        writer = io.StringIO()
        fl.FldExporter(input_values=False, output_values=False).write(
            fl.Engine(), writer, fl.array([])
        )
        self.assertEqual("", writer.getvalue())

        # Not enough values
        with self.assertRaises(ValueError) as error:
            fl.FldExporter().write(
                fl.Engine(input_variables=[fl.InputVariable()]),
                writer,
                fl.array([]),
            )
        self.assertEqual(
            "expected 1 input values (one per input variable), but got 0 instead",
            str(error.exception),
        )

        # input and output values
        writer = io.StringIO()
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        fl.FldExporter(input_values=True, output_values=True, headers=False).write(
            engine,
            writer,
            fl.array([0.25]),
        )
        self.assertEqual("0.250 0.750\n", writer.getvalue())

        # input values only
        writer = io.StringIO()
        fl.FldExporter(input_values=True, output_values=False, headers=False).write(
            engine,
            writer,
            fl.array([0.25]),
        )
        self.assertEqual("0.250\n", writer.getvalue())

        # output values only
        writer = io.StringIO()
        fl.FldExporter(input_values=False, output_values=True, headers=False).write(
            engine,
            writer,
            fl.array([0.25]),
        )
        self.assertEqual("0.750\n", writer.getvalue())

        # no values
        writer = io.StringIO()
        engine.process = MagicMock()  # type: ignore
        fl.FldExporter(input_values=False, output_values=False, headers=False).write(
            engine,
            writer,
            fl.array([0.25]),
        )
        self.assertEqual("", writer.getvalue())
        engine.process.assert_called_once()

        # active variables
        writer = io.StringIO()
        engine.input_variables[0].value = 0.250
        test_variable = fl.InputVariable("test")
        test_variable.value = 0.0
        engine.input_variables.append(test_variable)

        fl.FldExporter(headers=False).write(engine, writer, fl.array([fl.inf, fl.inf]))
        self.assertEqual("inf inf nan\n", writer.getvalue())

    def test_write_from_reader_empty_engine_empty(self) -> None:
        """Test exporting an empty engine."""
        engine = fl.Engine()

        writer = io.StringIO()
        fl.FldExporter().write_from_reader(engine, writer, io.StringIO())
        self.assertEqual("", writer.getvalue())

        writer = io.StringIO()
        fl.FldExporter(headers=False).write_from_reader(engine, writer, io.StringIO())
        self.assertEqual("", writer.getvalue())

    def test_write_from_reader_empty_engine_not_empty(self) -> None:
        """Test exporter can read an empty FLD and write it again."""
        engine = fl.Engine(
            input_variables=[fl.InputVariable("Input")],
            output_variables=[fl.OutputVariable("Output")],
        )

        writer = io.StringIO()
        with self.assertRaises(ValueError) as error:
            fl.FldExporter().write_from_reader(engine, writer, io.StringIO())
        self.assertEqual(
            "expected 1 input values (one per input variable), but got 0 instead",
            str(error.exception),
        )

    def test_write_from_reader_empty_or_commented(self) -> None:
        """Test exporter ignores comments."""
        reader = """\

# commented line 0.000
        """
        writer = io.StringIO()
        fl.FldExporter().write_from_reader(fl.Engine(), writer, io.StringIO(reader))
        self.assertEqual("", writer.getvalue())

    def test_write_from_reader(self) -> None:
        """Test exporter can read an FLD and export it again."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        reader = """\
Ambient Power
0.000000000 nan
0.499023438 0.501459144
#0.500000000 0.500000000

0.509765625 0.486065263
0.510742188 0.484743908
"""
        # Fails with headers
        with self.assertRaisesRegex(ValueError, r"could not convert string to float: 'Ambient'"):
            fl.FldExporter().write_from_reader(
                engine, io.StringIO(), io.StringIO(reader), skip_lines=0
            )

        # Success skipping headers
        writer = io.StringIO()
        fl.FldExporter().write_from_reader(engine, writer, io.StringIO(reader), skip_lines=1)
        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.499 0.501
0.510 0.486
0.511 0.485\n""",
            writer.getvalue(),
        )

    def test_to_file_from_reader(self) -> None:
        """Test exporter can read file and export it using default decimals."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        reader = """\
    Ambient Power
    0.000000000 nan
    0.499023438 0.501459144
    #0.500000000 0.500000000

    0.509765625 0.486065263
    0.510742188 0.484743908
    """

        file_name = (
            "file-" + "".join(random.choice(string.ascii_lowercase) for _ in range(5)) + ".fld"
        )
        fl.FldExporter().to_file_from_reader(
            Path(file_name), engine, io.StringIO(reader), skip_lines=1
        )
        self.assertTrue(Path(file_name).exists())
        obtained = Path(file_name).read_text()
        Path(file_name).unlink()
        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.499 0.501
0.510 0.486
0.511 0.485\n""",
            obtained,
        )

    def test_to_string_from_reader(self) -> None:
        """Test exporter can read from a reader and export to a string."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        reader = """\
        Ambient Power
        0.000000000 nan
        0.499023438 0.501459144
        #0.500000000 0.500000000

        0.509765625 0.486065263
        0.510742188 0.484743908
        """

        obtained = fl.FldExporter().to_string_from_reader(engine, io.StringIO(reader), skip_lines=1)
        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.499 0.501
0.510 0.486
0.511 0.485\n""",
            obtained,
        )

    def test_write_from_scope_each_variable_1(self) -> None:
        """Test exporter can write from a specific scope of specific variables."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=4,
            scope=fl.FldExporter.ScopeOfValues.EachVariable,
            active_variables=set(engine.input_variables),
        )

        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.333 0.659
0.667 0.341
1.000 nan
""",
            writer.getvalue(),
        )

    def test_write_from_scope_all_variables_on_empty_engine(self) -> None:
        """Test the exporter cannot export an empty engine."""
        engine = fl.Engine()
        writer = io.StringIO()
        with self.assertRaisesRegex(ValueError, "expected input variables in engine, but got none"):
            fl.FldExporter().write_from_scope(
                engine,
                writer,
                values=16,
                scope=fl.FldExporter.ScopeOfValues.AllVariables,
                active_variables=set(engine.input_variables),
            )

    def test_write_from_scope_all_variables_1(self) -> None:
        """Test the exporter can export the values of all variables in the AllVariables scope."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=16,
            scope=fl.FldExporter.ScopeOfValues.AllVariables,
            active_variables=set(engine.input_variables),
        )

        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.067 0.750
0.133 0.750
0.200 0.750
0.267 0.727
0.333 0.659
0.400 0.605
0.467 0.543
0.533 0.457
0.600 0.395
0.667 0.341
0.733 0.273
0.800 0.250
0.867 0.250
0.933 0.250
1.000 nan
""",
            writer.getvalue(),
        )

    def test_write_from_scope_each_variable_2(self) -> None:
        """Test the exporter can export the values of all variables in each variable scope."""
        engine = fl.FllImporter().from_string(str(hybrid.tipper.Tipper().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=4,
            scope=fl.FldExporter.ScopeOfValues.EachVariable,
            active_variables=set(engine.input_variables),
        )

        self.assertEqual(
            """\
service food mTip tsTip
0.000 0.000 5.000 5.000
0.000 3.333 7.754 6.538
0.000 6.667 12.950 10.882
0.000 10.000 13.571 11.667
3.333 0.000 8.571 7.500
3.333 3.333 10.110 8.673
3.333 6.667 13.769 12.925
3.333 10.000 14.367 13.889
6.667 0.000 12.895 11.000
6.667 3.333 13.205 12.797
6.667 6.667 17.987 20.636
6.667 10.000 21.154 22.778
10.000 0.000 13.571 11.667
10.000 3.333 13.710 13.889
10.000 6.667 20.217 22.778
10.000 10.000 25.000 25.000
""",
            writer.getvalue(),
        )

    def test_write_from_scope_all_variables_2(self) -> None:
        """Test the exporter can export the values of all variables in the AllVariables scope."""
        engine = fl.FllImporter().from_string(str(hybrid.tipper.Tipper().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=16,
            scope=fl.FldExporter.ScopeOfValues.AllVariables,
            active_variables=set(engine.input_variables),
        )

        self.assertEqual(
            """\
service food mTip tsTip
0.000 0.000 5.000 5.000
0.000 3.333 7.754 6.538
0.000 6.667 12.950 10.882
0.000 10.000 13.571 11.667
3.333 0.000 8.571 7.500
3.333 3.333 10.110 8.673
3.333 6.667 13.769 12.925
3.333 10.000 14.367 13.889
6.667 0.000 12.895 11.000
6.667 3.333 13.205 12.797
6.667 6.667 17.987 20.636
6.667 10.000 21.154 22.778
10.000 0.000 13.571 11.667
10.000 3.333 13.710 13.889
10.000 6.667 20.217 22.778
10.000 10.000 25.000 25.000
""",
            writer.getvalue(),
        )

    def test_write_from_scope_each_variable_one_inactive(self) -> None:
        """Test the exporter can export the values of only active variables in EachVariable scope."""
        engine = fl.FllImporter().from_string(str(hybrid.tipper.Tipper().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=4,
            scope=fl.FldExporter.ScopeOfValues.EachVariable,
            active_variables={engine.input_variables[0]},
        )

        self.assertEqual(
            """\
service food mTip tsTip
0.000 nan nan nan
3.333 nan 15.000 15.000
6.667 nan 15.000 15.000
10.000 nan nan nan
""",
            writer.getvalue(),
        )

    def test_write_from_scope_all_variables_one_inactive(self) -> None:
        """Test the exporter can export the values of only active variables in AllVariables scope."""
        engine = fl.FllImporter().from_string(str(hybrid.tipper.Tipper().engine))
        writer = io.StringIO()
        fl.FldExporter().write_from_scope(
            engine,
            writer,
            values=16,
            scope=fl.FldExporter.ScopeOfValues.AllVariables,
            active_variables={engine.input_variables[0]},
        )

        self.assertEqual(
            """\
service food mTip tsTip
0.000 nan nan nan
3.333 nan 15.000 15.000
6.667 nan 15.000 15.000
10.000 nan nan nan
""",
            writer.getvalue(),
        )

    def test_to_file_from_scope(self) -> None:
        """Test the exporter can export the values to a file from EachVariable scope."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))

        file_name = (
            "file-" + "".join(random.choice(string.ascii_lowercase) for _ in range(5)) + ".fld"
        )

        fl.FldExporter().to_file_from_scope(
            Path(file_name),
            engine,
            values=4,
            scope=fl.FldExporter.ScopeOfValues.EachVariable,
            active_variables=set(engine.input_variables),
        )

        self.assertTrue(Path(file_name).exists())
        obtained = Path(file_name).read_text()
        Path(file_name).unlink()

        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.333 0.659
0.667 0.341
1.000 nan
""",
            obtained,
        )

    def test_to_string_from_scope(self) -> None:
        """Test the exporter can export the values to a file from AllVariables scope."""
        engine = fl.FllImporter().from_string(str(mamdani.simple_dimmer.SimpleDimmer().engine))

        obtained = fl.FldExporter().to_string_from_scope(
            engine,
            values=4,
            scope=fl.FldExporter.ScopeOfValues.EachVariable,
            active_variables=set(engine.input_variables),
        )

        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.333 0.659
0.667 0.341
1.000 nan
""",
            obtained,
        )

    def test_to_string(self) -> None:
        """Test the exporter can export to string."""
        with self.assertRaisesRegex(ValueError, "expected an Engine, but got InputVariable"):
            fl.FldExporter().to_string(fl.InputVariable())

        engine = fl.FllImporter().from_string(
            str(takagi_sugeno.simple_dimmer.SimpleDimmer().engine)
        )

        obtained = fl.FldExporter().to_string(engine)
        self.assertEqual(1025 + 1, len(obtained.split("\n")))

        self.assertEqual(
            """\
Ambient Power
0.000 nan
0.001 0.750
0.002 0.750
0.003 0.750""",
            "\n".join(obtained.split("\n")[:5]),
        )


class TestExporters(unittest.TestCase):
    """Test exporters for every example."""

    @unittest.skip("Re-enable after test coverage improved independently")
    def test_exporters(self) -> None:
        """Test every FLL example can be exported."""
        import concurrent.futures

        with fl.settings.context(decimals=9):
            modules = [module for module in fl.Op.glob_examples("module")]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                threads = [executor.submit(TestExporters.export, module) for module in modules]
            concurrent.futures.wait(threads, return_when=concurrent.futures.FIRST_EXCEPTION)
            for t in threads:
                print(t.result())

        self.assertEqual(fl.settings.decimals, 3)

    @unittest.skip("Testing export single thread")
    def test_exporter(self) -> None:
        """Test exporting an arbitrary FLL file."""
        from fuzzylite.examples.terms import bell

        with fl.settings.context(decimals=3):
            TestExporters.export(bell)

    @staticmethod
    def export(example: ModuleType) -> None:
        """Given an FLL file or Python example, export to FLL, Python and FLD."""
        import time

        import numpy as np

        np.seterr(invalid="ignore", divide="ignore")

        engine, *_ = fl.Op.glob_examples("engine", module=example)
        exporters = [
            # fl.FllExporter(),
            # fl.PythonExporter(encapsulated=True),
            fl.FldExporter(),
        ]

        file_name = Path(f"{example.__file__}").stem
        package = Path(f"{example.__file__}").parent.relative_to(*fl.__path__)
        for exporter in exporters:
            start = time.time()
            target_path = Path("/tmp/fl/") / package
            target_path.mkdir(parents=True, exist_ok=True)
            fl.settings.logger.info(str(package) + f" -> {fl.Op.class_name(exporter)}")
            if isinstance(exporter, fl.FldExporter):
                exporter.to_file_from_scope(target_path / (file_name + ".fld"), engine, 1024)

            elif isinstance(exporter, fl.FllExporter):
                exporter.to_file(target_path / (file_name + ".fll"), engine)

            elif isinstance(exporter, fl.PythonExporter):
                exporter.to_file(target_path / (file_name + ".py"), engine)

            fl.settings.logger.info(
                str(package) + f" -> {fl.Op.class_name(exporter)}\t{time.time() - start}"
            )


if __name__ == "__main__":
    unittest.main()
