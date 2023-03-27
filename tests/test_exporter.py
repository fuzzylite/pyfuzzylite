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
import io
import os
import random
import string
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import fuzzylite as fl
from fuzzylite.examples.mamdani import SimpleDimmer


class TestExporter(unittest.TestCase):
    """Test exporters."""

    def setUp(self) -> None:
        """Display the entire diff in tests."""
        self.maxDiff = None

    def test_class_name(self) -> None:
        """Tests the base name."""
        self.assertEqual(fl.Exporter().class_name, "Exporter")

    def test_to_string(self) -> None:
        """Test the base method."""
        with self.assertRaises(NotImplementedError):
            fl.Exporter().to_string(None)

    def test_to_file(self) -> None:
        """Test the exporter saves to file."""
        exporter = fl.Exporter()
        exporter.to_string = MagicMock(return_value="MagicMock Test")  # type: ignore
        path = tempfile.mkstemp(text=True)[1]

        exporter.to_file(path, object())

        with open(path) as exported:
            self.assertEqual("MagicMock Test", exported.read())

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
        )
        self.assertEqual(
            fl.FllExporter().to_string(engine), fl.FllExporter().engine(engine)
        )
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
        self.assertEqual(
            fl.FllExporter().to_string(variable), fl.FllExporter().variable(variable)
        )
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
        self.assertEqual(
            fl.FllExporter().to_string(rb), fl.FllExporter().rule_block(rb)
        )
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
        self.assertEqual(
            fl.FllExporter().term(term), "term: A Triangle 0.000 1.000 2.000 0.500"
        )

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
        self.assertEqual(fl.FllExporter().defuzzifier(defuzzifier), "Centroid 100")

    def test_object(self) -> None:
        """Test a non-fuzzylite object cannot exported."""
        with self.assertRaisesRegex(
            ValueError, r"expected a fuzzylite object, but found 'object'"
        ):
            fl.FllExporter().to_string(object())


class TestPythonExporter(unittest.TestCase):
    """Test Python exporter."""

    def setUp(self) -> None:
        """Display the entire diff in tests."""
        self.maxDiff = None

    def test_empty_engine(self) -> None:
        """Test an empty engine is exported."""
        engine = fl.Engine(name="engine", description="an engine")
        self.assertEqual(
            fl.PythonExporter().to_string(engine), fl.PythonExporter().engine(engine)
        )
        self.assertEqual(
            second=fl.PythonExporter().engine(engine),
            first="""\
import fuzzylite as fl

engine = fl.Engine(
    name="engine",
    description="an engine"
)
engine.input_variables = []
engine.output_variables = []
engine.rule_blocks = []
""",
        )

    def test_engine(self) -> None:
        """Test a basic engine is exported."""
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
        )
        self.assertEqual(
            fl.PythonExporter().to_string(engine), fl.PythonExporter().engine(engine)
        )
        self.assertEqual(
            second=fl.PythonExporter().engine(engine),
            first="""\
import fuzzylite as fl

engine = fl.Engine(
    name="engine",
    description="an engine"
)
engine.input_variables = [
    fl.InputVariable(
        name="input_variable",
        description="an input variable",
        enabled=True,
        minimum=0,
        maximum=1,
        lock_range=False,
        terms=[fl.Triangle("A", nan, nan, nan)]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="output_variable",
        description="an output variable",
        enabled=True,
        minimum=0,
        maximum=1,
        lock_range=False,
        aggregation=None,
        defuzzifier=None,
        lock_previous=False,
        terms=[fl.Triangle("A", nan, nan, nan)]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="rb",
        description="a rule block",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=None,
        activation=None,
        rules=[fl.Rule.create("if a then z", engine)]
    )
]
""",
        )

    def test_input_variable(self) -> None:
        """Test input variables are exported."""
        iv = fl.InputVariable(
            name="input_variable",
            description="an input variable",
            minimum=0,
            maximum=1,
            terms=[fl.Triangle("A")],
        )
        self.assertEqual(
            fl.PythonExporter().to_string(iv), fl.PythonExporter().input_variable(iv)
        )
        self.assertEqual(
            fl.PythonExporter().input_variable(iv),
            """\
fl.InputVariable(
    name="input_variable",
    description="an input variable",
    enabled=True,
    minimum=0,
    maximum=1,
    lock_range=False,
    terms=[fl.Triangle("A", nan, nan, nan)]
)""",
        )
        iv.terms.append(fl.Triangle("Z"))
        self.assertEqual(
            fl.PythonExporter().input_variable(iv),
            """\
fl.InputVariable(
    name="input_variable",
    description="an input variable",
    enabled=True,
    minimum=0,
    maximum=1,
    lock_range=False,
    terms=[
        fl.Triangle("A", nan, nan, nan),
        fl.Triangle("Z", nan, nan, nan)
    ]
)""",
        )

    def test_output_variable(self) -> None:
        """Test output variables are exported."""
        ov = fl.OutputVariable(
            name="output_variable",
            description="an output variable",
            minimum=0.0,
            maximum=1.0,
            terms=[fl.Triangle("A")],
        )
        self.assertEqual(
            fl.PythonExporter().to_string(ov), fl.PythonExporter().output_variable(ov)
        )
        self.assertEqual(
            fl.PythonExporter().output_variable(ov),
            """\
fl.OutputVariable(
    name="output_variable",
    description="an output variable",
    enabled=True,
    minimum=0.000,
    maximum=1.000,
    lock_range=False,
    aggregation=None,
    defuzzifier=None,
    lock_previous=False,
    terms=[fl.Triangle("A", nan, nan, nan)]
)""",
        )
        ov.terms.append(fl.Triangle("Z"))
        self.assertEqual(
            fl.PythonExporter().output_variable(ov),
            """\
fl.OutputVariable(
    name="output_variable",
    description="an output variable",
    enabled=True,
    minimum=0.000,
    maximum=1.000,
    lock_range=False,
    aggregation=None,
    defuzzifier=None,
    lock_previous=False,
    terms=[
        fl.Triangle("A", nan, nan, nan),
        fl.Triangle("Z", nan, nan, nan)
    ]
)""",
        )

    def test_rule_block(self) -> None:
        """Test rule blocks are exported."""
        rb = fl.RuleBlock(
            name="rb", description="a rule block", rules=[fl.Rule.create("if a then z")]
        )
        self.assertEqual(
            fl.PythonExporter().to_string(rb), fl.PythonExporter().rule_block(rb)
        )
        self.assertEqual(
            fl.PythonExporter().rule_block(rb),
            """\
fl.RuleBlock(
    name="rb",
    description="a rule block",
    enabled=True,
    conjunction=None,
    disjunction=None,
    implication=None,
    activation=None,
    rules=[fl.Rule.create("if a then z", engine)]
)""",
        )
        rb.rules.append(fl.Rule.create("if b then y"))
        self.assertEqual(
            fl.PythonExporter().rule_block(rb),
            """\
fl.RuleBlock(
    name="rb",
    description="a rule block",
    enabled=True,
    conjunction=None,
    disjunction=None,
    implication=None,
    activation=None,
    rules=[
        fl.Rule.create("if a then z", engine),
        fl.Rule.create("if b then y", engine)
    ]
)""",
        )

    def test_term(self) -> None:
        """Test terms are exported."""
        term: fl.Term = fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)
        self.assertEqual(
            fl.PythonExporter().to_string(term), fl.PythonExporter().term(term)
        )
        self.assertEqual(
            fl.PythonExporter().term(term),
            'fl.Triangle("A", 0.000, 1.000, 2.000, 0.500)',
        )

        term = fl.Discrete("B", [0.0, 0.0, 0.5, 1.0, 1.0, 0.0])
        self.assertEqual(
            fl.PythonExporter().to_string(term), fl.PythonExporter().term(term)
        )
        self.assertEqual(
            fl.PythonExporter().term(term),
            'fl.Discrete("B", [0.000, 0.000, 0.500, 1.000, 1.000, 0.000])',
        )

        term = fl.Function("C", "x + 1")
        self.assertEqual(
            fl.PythonExporter().to_string(term), fl.PythonExporter().term(term)
        )
        self.assertEqual(
            fl.PythonExporter().term(term), 'fl.Function.create("C", "x + 1", engine)'
        )

        term = fl.Linear("D", [0.0, 1.0, 2.0, -fl.inf, fl.inf])
        self.assertEqual(
            fl.PythonExporter().to_string(term), fl.PythonExporter().term(term)
        )
        self.assertEqual(
            fl.PythonExporter().term(term),
            'fl.Linear("D", [0.000, 1.000, 2.000, -fl.inf, fl.inf], engine)',
        )

    def test_rule(self) -> None:
        """Test rules are exported."""
        rule = fl.Rule.create("if a then z")
        self.assertEqual(
            fl.PythonExporter().to_string(rule), fl.PythonExporter().rule(rule)
        )
        self.assertEqual(
            fl.PythonExporter().rule(rule), 'fl.Rule.create("if a then z", engine)'
        )

    def test_norm(self) -> None:
        """Test norms are exported."""
        self.assertEqual(fl.PythonExporter().norm(None), "None")
        norm = fl.AlgebraicProduct()
        self.assertEqual(
            fl.PythonExporter().to_string(norm), fl.PythonExporter().norm(norm)
        )
        self.assertEqual(fl.PythonExporter().norm(norm), "fl.AlgebraicProduct()")

    def test_activation(self) -> None:
        """Test activation methods are exported."""
        self.assertEqual(fl.PythonExporter().activation(None), "None")
        norm = fl.General()
        self.assertEqual(
            fl.PythonExporter().to_string(norm), fl.PythonExporter().activation(norm)
        )
        self.assertEqual(fl.PythonExporter().activation(norm), "fl.General()")

    def test_defuzzifier(self) -> None:
        """Test defuzzifiers are exported."""
        self.assertEqual(fl.PythonExporter().defuzzifier(None), "None")

        defuzzifier: fl.Defuzzifier = fl.Centroid()
        self.assertEqual(
            fl.PythonExporter().to_string(defuzzifier),
            fl.PythonExporter().defuzzifier(defuzzifier),
        )
        self.assertEqual(
            fl.PythonExporter().defuzzifier(defuzzifier), "fl.Centroid(100)"
        )

        defuzzifier = fl.WeightedAverage()
        self.assertEqual(
            fl.PythonExporter().to_string(defuzzifier),
            fl.PythonExporter().defuzzifier(defuzzifier),
        )
        self.assertEqual(
            fl.PythonExporter().defuzzifier(defuzzifier),
            'fl.WeightedAverage("Automatic")',
        )

    def test_object(self) -> None:
        """Test non-fuzzylite objects cannot be exported."""
        with self.assertRaisesRegex(
            ValueError, "expected a fuzzylite object, but found 'object'"
        ):
            fl.PythonExporter().to_string(object())


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
        fl.FldExporter().write(fl.Engine(), writer, [], set())
        self.assertEqual("\n", writer.getvalue())

        # Not enough values
        with self.assertRaisesRegex(ValueError, "not enough input values"):
            fl.FldExporter().write(
                fl.Engine(input_variables=[fl.InputVariable()]), writer, [], set()
            )

        # input and output values
        writer = io.StringIO()
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
        fl.FldExporter(input_values=True, output_values=True).write(
            engine, writer, [0.25], set(engine.input_variables)
        )
        self.assertEqual("0.250 0.750\n", writer.getvalue())

        # input values only
        writer = io.StringIO()
        fl.FldExporter(input_values=True, output_values=False).write(
            engine, writer, [0.25], set(engine.input_variables)
        )
        self.assertEqual("0.250\n", writer.getvalue())

        # output values only
        writer = io.StringIO()
        fl.FldExporter(input_values=False, output_values=True).write(
            engine, writer, [0.25], set(engine.input_variables)
        )
        self.assertEqual("0.750\n", writer.getvalue())

        # no values
        writer = io.StringIO()
        engine.process = MagicMock()  # type: ignore
        fl.FldExporter(input_values=False, output_values=False).write(
            engine, writer, [0.25], set(engine.input_variables)
        )
        self.assertEqual("\n", writer.getvalue())
        engine.process.assert_called_once()

        # active variables
        writer = io.StringIO()
        engine.input_variables[0].value = 0.250
        test_variable = fl.InputVariable("test")
        test_variable.value = 0.0
        engine.input_variables.append(test_variable)

        fl.FldExporter().write(engine, writer, [fl.inf, fl.inf], {test_variable})
        self.assertEqual("0.250 inf 0.750\n", writer.getvalue())

    def test_write_from_reader_empty_engine_empty(self) -> None:
        """Test exporting an empty engine."""
        engine = fl.Engine()

        writer = io.StringIO()
        fl.FldExporter().write_from_reader(engine, writer, io.StringIO())
        self.assertEqual("\n", writer.getvalue())

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
        fl.FldExporter().write_from_reader(engine, writer, io.StringIO())
        self.assertEqual("Input Output\n", writer.getvalue())

        writer = io.StringIO()
        fl.FldExporter(headers=False).write_from_reader(engine, writer, io.StringIO())
        self.assertEqual("", writer.getvalue())

    def test_write_from_reader_empty_or_commented(self) -> None:
        """Test exporter ignores comments."""
        reader = """\

# commented line 0.000
            """
        writer = io.StringIO()
        fl.FldExporter().write_from_reader(fl.Engine(), writer, io.StringIO(reader))
        self.assertEqual("\n", writer.getvalue())

    def test_write_from_reader(self) -> None:
        """Test exporter can read an FLD and export it again."""
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
        reader = """\
Ambient Power
0.000000000 nan
0.499023438 0.501459144
#0.500000000 0.500000000

0.509765625 0.486065263
0.510742188 0.484743908
"""
        # Fails with headers
        with self.assertRaisesRegex(
            ValueError, r"could not convert string to float: 'Ambient'"
        ):
            fl.FldExporter().write_from_reader(
                engine, io.StringIO(), io.StringIO(reader), skip_lines=0
            )

        # Success skipping headers
        writer = io.StringIO()
        fl.FldExporter().write_from_reader(
            engine, writer, io.StringIO(reader), skip_lines=1
        )
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
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
        reader = """\
        Ambient Power
        0.000000000 nan
        0.499023438 0.501459144
        #0.500000000 0.500000000

        0.509765625 0.486065263
        0.510742188 0.484743908
        """

        file_name = (
            "file-"
            + "".join(random.choice(string.ascii_lowercase) for _ in range(5))
            + ".fld"
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
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
        reader = """\
            Ambient Power
            0.000000000 nan
            0.499023438 0.501459144
            #0.500000000 0.500000000

            0.509765625 0.486065263
            0.510742188 0.484743908
            """

        obtained = fl.FldExporter().to_string_from_reader(
            engine, io.StringIO(reader), skip_lines=1
        )
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
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
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
        with self.assertRaisesRegex(
            ValueError, "expected input variables in engine, but got none"
        ):
            fl.FldExporter().write_from_scope(
                engine,
                writer,
                values=16,
                scope=fl.FldExporter.ScopeOfValues.AllVariables,
                active_variables=set(engine.input_variables),
            )

    def test_write_from_scope_all_variables_1(self) -> None:
        """Test the exporter can export the values of all variables in the AllVariables scope."""
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))
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
        from fuzzylite.examples.hybrid import tipper

        engine = fl.FllImporter().from_string(str(tipper.engine))
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
0.000 0.000 4.999 5.000
0.000 3.333 7.756 6.538
0.000 6.667 12.949 10.882
0.000 10.000 13.571 11.667
3.333 0.000 8.569 7.500
3.333 3.333 10.110 8.673
3.333 6.667 13.770 12.925
3.333 10.000 14.368 13.889
6.667 0.000 12.895 11.000
6.667 3.333 13.204 12.797
6.667 6.667 17.986 20.636
6.667 10.000 21.156 22.778
10.000 0.000 13.571 11.667
10.000 3.333 13.709 13.889
10.000 6.667 20.216 22.778
10.000 10.000 25.001 25.000
""",
            writer.getvalue(),
        )

    def test_write_from_scope_all_variables_2(self) -> None:
        """Test the exporter can export the values of all variables in the AllVariables scope."""
        from fuzzylite.examples.hybrid import tipper

        engine = fl.FllImporter().from_string(str(tipper.engine))
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
0.000 0.000 4.999 5.000
0.000 3.333 7.756 6.538
0.000 6.667 12.949 10.882
0.000 10.000 13.571 11.667
3.333 0.000 8.569 7.500
3.333 3.333 10.110 8.673
3.333 6.667 13.770 12.925
3.333 10.000 14.368 13.889
6.667 0.000 12.895 11.000
6.667 3.333 13.204 12.797
6.667 6.667 17.986 20.636
6.667 10.000 21.156 22.778
10.000 0.000 13.571 11.667
10.000 3.333 13.709 13.889
10.000 6.667 20.216 22.778
10.000 10.000 25.001 25.000
""",
            writer.getvalue(),
        )

    def test_write_from_scope_each_variable_one_inactive(self) -> None:
        """Test the exporter can export the values of only active variables in EachVariable scope."""
        from fuzzylite.examples.hybrid import tipper

        engine = fl.FllImporter().from_string(str(tipper.engine))
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
        from fuzzylite.examples.hybrid import tipper

        engine = fl.FllImporter().from_string(str(tipper.engine))
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
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))

        file_name = (
            "file-"
            + "".join(random.choice(string.ascii_lowercase) for _ in range(5))
            + ".fld"
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
        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))

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
        with self.assertRaisesRegex(
            ValueError, "expected an Engine, but got InputVariable"
        ):
            fl.FldExporter().to_string(fl.InputVariable())

        from fuzzylite.examples.takagi_sugeno import SimpleDimmer

        engine = fl.FllImporter().from_string(str(SimpleDimmer.engine))

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
        import logging
        import pathlib

        fl.lib.configure_logging(logging.INFO)

        fl.lib.decimals = 3
        import numpy as np

        np.seterr(divide="ignore", invalid="ignore")
        fl.lib.floating_point_type = np.float64  # type: ignore

        path = "/tmp/source/"
        examples = pathlib.Path(path)
        files = [str(example) for example in examples.rglob("*.fll")]
        print(files)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            threads = [executor.submit(TestExporters.export, file) for file in files]
        concurrent.futures.wait(threads, return_when=concurrent.futures.FIRST_EXCEPTION)
        for t in threads:
            print(t.result())

        self.assertEqual(fl.lib.decimals, 3)

    @unittest.skip("Testing export single thread")
    def test_exporter(self) -> None:
        """Test exporting an arbitrary FLL file."""
        import numpy as np

        np.seterr(divide="ignore", invalid="ignore")
        fl.lib.floating_point_type = np.float64  # type: ignore

        examples = "/tmp/source/takagi_sugeno/"
        TestExporters.export(examples + "/approximation.fll")

    @staticmethod
    def export(file_path: str) -> None:
        """Given an FLL file or Python example, export to FLL, Python and FLD."""
        import importlib
        import pathlib
        import time

        import numpy as np

        np.seterr(divide="ignore", invalid="ignore")
        fl.lib.floating_point_type = np.float64  # type: ignore

        path = pathlib.Path(file_path)
        if path.suffix == ".fll":
            with open(path) as file:
                import_fll = file.read()
                engine = fl.FllImporter().from_string(import_fll)
        elif path.suffix == ".py":
            package: List[str] = []
            for parent in path.parents:
                package.append(parent.name)
                if parent.name == "fuzzylite":
                    break
            module = ".".join(reversed(package)) + f".{path.stem}"
            engine = importlib.import_module(module).engine
        else:
            raise Exception(f"unknown importer of files like {path}")

        exporters = [fl.FllExporter(), fl.PythonExporter(), fl.FldExporter()]

        file_name = path.stem
        for exporter in exporters:
            start = time.time()
            target_path = Path("/tmp/fl/") / path.parent.parent.stem / path.parent.stem
            target_path.mkdir(parents=True, exist_ok=True)
            fl.lib.decimals = 3
            fl.lib.logger.info(str(path) + f" -> {exporter.class_name}")
            if isinstance(exporter, fl.FldExporter):
                fl.lib.decimals = 9
                exporter.to_file_from_scope(
                    target_path / (file_name + ".fld"), engine, 1024
                )

            elif isinstance(exporter, fl.FllExporter):
                exporter.to_file(target_path / (file_name + ".fll"), engine)

            elif isinstance(exporter, fl.PythonExporter):
                exporter.to_file(target_path / (file_name + ".py"), engine)

            fl.lib.logger.info(
                str(path) + f" -> {exporter.class_name}\t{time.time() - start}"
            )


if __name__ == "__main__":
    unittest.main()
