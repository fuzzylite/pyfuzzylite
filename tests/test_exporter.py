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
import glob
import io
import os
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import fuzzylite as fl


class TestExporter(unittest.TestCase):

    def test_class_name(self) -> None:
        self.assertEqual(fl.Exporter().class_name, "Exporter")

    def test_to_string(self) -> None:
        with self.assertRaises(NotImplementedError):
            fl.Exporter().to_string(None)

    def test_to_file(self) -> None:
        exporter = fl.Exporter()
        exporter.to_string = MagicMock(return_value="MagicMock Test")  # type: ignore
        path = tempfile.mkstemp(text=True)[1]

        exporter.to_file(path, object())

        with io.open(path, "r") as exported:
            self.assertEqual("MagicMock Test", exported.read())

        os.remove(path)


class TestFllExporter(unittest.TestCase):

    def test_single_line_indent(self) -> None:
        engine = fl.Engine("engine", "single line export to FLL",
                           [fl.InputVariable("A", "variable A")],
                           [fl.OutputVariable("Z", "variable Z")],
                           [fl.RuleBlock("R", "rule block R")])
        self.assertEqual(
            fl.FllExporter(separator="; ", indent='\t').engine(engine),
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
            "\tactivation: none; ")

    def test_engine(self) -> None:
        engine = fl.Engine(
            name="engine",
            description="an engine",
            input_variables=[
                fl.InputVariable(
                    name="input_variable",
                    description="an input variable",
                    minimum=0, maximum=1,
                    terms=[fl.Triangle("A")])],
            output_variables=[
                fl.OutputVariable(
                    name="output_variable",
                    description="an output variable",
                    minimum=0, maximum=1,
                    terms=[fl.Triangle("A")])
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="rb",
                    description="a rule block",
                    rules=[fl.Rule.create("if a then z")])
            ]
        )
        self.assertEqual(fl.FllExporter().to_string(engine),
                         fl.FllExporter().engine(engine))
        self.assertEqual(fl.FllExporter().engine(engine), """\
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
""")

    def test_variable(self) -> None:
        variable = fl.Variable(name="variable", description="a variable",
                               minimum=0, maximum=1, terms=[fl.Triangle("A")])
        self.assertEqual(fl.FllExporter().to_string(variable),
                         fl.FllExporter().variable(variable))
        self.assertEqual(fl.FllExporter().variable(variable), """\
Variable: variable
  description: a variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan""")

    def test_input_variable(self) -> None:
        variable = fl.InputVariable(name="input_variable",
                                    description="an input variable",
                                    minimum=0, maximum=1,
                                    terms=[fl.Triangle("A")])
        self.assertEqual(fl.FllExporter().to_string(variable),
                         fl.FllExporter().input_variable(variable))
        self.assertEqual(fl.FllExporter().input_variable(variable), """\
InputVariable: input_variable
  description: an input variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan""")

    def test_output_variable(self) -> None:
        variable = fl.OutputVariable(name="output_variable",
                                     description="an output variable",
                                     minimum=0, maximum=1,
                                     terms=[fl.Triangle("A")])
        self.assertEqual(fl.FllExporter().to_string(variable),
                         fl.FllExporter().output_variable(variable))
        self.assertEqual(fl.FllExporter().output_variable(variable), """\
OutputVariable: output_variable
  description: an output variable
  enabled: true
  range: 0 1
  lock-range: false
  aggregation: none
  defuzzifier: none
  default: nan
  lock-previous: false
  term: A Triangle nan nan nan""")

    def test_rule_block(self) -> None:
        rb = fl.RuleBlock(name="rb", description="a rule block",
                          rules=[fl.Rule.create("if a then z")])
        self.assertEqual(fl.FllExporter().to_string(rb),
                         fl.FllExporter().rule_block(rb))
        self.assertEqual(fl.FllExporter().rule_block(rb), """\
RuleBlock: rb
  description: a rule block
  enabled: true
  conjunction: none
  disjunction: none
  implication: none
  activation: none
  rule: if a then z""")

    def test_term(self) -> None:
        term = fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)
        self.assertEqual(fl.FllExporter().to_string(term),
                         fl.FllExporter().term(term))
        self.assertEqual(fl.FllExporter().term(term),
                         "term: A Triangle 0.000 1.000 2.000 0.500")

    def test_rule(self) -> None:
        rule = fl.Rule.create("if a then z")
        self.assertEqual(fl.FllExporter().to_string(rule),
                         fl.FllExporter().rule(rule))
        self.assertEqual(fl.FllExporter().rule(rule),
                         "rule: if a then z")

    def test_norm(self) -> None:
        self.assertEqual(fl.FllExporter().norm(None), "none")
        norm = fl.AlgebraicProduct()
        self.assertEqual(fl.FllExporter().to_string(norm),
                         fl.FllExporter().norm(norm))
        self.assertEqual(fl.FllExporter().norm(norm), "AlgebraicProduct")

    def test_activation(self) -> None:
        self.assertEqual(fl.FllExporter().activation(None), "none")
        activation = fl.General()
        self.assertEqual(fl.FllExporter().to_string(activation),
                         fl.FllExporter().activation(activation))
        self.assertEqual(fl.FllExporter().activation(activation), "General")

    def test_defuzzifier(self) -> None:
        self.assertEqual(fl.FllExporter().defuzzifier(None), "none")
        defuzzifier = fl.Centroid()
        self.assertEqual(fl.FllExporter().to_string(defuzzifier),
                         fl.FllExporter().defuzzifier(defuzzifier))
        self.assertEqual(fl.FllExporter().defuzzifier(defuzzifier), "Centroid 100")

    def test_object(self) -> None:
        with self.assertRaisesRegex(ValueError, rf"expected a fuzzylite object, but found 'object"):
            fl.FllExporter().to_string(object())


class TestPythonExporter(unittest.TestCase):

    def test_empty_engine(self) -> None:
        engine = fl.Engine(
            name="engine",
            description="an engine")
        self.assertEqual(fl.PythonExporter().to_string(engine),
                         fl.PythonExporter().engine(engine))
        self.assertEqual(fl.PythonExporter().engine(engine), """\
import fuzzylite as fl

engine = fl.Engine(
    name="engine",
    description="an engine",
    input_variables = [],
    output_variables = [],
    rule_blocks = []""")

    def test_engine(self) -> None:
        engine = fl.Engine(
            name="engine",
            description="an engine",
            input_variables=[
                fl.InputVariable(
                    name="input_variable",
                    description="an input variable",
                    minimum=0, maximum=1,
                    terms=[fl.Triangle("A")]
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="output_variable",
                    description="an output variable",
                    minimum=0, maximum=1,
                    terms=[fl.Triangle("A")]
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="rb",
                    description="a rule block",
                    rules=[fl.Rule.create("if a then z")])
            ]
        )
        self.assertEqual(fl.PythonExporter().to_string(engine),
                         fl.PythonExporter().engine(engine))
        self.assertEqual(second=fl.PythonExporter().engine(engine), first="""\
import fuzzylite as fl

engine = fl.Engine(
    name="engine",
    description="an engine",
    input_variables = [
        fl.InputVariable(
            name="input_variable",
            description="an input variable",
            enabled=True,
            minimum=0,
            maximum=1,
            lock_range=False,
            terms=[fl.Triangle("A", nan, nan, nan)]
        )
    ],
    output_variables = [
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
    ],
    rule_blocks = [
        fl.RuleBlock(
            name="rb",
            description="a rule block",
            enabled=True,
            conjunction=None,
            disjunction=None,
            implication=None,
            activation=None,
            rules=[fl.Rule.create("if a then z")]
        )
    ]""")

    def test_input_variable(self) -> None:
        iv = fl.InputVariable(name="input_variable",
                              description="an input variable",
                              minimum=0, maximum=1,
                              terms=[fl.Triangle("A")])
        self.assertEqual(fl.PythonExporter().to_string(iv),
                         fl.PythonExporter().input_variable(iv))
        self.assertEqual(fl.PythonExporter().input_variable(iv), """\
fl.InputVariable(
    name="input_variable",
    description="an input variable",
    enabled=True,
    minimum=0,
    maximum=1,
    lock_range=False,
    terms=[fl.Triangle("A", nan, nan, nan)]
)""")
        iv.terms.append(fl.Triangle("Z"))
        self.assertEqual(fl.PythonExporter().input_variable(iv), """\
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
)""")

    def test_output_variable(self) -> None:
        ov = fl.OutputVariable(name="output_variable",
                               description="an output variable",
                               minimum=0.0, maximum=1.0,
                               terms=[fl.Triangle("A")])
        self.assertEqual(fl.PythonExporter().to_string(ov),
                         fl.PythonExporter().output_variable(ov))
        self.assertEqual(fl.PythonExporter().output_variable(ov), """\
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
)""")
        ov.terms.append(fl.Triangle("Z"))
        self.assertEqual(fl.PythonExporter().output_variable(ov), """\
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
)""")

    def test_rule_block(self) -> None:
        rb = fl.RuleBlock(name="rb", description="a rule block",
                          rules=[fl.Rule.create("if a then z")])
        self.assertEqual(fl.PythonExporter().to_string(rb),
                         fl.PythonExporter().rule_block(rb))
        self.assertEqual(fl.PythonExporter().rule_block(rb), """\
fl.RuleBlock(
    name="rb",
    description="a rule block",
    enabled=True,
    conjunction=None,
    disjunction=None,
    implication=None,
    activation=None,
    rules=[fl.Rule.create("if a then z")]
)""")
        rb.rules.append(fl.Rule.create("if b then y"))
        self.assertEqual(fl.PythonExporter().rule_block(rb), """\
fl.RuleBlock(
    name="rb",
    description="a rule block",
    enabled=True,
    conjunction=None,
    disjunction=None,
    implication=None,
    activation=None,
    rules=[
        fl.Rule.create("if a then z"),
        fl.Rule.create("if b then y")
    ]
)""")

    def test_term(self) -> None:
        term = fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)
        self.assertEqual(fl.PythonExporter().to_string(term),
                         fl.PythonExporter().term(term))
        self.assertEqual(fl.PythonExporter().term(term),
                         "fl.Triangle(\"A\", 0.000, 1.000, 2.000, 0.500)")

    def test_rule(self) -> None:
        rule = fl.Rule.create("if a then z")
        self.assertEqual(fl.PythonExporter().to_string(rule),
                         fl.PythonExporter().rule(rule))
        self.assertEqual(fl.PythonExporter().rule(rule),
                         "fl.Rule.create(\"if a then z\")")

    def test_norm(self) -> None:
        self.assertEqual(fl.PythonExporter().norm(None), "None")
        norm = fl.AlgebraicProduct()
        self.assertEqual(fl.PythonExporter().to_string(norm),
                         fl.PythonExporter().norm(norm))
        self.assertEqual(fl.PythonExporter().norm(norm),
                         "fl.AlgebraicProduct()")

    def test_activation(self) -> None:
        self.assertEqual(fl.PythonExporter().activation(None), "None")
        norm = fl.General()
        self.assertEqual(fl.PythonExporter().to_string(norm),
                         fl.PythonExporter().activation(norm))
        self.assertEqual(fl.PythonExporter().activation(norm), "fl.General()")

    def test_defuzzifier(self) -> None:
        self.assertEqual(fl.PythonExporter().defuzzifier(None), "None")
        defuzzifier = fl.Centroid()
        self.assertEqual(fl.PythonExporter().to_string(defuzzifier),
                         fl.PythonExporter().defuzzifier(defuzzifier))
        self.assertEqual(fl.PythonExporter().defuzzifier(defuzzifier),
                         "fl.Centroid(100)")

    def test_object(self) -> None:
        with self.assertRaisesRegex(ValueError, rf"expected a fuzzylite object, but found 'object"):
            fl.PythonExporter().to_string(object())


class TestExporters(unittest.TestCase):

    # @unittest.skip("Re-enable after test coverage improved independently")
    def test_exporters(self) -> None:
        import concurrent.futures
        import logging

        fl.lib.configure_logging(logging.INFO)

        fl.lib.decimals = 3

        terms = next(iter(fl.examples.__path__)) + "/terms"  # type: ignore
        files = list(glob.iglob(terms + '/*.py', recursive=True))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(TestExporters.export, files)

        self.assertEqual(fl.lib.decimals, 3)

    def test_exporter(self) -> None:
        terms = next(iter(fl.examples.__path__)) + "/terms"  # type: ignore
        TestExporters.export(terms + "/Bell.py")

    @staticmethod
    def export(path: str) -> None:
        import io
        import time
        import pathlib
        import importlib

        fl.lib.decimals = 9
        path = pathlib.Path(path)
        if path.suffix == ".fll":
            with io.open(path, 'r') as file:
                import_fll = file.read()
                engine = fl.FllImporter().from_string(import_fll)
        elif path.suffix == ".py":
            package: List[str] = []
            for parent in path.parents:
                package.append(parent.name)
                if parent.name == fl.examples.__name__:
                    break
            module = ".".join(reversed(package)) + f".{path.stem}"
            engine = importlib.import_module(module).engine
            for rb in engine.rule_blocks:
                rb.load_rules(engine)
        else:
            raise Exception(f"unknown importer of files like {path}")

        exporters = [
            fl.FllExporter(), fl.PythonExporter(),  # fl.FldExporter()
        ]

        file_name = path.stem
        for exporter in exporters:
            start = time.time()
            if isinstance(exporter, fl.FldExporter):
                exporter.to_file_from_scope(
                    Path("/tmp/fl/" + file_name + ".fld"), engine, 100_000)

            elif isinstance(exporter, fl.FllExporter):
                exporter.to_file(Path("/tmp/fl/" + file_name + ".fll"), engine)

            elif isinstance(exporter, fl.PythonExporter):
                exporter.to_file(Path("/tmp/fl/" + file_name + ".py"), engine)

            fl.lib.logger.info(str(path) + f".fld\t{time.time() - start}")


if __name__ == '__main__':
    unittest.main()
