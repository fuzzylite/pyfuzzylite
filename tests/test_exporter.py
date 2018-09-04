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
import unittest
from pathlib import Path

import fuzzylite as fl


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
            "\tactivation: none")

    def test_variable(self) -> None:
        self.assertEqual(fl.FllExporter().variable(
            fl.Variable("variable", "a variable", 0, 1, [fl.Triangle("A")])),
            """\
Variable: variable
  description: a variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan\
""")

    def test_input_variable(self) -> None:
        self.assertEqual(fl.FllExporter().input_variable(
            fl.InputVariable("input_variable", "an input variable", 0, 1, [fl.Triangle("A")])),
            """\
InputVariable: input_variable
  description: an input variable
  enabled: true
  range: 0 1
  lock-range: false
  term: A Triangle nan nan nan\
""")

    def test_output_variable(self) -> None:
        self.assertEqual(fl.FllExporter().output_variable(
            fl.OutputVariable("output_variable", "an output variable", 0, 1, [fl.Triangle("A")])),
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
  term: A Triangle nan nan nan\
""")

    def test_rule_block(self) -> None:
        self.assertEqual(fl.FllExporter().rule_block(
            fl.RuleBlock("rb", "a ruleblock", [fl.Rule.parse("if a then z")])),
            """\
RuleBlock: rb
  description: a ruleblock
  enabled: true
  conjunction: none
  disjunction: none
  implication: none
  activation: none
  rule: if a then z\
""")

    def test_term(self) -> None:
        self.assertEqual(fl.FllExporter().term(fl.Triangle("A", 0.0, 1.0, 2.0, 0.5)),
                         "term: A Triangle 0.000 1.000 2.000 0.500")

    def test_rule(self) -> None:
        self.assertEqual(fl.FllExporter().rule(fl.Rule.parse("if a then z")),
                         "rule: if a then z")

    def test_norm(self) -> None:
        self.assertEqual(fl.FllExporter().norm(None), "none")
        self.assertEqual(fl.FllExporter().norm(fl.AlgebraicProduct()), "AlgebraicProduct")
        self.assertEqual(fl.FllExporter().norm(fl.AlgebraicSum()), "AlgebraicSum")

    def test_activation(self) -> None:
        self.assertEqual(fl.FllExporter().activation(None), "none")
        self.assertEqual(fl.FllExporter().activation(fl.General()), "General")

    def test_defuzzifier(self) -> None:
        self.assertEqual(fl.FllExporter().defuzzifier(None), "none")
        self.assertEqual(fl.FllExporter().defuzzifier(fl.Centroid()), "Centroid 100")


class TestPythonExporter(unittest.TestCase):
    @unittest.skip("")
    def test_export(self) -> None:
        pass


class TestFldExporter(unittest.TestCase):
    @unittest.skip("")
    def test_from_scope(self) -> None:
        import logging
        fl.lib.configure_logging(logging.INFO)

        files = list(glob.iglob('examples/terms/**/*.fll', recursive=True))
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(TestFldExporter.export, files)
        print(f"fl.lib.decimals: {fl.lib.decimals}")

    @staticmethod
    def export(path: str) -> None:
        import io
        fl.lib.decimals = 9
        importer = fl.FllImporter()
        fll_exporter = fl.FllExporter()
        fld_exporter = fl.FldExporter()

        import time
        with io.open(path, 'r') as file:
            import_fll = "".join(file.readlines())
            engine = importer.from_string(import_fll)
            file_name = file.name[file.name.rfind('/'):file.name.rfind('.')]
            fll_exporter.to_file(Path("/tmp/fl/" + file_name + ".fll"), engine)
            start = time.time()
            fld_exporter.to_file_from_scope(Path("/tmp/fl/" + file_name + ".fld"), engine, 100_000)
            fl.lib.logger.info(str(path) + f".fld\t{time.time() - start}")


if __name__ == '__main__':
    unittest.main()
