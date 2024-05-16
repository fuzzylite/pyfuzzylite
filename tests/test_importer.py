"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import glob
import os
import re
import tempfile
import unittest
from typing import cast

import fuzzylite as fl

BELL_FLL = """\
Engine: Bell
  description: obstacle avoidance for self-driving cars
InputVariable: obstacle
  description: location of obstacle relative to vehicle
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: left Triangle 0.000 0.333 0.666
  term: right Triangle 0.333 0.666 1.000
OutputVariable: steer
  description: direction to steer the vehicle to
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 100
  default: nan
  lock-previous: false
  term: left Bell 0.333 0.167 3.000
  term: right Bell 0.666 0.167 3.000
RuleBlock: steer_away
  description: steer away from obstacles
  enabled: true
  conjunction: none
  disjunction: none
  implication: Minimum
  activation: General
  rule: if obstacle is left then steer is right
  rule: if obstacle is right then steer is left
"""


class TestImporter(unittest.TestCase):
    """Test the importer class."""

    def test_from_file(self) -> None:
        """Test the importer can read files."""

        class BaseImporter(fl.Importer):
            """Base importer class for testing."""

            def from_string(self, fll: str) -> fl.Engine:
                """Returns an empty engine with the FLL as a description."""
                return fl.Engine(description=fll)

        path = tempfile.mkstemp(text=True)[1]
        with open(path, "w") as file:
            file.write(BELL_FLL)

        engine = BaseImporter().from_file(path)

        self.assertEqual(BELL_FLL, engine.description)

        os.remove(path)


class TestFllImporter(unittest.TestCase):
    """Test the FuzzyLite Language importer."""

    def test_from_string(self) -> None:
        """Test the importer can import from a string."""
        engine = fl.FllImporter().from_string(BELL_FLL)
        self.assertEqual(BELL_FLL, str(engine))

    def test_engine(self) -> None:
        """Test a simple engine can be imported."""
        engine = """\
Engine: Bell

  description: obstacle avoidance for self-driving cars
"""
        self.assertEqual(engine.replace("\n\n", "\n"), str(fl.FllImporter().engine(engine)))

    def test_input_variable(self) -> None:
        """Test input variables can be imported."""
        iv = """\
InputVariable: obstacle
  description: location of obstacle relative to vehicle
  enabled: true
  range: 0.000 1.000
  lock-range: false

  term: left Triangle 0.000 0.333 0.666
  term: right Triangle 0.333 0.666 1.000"""
        self.assertEqual(iv.replace("\n\n", "\n"), str(fl.FllImporter().input_variable(iv)))

    def test_output_variable(self) -> None:
        """Test output variables can be imported."""
        ov = """\
OutputVariable: steer
  description: direction to steer the vehicle to
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 100
  default: nan
  lock-previous: false

  term: left Bell 0.333 0.167 3.000
  term: right Bell 0.666 0.167 3.000"""
        self.assertEqual(ov.replace("\n\n", "\n"), str(fl.FllImporter().output_variable(ov)))

    def test_rule_block(self) -> None:
        """Test rule blocks can be imported."""
        rb = """\
RuleBlock: steer_away
  description: steer away from obstacles
  enabled: true
  conjunction: none
  disjunction: none
  implication: Minimum
  activation: General

  rule: if obstacle is left then steer is right
  rule: if obstacle is right then steer is left"""
        self.assertEqual(rb.replace("\n\n", "\n"), str(fl.FllImporter().rule_block(rb)))

    def test_term(self) -> None:
        """Test terms can be imported."""
        term = "term: left Triangle 0.000 0.333 0.666"
        self.assertEqual(term, str(fl.FllImporter().term(term)))
        engine = fl.Engine()
        term = "term: function Function 1 + 1"
        self.assertEqual(term, str(fl.FllImporter().term(term)))
        self.assertEqual(None, cast(fl.Function, fl.FllImporter().term(term)).engine)
        self.assertEqual(engine, cast(fl.Function, fl.FllImporter().term(term, engine)).engine)

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("expected format 'term: name Term [parameters]', but got 'term: name'"),
        ):
            fl.FllImporter().term("term: name")

    def test_rule(self) -> None:
        """Test rules can be imported."""
        rule = "rule: if obstacle is left then steer is right"
        self.assertEqual(rule, str(fl.FllImporter().rule(rule)))
        self.assertFalse(cast(fl.Rule, fl.FllImporter().rule(rule)).is_loaded())

        engine = fl.FllImporter().from_string(BELL_FLL)
        self.assertEqual(rule, str(fl.FllImporter().rule(rule, engine)))
        self.assertTrue(cast(fl.Rule, fl.FllImporter().rule(rule, engine)).is_loaded())

    def test_tnorm(self) -> None:
        """Test T-Norms can be imported."""
        tnorm = "AlgebraicProduct"
        self.assertEqual(tnorm, str(fl.FllImporter().tnorm(tnorm)))

        self.assertEqual(None, fl.FllImporter().tnorm(""))
        self.assertEqual(None, fl.FllImporter().tnorm("none"))

        with self.assertRaisesRegex(
            ValueError,
            re.escape("constructor of 'AlgebraicSum' not found in TNormFactory"),
        ):
            fl.FllImporter().tnorm("AlgebraicSum")

    def test_snorm(self) -> None:
        """Test S-Norms can be imported."""
        snorm = "AlgebraicSum"
        self.assertEqual(snorm, str(fl.FllImporter().snorm(snorm)))

        self.assertEqual(None, fl.FllImporter().snorm(""))
        self.assertEqual(None, fl.FllImporter().snorm("none"))

        with self.assertRaisesRegex(
            ValueError,
            re.escape("constructor of 'AlgebraicProduct' not found in SNormFactory"),
        ):
            fl.FllImporter().snorm("AlgebraicProduct")

    def test_activation(self) -> None:
        """Test activation methods can be imported."""
        activation = "General"
        self.assertEqual(activation, str(fl.FllImporter().activation(activation)))
        activation = "Highest 2"
        self.assertEqual(activation, str(fl.FllImporter().activation(activation)))

        self.assertEqual(None, fl.FllImporter().activation("none"))

        with self.assertRaisesRegex(
            ValueError,
            re.escape("constructor of 'Invalid' not found in ActivationFactory"),
        ):
            fl.FllImporter().activation("Invalid")

    def test_defuzzifier(self) -> None:
        """Test defuzzifiers can be imported."""
        defuzzifier = "Centroid"
        self.assertEqual(defuzzifier, str(fl.FllImporter().defuzzifier(defuzzifier)))
        defuzzifier = "WeightedAverage TakagiSugeno"
        self.assertEqual(defuzzifier, str(fl.FllImporter().defuzzifier(defuzzifier)))

        self.assertEqual(None, fl.FllImporter().defuzzifier("none"))

        with self.assertRaisesRegex(
            ValueError,
            re.escape("constructor of 'Invalid' not found in DefuzzifierFactory"),
        ):
            fl.FllImporter().defuzzifier("Invalid")

    def test_range(self) -> None:
        """Test range values are parsed correctly."""
        self.assertEqual((1.0, 1.0), fl.FllImporter().range("1.0000 1.0000"))
        self.assertEqual((-fl.inf, fl.inf), fl.FllImporter().range("-inf\tinf"))
        self.assertEqual(str((fl.nan, fl.nan)), str(fl.FllImporter().range("-nan nan")))
        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("expected range of two values, but got ['1', '2', '3']"),
        ):
            fl.FllImporter().range("1 2 3")

    def test_boolean(self) -> None:
        """Test boolean values can be parsed."""
        self.assertEqual(True, fl.FllImporter().boolean("true"))
        self.assertEqual(True, fl.FllImporter().boolean("  true  "))
        self.assertEqual(False, fl.FllImporter().boolean("false"))
        self.assertEqual(False, fl.FllImporter().boolean("  false  "))

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("expected boolean in ['true', 'false'], but got 'True'"),
        ):
            fl.FllImporter().boolean("True")
        with self.assertRaisesRegex(
            SyntaxError,
            re.escape(r"expected boolean in ['true', 'false'], but got 'False'"),
        ):
            fl.FllImporter().boolean("False")

    def test_extract_key_value(self) -> None:
        """Test correct extraction of keys and values."""
        self.assertEqual(("Key", "value"), fl.FllImporter().extract_key_value("Key: value"))
        self.assertEqual(("Key", "value"), fl.FllImporter().extract_key_value("Key : value"))
        self.assertEqual(
            ("Key", "value1 : value2"),
            fl.FllImporter().extract_key_value("Key: value1 : value2", "Key"),
        )
        self.assertEqual(
            ("Key", "value1 : value2 : value 3"),
            fl.FllImporter().extract_key_value("Key: value1 : value2 : value 3", "Key"),
        )

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("expected 'key: value' definition, but found 'key value1 value2'"),
        ):
            fl.FllImporter().extract_key_value("key value1 value2")

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape(
                "expected 'DESCRIPTION: value' definition, "
                "but found 'description: value1 value2'"
            ),
        ):
            fl.FllImporter().extract_key_value("description: value1 value2", "DESCRIPTION")

    def test_extract_value(self) -> None:
        """Test correct extraction of values from a key-value."""
        self.assertEqual("value", fl.FllImporter().extract_value("Key: value"))
        self.assertEqual("value", fl.FllImporter().extract_value("Key : value"))
        self.assertEqual(
            "value1 : value2",
            fl.FllImporter().extract_value("Key: value1 : value2", "Key"),
        )
        self.assertEqual(
            "value1 : value2 : value 3",
            fl.FllImporter().extract_value("Key: value1 : value2 : value 3", "Key"),
        )

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("expected 'key: value' definition, but found 'key value1 value2'"),
        ):
            fl.FllImporter().extract_value("key value1 value2")

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape(
                "expected 'DESCRIPTION: value' definition, "
                "but found 'description: value1 value2'"
            ),
        ):
            fl.FllImporter().extract_value("description: value1 value2", "DESCRIPTION")

    def test_invalid_components(self) -> None:
        """Test errors on invalid FuzzyLite Language components."""
        with self.assertRaisesRegex(
            SyntaxError, re.escape("'invalid' is not a valid component of 'Engine'")
        ):
            fl.FllImporter().engine("""Engine: name\n  invalid: component""")

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("'invalid' is not a valid component of 'InputVariable'"),
        ):
            fl.FllImporter().input_variable("""InputVariable: name\n  invalid: component""")

        with self.assertRaisesRegex(
            SyntaxError,
            re.escape("'invalid' is not a valid component of 'OutputVariable'"),
        ):
            fl.FllImporter().output_variable("""OutputVariable: name\n  invalid: component""")

        with self.assertRaisesRegex(
            SyntaxError, re.escape("'invalid' is not a valid component of 'RuleBlock'")
        ):
            fl.FllImporter().rule_block("""RuleBlock: name\n  invalid: component""")


class TestFllImporterBatch(unittest.TestCase):
    """Test batch imports."""

    @unittest.skip("Re-enable after test coverage improved independently")
    def test_import_examples(self) -> None:
        """Test all the examples can be imported correctly and exported to their initial form."""
        self.maxDiff = None  # show all differences
        importer = fl.FllImporter()
        exporter = fl.FllExporter()
        fl.settings.decimals = 9
        import logging

        fl.settings.logger.setLevel(logging.INFO)
        import fuzzylite.examples.terms

        terms = next(iter(fuzzylite.examples.terms.__path__))
        counter = 0
        for fll_file in glob.iglob(terms + "/*.fll", recursive=True):
            counter += 1
            with open(fll_file) as file:
                fl.settings.logger.info(fll_file)
                fll_from_string = file.read()
                engine = importer.from_string(fll_from_string)
                export_fll = exporter.to_string(engine)
                self.assertEqual(fll_from_string, export_fll)
        self.assertEqual(20, counter)
        fl.settings.decimals = 3


if __name__ == "__main__":
    unittest.main()
