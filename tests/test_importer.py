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

import fuzzylite as fl


class TestFllImporter(unittest.TestCase):

    def test_from_file(self) -> None:
        expected = """\
Engine: Bell
InputVariable: obstacle
  enabled: true
  range: 0.000000000 1.000000000
  lock-range: false
  term: left Triangle 0.000000000 0.333000000 0.666000000
  term: right Triangle 0.333000000 0.666000000 1.000000000
OutputVariable: steer
  enabled: true
  range: 0.000000000 1.000000000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 100
  default: nan
  lock-previous: false
  term: left Bell 0.333000000 0.166500000 3.000000000
  term: right Bell 0.666500000 0.166750000 3.000000000
RuleBlock: 
  enabled: true
  conjunction: none
  disjunction: none
  implication: Minimum
  activation: General
  rule: if obstacle is left then steer is right
  rule: if obstacle is right then steer is left\
"""
        fl.lib.decimals = 9
        engine = fl.importer.FllImporter().from_file("examples/terms/Bell.fll")
        self.assertEqual(expected, fl.exporter.FllExporter().to_string(engine))

        from pathlib import Path
        engine = fl.importer.FllImporter().from_file(Path("examples/terms/Bell.fll"))
        self.assertEqual(expected, fl.exporter.FllExporter().to_string(engine))
        fl.lib.decimals = 3

    def test_import_examples(self) -> None:
        self.maxDiff = None  # show all differences
        importer = fl.FllImporter()
        exporter = fl.FllExporter()
        fl.lib.decimals = 9
        import logging
        fl.lib.logger.setLevel(logging.INFO)
        examples = 0
        for fll_file in glob.iglob('examples/terms/*.fll', recursive=True):
            examples += 1
            with open(fll_file, 'r') as file:
                fl.lib.logger.info(fll_file)
                fll_from_string = "".join(file.readlines())
                engine = importer.from_string(fll_from_string)
                export_fll = exporter.to_string(engine)
                self.assertEqual(fll_from_string, export_fll)
        self.assertEqual(20, examples)
        fl.lib.decimals = 3


if __name__ == '__main__':
    unittest.main()
