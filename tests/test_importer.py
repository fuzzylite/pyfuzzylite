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
    @unittest.skip("")
    def test_import_examples(self) -> None:
        self.maxDiff = None  # show all differences
        importer = fl.FllImporter()
        exporter = fl.FllExporter()
        fl.lib.decimals = 9
        for fll_file in glob.iglob('examples/**/*.fll', recursive=True):
            with open(fll_file, 'r') as file:
                fl.lib.logger.info(fll_file)
                import_fll = "".join(file.readlines())
                engine = importer.from_string(import_fll)
                export_fll = exporter.to_string(engine)
                self.assertEqual(import_fll, export_fll)
        fl.lib.decimals = 3


if __name__ == '__main__':
    unittest.main()
