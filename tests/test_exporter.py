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
    def test_to_string(self) -> None:
        self.assertEqual(fl.FllExporter().term(fl.Term("X", 1.0)), "term: X Term")
        self.assertEqual(fl.FllExporter().rule(fl.Rule.parse("if x then y")), "rule: if x then y")

    def test_variables(self) -> None:
        self.assertEqual(fl.FllExporter().input_variable(
            fl.InputVariable("name", terms=[fl.Triangle("A"), fl.Trapezoid("B")])),
            "\n".join(
                ["InputVariable: name",
                 "  enabled: true",
                 "  range: -inf inf",
                 "  lock-range: false",
                 "  term: A Triangle nan nan nan",
                 "  term: B Trapezoid nan nan nan nan",
                 ])
        )


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
