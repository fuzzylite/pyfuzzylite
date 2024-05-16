"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import inspect
import logging
import pathlib
import unittest
from pathlib import Path

logger = logging.getLogger()


class TestExamples(unittest.TestCase):
    """Test the examples."""

    def test_imports(self) -> None:
        """Tests the examples can be imported."""
        from fuzzylite import examples  # noqa
        from fuzzylite.examples import hybrid, mamdani, takagi_sugeno  # noqa
        from fuzzylite.examples.mamdani import simple_dimmer  # noqa
        from fuzzylite.examples.mamdani.simple_dimmer import SimpleDimmer  # noqa

    def test_absolute_imports(self) -> None:
        """Tests the examples can be imported with absolute imports."""
        import fuzzylite  # noqa
        import fuzzylite.examples  # noqa
        import fuzzylite.examples.mamdani  # noqa
        import fuzzylite.examples.mamdani.simple_dimmer  # noqa
        from fuzzylite.examples.mamdani.simple_dimmer import SimpleDimmer  # noqa

    def test_examples(self) -> None:
        """Test all the examples are included and can be imported."""
        import fuzzylite as fl

        examples = pathlib.Path(*fl.examples.__path__)
        self.assertTrue(examples.exists() and examples.is_dir())

        expected = set(
            """\
fuzzylite.examples.hybrid.obstacle_avoidance
fuzzylite.examples.hybrid.tipper
fuzzylite.examples.mamdani.all_terms
fuzzylite.examples.mamdani.laundry
fuzzylite.examples.mamdani.obstacle_avoidance
fuzzylite.examples.mamdani.simple_dimmer
fuzzylite.examples.mamdani.simple_dimmer_chained
fuzzylite.examples.mamdani.simple_dimmer_inverse
fuzzylite.examples.mamdani.matlab.mam21
fuzzylite.examples.mamdani.matlab.mam22
fuzzylite.examples.mamdani.matlab.shower
fuzzylite.examples.mamdani.matlab.tank
fuzzylite.examples.mamdani.matlab.tank2
fuzzylite.examples.mamdani.matlab.tipper
fuzzylite.examples.mamdani.matlab.tipper1
fuzzylite.examples.mamdani.octave.investment_portfolio
fuzzylite.examples.mamdani.octave.mamdani_tip_calculator
fuzzylite.examples.takagi_sugeno.obstacle_avoidance
fuzzylite.examples.takagi_sugeno.simple_dimmer
fuzzylite.examples.takagi_sugeno.approximation
fuzzylite.examples.takagi_sugeno.matlab.fpeaks
fuzzylite.examples.takagi_sugeno.matlab.invkine1
fuzzylite.examples.takagi_sugeno.matlab.invkine2
fuzzylite.examples.takagi_sugeno.matlab.juggler
fuzzylite.examples.takagi_sugeno.matlab.membrn1
fuzzylite.examples.takagi_sugeno.matlab.membrn2
fuzzylite.examples.takagi_sugeno.matlab.slbb
fuzzylite.examples.takagi_sugeno.matlab.slcp
fuzzylite.examples.takagi_sugeno.matlab.slcp1
fuzzylite.examples.takagi_sugeno.matlab.slcpp1
fuzzylite.examples.takagi_sugeno.matlab.sltbu_fl
fuzzylite.examples.takagi_sugeno.matlab.sugeno1
fuzzylite.examples.takagi_sugeno.matlab.tanksg
fuzzylite.examples.takagi_sugeno.matlab.tippersg
fuzzylite.examples.takagi_sugeno.octave.cubic_approximator
fuzzylite.examples.takagi_sugeno.octave.heart_disease_risk
fuzzylite.examples.takagi_sugeno.octave.linear_tip_calculator
fuzzylite.examples.takagi_sugeno.octave.sugeno_tip_calculator
fuzzylite.examples.terms.arc
fuzzylite.examples.terms.bell
fuzzylite.examples.terms.binary
fuzzylite.examples.terms.concave
fuzzylite.examples.terms.constant
fuzzylite.examples.terms.cosine
fuzzylite.examples.terms.discrete
fuzzylite.examples.terms.function
fuzzylite.examples.terms.gaussian
fuzzylite.examples.terms.gaussian_product
fuzzylite.examples.terms.linear
fuzzylite.examples.terms.pi_shape
fuzzylite.examples.terms.ramp
fuzzylite.examples.terms.rectangle
fuzzylite.examples.terms.semi_ellipse
fuzzylite.examples.terms.sigmoid
fuzzylite.examples.terms.sigmoid_difference
fuzzylite.examples.terms.sigmoid_product
fuzzylite.examples.terms.spike
fuzzylite.examples.terms.trapezoid
fuzzylite.examples.terms.triangle
fuzzylite.examples.terms.zs_shape
fuzzylite.examples.tsukamoto.tsukamoto
""".split()
        )

        obtained = {module for module in fl.Op.glob_examples("module")}
        self.assertSetEqual(expected, {module.__name__ for module in obtained})

        engines = [engine for engine in fl.Op.glob_examples("engine")]
        self.assertEqual(len(engines), len(obtained))

    def test_examples_remain_the_same(self) -> None:
        """Test that all the examples remain the same when exported in Python, FLL and FLD."""
        import fuzzylite as fl

        examples = set(fl.Op.glob_examples("module"))
        self.assertTrue(bool(examples), msg="examples not found")

        for example_module in examples:
            example_class, *_ = inspect.getmembers(example_module, predicate=inspect.isclass)
            engine = example_class[1]().engine

            expected_python = Path(str(example_module.__file__)).read_text()
            obtained_python = fl.PythonExporter(formatted=True, encapsulated=True).to_string(engine)
            self.assertEqual(expected_python, obtained_python, msg=example_module.__name__)

            expected_fll = Path(str(example_module.__file__)).with_suffix(".fll").read_text()
            obtained_fll = fl.FllExporter().to_string(engine)
            self.assertEqual(expected_fll, obtained_fll, msg=example_module.__name__)

            expected_fld = Path(str(example_module.__file__)).with_suffix(".fld").read_text()
            with fl.settings.context(decimals=9):
                obtained_fld = fl.FldExporter().to_string(engine)
            self.assertEqual(expected_fld, obtained_fld, msg=example_module.__name__)


if __name__ == "__main__":
    unittest.main()
