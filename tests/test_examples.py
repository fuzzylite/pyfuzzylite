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
from __future__ import annotations

import logging
import pathlib
import unittest

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


if __name__ == "__main__":
    unittest.main()
