"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

import fuzzylite as fl


class TestLibrary(unittest.TestCase):
    """Test the library class."""

    def test_library_exports_dir(self) -> None:
        """Test the library exports expected components."""
        expected = """
__author__ __builtins__ __cached__ __copyright__ __doc__ __file__ __license__ __loader__
__name__ __package__ __path__ __spec__ __version__

activation Activation First General Highest Last Lowest Proportional Threshold

benchmark Benchmark

defuzzifier Bisector Centroid Defuzzifier IntegralDefuzzifier LargestOfMaximum MeanOfMaximum
SmallestOfMaximum WeightedAverage WeightedDefuzzifier WeightedSum

engine Engine

examples

exporter Exporter FldExporter FllExporter PythonExporter

factory ActivationFactory CloningFactory ConstructionFactory DefuzzifierFactory FactoryManager
FunctionFactory HedgeFactory SNormFactory TNormFactory TermFactory

hedge Any Extremely Hedge HedgeFunction HedgeLambda Maximum Minimum NilpotentMaximum
NilpotentMinimum Norm NormFunction NormLambda NormalizedSum Not Seldom Somewhat Very

importer FllImporter Importer

library Information Representation Settings information representation repr settings array to_float scalar inf nan

norm AlgebraicProduct AlgebraicSum BoundedDifference BoundedSum DrasticProduct DrasticSum
EinsteinProduct EinsteinSum HamacherProduct HamacherSum SNorm TNorm UnboundedSum

operation Op Operation

rule Antecedent Consequent Expression Operator Proposition Rule RuleBlock

term Activated Aggregated Arc Bell Binary Concave Constant Cosine Discrete Function Gaussian
GaussianProduct Linear PiShape Ramp Rectangle SShape SemiEllipse Sigmoid SigmoidDifference
SigmoidProduct Spike Term Trapezoid Triangle ZShape

types Array Scalar ScalarArray Self

variable InputVariable OutputVariable Variable
"""
        self.assertSetEqual(set(expected.split()), set(vars(fl)))

    def test_library_vars(self) -> None:
        """Test the library variables."""
        __version__ = "8.0.4"
        self.assertTrue("fuzzylite" == fl.__name__ == fl.information.name)
        self.assertTrue(__version__ == fl.__version__ == fl.information.version)
        self.assertTrue(
            "a fuzzy logic control library in Python" == fl.__doc__ == fl.information.description
        )
        self.assertTrue("Juan Rada-Vilela, PhD" == fl.__author__ == fl.information.author)
        self.assertTrue("FuzzyLite License" == fl.__license__ == fl.information.license)
        self.assertTrue(
            "Copyright (C) 2010-2024 FuzzyLite by Juan Rada-Vilela. All rights reserved."
            == fl.__copyright__
            == fl.information.copyright
        )

    def test_context(self) -> None:
        """Tests the context."""
        # float_type
        self.assertEqual(fl.settings.float_type, np.float64)
        with fl.settings.context(float_type=np.float16):
            self.assertEqual(fl.settings.float_type, np.float16)
        self.assertEqual(fl.settings.float_type, np.float64)

        # decimals
        self.assertEqual("0.333", fl.Op.str(1 / 3))
        with fl.settings.context(decimals=6):
            self.assertEqual(6, fl.settings.decimals)
            self.assertEqual("0.333333", fl.Op.str(1 / 3))
        self.assertEqual("0.333", fl.Op.str(1 / 3))

        # atol
        self.assertEqual(1e-3, fl.settings.atol)
        with fl.settings.context(atol=1e-6):
            self.assertEqual(1e-6, fl.settings.atol)
        self.assertEqual(1e-3, fl.settings.atol)

        # rtol
        self.assertEqual(0.0, fl.settings.rtol)
        with fl.settings.context(rtol=1e-6):
            self.assertEqual(1e-6, fl.settings.rtol)
        self.assertEqual(0.0, fl.settings.rtol)

        # alias
        self.assertEqual("fl", fl.settings.alias)
        alias_test = fl.Constant("c", fl.nan)
        self.assertEqual("fl.Constant('c', fl.nan)", repr(alias_test))

        with fl.settings.context(alias=""):
            self.assertEqual("", fl.settings.alias)
            self.assertEqual(
                "fuzzylite.term.Constant('c', fuzzylite.library.nan)",
                repr(alias_test),
            )

        with fl.settings.context(alias="*"):
            self.assertEqual("*", fl.settings.alias)
            self.assertEqual(
                "Constant('c', nan)",
                repr(alias_test),
            )

        with fl.settings.context(alias="fuzzylite"):
            self.assertEqual("fuzzylite", fl.settings.alias)
            self.assertEqual(
                "fuzzylite.Constant('c', fuzzylite.nan)",
                repr(alias_test),
            )

        self.assertEqual("fl", fl.settings.alias)
        self.assertEqual("fl.Constant('c', fl.nan)", repr(alias_test))

        # logger
        self.assertEqual(fl.settings.logger, logging.getLogger("fuzzylite"))
        with fl.settings.context(logger=logging.getLogger("fuzzylite.test")):
            self.assertEqual(fl.settings.logger, logging.getLogger("fuzzylite.test"))
        self.assertEqual(fl.settings.logger, logging.getLogger("fuzzylite"))

        # factory manager
        default_factory_manager = fl.settings.factory_manager
        test_factory_manager = fl.FactoryManager()
        with fl.settings.context(factory_manager=test_factory_manager):
            self.assertEqual(fl.settings.factory_manager, test_factory_manager)
        self.assertEqual(fl.settings.factory_manager, default_factory_manager)

    def test_repr_with_takagi_sugeno(self) -> None:
        """Tests the repr with terms referencing engines."""
        from fuzzylite.examples.takagi_sugeno.simple_dimmer import SimpleDimmer

        engine = SimpleDimmer().engine
        engine.input_variables[0].value = 1 / 3
        engine.process()
        self.assertEqual(2 / 3, engine.output_variables[0].value)
        engine.restart()

        engine = eval(repr(SimpleDimmer().engine))
        engine.input_variables[0].value = 1 / 3
        engine.process()
        self.assertEqual(2 / 3, engine.output_variables[0].value)

    def test_numpy_arrays(self) -> None:
        """Tests of numpy arrays."""
        x = fl.array(
            [
                [1, 2],
                [-1, -2],
            ]
        )
        rows = ...
        column = 0
        np.testing.assert_allclose(fl.array([1, -1]), x[rows, column])
        np.testing.assert_allclose(fl.array([2, -2]), x[rows, column + 1])
        np.testing.assert_allclose(fl.array([3, -3]), x.sum(axis=1))
        np.testing.assert_allclose(fl.array([0, 0]), x.sum(axis=0))
        np.testing.assert_allclose(fl.array([3, -3]), x.sum(axis=-1))

    def test_repr_with_takagi_sugeno_referencing_engines(self) -> None:
        """Tests the repr with terms referencing engines."""
        from fuzzylite.examples.takagi_sugeno.octave.linear_tip_calculator import (
            LinearTipCalculator,
        )

        engine = LinearTipCalculator().engine
        engine.input_values = fl.array(
            [
                [1, 2],
                [4, 6],
                [8, 10],
            ]
        )
        np.testing.assert_allclose(fl.array([1, 4, 8]), engine.input_variable(0).value)
        np.testing.assert_allclose(fl.array([2, 6, 10]), engine.input_variable(1).value)
        engine.process()
        np.testing.assert_allclose(engine.output_values, fl.array([[10, 15, 20]]).T)

        engine = eval(repr(LinearTipCalculator().engine))
        engine.input_values = fl.array(
            [
                [1, 2],
                [4, 6],
                [8, 10],
            ]
        )
        np.testing.assert_allclose(fl.array([1, 4, 8]), engine.input_variable(0).value)
        np.testing.assert_allclose(fl.array([2, 6, 10]), engine.input_variable(1).value)
        engine.process()
        np.testing.assert_allclose(engine.output_values, fl.array([[10, 15, 20]]).T)

        engine = eval(repr(LinearTipCalculator().engine))
        engine.input_values = fl.array([1, 2])
        np.testing.assert_allclose(1, engine.input_variable(0).value)
        np.testing.assert_allclose(2, engine.input_variable(1).value)
        engine.process()
        np.testing.assert_allclose(engine.output_values, 10)


if __name__ == "__main__":
    unittest.main()
