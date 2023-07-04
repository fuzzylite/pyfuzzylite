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
import unittest

import numpy as np

import fuzzylite as fl


class TestLibrary(unittest.TestCase):
    """Test the library class."""

    def test_library_exports_dir(self) -> None:
        """Test the library exports expected components."""
        expected = """
__builtins__ __cached__ __doc__ __file__ __loader__
__name__ __package__ __path__ __spec__ __version__

activation Activation First General Highest Last Lowest Proportional Threshold

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

types Array Scalar ScalarArray

variable InputVariable OutputVariable Variable

__getattr__
"""
        # TODO: remove __getattr__
        self.assertSetEqual(set(expected.split()), set(vars(fl)))

    def test_library_vars(self) -> None:
        """Test the library variables."""
        __version__ = "8.0.0"
        self.assertEqual(fl.__name__, "fuzzylite")
        self.assertEqual(fl.__version__, __version__)
        self.assertEqual(fl.__doc__, fl.information.description)

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
        self.assertEqual("fl.Constant(name='c', value=fl.nan)", repr(alias_test))

        with fl.settings.context(alias=""):
            self.assertEqual("", fl.settings.alias)
            self.assertEqual(
                "fuzzylite.term.Constant(name='c', value=fuzzylite.library.nan)",
                repr(alias_test),
            )

        with fl.settings.context(alias="*"):
            self.assertEqual("*", fl.settings.alias)
            self.assertEqual(
                "Constant(name='c', value=nan)",
                repr(alias_test),
            )

        with fl.settings.context(alias="fuzzylite"):
            self.assertEqual("fuzzylite", fl.settings.alias)
            self.assertEqual(
                "fuzzylite.Constant(name='c', value=fuzzylite.nan)",
                repr(alias_test),
            )

        self.assertEqual("fl", fl.settings.alias)
        self.assertEqual("fl.Constant(name='c', value=fl.nan)", repr(alias_test))

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


if __name__ == "__main__":
    unittest.main()
