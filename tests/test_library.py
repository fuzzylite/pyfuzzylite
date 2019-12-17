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

import unittest

import fuzzylite


class TestLibrary(unittest.TestCase):

    def test_library_exports_dir(self) -> None:
        expected = """
__annotations__ __builtins__ __cached__ __doc__ __file__ __loader__
__name__ __package__ __path__ __spec__ __version__

inf isinf isnan lib nan scalar

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

library Library

norm AlgebraicProduct AlgebraicSum BoundedDifference BoundedSum DrasticProduct DrasticSum
EinsteinProduct EinsteinSum HamacherProduct HamacherSum SNorm TNorm UnboundedSum

operation Op Operation

rule Antecedent Consequent Expression Operator Proposition Rule RuleBlock

term Activated Aggregated Bell Binary Concave Constant Cosine Discrete Function Gaussian
GaussianProduct Linear PiShape Ramp Rectangle SShape Sigmoid SigmoidDifference SigmoidProduct
Spike Term Trapezoid Triangle ZShape

variable InputVariable OutputVariable Variable
"""
        self.assertSetEqual(set(expected.split()), set(dir(fuzzylite)))

    def test_library_vars(self) -> None:
        self.assertEqual(fuzzylite.__name__, "pyfuzzylite")
        self.assertEqual(fuzzylite.__version__, "7.0b3")
        self.assertEqual(fuzzylite.__doc__, fuzzylite.lib.summary)


if __name__ == '__main__':
    unittest.main()
