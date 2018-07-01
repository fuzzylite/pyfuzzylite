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
        expected = {'Activated', 'Activation', 'ActivationFactory', 'Aggregated',
                    'AlgebraicProduct', 'AlgebraicSum', 'Antecedent', 'Any', 'Bell', 'Binary',
                    'Bisector', 'BoundedDifference', 'BoundedSum', 'Centroid', 'CloningFactory',
                    'Concave', 'Consequent', 'Constant', 'ConstructionFactory', 'Cosine',
                    'Defuzzifier', 'DefuzzifierFactory', 'Discrete', 'DrasticProduct', 'DrasticSum',
                    'EinsteinProduct', 'EinsteinSum', 'Engine', 'Exporter', 'Expression',
                    'Extremely', 'FactoryManager', 'First', 'FldExporter', 'FllExporter',
                    'FllImporter', 'Function', 'FunctionFactory', 'Gaussian',
                    'GaussianProduct', 'General', 'HamacherProduct', 'HamacherSum', 'Hedge',
                    'HedgeFactory', 'HedgeFunction', 'HedgeLambda', 'Highest', 'Importer',
                    'InputVariable', 'IntegralDefuzzifier', 'LargestOfMaximum', 'Last', 'Library',
                    'Linear', 'Lowest', 'Maximum', 'MeanOfMaximum', 'Minimum', 'NilpotentMaximum',
                    'NilpotentMinimum', 'Norm', 'NormFunction', 'NormLambda', 'NormalizedSum',
                    'Not', 'Op', 'Operation', 'Operator', 'OutputVariable', 'PiShape',
                    'Proportional', 'Proposition', 'Ramp', 'Rectangle', 'Rule', 'RuleBlock',
                    'SNorm', 'SNormFactory', 'SShape', 'Seldom', 'Sigmoid', 'SigmoidDifference',
                    'SigmoidProduct', 'SmallestOfMaximum', 'Somewhat', 'Spike', 'TNorm',
                    'TNormFactory', 'Term', 'TermFactory', 'Threshold', 'Trapezoid', 'Triangle',
                    'UnboundedSum', 'Variable', 'Very', 'WeightedAverage', 'WeightedDefuzzifier',
                    'WeightedSum', 'ZShape', '__annotations__', '__builtins__', '__cached__',
                    '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__',
                    '__spec__', '__version__', 'activation', 'defuzzifier', 'engine', 'exporter',
                    'factory', 'hedge', 'importer', 'inf', 'isnan', 'lib', 'library', 'nan', 'norm',
                    'operation', 'rule', 'scalar', 'term', 'variable'}

        self.assertSetEqual(expected, set(dir(fuzzylite)))

    def test_library_vars(self) -> None:
        self.assertEqual(fuzzylite.__name__, "pyfuzzylite")
        self.assertEqual(fuzzylite.__version__, "7.0")
        self.assertEqual(fuzzylite.__doc__, fuzzylite.Library().summary)


if __name__ == '__main__':
    unittest.main()
