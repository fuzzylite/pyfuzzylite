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
# flake8: noqa

# noinspection PyUnresolvedReferences
from math import nan, inf, isnan

# ACTIVATION
from fuzzylite.activation import (Activation, First, General, Highest, Last, Lowest, Proportional,
                                  Threshold)
# DEFUZZIFIER
from fuzzylite.defuzzifier import (Defuzzifier, WeightedDefuzzifier, IntegralDefuzzifier, Centroid,
                                   Bisector, LargestOfMaximum, MeanOfMaximum, SmallestOfMaximum,
                                   WeightedAverage, WeightedSum)
from fuzzylite.engine import Engine
# EXPORTER
from fuzzylite.exporter import Exporter, FllExporter
# FACTORY
from fuzzylite.factory import (ActivationFactory, CloningFactory, ConstructionFactory,
                               DefuzzifierFactory, FactoryManager, FunctionFactory, HedgeFactory,
                               SNormFactory, TNormFactory, TermFactory)
# HEDGES
from fuzzylite.hedge import (Any, Extremely, Hedge, HedgeFunction, HedgeLambda, Not, Seldom,
                             Somewhat, Very)
# IMPORTER
from fuzzylite.importer import Importer
# LIBRARY
from fuzzylite.library import Library
# NORM
from fuzzylite.norm import (AlgebraicProduct, BoundedDifference, DrasticProduct, EinsteinProduct,
                            HamacherProduct, Minimum, NilpotentMinimum)
from fuzzylite.norm import (AlgebraicSum, BoundedSum, DrasticSum, EinsteinSum, HamacherSum,
                            Maximum, NilpotentMaximum, NormalizedSum, UnboundedSum)
from fuzzylite.norm import Norm, NormFunction, NormLambda, SNorm, TNorm
# OPERATION
from fuzzylite.operation import Op, Operation
# RULE
from fuzzylite.rule import (Rule, Proposition, Operator, RuleBlock, Expression, Antecedent,
                            Consequent)
# TERM
from fuzzylite.term import (Activated, Aggregated, Bell, Binary, Concave, Constant, Cosine,
                            Discrete, Function, Gaussian, GaussianProduct, Linear, PiShape, Ramp,
                            Rectangle, SShape, Sigmoid, SigmoidDifference, SigmoidProduct,
                            Spike, Term, Trapezoid, Triangle, ZShape)
# VARIABLE
from fuzzylite.variable import InputVariable, OutputVariable, Variable

lib: Library = Library()
__name__ = lib.name
__version__ = lib.version
__doc__ = lib.summary
Float = lib.floating_point
