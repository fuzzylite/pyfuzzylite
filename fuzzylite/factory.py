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

__all__ = [
    "ConstructionFactory",
    "CloningFactory",
    "ActivationFactory",
    "DefuzzifierFactory",
    "HedgeFactory",
    "SNormFactory",
    "TNormFactory",
    "TermFactory",
    "FunctionFactory",
    "FactoryManager",
]

import copy
import math
from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar

from .activation import (
    Activation,
    First,
    General,
    Highest,
    Last,
    Lowest,
    Proportional,
    Threshold,
)
from .defuzzifier import (
    Bisector,
    Centroid,
    Defuzzifier,
    LargestOfMaximum,
    MeanOfMaximum,
    SmallestOfMaximum,
    WeightedAverage,
    WeightedSum,
)
from .hedge import Any, Extremely, Hedge, Not, Seldom, Somewhat, Very
from .norm import (
    AlgebraicProduct,
    AlgebraicSum,
    BoundedDifference,
    BoundedSum,
    DrasticProduct,
    DrasticSum,
    EinsteinProduct,
    EinsteinSum,
    HamacherProduct,
    HamacherSum,
    Maximum,
    Minimum,
    NilpotentMaximum,
    NilpotentMinimum,
    NormalizedSum,
    SNorm,
    TNorm,
    UnboundedSum,
)
from .operation import Op
from .rule import Rule
from .term import (
    Bell,
    Binary,
    Concave,
    Constant,
    Cosine,
    Discrete,
    Function,
    Gaussian,
    GaussianProduct,
    Linear,
    PiShape,
    Ramp,
    Rectangle,
    Sigmoid,
    SigmoidDifference,
    SigmoidProduct,
    Spike,
    SShape,
    Term,
    Trapezoid,
    Triangle,
    ZShape,
)

T = TypeVar("T")


class ConstructionFactory(Generic[T]):
    """The ConstructionFactory class is the base class for a factory whose
    objects are created from a registered ConstructionFactory::Constructor.

    @author Juan Rada-Vilela, Ph.D.
    @see FactoryManager
    @since 5.0
    """

    def __init__(self) -> None:
        """Create the construction factory."""
        self.constructors: Dict[str, Callable[[], T]] = {}

    def __iter__(self) -> Iterator[str]:
        """Gets the iterator of constructors."""
        return self.constructors.__iter__()

    @property
    def class_name(self) -> str:
        """Returns the class name of the factory
        @return the class name of the factory.
        """
        return self.__class__.__name__

    def construct(self, key: str) -> T:
        """Creates an object by executing the constructor associated to the given key
        @param key is the unique name by which constructors are registered
        @return an object by executing the constructor associated to the given key
        @throws ValueError if the key is not in the registered constructors.
        """
        if key in self.constructors:
            return self.constructors[key]()
        raise ValueError(f"constructor of '{key}' not found in {self.class_name}")


class CloningFactory(Generic[T]):
    """The CloningFactory class is the base class for a factory whose objects
    are created from a registered object by creating a deep copy.

    @author Juan Rada-Vilela, Ph.D.
    @see FactoryManager
    @since 5.0
    """

    def __init__(self) -> None:
        """Create cloning factory."""
        self.objects: Dict[str, T] = {}

    def __iter__(self) -> Iterator[str]:
        """Get iterator iterator of objects."""
        return self.objects.__iter__()

    @property
    def class_name(self) -> str:
        """Returns the class name of the factory
        @return the class name of the factory.
        """
        return self.__class__.__name__

    def copy(self, key: str) -> T:
        """Creates a deep copy of the registered object
        @param key is the unique name by which the object is registered
        @return a deep copy of the registered object
        @throws ValueError if the key is not in the registered objected.
        """
        if key in self.objects:
            return copy.deepcopy(self.objects[key])
        raise ValueError(f"object with key '{key}' not found in {self.class_name}")


class ActivationFactory(ConstructionFactory[Activation]):
    """e ActivationFactory class is a ConstructionFactory of Activation
    methods for RuleBlock%s.

    @author Juan Rada-Vilela, Ph.D.
    @see Activation
    @see RuleBlock
    @see ConstructionFactory
    @see FactoryManager
    @since 6.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            activation().class_name: activation
            for activation in [
                First,
                General,
                Highest,
                Last,
                Lowest,
                Proportional,
                Threshold,
            ]
        }


class DefuzzifierFactory(ConstructionFactory[Defuzzifier]):
    """The DefuzzifierFactory class is a ConstructionFactory of Defuzzifier%s.

    @author Juan Rada-Vilela, Ph.D.
    @see Defuzzifier
    @see ConstructionFactory
    @see FactoryManager
    @since 4.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            defuzzifier().class_name: defuzzifier
            for defuzzifier in [
                Bisector,
                Centroid,
                LargestOfMaximum,
                MeanOfMaximum,
                SmallestOfMaximum,
                WeightedAverage,
                WeightedSum,
            ]
        }

    # TODO: Implement?
    # def construct(self, key: str, parameter: Union[int, str]):
    #     raise NotImplementedError()


class HedgeFactory(ConstructionFactory[Hedge]):
    """The HedgeFactory class is a ConstructionFactory of Hedge%s.

    @author Juan Rada-Vilela, Ph.D.
    @see Hedge
    @see ConstructionFactory
    @see FactoryManager
    @since 4.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            hedge().name: hedge
            for hedge in [Any, Extremely, Not, Seldom, Somewhat, Very]
        }


class SNormFactory(ConstructionFactory[SNorm]):
    """The SNormFactory class is a ConstructionFactory of SNorm%s.

    @author Juan Rada-Vilela, Ph.D.
    @see SNorm
    @see ConstructionFactory
    @see FactoryManager
    @since 4.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            snorm().class_name: snorm
            for snorm in [
                AlgebraicSum,
                BoundedSum,
                DrasticSum,
                EinsteinSum,
                HamacherSum,
                Maximum,
                NilpotentMaximum,
                NormalizedSum,
                UnboundedSum,
            ]
        }


class TNormFactory(ConstructionFactory[TNorm]):
    """The TNormFactory class is a ConstructionFactory of TNorm%s.

    @author Juan Rada-Vilela, Ph.D.
    @see TNorm
    @see ConstructionFactory
    @see FactoryManager
    @since 4.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            tnorm().class_name: tnorm
            for tnorm in [
                AlgebraicProduct,
                BoundedDifference,
                DrasticProduct,
                EinsteinProduct,
                HamacherProduct,
                Minimum,
                NilpotentMinimum,
            ]
        }


class TermFactory(ConstructionFactory[Term]):
    """The TermFactory class is a ConstructionFactory of Term%s.

    @author Juan Rada-Vilela, Ph.D.
    @see Term
    @see ConstructionFactory
    @see FactoryManager
    @since 4.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self.constructors = {
            term().class_name: term
            for term in [
                Bell,
                Binary,
                Concave,
                Constant,
                Cosine,
                Discrete,
                Function,
                Gaussian,
                GaussianProduct,
                Linear,
                PiShape,
                Ramp,
                Rectangle,
                Sigmoid,
                SigmoidDifference,
                SigmoidProduct,
                Spike,
                SShape,
                Trapezoid,
                Triangle,
                ZShape,
            ]
        }


class FunctionFactory(CloningFactory[Function.Element]):
    """The FunctionFactory class is a CloningFactory of operators and functions
    utilized by the Function term.

    @author Juan Rada-Vilela, Ph.D.
    @see Function
    @see Element
    @see CloningFactory
    @see FactoryManager
    @since 5.0
    """

    def __init__(self) -> None:
        """Create the factory."""
        super().__init__()
        self._register_operators()
        self._register_functions()

    def _precedence(self, importance: int) -> int:
        maximum = 100
        step = 10
        return maximum - importance * step

    def _register_operators(self) -> None:
        import operator

        operator_type = Function.Element.Type.Operator
        p: Callable[[int], int] = self._precedence
        operators = [
            # First order: not, negate
            Function.Element(
                "!",
                "Logical NOT",
                operator_type,
                operator.not_,
                arity=1,
                precedence=p(0),
                associativity=1,
            ),
            Function.Element(
                "~",
                "Negate",
                operator_type,
                operator.neg,
                arity=1,
                precedence=p(0),
                associativity=1,
            ),
            # Second order: power, unary -, unary +
            Function.Element(
                "^",
                "Power",
                operator_type,
                operator.pow,
                arity=2,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                "**",
                "Power",
                operator_type,
                operator.pow,
                arity=2,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                ".-",
                "Unary minus",
                operator_type,
                operator.neg,
                arity=1,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                ".+",
                "Unary plus",
                operator_type,
                operator.pos,
                arity=1,
                precedence=p(1),
                associativity=1,
            ),
            # Third order: Multiplication, Division, and Modulo
            Function.Element(
                "*",
                "Multiplication",
                operator_type,
                operator.mul,
                arity=2,
                precedence=p(2),
            ),
            Function.Element(
                "/",
                "Division",
                operator_type,
                operator.truediv,
                arity=2,
                precedence=p(2),
            ),
            Function.Element(
                "%", "Modulo", operator_type, operator.mod, arity=2, precedence=p(2)
            ),
            # Fourth order: Addition, Subtraction
            Function.Element(
                "+", "Addition", operator_type, operator.add, arity=2, precedence=p(3)
            ),
            Function.Element(
                "-",
                "Subtraction",
                operator_type,
                operator.sub,
                arity=2,
                precedence=p(3),
            ),
            # Fifth order: logical and
            Function.Element(
                Rule.AND,
                "Logical AND",
                operator_type,
                Op.logical_and,
                arity=2,
                precedence=p(4),
            ),
            # Sixth order: logical or
            Function.Element(
                Rule.OR,
                "Logical OR",
                operator_type,
                Op.logical_or,
                arity=2,
                precedence=p(5),
            ),
        ]
        for op in operators:
            self.objects[op.name] = op

    def _register_functions(self) -> None:
        function_type = Function.Element.Type.Function

        functions = [
            Function.Element("gt", "Greater than (>)", function_type, Op.gt, arity=2),
            Function.Element(
                "ge", "Greater than or equal to (>=)", function_type, Op.ge, arity=2
            ),
            Function.Element("eq", "Equal to (==)", function_type, Op.eq, arity=2),
            Function.Element(
                "neq", "Not equal to (!=)", function_type, Op.neq, arity=2
            ),
            Function.Element(
                "le", "Less than or equal to (<=)", function_type, Op.le, arity=2
            ),
            Function.Element("lt", "Less than (>)", function_type, Op.lt, arity=2),
            Function.Element("min", "Minimum", function_type, min, arity=2),
            Function.Element("max", "Maximum", function_type, max, arity=2),
            Function.Element(
                "acos", "Inverse cosine", function_type, math.acos, arity=1
            ),
            Function.Element("asin", "Inverse sine", function_type, math.asin, arity=1),
            Function.Element(
                "atan", "Inverse tangent", function_type, math.atan, arity=1
            ),
            Function.Element("ceil", "Ceiling", function_type, math.ceil, arity=1),
            Function.Element("cos", "Cosine", function_type, math.cos, arity=1),
            Function.Element(
                "cosh", "Hyperbolic cosine", function_type, math.cosh, arity=1
            ),
            Function.Element("exp", "Exponential", function_type, math.exp, arity=1),
            Function.Element("abs", "Absolute", function_type, math.fabs, arity=1),
            Function.Element("fabs", "Absolute", function_type, math.fabs, arity=1),
            Function.Element("floor", "Floor", function_type, math.floor, arity=1),
            Function.Element(
                "log", "Natural logarithm", function_type, math.log, arity=1
            ),
            Function.Element(
                "log10", "Common logarithm", function_type, math.log10, arity=1
            ),
            Function.Element("round", "Round", function_type, round, arity=1),
            Function.Element("sin", "Sine", function_type, math.sin, arity=1),
            Function.Element(
                "sinh", "Hyperbolic sine", function_type, math.sinh, arity=1
            ),
            Function.Element("sqrt", "Square root", function_type, math.sqrt, arity=1),
            Function.Element("tan", "Tangent", function_type, math.tan, arity=1),
            Function.Element(
                "tanh", "Hyperbolic tangent", function_type, math.tanh, arity=1
            ),
            Function.Element(
                "log1p", "Natural logarithm plus one", function_type, math.log1p, 1
            ),
            Function.Element(
                "acosh", "Inverse hyperbolic cosine", function_type, math.acosh, 1
            ),
            Function.Element(
                "asinh", "Inverse hyperbolic sine", function_type, math.asinh, 1
            ),
            Function.Element(
                "atanh", "Inverse hyperbolic tangent", function_type, math.atanh, 1
            ),
            Function.Element("pow", "Power", function_type, math.pow, arity=2),
            Function.Element(
                "atan2", "Inverse tangent (y,x)", function_type, math.atan2, arity=2
            ),
            Function.Element(
                "fmod", "Floating-point remainder", function_type, math.fmod, arity=2
            ),
            Function.Element("pi", "Pi constant", function_type, Op.pi, arity=0),
        ]

        for f in functions:
            f.precedence = self._precedence(0)
            self.objects[f.name] = f

    def operators(self) -> Dict[str, Function.Element]:
        """Returns a dictionary of the operators available
        @return a dictionary of the operators available.
        """
        result = {
            key: prototype
            for key, prototype in self.objects.items()
            if prototype.is_operator()
        }
        return result

    def functions(self) -> Dict[str, Function.Element]:
        """Returns a dictionary of the functions available
        @return a dictionary of the functions available.
        """
        result = {
            key: prototype
            for key, prototype in self.objects.items()
            if prototype.is_function()
        }
        return result


class FactoryManager:
    """The FactoryManager class is a central class grouping different factories
    of objects, together with a singleton instance to access each of the
    factories throughout the library.

    @author Juan Rada-Vilela, Ph.D.
    @see TermFactory
    @see TNormFactory
    @see SNormFactory
    @see HedgeFactory
    @see ActivationFactory
    @see DefuzzifierFactory
    @see FunctionFactory
    @since 4.0
    """

    def __init__(
        self,
        tnorm: Optional[TNormFactory] = None,
        snorm: Optional[SNormFactory] = None,
        activation: Optional[ActivationFactory] = None,
        defuzzifier: Optional[DefuzzifierFactory] = None,
        term: Optional[TermFactory] = None,
        hedge: Optional[HedgeFactory] = None,
        function: Optional[FunctionFactory] = None,
    ) -> None:
        """Creates a factory manager with the given factories (or default factories if none supplied)
        @param tnorm is the factory of TNorm%s
        @param snorm is the factory of SNorm%s
        @param activation is the factory of Activation methods
        @param defuzzifier is the factory of Defuzzifier%s
        @param term is the factory of Term%s
        @param hedge is the factory of Hedge%s
        @param function is the factory of Function Element%s.
        """
        self.tnorm = tnorm if tnorm else TNormFactory()
        self.snorm = snorm if snorm else SNormFactory()
        self.activation = activation if activation else ActivationFactory()
        self.defuzzifier = defuzzifier if defuzzifier else DefuzzifierFactory()
        self.term = term if term else TermFactory()
        self.hedge = hedge if hedge else HedgeFactory()
        self.function = function if function else FunctionFactory()
