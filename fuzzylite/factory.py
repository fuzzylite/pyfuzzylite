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
import inspect
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from .activation import Activation
from .defuzzifier import Defuzzifier
from .hedge import Hedge
from .library import representation
from .norm import SNorm, TNorm
from .operation import Op
from .rule import Rule
from .term import Function, Term

T = TypeVar("T")


class ConstructionFactory(Generic[T]):
    """The ConstructionFactory class is the base class for a factory whose
    objects are created from a registered ConstructionFactory::Constructor.

    @author Juan Rada-Vilela, Ph.D.
    @see FactoryManager
    @since 5.0
    """

    def __init__(self, constructors: dict[str, type[T]] | None = None) -> None:
        """Create the construction factory."""
        self.constructors = constructors or {}

    def __iter__(self) -> Iterator[str]:
        """Gets the iterator of constructors."""
        return iter(self.constructors)

    def __getitem__(self, key: str) -> type[T]:
        """Get type by 'key'."""
        return self.constructors[key]

    def __setitem__(self, key: str, value: type[T]) -> None:
        """Add (or replace) the key with the given value."""
        self.constructors[key] = value

    def __len__(self) -> int:
        """@return number of constructors."""
        return len(self.constructors)

    def __str__(self) -> str:
        """@return class name of the factory."""
        return Op.class_name(self)

    def __repr__(self) -> str:
        """@return Python code to construct the factory."""
        return representation.as_constructor(self)

    def import_from(self, module: ModuleType, cls: type[T]) -> list[type[T]]:
        """Imports constructors from a module.
        @param module is the module from which constructors are imported
        @param cls is the class of the constructors to be imported
        @return a list of constructors imported from the module.
        """

        def constructable(obj: type[T]) -> bool:
            try:
                return (
                    issubclass(obj, cls) and not inspect.isabstract(obj) and bool(obj())
                )
            except:  # noqa: E722
                return False

        constructors = [
            constructor
            for _, constructor in inspect.getmembers(module, predicate=constructable)
        ]
        return constructors

    def construct(self, key: str, **kwargs: Any) -> T:
        """Creates an object by executing the constructor associated to the given key
        @param key is the unique name by which constructors are registered
        @return an object by executing the constructor associated to the given key
        @throws ValueError if the key is not in the registered constructors.
        """
        if key in self.constructors:
            return self.constructors[key](**kwargs)
        raise ValueError(f"constructor of '{key}' not found in {Op.class_name(self)}")


class CloningFactory(Generic[T]):
    """The CloningFactory class is the base class for a factory whose objects
    are created from a registered object by creating a deep copy.

    @author Juan Rada-Vilela, Ph.D.
    @see FactoryManager
    @since 5.0
    """

    def __init__(self, objects: dict[str, T] | None = None) -> None:
        """Create cloning factory."""
        self.objects = objects or {}

    def __iter__(self) -> Iterator[str]:
        """Get iterator of objects."""
        return iter(self.objects)

    def __getitem__(self, key: str) -> T:
        """Get object by 'key'."""
        return self.objects[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Add (or replace) key with the given value."""
        self.objects[key] = value

    def __len__(self) -> int:
        """@return number of objects in factory."""
        return len(self.objects)

    def __str__(self) -> str:
        """@return class name of the factory."""
        return Op.class_name(self)

    def __repr__(self) -> str:
        """@return Python code to construct the factory."""
        return representation.as_constructor(self)

    def copy(self, key: str) -> T:
        """Creates a deep copy of the registered object
        @param key is the unique name by which the object is registered
        @return a deep copy of the registered object
        @throws ValueError if the key is not in the registered objected.
        """
        if key in self.objects:
            return copy.deepcopy(self.objects[key])
        raise ValueError(f"object with key '{key}' not found in {Op.class_name(self)}")


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
        from . import activation

        activations = {
            Op.class_name(a): a for a in self.import_from(activation, Activation)
        }
        super().__init__(constructors=activations)


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
        from . import defuzzifier

        defuzzifiers = {
            Op.class_name(d): d for d in self.import_from(defuzzifier, Defuzzifier)
        }
        super().__init__(constructors=defuzzifiers)


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
        from . import hedge

        hedges = {h().name: h for h in self.import_from(hedge, Hedge)}
        super().__init__(constructors=hedges)


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
        from . import norm as norm

        snorms = {Op.class_name(n): n for n in self.import_from(norm, SNorm)}
        super().__init__(constructors=snorms)


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
        from . import norm as norm

        tnorms = {Op.class_name(n): n for n in self.import_from(norm, TNorm)}
        super().__init__(constructors=tnorms)


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
        from . import term as term

        terms = {
            Op.class_name(t): t
            for t in self.import_from(term, Term)
            if t not in {term.Activated, term.Aggregated}
        }
        super().__init__(constructors=terms)


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
        elements = {
            element.name: element
            for element in self._create_operators() + self._create_functions()
        }
        super().__init__(objects=elements)

    def _precedence(self, importance: int) -> int:
        maximum = 100
        step = 10
        return maximum - importance * step

    def _create_operators(self) -> list[Function.Element]:
        operator_type = Function.Element.Type.Operator
        p: Callable[[int], int] = self._precedence
        operators = [
            # First order: not, negate
            Function.Element(
                "!",
                "Logical NOT",
                operator_type,
                np.logical_not,
                arity=1,
                precedence=p(0),
                associativity=1,
            ),
            Function.Element(
                "~",
                "Negate",
                operator_type,
                np.negative,
                arity=1,
                precedence=p(0),
                associativity=1,
            ),
            # Second order: power, unary -, unary +
            Function.Element(
                "^",
                "Power",
                operator_type,
                np.float_power,
                arity=2,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                "**",
                "Power",
                operator_type,
                np.float_power,
                arity=2,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                ".-",
                "Unary minus",
                operator_type,
                np.negative,
                arity=1,
                precedence=p(1),
                associativity=1,
            ),
            Function.Element(
                ".+",
                "Unary plus",
                operator_type,
                np.positive,
                arity=1,
                precedence=p(1),
                associativity=1,
            ),
            # Third order: Multiplication, Division, and Modulo
            Function.Element(
                "*",
                "Multiplication",
                operator_type,
                np.multiply,
                arity=2,
                precedence=p(2),
            ),
            Function.Element(
                "/",
                "Division",
                operator_type,
                np.true_divide,
                arity=2,
                precedence=p(2),
            ),
            Function.Element(
                "%",
                "Modulo",
                operator_type,
                np.remainder,
                arity=2,
                precedence=p(2),
            ),
            # Fourth order: Addition, Subtraction
            Function.Element(
                "+",
                "Addition",
                operator_type,
                np.add,
                arity=2,
                precedence=p(3),
            ),
            Function.Element(
                "-",
                "Subtraction",
                operator_type,
                np.subtract,
                arity=2,
                precedence=p(3),
            ),
            # Fifth order: logical and
            Function.Element(
                Rule.AND,
                "Logical AND",
                operator_type,
                np.logical_and,
                arity=2,
                precedence=p(4),
            ),
            # Sixth order: logical or
            Function.Element(
                Rule.OR,
                "Logical OR",
                operator_type,
                np.logical_or,
                arity=2,
                precedence=p(5),
            ),
        ]
        return operators

    def _create_functions(self) -> list[Function.Element]:
        function_type = Function.Element.Type.Function
        p: Callable[[int], int] = self._precedence
        functions = [
            Function.Element(
                "gt",
                "Greater than (>)",
                function_type,
                Op.gt,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "ge",
                "Greater than or equal to (>=)",
                function_type,
                Op.ge,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "eq",
                "Equal to (==)",
                function_type,
                Op.eq,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "neq",
                "Not equal to (!=)",
                function_type,
                Op.neq,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "le",
                "Less than or equal to (<=)",
                function_type,
                Op.le,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "lt",
                "Less than (<)",
                function_type,
                Op.lt,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "min",
                "Minimum",
                function_type,
                min,  # because is variadiac, whereas np.min takes arrays as args
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "max",
                "Maximum",
                function_type,
                max,  # because is variadiac, whereas np.max takes arrays as args
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "acos",
                "Inverse cosine",
                function_type,
                np.arccos,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "asin",
                "Inverse sine",
                function_type,
                np.arcsin,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "atan",
                "Inverse tangent",
                function_type,
                np.arctan,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "ceil",
                "Ceiling",
                function_type,
                np.ceil,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "cos",
                "Cosine",
                function_type,
                np.cos,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "cosh",
                "Hyperbolic cosine",
                function_type,
                np.cosh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "exp",
                "Exponential",
                function_type,
                np.exp,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "abs",
                "Absolute",
                function_type,
                np.fabs,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "fabs",
                "Absolute",
                function_type,
                np.fabs,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "floor",
                "Floor",
                function_type,
                np.floor,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "log",
                "Natural logarithm",
                function_type,
                np.log,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "log10",
                "Common logarithm",
                function_type,
                np.log10,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "round",
                "Round",
                function_type,
                np.round,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "sin",
                "Sine",
                function_type,
                np.sin,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "sinh",
                "Hyperbolic sine",
                function_type,
                np.sinh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "sqrt",
                "Square root",
                function_type,
                np.sqrt,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "tan",
                "Tangent",
                function_type,
                np.tan,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "tanh",
                "Hyperbolic tangent",
                function_type,
                np.tanh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "log1p",
                "Natural logarithm plus one",
                function_type,
                np.log1p,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "acosh",
                "Inverse hyperbolic cosine",
                function_type,
                np.arccosh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "asinh",
                "Inverse hyperbolic sine",
                function_type,
                np.arcsinh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "atanh",
                "Inverse hyperbolic tangent",
                function_type,
                np.arctanh,
                arity=1,
                precedence=p(0),
            ),
            Function.Element(
                "pow",
                "Power",
                function_type,
                np.float_power,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "atan2",
                "Inverse tangent (y,x)",
                function_type,
                np.arctan2,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "fmod",
                "Floating-point remainder",
                function_type,
                np.fmod,
                arity=2,
                precedence=p(0),
            ),
            Function.Element(
                "pi",
                "Pi constant",
                function_type,
                lambda: np.pi,
                arity=0,
                precedence=p(0),
            ),
        ]
        return functions

    def operators(self) -> dict[str, Function.Element]:
        """Returns a dictionary of the operators available
        @return a dictionary of the operators available.
        """
        result = {
            key: prototype
            for key, prototype in self.objects.items()
            if prototype.is_operator()
        }
        return result

    def functions(self) -> dict[str, Function.Element]:
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
        tnorm: TNormFactory | None = None,
        snorm: SNormFactory | None = None,
        activation: ActivationFactory | None = None,
        defuzzifier: DefuzzifierFactory | None = None,
        term: TermFactory | None = None,
        hedge: HedgeFactory | None = None,
        function: FunctionFactory | None = None,
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
