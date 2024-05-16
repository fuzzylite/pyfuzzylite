"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
    """Base class for a factory whose objects are created from a registered constructor.

    info: related
        - [fuzzylite.factory.ActivationFactory][]
        - [fuzzylite.factory.DefuzzifierFactory][]
        - [fuzzylite.factory.HedgeFactory][]
        - [fuzzylite.factory.SNormFactory][]
        - [fuzzylite.factory.TermFactory][]
        - [fuzzylite.factory.TNormFactory][]
        - [fuzzylite.factory.FactoryManager][]
        - [fuzzylite.factory.CloningFactory][]
    """

    def __init__(self, constructors: dict[str, type[T]] | None = None) -> None:
        """Constructor.

        Args:
            constructors: dictionary of constructors
        """
        self.constructors = constructors or {}

    def __iter__(self) -> Iterator[str]:
        """Return the iterator of the factory.

        Returns:
             iterator of the factory.
        """
        return iter(self.constructors)

    def __getitem__(self, key: str) -> type[T]:
        """Return the type by the key.

        Returns:
            type by the key.

        Raises:
            KeyError: when the key is not in the constructors
        """
        return self.constructors[key]

    def __setitem__(self, key: str, value: type[T]) -> None:
        """Set the value for the key.

        Args:
            key: name of the constructor
            value: type of the constructor
        """
        self.constructors[key] = value

    def __len__(self) -> int:
        """Return the number of constructors in the factory.

        Returns:
            number of constructors in the factory.
        """
        return len(self.constructors)

    def __str__(self) -> str:
        """Return the class name of the factory.

        Returns:
            class name of the factory.
        """
        return Op.class_name(self)

    def __repr__(self) -> str:
        """Return the Python code to construct the factory.

        Returns:
            Python code to construct the factory.
        """
        return representation.as_constructor(self)

    def import_from(self, module: ModuleType, cls: type[T]) -> list[type[T]]:
        """Import constructors from the module.

        Args:
            module: module to import constructors
            cls: class of constructors to import

        Returns:
             list of constructors imported from the module.
        """

        def constructable(obj: type[T]) -> bool:
            try:
                return issubclass(obj, cls) and not inspect.isabstract(obj) and bool(obj())
            except:  # noqa: E722
                return False

        constructors = [
            constructor for _, constructor in inspect.getmembers(module, predicate=constructable)
        ]
        return constructors

    def construct(self, key: str, **kwargs: Any) -> T:
        """Create an object from the constructor registered by the key.

        Args:
            key: name of the constructor
            **kwargs: parameters to pass to the constructor

        Returns:
             object created from the constructor registered by the key

        Raises:
            ValueError: when the key is not registered
        """
        if key in self.constructors:
            return self.constructors[key](**kwargs)
        raise ValueError(f"constructor of '{key}' not found in {Op.class_name(self)}")


class CloningFactory(Generic[T]):
    """Base class for a factory whose objects are created by a deep copy of registered instances.

    info: related
        - [fuzzylite.factory.FunctionFactory][]
        - [fuzzylite.factory.FactoryManager][]
        - [fuzzylite.factory.ConstructionFactory][]
    """

    def __init__(self, objects: dict[str, T] | None = None) -> None:
        """Constructor."""
        self.objects = objects or {}

    def __iter__(self) -> Iterator[str]:
        """Return the iterator of the factory.

        Returns:
             iterator of the factory.
        """
        return iter(self.objects)

    def __getitem__(self, key: str) -> T:
        """Return the object by the key.

        Returns:
            object by the key.

        Raises:
        KeyError: when the key is not in the factory
        """
        return self.objects[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Set the value for the key.

        Args:
            key: name of the object
            value: instance to be deep copied
        """
        self.objects[key] = value

    def __len__(self) -> int:
        """Return the number of objects in the factory.

        Returns:
            number of objects in the factory.
        """
        return len(self.objects)

    def __str__(self) -> str:
        """Return the class name of the factory.

        Returns:
            class name of the factory.
        """
        return Op.class_name(self)

    def __repr__(self) -> str:
        """Return the Python code to construct the factory.

        Returns:
            Python code to construct the factory.
        """
        return representation.as_constructor(self)

    def copy(self, key: str) -> T:
        """Create a deep copy of the object registered by the key.

        Args:
            key: name of the object

        Returns:
             deep copy of the object registered by the key

        Raises:
            ValueError: when the key is not registered.
        """
        if key in self.objects:
            return copy.deepcopy(self.objects[key])
        raise ValueError(f"object with key '{key}' not found in {Op.class_name(self)}")


class ActivationFactory(ConstructionFactory[Activation]):
    """Construction factory of activation methods for rule blocks.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.activation.Activation][]
        - [fuzzylite.rule.RuleBlock][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import activation

        activations = {Op.class_name(a): a for a in self.import_from(activation, Activation)}
        super().__init__(constructors=activations)


class DefuzzifierFactory(ConstructionFactory[Defuzzifier]):
    """Factory of defuzzifiers.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import defuzzifier

        defuzzifiers = {Op.class_name(d): d for d in self.import_from(defuzzifier, Defuzzifier)}
        super().__init__(constructors=defuzzifiers)


class HedgeFactory(ConstructionFactory[Hedge]):
    """Factory of hedges.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import hedge

        hedges = {h().name: h for h in self.import_from(hedge, Hedge)}
        super().__init__(constructors=hedges)


class SNormFactory(ConstructionFactory[SNorm]):
    """Factory of SNorms.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import norm as norm

        snorms = {Op.class_name(n): n for n in self.import_from(norm, SNorm)}
        super().__init__(constructors=snorms)


class TNormFactory(ConstructionFactory[TNorm]):
    """Factory of TNorms.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import norm as norm

        tnorms = {Op.class_name(n): n for n in self.import_from(norm, TNorm)}
        super().__init__(constructors=tnorms)


class TermFactory(ConstructionFactory[Term]):
    """Factory of terms.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.term.Term][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        from . import term as term

        terms = {
            Op.class_name(t): t
            for t in self.import_from(term, Term)
            if t not in {term.Activated, term.Aggregated}
        }
        super().__init__(constructors=terms)


class FunctionFactory(CloningFactory[Function.Element]):
    """Factory of operators and functions used by the Function term.

    info: related
        - [fuzzylite.factory.CloningFactory][]
        - [fuzzylite.term.Function.Element][]
        - [fuzzylite.term.Function][]
        - [fuzzylite.factory.FactoryManager][]
    """

    def __init__(self) -> None:
        """Constructor."""
        elements = {
            element.name: element for element in self._create_operators() + self._create_functions()
        }
        super().__init__(objects=elements)

    def _precedence(self, importance: int) -> int:
        """Inverts the priority of precedence of operations, mapping 0-10 in ascending order to 100-0 in descending order.

        Args:
            importance: value between 0 and 10, where 0 is the most important

        Returns:
             precedence between 100 and 0, where 100 is the most important
        """
        maximum = 100
        step = 10
        return maximum - importance * step

    def _create_operators(self) -> list[Function.Element]:
        """Return the list of function operators.

        Returns:
            list of function operators
        """
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
        """Return the list of functions.

        Returns:
            list of functions
        """
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
        """Return a dictionary of the operators available.

        Returns:
             dictionary of the operators available.
        """
        result = {
            key: prototype for key, prototype in self.objects.items() if prototype.is_operator()
        }
        return result

    def functions(self) -> dict[str, Function.Element]:
        """Return a dictionary of the functions available.

        Returns:
            dictionary of the functions available.
        """
        result = {
            key: prototype for key, prototype in self.objects.items() if prototype.is_function()
        }
        return result


class FactoryManager:
    """Manager that groups different factories to facilitate access across the library.

    info: related
        - [fuzzylite.factory.ConstructionFactory][]
        - [fuzzylite.factory.CloningFactory][]
        - [fuzzylite.factory.TermFactory][]
        - [fuzzylite.factory.TNormFactory][]
        - [fuzzylite.factory.SNormFactory][]
        - [fuzzylite.factory.HedgeFactory][]
        - [fuzzylite.factory.ActivationFactory][]
        - [fuzzylite.factory.DefuzzifierFactory][]
        - [fuzzylite.factory.FunctionFactory][]
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
        """Constructor.

        Args:
            tnorm: factory of TNorms
            snorm: factory of SNorms
            activation: factory of activation methods
            defuzzifier: factory of defuzzifiers
            term: factory of terms
            hedge: factory of hedges
            function: factory of functions
        """
        self.tnorm = tnorm or TNormFactory()
        self.snorm = snorm or SNormFactory()
        self.activation = activation or ActivationFactory()
        self.defuzzifier = defuzzifier or DefuzzifierFactory()
        self.term = term or TermFactory()
        self.hedge = hedge or HedgeFactory()
        self.function = function or FunctionFactory()
