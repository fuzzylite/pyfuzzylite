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
    "Norm",
    "TNorm",
    "AlgebraicProduct",
    "BoundedDifference",
    "DrasticProduct",
    "EinsteinProduct",
    "HamacherProduct",
    "Minimum",
    "NilpotentMinimum",
    "SNorm",
    "AlgebraicSum",
    "BoundedSum",
    "DrasticSum",
    "EinsteinSum",
    "HamacherSum",
    "Maximum",
    "NilpotentMaximum",
    "NormalizedSum",
    "UnboundedSum",
    "NormLambda",
    "NormFunction",
]

import typing
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .library import representation, scalar
from .operation import Op

if typing.TYPE_CHECKING:
    from .term import Function
    from .types import Scalar


class Norm(ABC):
    """The Norm class is the abstract class for norms.
    @author Juan Rada-Vilela, Ph.D.
    @see TNorm
    @see SNorm
    @see TNormFactory
    @see SNormFactory
    @since 4.0.
    """

    def __str__(self) -> str:
        """Gets a string representation in the FuzzyLite Language."""
        return Op.to_fll(self)

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return representation.as_constructor(self)

    @abstractmethod
    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        """Computes the norm for $a$ and $b$
        @param a is a membership function value
        @param b is a membership function value
        @return the norm between $a$ and $b$.
        """
        raise NotImplementedError()


class TNorm(Norm):
    """The TNorm class is the base class for T-Norms, and it is utilized as the
    conjunction fuzzy logic operator and as the implication (or `activation`
    in versions 5.0 and earlier) fuzzy logic operator.
    @author Juan Rada-Vilela, Ph.D.
    @see RuleBlock::getConjunction()
    @see RuleBlock::getImplication()
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    pass


class SNorm(Norm):
    """The SNorm class is the base class for all S-Norms, and it is utilized as
    the disjunction fuzzy logic operator and as the aggregation (or
    `accumulation` in versions 5.0 and earlier) fuzzy logic operator.
    @author Juan Rada-Vilela, Ph.D.
    @see RuleBlock::getDisjunction()
    @see OutputVariable::fuzzyOutput()
    @see Aggregated::getAggregation()
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    pass


class AlgebraicProduct(TNorm):
    """The AlgebraicProduct class is a TNorm that computes the algebraic product
    of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see AlgebraicSum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the algebraic product of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $a\times b$.
        """
        a = scalar(a)
        b = scalar(b)
        return a * b


class BoundedDifference(TNorm):
    """The BoundedDifference class is a TNorm that computes the bounded
    difference between any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see BoundedSum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the bounded difference between two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\max(0, a+b - 1)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.maximum(0, a + b - 1)


class DrasticProduct(TNorm):
    """The DrasticProduct class is a TNorm that computes the drastic product of
    any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see DrasticSum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the drastic product of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\begin{cases}
        \min(a,b) & \mbox{if $\max(a,b)=1$} \cr
        0 & \mbox{otherwise}
        \end{cases}$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(np.maximum(a, b) == 1.0, np.minimum(a, b), 0.0)


class EinsteinProduct(TNorm):
    """The EinsteinProduct class is a TNorm that computes the Einstein product
    of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see EinsteinSum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the Einstein product of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $(a\times b)/(2-(a+b-a\times b))$.
        """
        a = scalar(a)
        b = scalar(b)
        return (a * b) / (2.0 - (a + b - a * b))


class HamacherProduct(TNorm):
    """The HamacherProduct class is a TNorm that computes the Hamacher product
    of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see HamacherSum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.

    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the Hamacher product of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $(a \times b) / (a+b- a \times b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b != 0.0, (a * b) / (a + b - a * b), 0.0)


class Minimum(TNorm):
    """The Minimum class is a TNorm that computes the minimum of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see Maximum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the minimum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\min(a,b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.minimum(a, b)


class NilpotentMinimum(TNorm):
    """The NilpotentMinimum class is a TNorm that computes the nilpotent minimum
    of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see NilpotentMaximum
    @see TNorm
    @see TNormFactory
    @see Norm
    @since 5.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the nilpotent minimum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\begin{cases}
        \min(a,b) & \mbox{if $a+b>1$} \cr
        0 & \mbox{otherwise}
        \end{cases}$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b > 1.0, np.minimum(a, b), 0.0)


class AlgebraicSum(SNorm):
    """The AlgebraicSum class is an SNorm that computes the algebraic sum of
    values any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see AlgebraicProduct
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the algebraic sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $a+b-(a \times b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return a + b - (a * b)


class BoundedSum(SNorm):
    """The BoundedSum class is an SNorm that computes the bounded sum of any two
    values.
    @author Juan Rada-Vilela, Ph.D.
    @see BoundedDifference
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the bounded sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\min(1, a+b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.minimum(1.0, a + b)


class DrasticSum(SNorm):
    """The DrasticSum class is an SNorm that computes the drastic sum of any two
    values.
    @author Juan Rada-Vilela, Ph.D.
    @see DrasticProduct
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the drastic sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\begin{cases}
        \max(a,b) & \mbox{if $\min(a,b)=0$} \cr
        1 & \mbox{otherwise}
        \end{cases}$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(np.minimum(a, b) == 0.0, np.maximum(a, b), 1.0)


class EinsteinSum(SNorm):
    """The EinsteinSum class is an SNorm that computes the einstein sum of any
    two values.
    @author Juan Rada-Vilela, Ph.D.
    @see EinsteinProduct
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the Einstein sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $a+b/(1+a \times b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return (a + b) / (1.0 + a * b)


class HamacherSum(SNorm):
    """The HamacherSum class is an SNorm that computes the Hamacher sum of any
    two values.
    @author Juan Rada-Vilela, Ph.D.
    @see HamacherProduct
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the Hamacher sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $a+b-(2\times a \times b)/(1-a\times b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a * b != 1.0, (a + b - 2.0 * a * b) / (1.0 - a * b), 1.0)


class Maximum(SNorm):
    """The Maximum class is an SNorm that computes the maximum of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see Minimum
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the maximum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\max(a,b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.maximum(a, b)


class NilpotentMaximum(SNorm):
    """The NilpotentMaximum class is an SNorm that computes the nilpotent
    maximum of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see NilpotentMinimum
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 5.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the nilpotent maximum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\begin{cases}
        \max(a,b) & \mbox{if $a+b<0$} \cr
        1 & \mbox{otherwise}
        \end{cases}$.
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b < 1.0, np.maximum(a, b), 1.0)


class NormalizedSum(SNorm):
    """The NormalizedSum class is an SNorm that computes the normalized sum of
    any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the normalized sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $(a+b)/\max(1, a + b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return (a + b) / np.maximum(1.0, a + b)


class UnboundedSum(SNorm):
    """The UnboundedSum class is an SNorm that computes the sum of any two values.
    @author Juan Rada-Vilela, Ph.D.
    @see BoundedSum
    @see SNorm
    @see SNormFactory
    @see Norm
    @since 4.0.
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the bounded sum of two membership function values
        @param a is a membership function value
        @param b is a membership function value
        @return $\min(1, a+b)$.
        """
        a = scalar(a)
        b = scalar(b)
        return a + b


class NormLambda(TNorm, SNorm):
    """The NormLambda class is a customizable Norm via Lambda, which
    computes any lambda based on the $a$ and $b$ values.
    This Norm is not registered with the SNormFactory or the TNormFactory.
    @author Juan Rada-Vilela, Ph.D.
    @see SNorm
    @see TNorm
    @see Norm
    @see SNormFactory
    @see TNormFactory
    @since 7.0.

    """

    def __init__(self, function: Callable[[Scalar, Scalar], Scalar]) -> None:
        """Create the norm.
        @param function is a binary function.

        """
        self.function = function

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        """Computes the Norm utilizing the given lambda, which automatically assigns the values
        of $a$ and $b$.
        @param a is a membership function value
        @param b is a membership function value
        @return the evaluation of the function.

        """
        return self.function(a, b)


class NormFunction(TNorm, SNorm):
    """The NormFunction class is a customizable Norm via Function, which
    computes any function based on the $a$ and $b$ values.
    This Norm is not registered with the SNormFactory or the TNormFactory.
    @author Juan Rada-Vilela, Ph.D.
    @see Function
    @see SNorm
    @see TNorm
    @see Norm
    @see SNormFactory
    @see TNormFactory
    @since 6.0.

    """

    def __init__(self, function: Function) -> None:
        """Create the norm.
        @param function is a binary function.

        """
        self.function = function

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        """Computes the Norm utilizing the given function, which automatically assigns the values
        of $a$ and $b$.
        @param a is a membership function value
        @param b is a membership function value
        @return the evaluation of the function.

        """
        return self.function.evaluate({"a": a, "b": b})
