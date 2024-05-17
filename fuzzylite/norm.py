"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
from .types import Scalar

if typing.TYPE_CHECKING:
    from .term import Function


class Norm(ABC):
    """Abstract class for norms.

    | [fuzzylite.norm.TNorm][]                      	| [fuzzylite.norm.SNorm][]                     	|
    |-----------------------------------------------	|----------------------------------------------	|
    | [fuzzylite.norm.AlgebraicProduct][]           	| [fuzzylite.norm.AlgebraicSum][]              	|
    | ![](../../image/norm/T-AlgebraicProduct.svg)  	| ![](../../image/norm/S-AlgebraicSum.svg)     	|
    | [fuzzylite.norm.BoundedDifference][]          	| [fuzzylite.norm.BoundedSum][]                	|
    | ![](../../image/norm/T-BoundedDifference.svg) 	| ![](../../image/norm/S-BoundedSum.svg)       	|
    | [fuzzylite.norm.DrasticProduct][]             	| [fuzzylite.norm.DrasticSum][]                	|
    | ![](../../image/norm/T-DrasticProduct.svg)    	| ![](../../image/norm/S-DrasticSum.svg)       	|
    | [fuzzylite.norm.EinsteinProduct][]            	| [fuzzylite.norm.EinsteinSum][]               	|
    | ![](../../image/norm/T-EinsteinProduct.svg)   	| ![](../../image/norm/S-EinsteinSum.svg)      	|
    | [fuzzylite.norm.HamacherProduct][]            	| [fuzzylite.norm.HamacherSum][]               	|
    | ![](../../image/norm/T-HamacherProduct.svg)   	| ![](../../image/norm/S-HamacherSum.svg)      	|
    | [fuzzylite.norm.Minimum][]                    	| [fuzzylite.norm.Maximum][]                   	|
    | ![](../../image/norm/T-Minimum.svg)           	| ![](../../image/norm/S-Maximum.svg)          	|
    | [fuzzylite.norm.NilpotentMinimum][]           	| [fuzzylite.norm.NilpotentMaximum][]          	|
    | ![](../../image/norm/T-NilpotentMinimum.svg)  	| ![](../../image/norm/S-NilpotentMaximum.svg) 	|
    |                                               	| [fuzzylite.norm.NormalizedSum][]             	|
    |                                               	| ![](../../image/norm/S-NormalizedSum.svg)    	|
    |                                               	| [fuzzylite.norm.UnboundedSum][]              	|
    |                                               	| ![](../../image/norm/S-UnboundedSum.svg)     	|

    info: related
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.factory.SNormFactory][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def __str__(self) -> str:
        """Return the code to construct the norm in the FuzzyLite Language.

        Returns:
            code to construct the norm in the FuzzyLite Language.
        """
        return representation.fll.norm(self)

    def __repr__(self) -> str:
        """Return the code to construct the norm in Python.

        Returns:
            code to construct the norm in Python.
        """
        return representation.as_constructor(self)

    @abstractmethod
    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Implement the norm.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             norm between $a$ and $b$
        """


class TNorm(Norm):
    """Base class for T-Norms, used as fuzzy logic operator for conjunction and implication in rule blocks.

    | [fuzzylite.norm.TNorm][]                      	|
    |-----------------------------------------------	|
    | [fuzzylite.norm.AlgebraicProduct][]           	|
    | ![](../../image/norm/T-AlgebraicProduct.svg)  	|
    | [fuzzylite.norm.BoundedDifference][]          	|
    | ![](../../image/norm/T-BoundedDifference.svg) 	|
    | [fuzzylite.norm.DrasticProduct][]             	|
    | ![](../../image/norm/T-DrasticProduct.svg)    	|
    | [fuzzylite.norm.EinsteinProduct][]            	|
    | ![](../../image/norm/T-EinsteinProduct.svg)   	|
    | [fuzzylite.norm.HamacherProduct][]            	|
    | ![](../../image/norm/T-HamacherProduct.svg)   	|
    | [fuzzylite.norm.Minimum][]                    	|
    | ![](../../image/norm/T-Minimum.svg)           	|
    | [fuzzylite.norm.NilpotentMinimum][]           	|
    | ![](../../image/norm/T-NilpotentMinimum.svg)  	|

    info: related
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.term.Activated][]
        - [fuzzylite.rule.RuleBlock][]
        - [fuzzylite.factory.TNormFactory][]
    """

    @abstractmethod
    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Implement the T-Norm $a \otimes b$.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \otimes b$
        """


class SNorm(Norm):
    """Base class for S-Norms, used as fuzzy logic operator for disjunction and aggregation in rule blocks.

    | [fuzzylite.norm.SNorm][]                     	|
    |----------------------------------------------	|
    | [fuzzylite.norm.AlgebraicSum][]              	|
    | ![](../../image/norm/S-AlgebraicSum.svg)     	|
    | [fuzzylite.norm.BoundedSum][]                	|
    | ![](../../image/norm/S-BoundedSum.svg)       	|
    | [fuzzylite.norm.DrasticSum][]                	|
    | ![](../../image/norm/S-DrasticSum.svg)       	|
    | [fuzzylite.norm.EinsteinSum][]               	|
    | ![](../../image/norm/S-EinsteinSum.svg)      	|
    | [fuzzylite.norm.HamacherSum][]               	|
    | ![](../../image/norm/S-HamacherSum.svg)      	|
    | [fuzzylite.norm.Maximum][]                   	|
    | ![](../../image/norm/S-Maximum.svg)          	|
    | [fuzzylite.norm.NilpotentMaximum][]          	|
    | ![](../../image/norm/S-NilpotentMaximum.svg) 	|
    | [fuzzylite.norm.NormalizedSum][]             	|
    | ![](../../image/norm/S-NormalizedSum.svg)    	|
    | [fuzzylite.norm.UnboundedSum][]              	|
    | ![](../../image/norm/S-UnboundedSum.svg)     	|

    info: related
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.term.Aggregated][]
        - [fuzzylite.rule.RuleBlock][]
        - [fuzzylite.factory.TNormFactory][]
    """

    @abstractmethod
    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Implement the S-Norm $a \oplus b$.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \oplus b$
        """


class AlgebraicProduct(TNorm):
    r"""TNorm to compute the algebraic product of any two values.

    ![](../../image/norm/T-AlgebraicProduct.svg)

    Note: Equation
        $a \otimes b=a\times b$

    info: related
        - [fuzzylite.norm.AlgebraicSum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the algebraic product of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \otimes b=a\times b$
        """
        a = scalar(a)
        b = scalar(b)
        return a * b


class BoundedDifference(TNorm):
    r"""TNorm to compute the bounded difference between any two values.

    ![](../../image/norm/T-BoundedDifference.svg)

    Note: Equation
        $a \otimes b=\max(0, a + b - 1)$

    info: related
        - [fuzzylite.norm.BoundedSum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the bounded difference between two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \otimes b=\max(0, a + b - 1)$
        """
        a = scalar(a)
        b = scalar(b)
        return np.maximum(0, a + b - 1)


class DrasticProduct(TNorm):
    r"""TNorm to compute the drastic product of any two values.

    ![](../../image/norm/T-DrasticProduct.svg)

    Note: Equation
        $a \otimes b = \begin{cases}
            \min(a,b) & \mbox{if } \max(a,b)=1 \cr
            0 & \mbox{otherwise}
        \end{cases}$

    info: related
        - [fuzzylite.norm.DrasticSum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the drastic product of two membership function values.

        Args:
            a: membership function value
            b: membership function value.

        Returns:
            $a \otimes b = \begin{cases} \min(a,b) & \mbox{if } \max(a,b)=1 \cr 0 & \mbox{otherwise} \end{cases}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(np.maximum(a, b) == 1.0, np.minimum(a, b), 0.0)


class EinsteinProduct(TNorm):
    r"""TNorm to compute the Einstein product of any two values.

    ![](../../image/norm/T-EinsteinProduct.svg)

    Note: Equation
        $a \otimes b=\dfrac{a\times b}{2-(a+b-a\times b)}$

    info: related
        - [fuzzylite.norm.EinsteinSum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the Einstein product of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \otimes b=\dfrac{a\times b}{2-(a+b-a\times b)}$
        """
        a = scalar(a)
        b = scalar(b)
        return (a * b) / (2.0 - (a + b - a * b))


class HamacherProduct(TNorm):
    r"""TNorm to compute the Hamacher product of any two values.

    ![](../../image/norm/T-HamacherProduct.svg)

    Note: Equation
        $a \otimes b=\dfrac{a \times b}{a+b- a \times b}$

    info: related
        - [fuzzylite.norm.HamacherSum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the Hamacher product of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \otimes b=\dfrac{a \times b}{a+b- a \times b}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b != 0.0, (a * b) / (a + b - a * b), 0.0)


class Minimum(TNorm):
    r"""TNorm that computes the minimum of any two values.

    ![](../../image/norm/T-Minimum.svg)

    Note: Equation
        $a \otimes b=\min(a,b)$

    info: related
        - [fuzzylite.norm.Maximum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the minimum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \otimes b=\min(a,b)$
        """
        a = scalar(a)
        b = scalar(b)
        return np.minimum(a, b)


class NilpotentMinimum(TNorm):
    r"""TNorm to compute the nilpotent minimum of any two values.

    ![](../../image/norm/T-NilpotentMinimum.svg)

    Note: Equation
        $a \otimes b=\begin{cases}
            \min(a,b) & \mbox{if }a+b>1 \cr
            0 & \mbox{otherwise}
        \end{cases}$

    info: related
        - [fuzzylite.norm.NilpotentMaximum][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the nilpotent minimum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \otimes b=\begin{cases}  \min(a,b) & \mbox{if }a+b>1 \cr  0 & \mbox{otherwise}  \end{cases}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b > 1.0, np.minimum(a, b), 0.0)


class AlgebraicSum(SNorm):
    r"""SNorm to compute the algebraic sum of values any two values.

    ![](../../image/norm/S-AlgebraicSum.svg)

    Note: Equation
        $a \oplus b=a+b-(a \times b)$

    info: related
        - [fuzzylite.norm.AlgebraicProduct][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the algebraic sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \oplus b=a+b-(a \times b)$
        """
        a = scalar(a)
        b = scalar(b)
        return a + b - (a * b)


class BoundedSum(SNorm):
    r"""SNorm to compute the bounded sum of any two values.

    ![](../../image/norm/S-BoundedSum.svg)

    Note: Equation
        $a \oplus b=\min(1, a+b)$

    info: related
        - [fuzzylite.norm.BoundedDifference][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the bounded sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\min(1, a+b)$
        """
        a = scalar(a)
        b = scalar(b)
        return np.minimum(1.0, a + b)


class DrasticSum(SNorm):
    r"""SNorm to compute the drastic sum of any two values.

    ![](../../image/norm/S-DrasticSum.svg)

    Note: Equation
        $a \oplus b=\begin{cases}
            \max(a,b) & \mbox{if } \min(a,b)=0 \cr
             1 & \mbox{otherwise}
        \end{cases}$

    info: related
        - [fuzzylite.norm.DrasticProduct][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the drastic sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\begin{cases}  \max(a,b) & \mbox{if } \min(a,b)=0 \cr  1 & \mbox{otherwise}  \end{cases}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(np.minimum(a, b) == 0.0, np.maximum(a, b), 1.0)


class EinsteinSum(SNorm):
    r"""SNorm to compute the einstein sum of any two values.

    ![](../../image/norm/S-EinsteinSum.svg)

    Note: Equation
        $a \oplus b=\dfrac{a+b}{1+a \times b}$

    info: related
        - [fuzzylite.norm.EinsteinProduct][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the Einstein sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\dfrac{a+b}{1+a \times b}$
        """
        a = scalar(a)
        b = scalar(b)
        return (a + b) / (1.0 + a * b)


class HamacherSum(SNorm):
    r"""SNorm to compute the Hamacher sum of any two values.

    ![](../../image/norm/S-HamacherSum.svg)

    Note: Equation
        $a \oplus b=\dfrac{a+b-2(\times a \times b)}{1-a\times b}$

    info: related
        - [fuzzylite.norm.HamacherProduct][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the Hamacher sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\dfrac{a+b-2(\times a \times b)}{1-a\times b}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a * b != 1.0, (a + b - 2.0 * a * b) / (1.0 - a * b), 1.0)


class Maximum(SNorm):
    r"""SNorm to compute the maximum of any two values.

    ![](../../image/norm/S-Maximum.svg)

    Note: Equation
        $a \oplus b=\max(a,b)$

    info: related
       - [fuzzylite.norm.Minimum][]
       - [fuzzylite.norm.SNorm][]
       - [fuzzylite.norm.Norm][]
       - [fuzzylite.factory.SNormFactory][]

    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Computes the maximum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\max(a,b)$
        """
        a = scalar(a)
        b = scalar(b)
        return np.maximum(a, b)


class NilpotentMaximum(SNorm):
    r"""SNorm to compute the nilpotent maximum of any two values.

    ![](../../image/norm/S-NilpotentMaximum.svg)

    Note: Equation
        $a \oplus b=\begin{cases}
        \max(a,b) & \mbox{if } a+b<0 \cr
        1 & \mbox{otherwise}
        \end{cases}$

    info: related
       - [fuzzylite.norm.NilpotentMinimum][]
       - [fuzzylite.norm.SNorm][]
       - [fuzzylite.norm.Norm][]
       - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the nilpotent maximum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=\begin{cases} \max(a,b) & \mbox{if } a+b<0 \cr 1 & \mbox{otherwise} \end{cases}$
        """
        a = scalar(a)
        b = scalar(b)
        return np.where(a + b < 1.0, np.maximum(a, b), 1.0)


class NormalizedSum(SNorm):
    r"""SNorm to compute the normalized sum of any two values.

    ![](../../image/norm/S-NormalizedSum.svg)

    Note: Equation
        $a \oplus b=\dfrac{a+b}{\max(1, a + b)}$

    info: related
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the normalized sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
             $a \oplus b=\dfrac{a+b}{\max(1, a + b)}$
        """
        a = scalar(a)
        b = scalar(b)
        return (a + b) / np.maximum(1.0, a + b)


class UnboundedSum(SNorm):
    r"""SNorm to compute the sum of any two values.

    ![](../../image/norm/S-UnboundedSum.svg)

    Note: Equation
        $a \oplus b=a+b$

    info: related
        - [fuzzylite.norm.BoundedSum][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
    """

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the sum of two membership function values.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=a+b$
        """
        a = scalar(a)
        b = scalar(b)
        return a + b


class NormLambda(TNorm, SNorm):
    r"""TNorm or SNorm based on a $\lambda$ function on any two values.

    Note: Equation
        $a \oplus b = a \otimes b = \lambda(a,b)$

    This Norm is not registered in the SNormFactory or TNormFactory.

    info: related
        - [fuzzylite.norm.NormFunction][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def __init__(self, function: Callable[[Scalar, Scalar], Scalar]) -> None:
        r"""Constructor.

        Args:
            function: function $\lambda(a,b)$.
        """
        self.function = function

    def __repr__(self) -> str:
        """Return the code to construct the norm in Python.

        Returns:
            code to construct the norm in Python.
        """
        return f"{Op.class_name(self, qualname=True)}(lambda a, b: ...)"

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the norm using $\lambda(a,b)$.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b = a \otimes b = \lambda(a,b)$
        """
        return self.function(a, b)


class NormFunction(TNorm, SNorm):
    r"""TNorm or SNorm based on a term function on any two values.

    Note: Equation
        $a \oplus b = a \otimes b = f(a,b)$

    This Norm is not registered in the SNormFactory or TNormFactory.

    info: related
        - [fuzzylite.norm.NormLambda][]
        - [fuzzylite.norm.SNorm][]
        - [fuzzylite.norm.TNorm][]
        - [fuzzylite.norm.Norm][]
        - [fuzzylite.factory.SNormFactory][]
        - [fuzzylite.factory.TNormFactory][]
    """

    def __init__(self, function: Function) -> None:
        r"""Constructor.

        Args:
            function: function $f(a,b)$.
        """
        self.function = function

    def __repr__(self) -> str:
        """Return the code to construct the norm in Python.

        Returns:
            code to construct the norm in Python.
        """
        return representation.as_constructor(self, positional=True)

    def compute(self, a: Scalar, b: Scalar) -> Scalar:
        r"""Compute the Norm using $f(a,b)$.

        Args:
            a: membership function value
            b: membership function value

        Returns:
            $a \oplus b=f(a,b)$
        """
        return self.function.evaluate({"a": a, "b": b})
