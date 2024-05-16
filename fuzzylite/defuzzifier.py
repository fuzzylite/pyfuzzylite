"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = [
    "Defuzzifier",
    "IntegralDefuzzifier",
    "Bisector",
    "Centroid",
    "LargestOfMaximum",
    "MeanOfMaximum",
    "SmallestOfMaximum",
    "WeightedDefuzzifier",
    "WeightedAverage",
    "WeightedSum",
]

import enum
import typing
import warnings
from abc import ABC, abstractmethod
from typing import Final

import numpy as np

from .library import nan, representation, scalar
from .operation import Op
from .term import Activated, Aggregated, Constant, Function, Linear, Term
from .types import Scalar

if typing.TYPE_CHECKING:
    from .variable import Variable


class Defuzzifier(ABC):
    """Abstract class for defuzzifiers.

    info: related
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
            - [fuzzylite.defuzzifier.Bisector][]
            - [fuzzylite.defuzzifier.Centroid][]
            - [fuzzylite.defuzzifier.LargestOfMaximum][]
            - [fuzzylite.defuzzifier.MeanOfMaximum][]
            - [fuzzylite.defuzzifier.SmallestOfMaximum][]
        - [fuzzylite.defuzzifier.WeightedDefuzzifier][]
            - [fuzzylite.defuzzifier.WeightedAverage][]
            - [fuzzylite.defuzzifier.WeightedSum][]
        - [fuzzylite.variable.OutputVariable][]
        - [fuzzylite.term.Aggregated][]
        - [fuzzylite.term.Activated][]
    """

    def __str__(self) -> str:
        """Return the code to construct the defuzzifier in the FuzzyLite Language.

        Returns:
            code to construct the defuzzifier in the FuzzyLite Language.
        """
        return representation.fll.defuzzifier(self)

    def __repr__(self) -> str:
        """Return the code to construct the defuzzifier in Python.

        Returns:
            code to construct the defuzzifier in Python.
        """
        return representation.as_constructor(self)

    def configure(  # noqa: B027 empty method in an abstract base class
        self, parameters: str
    ) -> None:
        """Configure the defuzzifier with the parameters.

        Args:
            parameters: space-separated parameter values
        """
        pass

    def parameters(self) -> str:
        """Return the space-separated parameters of the defuzzifier.

        Returns:
            space-separated parameters of the defuzzifier.
        """
        return ""

    @abstractmethod
    def defuzzify(  # noqa: B027 empty method in an abstract base class
        self, term: Term, minimum: float, maximum: float
    ) -> Scalar:
        """Defuzzify the term using the range `[minimum,maximum]`.

        Args:
            term: term to defuzzify, typically an Aggregated term
            minimum: minimum value of the range
            maximum: maximum value of the range

        Returns:
            defuzzified value of the term.
        """


class IntegralDefuzzifier(Defuzzifier):
    """Abstract class for defuzzifiers that integrate over the fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.Bisector][]
        - [fuzzylite.defuzzifier.Centroid][]
        - [fuzzylite.defuzzifier.LargestOfMaximum][]
        - [fuzzylite.defuzzifier.MeanOfMaximum][]
        - [fuzzylite.defuzzifier.SmallestOfMaximum][]
    """

    default_resolution: Final[int] = 1000

    def __init__(self, resolution: int | None = None) -> None:
        """Constructor.

        Args:
            resolution: number of divisions to discretize the range and compute the area under the curve.
        """
        self.resolution = resolution or IntegralDefuzzifier.default_resolution

    def __repr__(self) -> str:
        """Return the code to construct the defuzzifier in Python.

        Returns:
            code to construct the defuzzifier in Python.
        """
        fields = vars(self).copy()
        if self.resolution == IntegralDefuzzifier.default_resolution:
            fields.pop("resolution")
        return representation.as_constructor(self, fields)

    def parameters(self) -> str:
        """Return the parameters to configure the defuzzifier.

        Returns:
            parameters to configure the defuzzifier.
        """
        if self.resolution != IntegralDefuzzifier.default_resolution:
            return Op.str(self.resolution)
        return ""

    def configure(self, parameters: str) -> None:
        """Configure the defuzzifier with the parameters.

        Args:
             parameters: list of space-separated parameter values
        """
        if parameters:
            self.resolution = int(parameters)

    @abstractmethod
    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Implement the defuzzification of the term using the given range.

        Args:
            term: term to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
            defuzzified value.
        """


class Bisector(IntegralDefuzzifier):
    """Integral defuzzifier that computes the bisector of a fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
        - [fuzzylite.defuzzifier.Centroid][]
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Compute the bisector of a fuzzy set, that is, the x-coordinate such that the area to its left is approximately equal to the area to its right.

        The defuzzification process integrates over the fuzzy set using the given range.
        The integration algorithm is the midpoint rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            term: fuzzy set to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
            $x$-coordinate of the bisector of the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        area = np.nancumsum(y, axis=1)
        # normalising the cumulative sum is not necessary, but it is convenient because it results in nan
        # when arrays are full of nans (ie, area = 0). Otherwise, result would be minimum + (maximum-minimum)/2
        area = np.abs((area / area[:, [-1]]) - 0.5)
        index = area == area.min(axis=1, keepdims=True)
        bisectors = np.where(index, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmean(bisectors, axis=1).squeeze()
            return z  # type: ignore


class Centroid(IntegralDefuzzifier):
    """Integral defuzzifier that computes the centroid of a fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
        - [fuzzylite.defuzzifier.Bisector][]
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Compute the centroid of a fuzzy set.

        The defuzzification process integrates over the fuzzy set using the given range.
        The integration algorithm is the midpoint rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            term: fuzzy set to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
            $x$-coordinate of the centroid of the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        z = ((x * y).sum(axis=1) / y.sum(axis=1)).squeeze()
        return z  # type: ignore


class LargestOfMaximum(IntegralDefuzzifier):
    """Integral defuzzifier that computes the largest value of the maximum membership function of a fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
        - [fuzzylite.defuzzifier.MeanOfMaximum][]
        - [fuzzylite.defuzzifier.SmallestOfMaximum][]
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Compute the largest value of the maximum membership function of a fuzzy set.

        The defuzzification process integrates over the fuzzy set using the given range.
        The integration algorithm is the midpoint rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            term: fuzzy set to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
            largest $x$-coordinate of the maximum membership function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        lom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmax(lom, axis=1).squeeze()
            return z  # type: ignore


class MeanOfMaximum(IntegralDefuzzifier):
    """Integral defuzzifier that computes the mean value of the maximum membership function of a fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
        - [fuzzylite.defuzzifier.SmallestOfMaximum][]
        - [fuzzylite.defuzzifier.LargestOfMaximum][]
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Compute the mean value of the maximum membership function of a fuzzy set.

        The defuzzification process integrates over the fuzzy set using the given range.
        The integration algorithm is the midpoint rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            term: fuzzy set to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
             mean $x$-coordinate of the maximum membership function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        mom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmean(mom, axis=1).squeeze()
            return z  # type: ignore


class SmallestOfMaximum(IntegralDefuzzifier):
    """Integral defuzzifier that computes the smallest value of the maximum membership function of a fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.IntegralDefuzzifier][]
        - [fuzzylite.defuzzifier.MeanOfMaximum][]
        - [fuzzylite.defuzzifier.LargestOfMaximum][]
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Compute the smallest value of the maximum membership function in the fuzzy set.

        The defuzzification process integrates over the fuzzy set using the given range.
        The integration algorithm is the midpoint rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            term: fuzzy set to defuzzify
            minimum: value to start defuzzification
            maximum: value to end defuzzification

        Returns:
            smallest $x$-coordinate of the maximum membership function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        som = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmin(som, axis=1).squeeze()
            return z  # type: ignore


class WeightedDefuzzifier(Defuzzifier):
    """Abstract class for defuzzifiers that compute a weighted function on the fuzzy set.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.WeightedAverage][]
        - [fuzzylite.defuzzifier.WeightedSum][]
    """

    @enum.unique
    class Type(enum.Enum):
        """Type of the weighted defuzzifier based on the terms in the fuzzy set.

        - `Automatic`: Automatically inferred from the terms
        - `TakagiSugeno`: Manually set to TakagiSugeno (or Inverse Tsukamoto)
        - `Tsukamoto`: Manually set to Tsukamoto
        """

        Automatic = enum.auto()
        TakagiSugeno = enum.auto()
        Tsukamoto = enum.auto()

        def __repr__(self) -> str:
            """Return the code to identify the type of defuzzifier in Python.

            Returns:
                code to identify the type of defuzzifier in Python.
            """
            return f"'{self.name}'"

    def __init__(
        self,
        type: str | WeightedDefuzzifier.Type = Type.Automatic,
    ) -> None:
        """Constructor.

        Args:
            type: name or type of the weighted defuzzifier.
        """
        if isinstance(type, str):
            self.type = WeightedDefuzzifier.Type[type]
        else:
            self.type = type

    def __repr__(self) -> str:
        """Return the code to construct the defuzzifier in Python.

        Returns:
            code to construct the defuzzifier in Python.
        """
        fields = vars(self).copy()
        if self.type == WeightedDefuzzifier.Type.Automatic:
            fields.pop("type")
        return representation.as_constructor(self, fields)

    def parameters(self) -> str:
        """Return the parameters to configure the defuzzifier.

        Returns:
            parameters to configure the defuzzifier.
        """
        if self.type == WeightedDefuzzifier.Type.Automatic:
            return ""
        return self.type.name

    def configure(self, parameters: str) -> None:
        """Configure the defuzzifier with the parameters.

        Args:
             parameters: list of space-separated parameter values
        """
        if parameters:
            self.type = WeightedDefuzzifier.Type[parameters]

    @classmethod
    def infer_type(cls, component: Term | Variable, /) -> WeightedDefuzzifier.Type:
        """Infer the type of the defuzzifier based on the component.

        Args:
            component: term or variable to infer the type for

        Returns:
             inferred type of the defuzzifier based on the component.
        """
        from .variable import Variable

        if isinstance(component, (Aggregated, Variable)):
            types = {cls.infer_type(t_i) for t_i in component.terms}
            if len(types) == 1:
                return types.pop()
            if len(types) == 0:
                # cannot infer type of empty term, and won't matter anyway,
                return WeightedDefuzzifier.Type.Automatic
            raise TypeError(
                f"cannot infer type of {cls.__name__}, got multiple types: {sorted(str(t) for t in types)}"
            )
        elif isinstance(component, Activated):  # noqa: RET506 - False Positive
            return cls.infer_type(component.term)
        elif isinstance(component, (Constant, Linear, Function)):
            return WeightedDefuzzifier.Type.TakagiSugeno
        elif component.is_monotonic():
            return WeightedDefuzzifier.Type.Tsukamoto
        else:
            # Inverse Tsukamoto: non-monotonic terms that are not TakagiSugeno
            return WeightedDefuzzifier.Type.Automatic

    @abstractmethod
    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        """Implement the defuzzification of the term.

        Warning:
            From version 8, the aggregation operator is used to aggregate multiple activations of the same term.

            In previous versions, the implication and aggregation operators are not used for weighted defuzzification.

        Args:
            term: term to defuzzify
            minimum: irrelevant
            maximum: irrelevant

        Returns:
            defuzzified value
        """


class WeightedAverage(WeightedDefuzzifier):
    """Weighted defuzzifier that computes the weighted average of a fuzzy set represented by an Aggregated term.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.WeightedDefuzzifier][]
        - [fuzzylite.defuzzifier.WeightedSum][]
        - [fuzzylite.term.Aggregated][]
    """

    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        r"""Computes the weighted average of the fuzzy set.

        The fuzzy set is represented by an Aggregated Term as $y = \sum_i{w_iz_i}$,
        where $w_i$ is the activation degree of term $i$, and $z_i = \mu_i(w_i)$.

        In Takagi-Sugeno controllers, the membership function $\mu_i(w_i)$ is generally a Constant, Linear, or Function
        term, which typically disregards the $w_i$ value.

        Warning:
            From version 8, the aggregation operator is used to aggregate multiple activations of the same term.

            In previous versions, the implication and aggregation operators are not used for weighted defuzzification.

        Args:
            term: term to defuzzify
            minimum: irrelevant
            maximum: irrelevant

        Returns:
            weighted average of the fuzzy set
        """
        fuzzy_output = term
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(f"expected an Aggregated term, but found {type(fuzzy_output)}")

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output)

        weighted_sum = scalar(0.0 if fuzzy_output.terms else nan)
        weights = scalar(0.0)
        membership = (
            Term.tsukamoto.__name__
            if this_type == WeightedDefuzzifier.Type.Tsukamoto
            else Term.membership.__name__
        )
        for activated in fuzzy_output.grouped_terms().values():
            w = activated.degree
            z = activated.term.__getattribute__(membership)(w)
            weighted_sum = weighted_sum + w * z
            weights = weights + w

        y = (weighted_sum / weights).squeeze()  # type: ignore
        return y


class WeightedSum(WeightedDefuzzifier):
    """Weighted defuzzifier that computes the weighted sum of a fuzzy set represented by an Aggregated term.

    info: related
        - [fuzzylite.defuzzifier.Defuzzifier][]
        - [fuzzylite.defuzzifier.WeightedDefuzzifier][]
        - [fuzzylite.defuzzifier.WeightedAverage][]
        - [fuzzylite.term.Aggregated][]
    """

    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        r"""Computes the weighted sum of the fuzzy set.

        The fuzzy set is represented by Aggregated term as $y = \sum_i{w_iz_i}$,
        where $w_i$ is the activation degree of term $i$, and $z_i = \mu_i(w_i)$.

        In Takagi-Sugeno controllers, the membership function $\mu_i(w_i)$ is generally a Constant, Linear, or Function
        term, which typically disregards the $w_i$ value.

        Warning:
            From version 8, the aggregation operator is used to aggregate multiple activations of the same term.

            In previous versions, the implication and aggregation operators are not used for weighted defuzzification.

        Args:
            term: term to defuzzify
            minimum: irrelevant
            maximum: irrelevant

        Returns:
            weighted sum of the fuzzy set
        """
        fuzzy_output = term
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(f"expected an Aggregated term, but found {type(fuzzy_output)}")

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output)

        weighted_sum = scalar(0.0 if fuzzy_output.terms else nan)
        weights = scalar(0.0)
        membership = (
            Term.tsukamoto.__name__
            if this_type == WeightedDefuzzifier.Type.Tsukamoto
            else Term.membership.__name__
        )
        for activated in fuzzy_output.grouped_terms().values():
            w = activated.degree
            z = activated.term.__getattribute__(membership)(w)
            weighted_sum = weighted_sum + w * z
            weights = weights + w

        y = weighted_sum / weights
        # This is done to get "invalid" output values from activated terms with zero activation degrees.
        # Thus, returning nan values in those cases. A regular weighted sum would result in zero.
        y = (y * weights).squeeze()  # type: ignore
        return y
