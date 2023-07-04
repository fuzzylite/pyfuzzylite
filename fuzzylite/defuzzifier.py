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
import warnings
from abc import ABC, abstractmethod

import numpy as np

from .library import nan, representation, scalar
from .operation import Op
from .term import Activated, Aggregated, Constant, Function, Linear, Term
from .types import Scalar


class Defuzzifier(ABC):
    """The Defuzzifier class is the abstract class for defuzzifiers.

    @author Juan Rada-Vilela, Ph.D.
    @see IntegralDefuzzifier
    @see WeightedDefuzzifier
    @since 4.0
    """

    def __str__(self) -> str:
        """Returns a string representation of the object in the FuzzyLite Language."""
        return representation.fll.defuzzifier(self)

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return representation.as_constructor(self)

    def configure(  # noqa: B027 empty method in an abstract base class
        self, parameters: str
    ) -> None:
        """Configures the defuzzifier with the given parameters.
        @param parameters contains a list of space-separated parameter values.
        """
        pass

    def parameters(self) -> str:
        """Returns the parameters of the defuzzifier.
        @return the parameters of the defuzzifier.
        """
        return ""

    @abstractmethod
    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Defuzzifies the given fuzzy term utilizing the range `[minimum,maximum]`
        @param term is the term to defuzzify, typically an Aggregated term
        @param minimum is the minimum value of the range
        @param maximum is the maximum value of the range
        @return the defuzzified value of the given fuzzy term.
        """
        raise NotImplementedError()


class IntegralDefuzzifier(Defuzzifier):
    """The IntegralDefuzzifier class is the base class for defuzzifiers which integrate
    over the fuzzy set.

    @author Juan Rada-Vilela, Ph.D.
    @since 4.0
    """

    # Default resolution for integral defuzzifiers
    default_resolution = 1000

    def __init__(self, resolution: int | None = None) -> None:
        """Creates an integral defuzzifier, where the resolution refers to the
        number of divisions in which the range `[minimum,maximum]` is divided
        in order to integrate the area under the curve.

        @param resolution is the resolution of the defuzzifier
        """
        self.resolution = (
            resolution if resolution else IntegralDefuzzifier.default_resolution
        )

    def parameters(self) -> str:
        """Returns the parameters to configure the defuzzifier
        @return the parameters to configure the defuzzifier.
        """
        return Op.str(self.resolution)

    def configure(self, parameters: str) -> None:
        """Configures the defuzzifier with the given parameters
        @param parameters to configure the defuzzifier.
        """
        if parameters:
            self.resolution = int(parameters)

    @abstractmethod
    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Defuzzify the term on the given range.

        Args:
            term: the term to defuzzify
            minimum: the minimum range value to start defuzzification
            maximum: the maximum range value to end defuzzification
        Retur:
            scalar: defuzzified value.
        """
        pass


class Bisector(IntegralDefuzzifier):
    """The Bisector class is an IntegralDefuzzifier that computes the bisector
    of a fuzzy set represented in a Term.

    @author Juan Rada-Vilela, Ph.D.
    @see Centroid
    @see IntegralDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Computes the bisector of a fuzzy set, that is, the x-coordinate such that
        the area to its left is approximately equal to the area to its right.
        The defuzzification process integrates over the fuzzy set utilizing the boundaries given as
        parameters. The integration algorithm is the midpoint rectangle
        method (https://en.wikipedia.org/wiki/Rectangle_method).

        @param term is the fuzzy set
        @param minimum is the minimum value of the fuzzy set
        @param maximum is the maximum value of the fuzzy set
        @return the $x$-coordinate of the bisector of the fuzzy set
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
    """The Centroid class is an IntegralDefuzzifier that computes the centroid
    of a fuzzy set represented in a Term.

    @author Juan Rada-Vilela, Ph.D.
    @see BiSector
    @see IntegralDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Computes the centroid of a fuzzy set. The defuzzification process
        integrates over the fuzzy set utilizing the boundaries given as
        parameters. The integration algorithm is the midpoint rectangle
        method (https://en.wikipedia.org/wiki/Rectangle_method).

        @param term is the fuzzy set
        @param minimum is the minimum value of the fuzzy set
        @param maximum is the maximum value of the fuzzy set
        @return the $x$-coordinate of the centroid of the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        z = ((x * y).sum(axis=1) / y.sum(axis=1)).squeeze()
        return z  # type: ignore


class LargestOfMaximum(IntegralDefuzzifier):
    """The LargestOfMaximum class is an IntegralDefuzzifier that computes the
    largest value of the maximum membership function of a fuzzy set
    represented in a Term.

    @author Juan Rada-Vilela, Ph.D.
    @see SmallestOfMaximum
    @see MeanOfMaximum
    @see IntegralDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Computes the largest value of the maximum membership function of a
        fuzzy set. The largest value is computed by integrating over the
        fuzzy set. The integration algorithm is the midpoint rectangle method
        (https://en.wikipedia.org/wiki/Rectangle_method).

        @param term is the fuzzy set
        @param minimum is the minimum value of the fuzzy set
        @param maximum is the maximum value of the fuzzy set
        @return the largest $x$-coordinate of the maximum membership
        function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        lom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmax(lom, axis=1, keepdims=True).squeeze()
            return z  # type: ignore


class MeanOfMaximum(IntegralDefuzzifier):
    """The MeanOfMaximum class is an IntegralDefuzzifier that computes the mean
    value of the maximum membership function of a fuzzy set represented in a
    Term.

    @author Juan Rada-Vilela, Ph.D.
    @see SmallestOfMaximum
    @see MeanOfMaximum
    @see IntegralDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Computes the mean value of the maximum membership function
        of a fuzzy set. The mean value is computed while integrating
        over the fuzzy set. The integration algorithm is the midpoint
        rectangle method (https://en.wikipedia.org/wiki/Rectangle_method).

        @param term is the fuzzy set
        @param minimum is the minimum value of the fuzzy set
        @param maximum is the maximum value of the fuzzy set
        @return the mean $x$-coordinate of the maximum membership
        function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        mom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmean(mom, axis=1, keepdims=True).squeeze()
            return z  # type: ignore


class SmallestOfMaximum(IntegralDefuzzifier):
    """The SmallestOfMaximum class is an IntegralDefuzzifier that computes the
    smallest value of the maximum membership function of a fuzzy set
    represented in a Term.

    @author Juan Rada-Vilela, Ph.D.
    @see LargestOfMaximum
    @see MeanOfMaximum
    @see IntegralDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Computes the smallest value of the maximum membership function in the
        fuzzy set. The smallest value is computed while integrating over the
        fuzzy set. The integration algorithm is the midpoint rectangle method
        (https://en.wikipedia.org/wiki/Rectangle_method).

        @param term is the fuzzy set
        @param minimum is the minimum value of the fuzzy set
        @param maximum is the maximum value of the fuzzy set
        @return the smallest $x$-coordinate of the maximum membership
        function value in the fuzzy set
        """
        x = np.atleast_2d(Op.midpoints(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = (y > 0) & (y == y.max(axis=1, keepdims=True))
        som = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmin(som, axis=1, keepdims=True).squeeze()
            return z  # type: ignore


class WeightedDefuzzifier(Defuzzifier):
    """The WeightedDefuzzifier class is the base class for defuzzifiers which
    compute a weighted function on the fuzzy set without requiring to
    integrate over the fuzzy set.

    @author Juan Rada-Vilela, Ph.D.
    @since 5.0
    """

    @enum.unique
    class Type(enum.Enum):
        """The Type enum indicates the type of the WeightedDefuzzifier based
        the terms included in the fuzzy set.

        Automatic: Automatically inferred from the terms
        TakagiSugeno: Manually set to TakagiSugeno (or Inverse Tsukamoto)
        Tsukamoto: Manually set to Tsukamoto
        """

        Automatic = enum.auto()
        TakagiSugeno = enum.auto()
        Tsukamoto = enum.auto()

    def __init__(
        self,
        type: str | WeightedDefuzzifier.Type = Type.Automatic,
    ) -> None:
        """Creates a WeightedDefuzzifier
        @param type of the WeightedDefuzzifier based the terms included in the fuzzy set.
        """
        if isinstance(type, str):
            self.type = WeightedDefuzzifier.Type[type]
        else:
            self.type = type

    def parameters(self) -> str:
        """Gets the type of weighted defuzzifier."""
        return self.type.name

    def configure(self, parameters: str) -> None:
        """Configure defuzzifier based on parameters.
        @params parameters is type of defuzzifier.
        """
        if parameters:
            self.type = WeightedDefuzzifier.Type[parameters]

    @classmethod
    def infer_type(cls, term: Term) -> WeightedDefuzzifier.Type:
        """Infers the type of the defuzzifier based on the given term.
        @param term is the given term
        @return the inferred type of the defuzzifier based on the given term.
        """
        if isinstance(term, Aggregated):
            types = {cls.infer_type(t_i) for t_i in term.terms}
            if len(types) == 1:
                return types.pop()
            if len(types) == 0:
                # cannot infer type of empty term, and won't matter anyway,
                return WeightedDefuzzifier.Type.Automatic
            raise TypeError(
                f"cannot infer type of {cls.__name__}, got multiple types: {sorted(str(t) for t in types)}"
            )
        elif isinstance(term, Activated):  # noqa: RET506 - False Positive
            return cls.infer_type(term.term)
        elif isinstance(term, (Constant, Linear, Function)):
            return WeightedDefuzzifier.Type.TakagiSugeno
        elif term.is_monotonic():
            return WeightedDefuzzifier.Type.Tsukamoto
        raise TypeError(f"cannot infer type of {cls.__name__} from {term}")

    @abstractmethod
    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        r"""Computes the weighted fuzzy set represented as an
        Aggregated Term as.

        From version 6.0, the implication and aggregation operators are not
        utilized for defuzzification.

        @param term is the fuzzy set represented as an AggregatedTerm
        @param minimum is the minimum value of the range (only used for Tsukamoto)
        @param maximum is the maximum value of the range (only used for Tsukamoto)
        @return the weighted sum of the given fuzzy set
        """
        pass


class WeightedAverage(WeightedDefuzzifier):
    """The WeightedAverage class is a WeightedDefuzzifier that computes the
    weighted average of a fuzzy set represented in an Aggregated Term.

    @author Juan Rada-Vilela, Ph.D.
    @see WeightedAverageCustom
    @see WeightedSum
    @see WeightedSumCustom
    @see WeightedDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        r"""Computes the weighted sum of the given fuzzy set represented as an
        Aggregated Term as $y = \sum_i{w_iz_i} $,
        where $w_i$ is the activation degree of term $i$, and $z_i
        = \mu_i(w_i) $.

        From version 6.0, the implication and aggregation operators are not
        utilized for defuzzification.

        @param term is the fuzzy set represented as an AggregatedTerm
        @param minimum is the minimum value of the range (only used for Tsukamoto)
        @param maximum is the maximum value of the range (only used for Tsukamoto)
        @return the weighted sum of the given fuzzy set
        """
        fuzzy_output = term
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(
                f"expected an Aggregated term, but found {type(fuzzy_output)}"
            )

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
        for activated in fuzzy_output.terms:
            w = activated.degree
            z = activated.term.__getattribute__(membership)(w)
            weighted_sum = weighted_sum + w * z
            weights = weights + w

        y = (weighted_sum / weights).squeeze()  # type: ignore
        return y


class WeightedSum(WeightedDefuzzifier):
    """The WeightedSum class is a WeightedDefuzzifier that computes the
    weighted sum of a fuzzy set represented in an Aggregated Term.

    @author Juan Rada-Vilela, Ph.D.
    @see WeightedSumCustom
    @see WeightedAverage
    @see WeightedAverageCustom
    @see WeightedDefuzzifier
    @see Defuzzifier
    @since 4.0
    """

    def defuzzify(
        self,
        term: Term,
        minimum: float = nan,
        maximum: float = nan,
    ) -> Scalar:
        r"""Computes the weighted sum of the given fuzzy set represented as an
        Aggregated Term as $y = \sum_i{w_iz_i} $,
        where $w_i$ is the activation degree of term $i$, and $z_i
        = \mu_i(w_i) $.

        From version 6.0, the implication and aggregation operators are not
        utilized for defuzzification.

        @param term is the fuzzy set represented as an AggregatedTerm
        @param minimum is the minimum value of the range (only used for Tsukamoto)
        @param maximum is the maximum value of the range (only used for Tsukamoto)
        @return the weighted sum of the given fuzzy set
        """
        fuzzy_output = term
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(
                f"expected an Aggregated term, but found {type(fuzzy_output)}"
            )

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
        for activated in fuzzy_output.terms:
            w = activated.degree
            z = activated.term.__getattribute__(membership)(w)
            weighted_sum = weighted_sum + w * z
            weights = weights + w

        y = weighted_sum / weights
        # This is done to get "invalid" output values from activated terms with zero activation degrees.
        # Thus, returning nan values in those cases. A regular weighted sum would result in zero.
        y = (y * weights).squeeze()  # type: ignore
        return y
