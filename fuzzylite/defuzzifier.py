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
from math import nan
from typing import Optional, Union

import numpy as np

from .operation import Op, scalar
from .term import Aggregated, Constant, Function, Linear, Term
from .types import Scalar


class Defuzzifier:
    """The Defuzzifier class is the abstract class for defuzzifiers.

    @author Juan Rada-Vilela, Ph.D.
    @see IntegralDefuzzifier
    @see WeightedDefuzzifier
    @since 4.0
    """

    @property
    def class_name(self) -> str:
        """Returns the name of the class of the defuzzifier
        @return the name of the class of the defuzzifier.
        """
        return self.__class__.__name__

    def configure(self, parameters: str) -> None:
        """Configures the defuzzifier with the given parameters.
        @param parameters contains a list of space-separated parameter values.
        """
        raise NotImplementedError()

    def parameters(self) -> str:
        """Returns the parameters of the defuzzifier.
        @return the parameters of the defuzzifier.
        """
        raise NotImplementedError()

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

    def __init__(self, resolution: Optional[int] = None) -> None:
        """Creates an integral defuzzifier, where the resolution refers to the
        number of divisions in which the range `[minimum,maximum]` is divided
        in order to integrate the area under the curve.

        @param resolution is the resolution of the defuzzifier
        """
        self.resolution = (
            resolution if resolution else IntegralDefuzzifier.default_resolution
        )

    def __str__(self) -> str:
        """Returns the FLL code for the defuzzifier
        @return FLL code for the activation method.
        """
        return f"{self.class_name} {self.parameters()}"

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

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Defuzzify the term on the given range.

        Args:
            term: the term to defuzzify
            minimum: the minimum range value to start defuzzification
            maximum: the maximum range value to end defuzzification
        Retur:
            scalar: defuzzified value.
        """
        raise NotImplementedError()


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
        if np.any(~np.isfinite([minimum, maximum])):
            return nan
        x = np.atleast_2d(Op.linspace(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        area_cumsum = np.nancumsum(y, axis=1)
        area = np.abs(area_cumsum - area_cumsum[:, [-1]] / 2)
        index = area == area.min(axis=1, keepdims=True)
        bisectors = np.where(index, x, np.nan)
        return np.nanmean(bisectors, axis=1).squeeze()  # type: ignore


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
        if np.any(~np.isfinite([minimum, maximum])):
            return nan
        x = np.atleast_2d(Op.linspace(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        return ((x * y).sum(axis=1) / y.sum(axis=1)).squeeze()  # type: ignore


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
        if np.any(~np.isfinite([minimum, maximum])):
            return nan
        x = np.atleast_2d(Op.linspace(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = y == y.max(axis=1, keepdims=True)
        lom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmax(lom, axis=1, keepdims=True).squeeze()  # type: ignore


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
        if np.any(~np.isfinite([minimum, maximum])):
            return nan
        x = np.atleast_2d(Op.linspace(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = y == y.max(axis=1, keepdims=True)
        mom = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(mom, axis=1, keepdims=True).squeeze()  # type: ignore


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
        if np.any(~np.isfinite([minimum, maximum])):
            return nan
        x = np.atleast_2d(Op.linspace(minimum, maximum, self.resolution))
        y = np.atleast_2d(term.membership(x))
        y_max = y == y.max(axis=1, keepdims=True)
        som = np.where(y_max, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmin(som, axis=1, keepdims=True).squeeze()  # type: ignore


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

        Automatic, TakagiSugeno, Tsukamoto = range(3)

    def __init__(
        self, type: Optional[Union[str, "WeightedDefuzzifier.Type"]] = None
    ) -> None:
        """Creates a WeightedDefuzzifier
        @param type of the WeightedDefuzzifier based the terms included in the fuzzy set.
        """
        if type is None:
            type = WeightedDefuzzifier.Type.Automatic
        elif isinstance(type, str):
            type = WeightedDefuzzifier.Type[type]
        self.type = type

    def __str__(self) -> str:
        """Gets a string representation of the defuzzifier."""
        return f"{self.class_name} {self.parameters()}"

    def parameters(self) -> str:
        """Gets the type of weighted defuzzifier."""
        return self.type.name

    def configure(self, parameters: str) -> None:
        """Configure defuzzifier based on parameters.
        @params parameters is type of defuzzifier.
        """
        if parameters:
            self.type = WeightedDefuzzifier.Type[parameters]

    def defuzzify(self, term: Term, minimum: float, maximum: float) -> Scalar:
        """Not implemented."""
        raise NotImplementedError()

    def infer_type(self, term: Term) -> "WeightedDefuzzifier.Type":
        """Infers the type of the defuzzifier based on the given term. If the
        given term is Constant, Linear or Function, then the type is
        TakagiSugeno; otherwise, the type is Tsukamoto.

        @param term is the given term
        @return the inferred type of the defuzzifier based on the given term
        """
        if isinstance(term, (Constant, Linear, Function)):
            return WeightedDefuzzifier.Type.TakagiSugeno
        return WeightedDefuzzifier.Type.Tsukamoto


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
        r"""Computes the weighted average of the given fuzzy set represented in
        an Aggregated term as $y = \dfrac{\sum_i w_iz_i}{\sum_i w_i} $,
        where $w_i$ is the activation degree of term $i$, and
        $z_i = \mu_i(w_i) $.

        From version 6.0, the implication and aggregation operators are not
        utilized for defuzzification.

        @param term is the fuzzy set represented as an Aggregated Term
        @param minimum is the minimum value of the range (only used for Tsukamoto)
        @param maximum is the maximum value of the range (only used for Tsukamoto)
        @return the weighted average of the given fuzzy set
        """
        fuzzy_output = term
        if not isinstance(fuzzy_output, Aggregated):
            raise ValueError(
                f"expected an Aggregated term, but found {type(fuzzy_output)}"
            )

        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        if not fuzzy_output.terms:
            return nan

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output.terms[0])

        weighted_sum = scalar(0.0)
        weights = scalar(0.0)
        if this_type == WeightedDefuzzifier.Type.TakagiSugeno:
            # Provides Takagi-Sugeno and Inverse Tsukamoto of Functions
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.membership(w)
                weighted_sum += w * z
                weights += w
        else:
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.tsukamoto(
                    w, fuzzy_output.minimum, fuzzy_output.maximum
                )
                weighted_sum += w * z
                weights += w

        return weighted_sum / weights


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

        if not self.type:
            raise ValueError("expected a type of defuzzifier, but found none")

        if not fuzzy_output.terms:
            return nan

        this_type = self.type
        if self.type == WeightedDefuzzifier.Type.Automatic:
            this_type = self.infer_type(fuzzy_output.terms[0])

        weighted_sum = scalar(0.0)
        if this_type == WeightedDefuzzifier.Type.TakagiSugeno:
            # Provides Takagi-Sugeno and Inverse Tsukamoto of Functions
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.membership(w)
                weighted_sum += w * z
        else:
            for activated in fuzzy_output.terms:
                w = activated.degree
                z = activated.term.tsukamoto(
                    w, fuzzy_output.minimum, fuzzy_output.maximum
                )
                weighted_sum += w * z

        return weighted_sum
