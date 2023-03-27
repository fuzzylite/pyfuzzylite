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

__all__ = ["Variable", "InputVariable", "OutputVariable"]

import contextlib
import math
import typing
from math import inf, isnan, nan
from typing import Iterable, List, Optional, Tuple

from .exporter import FllExporter
from .norm import SNorm
from .operation import Op
from .term import Aggregated

if typing.TYPE_CHECKING:
    from .defuzzifier import Defuzzifier
    from .term import Term


class Variable:
    """The Variable class is the base class for linguistic variables.
    @author Juan Rada-Vilela, Ph.D.
    @see InputVariable
    @see OutputVariable
    @see Term
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        terms: Optional[Iterable["Term"]] = None,
    ) -> None:
        """Create the variable.
        @param name is the name of the variable
        @param description is the description of the variable
        @param enabled determines whether to enable the variable
        @param minimum is the minimum value of the range
        @param maximum is the maximum value of the range
        @param lockValueInRange indicates whether to lock the value to the
          range of the variable
        @param terms is the list of terms.
        # TODO: Explore converting list of terms to a dictionary of terms.
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.minimum = minimum
        self.maximum = maximum
        self.lock_range = lock_range
        self.terms: List["Term"] = []
        if terms:
            self.terms.extend(terms)
        self._value = nan

    def __str__(self) -> str:
        """Gets a string representation of the variable in the FuzzyLite Language
        @return a string representation of the variable in the FuzzyLite
        Language
        @see FllExporter.
        """
        return FllExporter().variable(self)

    def term(self, name: str) -> "Term":
        """Gets the term by the name.
        @param name is the name of the term
        @return the term of the given name.

        """
        for term in self.terms:
            if term.name == name:
                return term
        raise ValueError(f"term '{name}' not found in {[t.name for t in self.terms]}")

    @property
    def drange(self) -> float:
        """Gets the magnitude of the range of the variable
        @return `maximum - minimum`.

        """
        return self.maximum - self.minimum

    @property
    def range(self) -> Tuple[float, float]:
        """Gets the range of the variable
        @return tuple of (minimum, maximum).

        """
        return self.minimum, self.maximum

    @range.setter
    def range(self, min_max: Tuple[float, float]) -> None:
        """Sets the range of the variable
        @param min_max is the range of the variable (minimum, maximum).

        """
        self.minimum, self.maximum = min_max

    @property
    def value(self) -> float:
        """Gets the value of the variable
        @return the input value of an InputVariable, or the output value of
        an OutputVariable.
        """
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        """Sets the value of the variable
        @param value is the input value of an InputVariable, or the output
        value of an OutputVariable.
        """
        self._value = (
            Op.bound(value, self.minimum, self.maximum) if self.lock_range else value
        )

    def fuzzify(self, x: float) -> str:
        r"""Evaluates the membership function of value $x$ for each
        term $i$, resulting in a fuzzy value in the form
        $\tilde{x}=\sum_i{\mu_i(x)/i}$
        @param x is the value to fuzzify
        @return the fuzzy value expressed as $\sum_i{\mu_i(x)/i}$.
        """
        result: List[str] = []
        for term in self.terms:
            fx = nan
            with contextlib.suppress(ValueError):
                fx = term.membership(x)
            if not result:
                result.append(f"{Op.str(fx)}/{term.name}")
            else:
                pm = "+" if Op.ge(fx, 0.0) or isnan(fx) else "-"
                result.append(f" {pm} {Op.str(fx)}/{term.name}")

        return "".join(result)

    def highest_membership(self, x: float) -> Tuple[float, Optional["Term"]]:
        r"""Gets the term which has the highest membership function value for
        $x$.
        @param x is the value of interest
        @param[out] yhighest is a pointer where the highest membership
        function value will be stored
        @return the term $i$ which maximimizes $\mu_i(x)$.
        """
        result: Tuple[float, Optional["Term"]] = (0.0, None)
        for term in self.terms:
            y = nan
            with contextlib.suppress(ValueError):
                y = term.membership(x)
            if y > result[0]:
                result = (y, term)
        return result


class InputVariable(Variable):
    """The InputVariable class is a Variable that represents an input of the
    fuzzy logic controller.
    @author Juan Rada-Vilela, Ph.D.
    @see Variable
    @see OutputVariable
    @see Term
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        terms: Optional[Iterable["Term"]] = None,
    ) -> None:
        """Create the input variable.
        @param name is the name of the variable
        @param description is the description of the variable
        @param enabled whether the variable is enabled
        @param minimum is the minimum value of the variable
        @param maximum is the maximum value of the variable
        @param lockValueInRange indicates whether to lock the value to the
          range of the variable
        @param terms is the list of terms.
        """
        super().__init__(
            name=name,
            description=description,
            enabled=enabled,
            minimum=minimum,
            maximum=maximum,
            lock_range=lock_range,
            terms=terms,
        )

    def __str__(self) -> str:
        """Gets a string representation of the variable in the FuzzyLite Language
        @return a string representation of the variable in the FuzzyLite
        Language
        @see FllExporter.
        """
        return FllExporter().input_variable(self)

    def fuzzy_value(self) -> str:
        r"""Evaluates the membership function of the current input value $x$
        for each term $i$, resulting in a fuzzy input value in the form
        $\tilde{x}=\sum_i{\mu_i(x)/i}$. This is equivalent to a call to
        Variable::fuzzify() passing $x$ as input value
        @return the fuzzy input value expressed as $\sum_i{\mu_i(x)/i}$.
        """
        return super().fuzzify(self.value)


class OutputVariable(Variable):
    r"""The OutputVariable class is a Variable that represents an output of the
    fuzzy logic controller. During the activation of a RuleBlock, the
    Activated terms of each Rule will be Aggregated in the
    OutputVariable::fuzzyOutput(), which represents a fuzzy set hereinafter
    referred to as $\tilde{y}$. The defuzzification of $\tilde{y}$
    translates the fuzzy output value $\tilde{y}$ into a crisp output
    value $y$, which can be retrieved using Variable::getValue(). The
    value of the OutputVariable is computed and automatically stored when
    calling OutputVariable::defuzzify(), but the value depends on the
    following properties (expressed in the FuzzyLite Language):
      - Property `default: scalar` overrides the output value $y$ with
        the given fl::scalar whenever the defuzzification process results in
        a non-finite value (i.e., fl::nan and fl::inf). For example,
        considering `default: 0.0`, if RuleBlock::activate() does not
        activate any rules whose Consequent contribute to the OutputVariable,
        then the fuzzy output value is empty, the Defuzzifier does not
        operate, and hence $y=0.0$. By default, `default: NaN`. Relevant
        methods are OutputVariable::getDefaultValue() and
        OutputVariable::setDefaultValue().
      - Property `lock-previous: boolean`, if enabled, overrides the output
        value $y^t$ at time $t$ with the previously defuzzified valid
        output value $y^{t-1}$ if defuzzification process results in a
        non-finite value (i.e., fl::nan and fl::inf). When enabled, the
        property takes precedence over `default` if $y^{t-1}$ is a finite
        value. By default, `lock-previous: false`, $y^{t-1}=\mbox{NaN}$
        for $t=0$, and $y^{t-1}=\mbox{NaN}$ when
        OutputVariable::clear(). Relevant methods are
        OutputVariable::lockPreviousValue(),
        OutputVariable::isLockPreviousValue,
        OutputVariable::getPreviousValue(), and
        OutputVariable::setPreviousValue().
      - Property `lock-range: boolean` overrides the output value $y$ to
        enforce it lies within the range of the variable determined by
        Variable::getMinimum() and Variable::getMaximum(). When enabled, this
        property takes precedence over `lock-previous` and `default`. For
        example, considering `range: -1.0 1.0` and `lock-range: true`,
        $y=-1.0$ if the result from the Defuzzifier is smaller than
        `-1.0`, and $y=1.0$ if the result from the Defuzzifier is greater
        than `1.0`. The property `lock-range` was introduced in version 5.0
        to substitute the property `lock-valid` in version 4.0. By default,
        `lock-range: false`. Relevant methods are
        Variable::lockValueInRange(), Variable::isLockValueInRange(),
        Variable::getMinimum(), and Variable::getMaximum()
    @author Juan Rada-Vilela, Ph.D.
    @see Variable
    @see InputVariable
    @see RuleBlock::activate()
    @see Term
    @since 4.0.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        enabled: bool = True,
        minimum: float = -inf,
        maximum: float = inf,
        lock_range: bool = False,
        lock_previous: bool = False,
        default_value: float = nan,
        aggregation: Optional[SNorm] = None,
        defuzzifier: Optional["Defuzzifier"] = None,
        terms: Optional[Iterable["Term"]] = None,
    ) -> None:
        """Create the output variable.
        @param name is the name of the variable
        @param description is the description of the variable
        @param enabled whether the variable is enabled
        @param minimum is the minimum value of the variable
        @param maximum is the maximum value of the variable
        @param lockValueInRange indicates whether to lock the value to the
          range of the variable
        @param lockPreviousValue indicates whether to lock the previous value
          of the output variable
        @param defaultValue is the default value of the output variable
        @param aggregation is the aggregation norm
        @param defuzzifier is the defuzzifier of the output variable
        @param terms is the list of terms.
        """
        # name, minimum, and maximum are properties in this class, replacing the inherited members
        # to point to the Aggregated object named fuzzy. Thus, first we need to set up the fuzzy
        # object such that initializing the parent object will use the respective replacements.
        self.fuzzy = Aggregated(aggregation=aggregation)
        # initialize parent members
        super().__init__(
            name=name,
            description=description,
            enabled=enabled,
            minimum=minimum,
            maximum=maximum,
            lock_range=lock_range,
            terms=terms,
        )
        # set values of output variable
        self.defuzzifier = defuzzifier
        self.lock_previous = lock_previous
        self.default_value = default_value
        self.previous_value = nan

    def __str__(self) -> str:
        """Gets a string representation of the variable in the FuzzyLite Language
        @return a string representation of the variable in the FuzzyLite
        Language
        @see FllExporter.
        """
        return FllExporter().output_variable(self)

    @property
    def name(self) -> str:
        """Gets the name of the output variable."""
        return self.fuzzy.name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the output variable
        @param value is the name of the output variable.

        """
        self.fuzzy.name = value

    @property
    def minimum(self) -> float:
        """Gets the minimum value of the range of the output variable."""
        return self.fuzzy.minimum

    @minimum.setter
    def minimum(self, value: float) -> None:
        """Sets the minimum value of the range of the output variable
        @param value is the minimum value of the output variable.

        """
        self.fuzzy.minimum = value

    @property
    def maximum(self) -> float:
        """Gets the maximum value of the range of the output variable."""
        return self.fuzzy.maximum

    @maximum.setter
    def maximum(self, value: float) -> None:
        """Sets the maximum value of the range of the output variable
        @param value is the maximum value of the output variable.
        """
        self.fuzzy.maximum = value

    @property
    def aggregation(self) -> Optional[SNorm]:
        """Gets the aggregation operator
        @return the aggregation operator.

        """
        return self.fuzzy.aggregation

    @aggregation.setter
    def aggregation(self, value: SNorm) -> None:
        """Sets the aggregation operator
        @param aggregation is the aggregation.

        """
        self.fuzzy.aggregation = value

    def defuzzify(self) -> None:
        """Defuzzifies the output variable and stores the output value and the
        previous output value.
        """
        if not self.enabled:
            return
        if math.isfinite(self.value):
            self.previous_value = self.value

        result = nan
        exception = None
        is_valid = bool(self.fuzzy.terms)
        if is_valid:
            # Checks whether the variable can be defuzzified without exceptions.
            # If it cannot be defuzzified, be that due to a missing defuzzifier
            # or aggregation operator, the expected behaviour is to leave the
            # variable in a state that reflects an invalid defuzzification,
            # that is, apply logic of default values and previous values.
            is_valid = False
            if self.defuzzifier:
                try:
                    result = self.defuzzifier.defuzzify(
                        self.fuzzy, self.minimum, self.maximum
                    )
                    is_valid = True
                except ValueError as ex:
                    exception = ex
            else:
                exception = ValueError(
                    f"expected a defuzzifier in output variable {self.name}, "
                    "but found none"
                )

        if not is_valid:
            # if a previous defuzzification was successfully performed and
            # and the output value is supposed not to change when the output is empty
            if self.lock_previous and not isnan(self.previous_value):
                result = self.previous_value
            else:
                result = self.default_value

        self.value = result

        if exception:
            raise exception

    def clear(self) -> None:
        r"""Clears the output variable by setting $\tilde{y}=\{\}$,
        $y^{t}=\mbox{NaN}$, $y^{t-1}=\mbox{NaN}$.
        """
        self.fuzzy.clear()
        self.previous_value = nan
        self._value = nan

    def fuzzy_value(self) -> str:
        r"""Returns: string representation of the fuzzy output value $\tilde{y}$."""
        result: List[str] = []
        for term in self.terms:
            degree = self.fuzzy.activation_degree(term)

            if not result:
                result.append(f"{Op.str(degree)}/{term.name}")
            else:
                result.append(
                    f" {'+' if isnan(degree) or degree >= 0 else '-'} {Op.str(math.fabs(degree))}/{term.name}"
                )
        return "".join(result)
