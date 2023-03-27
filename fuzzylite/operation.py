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

__all__ = ["Operation", "Op"]

import inspect
import math
from typing import Callable, List, Optional, SupportsFloat, Union


class Operation:
    """The Operation class contains methods for numeric operations, string
    manipulation, and other functions, all of which are also accessible via
    fl.Op.
    @author Juan Rada-Vilela, Ph.D.
    @since 4.0.
    """

    @staticmethod
    def eq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is equal to $b$ at the given tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is equal to $b$ at the given tolerance.
        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return a == b or abs(a - b) < abs_tolerance or (a != a and b != b)

    @staticmethod
    def neq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is not equal to $b$ at the given tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is equal to $b$ at the given tolerance.

        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return not (a == b or abs(a - b) < abs_tolerance or (a != a and b != b))

    @staticmethod
    def gt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is greater than $b$ at the given tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is greater than $b$ at the given tolerance.
        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return (
            not (a == b or abs(a - b) < abs_tolerance or (a != a and b != b)) and a > b
        )

    @staticmethod
    def ge(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is greater than or equal to $b$ at the
        given tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is greater than or equal to $b$ at the
        given tolerance.
        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return a == b or abs(a - b) < abs_tolerance or (a != a and b != b) or a > b

    @staticmethod
    def le(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is less than or equal to $b$ at the given
        tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is less than or equal to $b$ at the given
        tolerance.
        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return a == b or abs(a - b) < abs_tolerance or (a != a and b != b) or a < b

    @staticmethod
    def lt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        """Returns whether $a$ is less than $b$ at the given tolerance
        @param a
        @param b
        @param macheps is the minimum difference upon which two
        floating-point values are considered equivalent
        @return whether $a$ is less than $b$ at the given tolerance.
        """
        if abs_tolerance is None:
            from . import lib

            abs_tolerance = lib.abs_tolerance
        return (
            not (a == b or abs(a - b) < abs_tolerance or (a != a and b != b)) and a < b
        )

    @staticmethod
    def logical_and(a: float, b: float) -> bool:
        r"""Computes the logical AND
        @param a
        @param b
        @return $
        \begin{cases}
        1.0 & \mbox{if $a=1 \wedge b=1$}\cr
        0.0 & \mbox{otherwise}
        \end{cases}
        $.
        """
        return Operation.eq(a, 1.0) and Operation.eq(b, 1.0)

    @staticmethod
    def logical_or(a: float, b: float) -> bool:
        r"""Computes the logical OR
        @param a
        @param b
        @return $
        \begin{cases}
        1.0 & \mbox{if $a=1 \vee b=1$}\cr
        0.0 & \mbox{otherwise}
        \end{cases}
        $.
        """
        return Operation.eq(a, 1.0) or Operation.eq(b, 1.0)

    @staticmethod
    def as_identifier(name: str) -> str:
        """Convert the name into a valid FuzzyLite identifier
        @param name is the name to convert
        @returns the name as a valid identifier.

        """
        result = "".join([x for x in name if x in ("_", ".") or x.isalnum()])
        return result if result else "unnamed"

    @staticmethod
    def scale(
        x: float,
        from_minimum: float,
        from_maximum: float,
        to_minimum: float,
        to_maximum: float,
    ) -> float:
        r"""Linearly interpolates the parameter $x$ in range
        `[fromMin,fromMax]` to a new value in the range `[toMin,toMax]`,
        truncated to the range `[toMin,toMax]` if bounded is `true`.
        @param x is the source value to interpolate
        @param fromMin is the minimum value of the source range
        @param fromMax is the maximum value of the source range
        @param toMin is the minimum value of the target range
        @param toMax is the maximum value of the target range
        @param bounded determines whether the resulting value is bounded to
        the range
        @return the source value linearly interpolated to the target range:
        $ y = y_a + (y_b - y_a) \dfrac{x-x_a}{x_b-x_a} $.
        """
        return (to_maximum - to_minimum) / (from_maximum - from_minimum) * (
            x - from_minimum
        ) + to_minimum

    @staticmethod
    def bound(x: float, minimum: float, maximum: float) -> float:
        r"""Returns $x$ bounded in $[\min,\max]$
        @param x is the value to be bounded
        @param min is the minimum value of the range
        @param max is the maximum value of the range
        @return $
        \begin{cases}
        \min & \mbox{if $x < \min$} \cr
        \max & \mbox{if $x > \max$} \cr
        x & \mbox{otherwise}
        \end{cases}
        $.
        """
        if x > maximum:
            return maximum
        if x < minimum:
            return minimum
        return x

    @staticmethod
    def arity_of(method: Callable) -> int:  # type: ignore
        """Gets the arity of the given method.
        @param method is the method to get the arity from
        @returns the arity of the method.

        """
        signature = inspect.signature(method)
        required_parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default == inspect.Parameter.empty
        ]
        return len(required_parameters)

    @staticmethod
    def pi() -> float:
        """Gets the value of Pi."""
        return math.pi

    @staticmethod
    def describe(
        instance: object,
        slots: bool = True,
        variables: bool = True,
        class_hierarchy: bool = False,
    ) -> str:
        """Describes the instance in terms of its slots, variables, and class hierarchy.
        @param instance is the instance to describe
        @param slots whether to include slots in the description
        @param variables whether to include variables in the description
        @param class_hierarchy whether to include class hierarchy in the description.

        @return the description of the instance
        """
        if not instance:
            return str(None)
        key_values = {}
        if instance:
            if slots and hasattr(instance, "__slots__") and instance.__slots__:
                for slot in instance.__slots__:
                    key_values[slot] = str(getattr(instance, slot))

            if variables and hasattr(instance, "__dict__") and instance.__dict__:
                for variable in instance.__dict__:
                    key_values[variable] = str(getattr(instance, variable))

            if class_hierarchy:
                key_values["__hierarchy__"] = ", ".join(
                    f"{cls.__module__}.{cls.__name__}"
                    for cls in inspect.getmro(instance.__class__)
                )

        class_name = instance.__class__.__name__
        sorted_dict = {key: key_values[key] for key in sorted(key_values.keys())}
        return f"{class_name}[{sorted_dict}]"

    @staticmethod
    def strip_comments(fll: str, delimiter: str = "#") -> str:
        """Removes the comments from the text.
        @param fll is the text to strip comments from
        @param delimiter is the start delimiter to denote a comment.

        @returns the text with comments stripped out.
        """
        lines: List[str] = []
        for line in fll.split("\n"):
            ignore = line.find(delimiter)
            if ignore != -1:
                line = line[:ignore]
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def scalar(x: Union[SupportsFloat, str, bytes]) -> float:
        """Convert the value into a floating point defined by the library
        @param x is the value to convert.
        """
        from . import lib

        return lib.floating_point(x)

    @staticmethod
    def increment(
        x: List[int],
        minimum: List[int],
        maximum: List[int],
        position: Optional[int] = None,
    ) -> bool:
        """Increments the list by the unit.
        @param x is the list to increment
        @param minimum is the list of minimum values for each element in the list
        @param maximum is the list of maximum values for each element in the list
        @param position is the position in the list to increment
        @returns boolean whether it was incremented.
        """
        if position is None:
            position = len(x) - 1
        if not x or position < 0:
            return False

        incremented = True
        if x[position] < maximum[position]:
            x[position] += 1
        else:
            incremented = position != 0
            x[position] = minimum[position]
            position -= 1
            if position >= 0:
                incremented = Op.increment(x, minimum, maximum, position)
        return incremented

    # Last method of class such that it does not replace builtins.str
    @staticmethod
    def str(x: Union[float, object], decimals: Optional[int] = None) -> str:
        """Returns a string representation of the given value
        @param x is the value
        @param decimals is the number of decimals to display
        @return a string representation of the given value.
        """
        if not decimals:
            from . import lib

            decimals = lib.decimals
        if isinstance(x, float):
            return f"{x:.{decimals}f}"
        return str(x)


Op = Operation
