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

__all__ = ["Operation", "Op"]

import builtins
import inspect
from typing import Any, Callable

import numpy as np

from .library import scalar, settings
from .types import Array, Scalar, ScalarArray


class Operation:
    """The Operation class contains methods for numeric operations, string
    manipulation, and other functions, all of which are also accessible via
    fl.Op.
    @author Juan Rada-Vilela, Ph.D.
    @since 4.0.
    """

    isinf = np.isinf
    isnan = np.isnan

    @staticmethod
    def eq(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is equal to $b$, considering nan == nan is True
        @param a
        @param b
        @return whether $a$ is equal to $b$.
        """
        return np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def neq(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is not equal to $b$
        @param a
        @param b
        @return whether $a$ is equal to $b$.

        """
        return ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def gt(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is greater than $b$ at the given tolerance
        @param a
        @param b
        @return whether $a$ is greater than $b$.
        """
        return scalar(a > b)

    @staticmethod
    def ge(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is greater than or equal to $b$ at the
        given tolerance
        @param a
        @param b
        @return whether $a$ is greater than or equal to $b$.
        """
        return (a >= b) | np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def le(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is less than or equal to $b$ at the given
        tolerance
        @param a
        @param b
        @return whether $a$ is less than or equal to $b$.
        """
        return (a <= b) | np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def lt(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Returns whether $a$ is less than $b$ at the given tolerance
        @param a
        @param b
        floating-point values are considered equivalent
        @return whether $a$ is less than $b$ at the given tolerance.
        """
        return scalar(a < b)

    @staticmethod
    def logical_and(a: Scalar, b: Scalar) -> Scalar:
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
        return np.logical_and(a, b)

    @staticmethod
    def logical_or(a: Scalar, b: Scalar) -> Scalar:
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
        return np.logical_or(a, b)

    @staticmethod
    def is_close(a: Scalar, b: Scalar) -> bool | Array[np.bool_]:
        """Return whether a and b are close within the library's absolute and relative tolerances."""
        z = np.isclose(a, b, atol=settings.atol, rtol=settings.rtol, equal_nan=True)
        return z

    @staticmethod
    def as_identifier(name: str) -> str:
        """Convert the name into a valid FuzzyLite identifier
        @param name is the name to convert
        @returns the name as a valid identifier.
        """
        name = "".join([x for x in name if x.isalnum() or x == "_"]) or "_"
        if name[0].isnumeric():
            name = f"_{name}"
        return name

    @staticmethod
    def snake_case(text: str) -> str:
        """Converts the string to snake_case."""
        result = [" "]
        for character in text:
            if character.isalpha() and character.isupper():
                if result[-1] != " ":
                    result.append(" ")
                result.append(character.lower())
            elif character.isalnum():
                result.append(character)
            elif (character.isspace() or not character.isalnum()) and result[-1] != " ":
                result.append(" ")
        sc_text = "".join(result).strip().replace(" ", "_")
        return sc_text

    @staticmethod
    def pascal_case(text: str) -> str:
        """Converts the string to PascalCase."""
        result = Op.snake_case(text)
        cc_text = "".join(word.capitalize() for word in result.split("_"))
        return cc_text

    @staticmethod
    def scale(
        x: Scalar,
        from_minimum: float,
        from_maximum: float,
        to_minimum: float,
        to_maximum: float,
    ) -> Scalar:
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
        x = scalar(x)
        return (to_maximum - to_minimum) / (from_maximum - from_minimum) * (
            x - from_minimum
        ) + to_minimum

    @staticmethod
    def bound(x: Scalar, minimum: float, maximum: float) -> Scalar:
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
        return np.clip(scalar(x), minimum, maximum)

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
    def describe(
        instance: object,
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
            if variables and hasattr(instance, "__dict__") and instance.__dict__:
                for variable, value in vars(instance).items():
                    key_values[variable] = str(value)

            if class_hierarchy:
                key_values["__hierarchy__"] = ", ".join(
                    f"{cls.__module__}.{cls.__name__}"
                    for cls in inspect.getmro(instance.__class__)
                )

        sorted_dict = {key: key_values[key] for key in sorted(key_values.keys())}
        return f"{Op.class_name(instance)}[{sorted_dict}]"

    @staticmethod
    def strip_comments(fll: str, /, delimiter: str = "#") -> str:
        """Removes the comments from the text.
        @param fll is the text to strip comments from
        @param delimiter is the start delimiter to denote a comment.

        @returns the text with comments stripped out.
        """
        lines: list[str] = []
        for line in fll.split("\n"):
            ignore = line.find(delimiter)
            if ignore != -1:
                line = line[:ignore]
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def midpoints(start: float, end: float, resolution: int = 1000) -> ScalarArray:
        """Returns a list of values from start to end with the given resolution using midpoint method."""
        # dx = ((end - start) / resolution)
        # result = start + (i + 0.5) * dx
        return start + (np.array(range(resolution)) + 0.5) * (
            (end - start) / resolution
        )

    @staticmethod
    def increment(
        x: list[int],
        minimum: list[int],
        maximum: list[int],
        position: int | None = None,
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

    @staticmethod
    def class_name(x: Any, /, qualname: bool = False) -> str:
        """Returns the class name of the given object.
        @param x is the object
        @return the class name of the given object.
        """
        package = ""
        if qualname:
            from .library import representation

            package = representation.package_of(x)

        if inspect.isclass(x):
            return f"{package}{x.__name__}"
        return f"{package}{x.__class__.__name__}"

    @staticmethod
    def to_fll(x: Any, /) -> str:
        """Returns a string representation of the given value."""
        from .library import representation

        return representation.fll.to_string(x)

    @staticmethod
    def str(x: Any, /, delimiter: str = " ") -> str:
        """Returns a string representation of the given value
        @param x is the value
        @param decimals is the number of decimals to display
        @return a string representation of the given value.
        """
        if isinstance(x, (float, np.floating)):
            return f"{x:.{settings.decimals}f}"
        if isinstance(x, (np.ndarray, list)):
            return delimiter.join([Op.str(x_i) for x_i in np.atleast_1d(x)])
        return builtins.str(x)


Op = Operation
