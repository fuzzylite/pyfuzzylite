"""
 pyfuzzylite (TM), a fuzzy logic control library in Python.
 Copyright (C) 2010-2017 FuzzyLite Limited. All rights reserved.
 Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>

 This file is part of pyfuzzylite.

 pyfuzzylite is free software: you can redistribute it and/or modify it under
 the terms of the FuzzyLite License included with the software.

 You should have received a copy of the FuzzyLite License along with
 pyfuzzylite. If not, see <http://www.fuzzylite.com/license/>.

 pyfuzzylite is a trademark of FuzzyLite Limited
 fuzzylite is a registered trademark of FuzzyLite Limited.
"""

__all__ = ["Operation", "Op"]

import inspect
import math
from typing import Callable, List, Optional, SupportsFloat, Text, Union


class Operation:
    """
    Operation
    """

    @staticmethod
    def eq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b))

    @staticmethod
    def neq(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b))

    @staticmethod
    def gt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b)
                    ) and a > b

    @staticmethod
    def ge(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b)
                or a > b)

    @staticmethod
    def le(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return (a == b
                or abs(a - b) < abs_tolerance
                or (a != a and b != b)
                or a < b)

    @staticmethod
    def lt(a: float, b: float, abs_tolerance: Optional[float] = None) -> bool:
        if abs_tolerance is None:
            from . import lib
            abs_tolerance = lib.abs_tolerance
        return not (a == b
                    or abs(a - b) < abs_tolerance
                    or (a != a and b != b)
                    ) and a < b

    @staticmethod
    def logical_and(a: float, b: float) -> bool:
        return Operation.eq(a, 1.0) and Operation.eq(b, 1.0)

    @staticmethod
    def logical_or(a: float, b: float) -> bool:
        return Operation.eq(a, 1.0) or Operation.eq(b, 1.0)

    @staticmethod
    def as_identifier(name: str) -> str:
        result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
        return result if result else "unnamed"

    @staticmethod
    def scale(x: float, from_minimum: float, from_maximum: float, to_minimum: float,
              to_maximum: float) -> float:
        return ((to_maximum - to_minimum) / (from_maximum - from_minimum) * (x - from_minimum)
                + to_minimum)

    @staticmethod
    def bound(x: float, minimum: float, maximum: float) -> float:
        if x > maximum:
            return maximum
        if x < minimum:
            return minimum
        return x

    @staticmethod
    def arity_of(method: Callable) -> int:  # type: ignore
        signature = inspect.signature(method)
        required_parameters = [parameter for parameter in signature.parameters.values()
                               if parameter.default == inspect.Parameter.empty]
        return len(required_parameters)

    @staticmethod
    def pi() -> float:
        return math.pi

    @staticmethod
    def describe(instance: object, slots: bool = True, variables: bool = True,
                 class_hierarchy: bool = False) -> str:
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
                    for cls in inspect.getmro(instance.__class__))

        class_name = instance.__class__.__name__
        sorted_dict = {key: key_values[key] for key in sorted(key_values.keys())}
        return f"{class_name}[{sorted_dict}]"

    @staticmethod
    def strip_comments(fll: str, delimiter: str = "#") -> str:
        lines: List[str] = []
        for line in fll.split('\n'):
            ignore = line.find(delimiter)
            if ignore != -1:
                line = line[:ignore]
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def scalar(x: Union[SupportsFloat, str, bytes]) -> float:
        from . import lib
        return lib.floating_point(x)

    @staticmethod
    def increment(x: List[int], minimum: List[int], maximum: List[int],
                  position: Optional[int] = None) -> bool:
        if position is None:
            position = len(x) - 1
        if not x or position < 0:
            return False

        incremented = True
        if x[position] < maximum[position]:
            x[position] += 1
        else:
            incremented = not (position == 0)
            x[position] = minimum[position]
            position -= 1
            if position >= 0:
                incremented = Op.increment(x, minimum, maximum, position)
        return incremented

    # Last method of class such that it does not replace builtins.str
    @staticmethod
    def str(x: Union[float, object], decimals: Optional[int] = None) -> Text:
        if not decimals:
            from . import lib
            decimals = lib.decimals
        if isinstance(x, float):
            return f"{x:.{decimals}f}"
        return str(x)


Op = Operation
