"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = ["Operation", "Op"]

import builtins
import importlib
import importlib.util
import inspect
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, overload

import numpy as np

from .library import scalar, settings
from .types import Array, Scalar, ScalarArray

if typing.TYPE_CHECKING:
    from .engine import Engine


class Operation:
    """Methods for numeric operations, string manipulation, and other functions.

    `fl.Op` is a shortcut to this class.
    """

    isinf = np.isinf
    isnan = np.isnan

    @staticmethod
    def eq(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        r"""Return $a = b$ (with NaN's as equal).

        Args:
            a: scalar
            b: scalar

        Returns:
             $a=b$
        """
        return np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def neq(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        r"""Return $a \not= b$ (with NaN's as equal).

        Args:
            a: scalar
            b: scalar

        Returns:
             $a\not=b$
        """
        return ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def gt(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        """Return $a > b$.

        Args:
            a: scalar
            b: scalar

        Returns:
             $a>b$
        """
        return scalar(a > b)

    @staticmethod
    def ge(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        r"""Return $a \ge b$ (with NaN's as equal).

        Args:
            a: scalar
            b: scalar

        Returns:
             $a \ge b$
        """
        return (a >= b) | np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def le(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        r"""Return $a \le b$ (with NaN's as equal).

        Args:
            a: scalar
            b: scalar

        Returns:
             $a \le b$
        """
        return (a <= b) | np.isclose(a, b, rtol=0, atol=0, equal_nan=True)  # type: ignore

    @staticmethod
    def lt(
        a: Scalar,
        b: Scalar,
    ) -> Scalar:
        r"""Return $a < b$.

        Args:
            a: scalar
            b: scalar

        Returns:
             $a < b$
        """
        return scalar(a < b)

    @staticmethod
    def is_close(a: Scalar, b: Scalar) -> bool | Array[np.bool_]:
        r"""Return $a \approx b$ (with NaN's as equal) using the absolute and relative tolerances of the library.

        Args:
            a: scalar
            b: scalar

        Returns:
             $a \approx b$

        info: related
            - [fuzzylite.library.Settings][]
        """
        z = np.isclose(a, b, atol=settings.atol, rtol=settings.rtol, equal_nan=True)
        return z

    @staticmethod
    def as_identifier(name: str) -> str:
        """Convert the name into a valid FuzzyLite and Python identifier by removing non-alphanumeric characters and prepending `_` to names starting with a number.

        Args:
            name: name to convert

        Returns:
             name as a valid identifier.
        """
        name = "".join([x for x in name if x.isalnum() or x == "_"]) or "_"
        if name[0].isnumeric():
            name = f"_{name}"
        return name

    @staticmethod
    def snake_case(text: str) -> str:
        """Converts the string to snake_case.

        Args:
            text: any string

        Returns:
            text in `snake_case`
        """
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
        """Converts the string to PascalCase.

        Args:
            text: any string

        Returns:
            text in `PascalCase`
        """
        result = Op.snake_case(text)
        cc_text = "".join(word.capitalize() for word in result.split("_"))
        return cc_text

    @staticmethod
    def scale(
        x: Scalar,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> Scalar:
        r"""Linearly interpolates $x$ from the source range `[from_minimum, from_maximum]` to its new value in the target range `[to_minimum, to_maximum]`.

        Args:
            x: value to interpolate
            x_min: minimum value of the source range
            x_max: maximum value of the source range
            y_min: minimum value of the target range
            y_max: maximum value of the target range

        Returns:
             $x$ linearly interpolated to the target range as: $y = \dfrac{y_\max - y_\min}{x_\max-x_\min} (x-x_\min) + y_\min$
        """
        x = scalar(x)
        return (y_max - y_min) / (x_max - x_min) * (x - x_min) + y_min

    @staticmethod
    def bound(x: Scalar, minimum: float, maximum: float) -> Scalar:
        r"""Return $x$ clipped between `[minimum, maximum]`.

        Args:
           x: value to be clipped
           minimum: minimum value of the range
           maximum: maximum value of the range

        Returns:
             $$\begin{cases} \min & \mbox{if $x < \min$} \cr \max & \mbox{if $x > \max$} \cr x & \mbox{otherwise} \end{cases}$$
        """
        return np.clip(scalar(x), minimum, maximum)

    @staticmethod
    def arity_of(method: Callable) -> int:  # type: ignore
        """Gets the arity of the method.

        Args:
            method: method to get the arity from

        Returns:
            arity of the method.
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
        """Describe the instance based on its variables and class hierarchy.

        Args:
            instance: instance to describe
            variables: include variables in the description
            class_hierarchy: include class hierarchy in the description.

        Returns:
             description of the instance
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
                    f"{cls.__module__}.{cls.__name__}" for cls in inspect.getmro(instance.__class__)
                )

        sorted_dict = {key: key_values[key] for key in sorted(key_values.keys())}
        return f"{Op.class_name(instance)}[{sorted_dict}]"

    @staticmethod
    def strip_comments(fll: str, /, delimiter: str = "#") -> str:
        """Remove the comments from the text.

        Args:
            fll: text to strip comments from
            delimiter: delimiter that indicates the start of a comment.

        Returns:
            text with comments stripped out.
        """
        # todo: Move to FllImporter
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
        """Return the list of values in the range at the given resolution using the [midpoint rule](https://en.wikipedia.org/wiki/Rectangle_method).

        Args:
            start: start of range
            end: end of range
            resolution: number of divisions to discretize the range

        Returns:
            list of values in the range at the given resolution using the [midpoint rule](https://en.wikipedia.org/wiki/Rectangle_method)
        """
        # dx = ((end - start) / resolution)
        # result = start + (i + 0.5) * dx
        return start + (np.array(range(resolution)) + 0.5) * ((end - start) / resolution)

    @staticmethod
    def increment(
        x: list[int],
        minimum: list[int],
        maximum: list[int],
        position: int | None = None,
    ) -> bool:
        """Increment the list by the unit.

        Args:
            x: list to increment
            minimum: list of minimum values for each element in the list
            maximum: list of maximum values for each element in the list
            position: position in the list to increment

        Returns:
             whether the list was incremented.
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
        """Return the class name of the object.

        Args:
            x: object to get the class name
            qualname: use fully qualified classes

        Returns:
            class name of the given object.
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
        """Return the string representation of the object in the FuzzyLite Language.

        Args:
            x: object

        Returns:
            string representation of the object in the FuzzyLite Language.
        """
        from .library import representation

        return representation.fll.to_string(x)

    @staticmethod
    @overload
    def glob_examples(
        return_type: Literal["module"],
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[ModuleType]: ...

    @staticmethod
    @overload
    def glob_examples(
        return_type: Literal["engine"],
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[Engine]: ...

    @staticmethod
    @overload
    def glob_examples(
        return_type: Literal["dataset"] | Literal["fld"],
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[ScalarArray]: ...

    @staticmethod
    @overload
    def glob_examples(
        return_type: Literal["language"] | Literal["fll"],
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[str]: ...

    @staticmethod
    @overload
    def glob_examples(
        return_type: Literal["files"],
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[Path]: ...

    @staticmethod
    def glob_examples(
        return_type: (
            Literal["module"]
            | Literal["engine"]
            | Literal["dataset"]
            | Literal["fld"]
            | Literal["language"]
            | Literal["fll"]
            | Literal["files"]
        ) = "engine",
        module: ModuleType | None = None,
        recursive: bool = True,
    ) -> Iterable[ModuleType | Engine | ScalarArray | str | Path]:
        """Glob the examples (alphabetically and in ascending order) returning the specified type.

        Args:
            return_type: type of objects to return
            module: package (eg, `fuzzylite.examples`) or module (eg, `fuzzylite.examples.terms.arc`) to glob
            recursive: recursively glob into subdirectories

        Yields:
            Iterable of the specified type.
        """
        if module is None:
            import fuzzylite.examples

            module = fuzzylite.examples

        pattern = "**/" if recursive else ""

        # A package is a module with a __path__ attribute: https://docs.python.org/3/reference/import.html#packages
        is_package = hasattr(module, "__path__")

        if is_package:
            # module is a package (directory) (eg, fuzzylite.examples)
            package = Path(*module.__path__)
            pattern += "*"
        else:
            # module is a module (eg, fuzzylite.examples.terms.arc)
            package = Path(f"{module.__file__}").parent
            pattern += Path(f"{module.__file__}").stem

        if return_type in {"module", "engine"}:
            pattern += ".py"
            for file in sorted(package.glob(pattern)):
                if file.stem != "__init__":
                    submodule = ".".join(
                        Op.as_identifier(part)
                        for part in file.with_suffix("").relative_to(package).parts
                    )
                    import_name = (
                        f"{module.__name__}.{submodule}" if is_package else module.__name__
                    )
                    example_module = importlib.import_module(import_name)
                    if return_type == "module":
                        yield example_module
                    else:
                        example_class, *_ = inspect.getmembers(
                            example_module, predicate=inspect.isclass
                        )
                        # example_class: tuple[str, type]
                        engine = example_class[1]().engine
                        yield engine

        elif return_type in {"dataset", "fld"}:
            pattern += ".fld"
            for file in sorted(package.glob(pattern)):
                yield np.loadtxt(file, skiprows=1)

        elif return_type in {"language", "fll"}:
            pattern += ".fll"
            for file in sorted(package.glob(pattern)):
                yield file.read_text()

        elif return_type == "files":
            pattern += ".*"
            for file in sorted(package.glob(pattern)):
                if file.suffix in {".py", ".fll", ".fld"} and file.name != "__init__.py":
                    yield file

        else:
            raise ValueError(
                f"expected 'return_type' in {'module engine dataset fld language fll files'.split()}, "
                f"but got '{return_type}'"
            )

    @staticmethod
    def str(x: Any, /, delimiter: str = " ") -> builtins.str:
        """Returns a string representation of the value.

        Args:
            x: value
            delimiter: delimiter to use when `x` is a `Sequence` or `ScalarArray`

        Returns:
             string representation of the value.
        """
        if isinstance(x, str):
            return x
        if isinstance(x, (float, np.floating)):
            return f"{x:.{settings.decimals}f}"
        if isinstance(x, Sequence):
            return delimiter.join([Op.str(x_i) for x_i in x])
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return f"{x.item():.{settings.decimals}f}"
            if x.ndim == 1:
                return delimiter.join([Op.str(x_i) for x_i in np.atleast_1d(x)])
            if x.ndim == 2:
                return "\n".join(Op.str(x[i, :]) for i in range(len(x)))
            return np.array2string(x, precision=settings.decimals, floatmode="fixed")
        return builtins.str(x)


Op = Operation
