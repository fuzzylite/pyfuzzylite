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
    "information",
    "Information",
    "settings",
    "Settings",
    "repr",
    "representation",
    "Representation",
    "scalar",
    "to_float",
    "array",
    "inf",
    "nan",
]

import builtins
import inspect
import logging
import reprlib
import typing
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Final, overload

import numpy as np
from numpy import array, inf, nan

from .types import Array, Scalar, ScalarArray

if typing.TYPE_CHECKING:
    from .factory import FactoryManager

np.seterr(invalid="ignore", divide="ignore")

logging.basicConfig(
    level=logging.ERROR,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)s "
    "%(module)s::%(funcName)s()[%(lineno)d]"
    "\n%(message)s",
)


def to_float(x: Any, /) -> float:
    """Convert the value into a floating point defined by the library
    @param x is the value to convert.
    """
    return settings.float_type(x)  # type: ignore


@overload
def scalar(x: Sequence[Any] | Array[Any], /) -> ScalarArray:
    ...


@overload
def scalar(x: Any, /) -> Scalar:
    ...


def scalar(
    x: Sequence[Any] | Array[Any] | Any, /, **kwargs: Any
) -> ScalarArray | Scalar:
    """Convert the values into a floating point value  defined by the library
    @param x is the value to convert.
    """
    return np.asarray(x, dtype=settings.float_type, **kwargs)


class Settings:
    """Settings for the library."""

    def __init__(
        self,
        float_type: Any = np.float64,
        decimals: int = 3,
        atol: float = 1e-03,
        rtol: float = 0.0,
        alias: str = "fl",
        logger: logging.Logger | None = None,
        factory_manager: FactoryManager | None = None,
    ) -> None:
        """@param float_type is the floating point type to use.
        @param decimals is the number of decimals to use.
        @param atol is the absolute tolerance.
        @param rtol is the relative tolerance.
        @param alias is the alias for the library used when representing objects (ie, repr).
        Cases:
            - fully qualified package when alias == "" (eg, `fuzzylite.term.Constant(name="A", height=1.0)`)
            - no prefixes when alias == "*" (eg, `Constant(name="A", height=1.0)`)
            - alias otherwise (eg, `{alias}.Constant(name="A", height=1.0)`
        @param logger is the logger to use.
        @param factory_manager is the factory manager to use.
        @return the settings.
        """
        self.float_type = float_type
        self.decimals = decimals
        self.atol = atol
        self.rtol = rtol
        self.alias = alias
        self.logger = logger or logging.getLogger("fuzzylite")
        self._factory_manager = factory_manager

    def __repr__(self) -> str:
        """@return Python code to construct the settings."""
        fields = vars(self).copy()
        fields["factory_manager"] = fields.pop("_factory_manager")
        return representation.as_constructor(self, fields)

    @property
    def factory_manager(self) -> FactoryManager:
        """Get the factory manager."""
        if self._factory_manager is None:
            # done here to avoid partially initialised class during __init__ using setter
            from .factory import FactoryManager

            self._factory_manager = FactoryManager()
        return self._factory_manager

    @factory_manager.setter
    def factory_manager(self, value: FactoryManager) -> None:
        """Set the factory manager."""
        self._factory_manager = value

    @property
    def debugging(self) -> bool:
        """Whether the library is in debug mode."""
        return self.logger.level == logging.DEBUG

    @debugging.setter
    def debugging(self, value: bool) -> None:
        """Set the library debugging mode."""
        self.logger.setLevel(logging.DEBUG if value else logging.ERROR)

    @contextmanager
    def context(
        self,
        *,
        float_type: Any | None = None,
        decimals: int | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        alias: str | None = None,
        logger: logging.Logger | None = None,
        factory_manager: FactoryManager | None = None,
    ) -> Generator[None, None, None]:
        """Creates a context with the given settings.
        @param float_type is the floating point type to use.
        @param decimals is the number of decimals to use.
        @param atol is the absolute tolerance.
        @param rtol is the relative tolerance.
        @param alias is the alias for the library.
        @param logger is the logger to use.
        @param factory_manager is the factory manager to use.
        """
        context_settings = {
            key: value
            for key, value in locals().items()
            if not (key == "self" or value is None)
        }
        if "factory_manager" in context_settings:
            context_settings["_factory_manager"] = context_settings.pop(
                "factory_manager"
            )
        rollback_settings = vars(self).copy()
        for key, value in context_settings.items():
            setattr(self, key, value)
        try:
            yield
        finally:
            for key, value in context_settings.items():
                setattr(self, key, rollback_settings[key])


settings: Final[Settings] = Settings()


@dataclass(frozen=True, repr=False)
class Information:
    """Information about the library."""

    name: str = "fuzzylite"
    description: str = "a fuzzy logic control library in Python"
    license: str = "FuzzyLite License"
    author: str = "Juan Rada-Vilela, Ph.D."
    author_email: str = "jcrada@fuzzylite.com"
    company: str = "FuzzyLite Limited"
    website: str = "https://fuzzylite.com/"

    def __repr__(self) -> str:
        """@return Python code to construct the information."""
        fields = vars(self).copy()
        fields["version"] = self.version
        return representation.as_constructor(self, fields)

    @property
    def version(self) -> str:
        """The version of the library."""
        __version__ = "8.0.0"
        return __version__


information: Final[Information] = Information()


class Representation(reprlib.Repr):
    """Representation class for the library."""

    T = typing.TypeVar("T")

    def __init__(self) -> None:
        """Initialize the representation class."""
        super().__init__()
        limits = {k: v for k, v in vars(self).items()}
        increase_factor = int(1e9)  # virtually limitless
        for variable, value in limits.items():
            setattr(self, variable, value * increase_factor)
        self.maxlevel = 10  # except for recursion level
        from .exporter import FllExporter

        self.fll: Final[FllExporter] = FllExporter()

    def __repr__(self) -> str:
        """@return Python code to construct the representation."""
        return representation.as_constructor(self)

    def package_of(self, x: Any, /) -> str:
        """Returns the qualified class name of the given object.
        @param x is the object
        @return the qualified class name of the given object.
        """
        package = ""
        module = inspect.getmodule(x)
        if module:
            if not settings.alias:
                package = module.__name__
            elif settings.alias == "*":
                package = ""
            elif module.__name__.startswith("fuzzylite."):
                package = settings.alias
            else:
                package = module.__name__
            if module.__name__.startswith("fuzzylite.examples.") and settings.alias:
                # use fully qualified package
                package += module.__name__[len("fuzzylite") :]
        if package and not package.endswith("."):
            package += "."
        return package

    def import_statement(self) -> str:
        """@return fuzzylite import statement based on the alias in the settings."""
        if not settings.alias:
            return "import fuzzylite"
        elif settings.alias == "*":
            return "from fuzzylite import *"
        else:
            return f"import fuzzylite as {settings.alias}"

    def as_constructor(
        self,
        x: T,
        /,
        fields: dict[str, Any] | None = None,
        *,
        positional: bool = False,
        cast_as: type[T] | None = None,
    ) -> str:
        """Returns the Python code representing the constructor of the given object using its signature.
        @param x is the object to construct
        @param fields overrides the parameters and arguments to use in the constructor
        @param positional indicates whether to use positional parameters or keyword parameters
        @param cast_as indicates the type to upcast the object (useful in inheritance approaches)
        @return the Python code representing the constructor of the given object.
        """
        arguments = self.construction_arguments(
            x, fields=fields, positional=positional, cast_as=cast_as
        )
        return f"{self.package_of((cast_as or x))}{(cast_as or x.__class__).__name__}({', '.join(arguments)})"

    def construction_arguments(
        self,
        x: T,
        /,
        fields: dict[str, Any] | None = None,
        *,
        positional: bool = False,
        cast_as: type[T] | None = None,
    ) -> list[str]:
        """Returns the list of parameters and arguments for the constructor of the given object using its signature.
        @param x is the object to construct
        @param fields overrides the parameters and arguments to use in the constructor
        @param positional indicates whether to use positional parameters or keyword parameters
        @param cast_as indicates the type to upcast the object (useful in inheritance approaches)
        @return the list of parameters and arguments for the constructor of the given object.
        """
        if fields is None:
            fields = vars(x) or {}
        arguments = []
        if x.__class__.__init__ == object.__init__:
            # there is no constructor in fuzzylite class hierarchy
            constructor = []
        else:
            constructor = list(
                inspect.signature((cast_as or x.__class__).__init__).parameters.values()
            )
        for parameter in constructor:
            if parameter.name == "self":
                continue
            if parameter.name in fields:
                value = self.repr(fields[parameter.name])
                argument = ("" if positional else f"{parameter.name}=") + value
                arguments.append(argument)
            else:
                if parameter.default != parameter.empty:
                    # if argument is not given for the parameter and the parameter has a default value,
                    # we can ignore it, but next parameter values need to use keywords.
                    positional = False
                else:
                    # if the parameter does not have a default value, then the constructor will not be valid code.
                    raise ValueError(
                        f"expected argument for parameter `{parameter.name}` in constructor of {x.__class__.__name__}, "
                        f"but it was missing from the fields context: {fields}"
                    )
        return arguments

    def repr_float(self, obj: float | np.floating[Any], level: int) -> Any:
        """Returns the representation of floats in fuzzylite."""
        from .operation import Op

        if Op.isinf(obj):
            infinity = f"{self.package_of(settings)}{np.abs(obj)!r}"
            return infinity if obj > 0 else f"-{infinity}"
        if Op.isnan(obj):
            return f"{self.package_of(settings)}{obj!r}"
        return builtins.repr(obj)

    repr_float16 = repr_float
    repr_float32 = repr_float
    repr_float64 = repr_float
    repr_float128 = repr_float

    def repr_ndarray(self, x: Array[Any], level: int) -> str:
        """Returns the representation of numpy arrays in fuzzylite."""
        if x.ndim == 0:
            return self.repr1(x.item(), level)
        elements = ", ".join(self.repr1(y, level) for y in x)
        return f"{self.package_of(settings)}{array.__name__}([{elements}])"


representation: Final[Representation] = Representation()
repr: Final[Callable[[Any], str]] = representation.repr
