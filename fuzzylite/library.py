"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
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
from typing import Any, Final, overload

import numpy as np

from .types import Array, Scalar, ScalarArray

if typing.TYPE_CHECKING:
    from .factory import FactoryManager

np.seterr(invalid="ignore", divide="ignore")

logging.basicConfig(
    level=logging.ERROR,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)s %(module)s::%(funcName)s()[%(lineno)d]\n%(message)s",
)

array: Final = np.array
inf: Final = np.inf
nan: Final = np.nan


def to_float(x: Any, /) -> float:
    """Convert the value into a floating point defined by the library.

    Args:
        x: value to convert.

    Returns:
        converted value
    """
    return settings.float_type(x)  # type: ignore


@overload
def scalar(x: Sequence[Any] | Array[Any], /) -> ScalarArray: ...


@overload
def scalar(x: Any, /) -> Scalar: ...


def scalar(x: Sequence[Any] | Array[Any] | Any, /, **kwargs: Any) -> ScalarArray | Scalar:
    """Convert the values into a floating point value defined by the library.

    Args:
        x: value to convert.
        **kwargs: keyword arguments to pass to [numpy.asarray][]

    Returns:
        array of converted values

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
        """Constructor.

        Args:
            float_type: floating point type.
            decimals: number of decimals.
            atol: absolute tolerance.
            rtol: relative tolerance.
            alias: alias to use when representing objects (ie, `__repr__()`).
                Cases:
                    - fully qualified package when alias == "" (eg, `fuzzylite.term.Constant(name="A", height=1.0)`)
                    - no prefixes when alias == "*" (eg, `Constant(name="A", height=1.0)`)
                    - alias otherwise (eg, `{alias}.Constant(name="A", height=1.0)`
            logger: logger.
            factory_manager: factory manager.
        """
        self.float_type = float_type
        self.decimals = decimals
        self.atol = atol
        self.rtol = rtol
        self.alias = alias
        self.logger = logger or logging.getLogger("fuzzylite")
        self._factory_manager = factory_manager

    def __repr__(self) -> str:
        """Return code to construct the settings in Python.

        Returns:
            code to construct the settings in Python
        """
        fields = vars(self).copy()
        fields["factory_manager"] = fields.pop("_factory_manager")
        return representation.as_constructor(self, fields)

    @property
    def factory_manager(self) -> FactoryManager:
        """Get/Set the factory manager.

        # Getter

        Returns:
            factory manager

        # Setter

        Args:
            value (FactoryManager): factory manager
        """
        if self._factory_manager is None:
            # done here to avoid partially initialised class during __init__ using setter
            from .factory import FactoryManager

            self._factory_manager = FactoryManager()
        return self._factory_manager

    @factory_manager.setter
    def factory_manager(self, value: FactoryManager) -> None:
        """Set the factory manager.

        Args:
            value: factory manager
        """
        self._factory_manager = value

    @property
    def debugging(self) -> bool:
        """Get/Set the library in debug mode.

        # Getter

        Returns:
            whether the library is in debug mode

        # Setter

        Args:
            value (bool): set logging level to `DEBUG` if `true`, and to `ERROR` otherwise
        """
        return self.logger.level == logging.DEBUG

    @debugging.setter
    def debugging(self, value: bool) -> None:
        """Set the library debugging mode.

        Args:
            value: set logging level to `DEBUG` if `true`, and to `ERROR` otherwise
        """
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
        """Create a context with specific settings.

        Args:
            float_type: floating point type
            decimals: number of decimals.
            atol: absolute tolerance.
            rtol: relative tolerance.
            alias: alias for the library.
            logger: logger.
            factory_manager: factory manager.

        Returns:
            context with specific settings.
        """
        context_settings = {
            key: value for key, value in locals().items() if not (key == "self" or value is None)
        }
        if "factory_manager" in context_settings:
            context_settings["_factory_manager"] = context_settings.pop("factory_manager")
        rollback_settings = vars(self).copy()
        for key, value in context_settings.items():
            setattr(self, key, value)
        try:
            yield
        finally:
            for key, value in context_settings.items():
                setattr(self, key, rollback_settings[key])


settings: Final = Settings()


@dataclass(frozen=True, repr=False)
class Information:
    """Information about the library."""

    name: Final[str] = "fuzzylite"
    description: Final[str] = "a fuzzy logic control library in Python"
    license: Final[str] = "FuzzyLite License"
    author: Final[str] = "Juan Rada-Vilela, PhD"
    author_email: Final[str] = "jcrada@fuzzylite.com"
    company: Final[str] = "FuzzyLite"
    website: Final[str] = "https://fuzzylite.com/"
    copyright: Final[str] = (
        "Copyright (C) 2010-2024 FuzzyLite by Juan Rada-Vilela. All rights reserved."
    )

    def __repr__(self) -> str:
        """Return code to construct the information in Python.

        Returns:
            code to construct the information in Python
        """
        fields = vars(self).copy()
        fields["version"] = self.version
        return representation.as_constructor(self, fields)

    @property
    def version(self) -> str:
        """Automatic version of the library handled by poetry using `[tool.poetry_bumpversion.file."fuzzylite/library.py"]`.

        Returns:
            version of the library
        """
        __version__ = "8.0.4"
        return __version__


information: Final = Information()


class Representation(reprlib.Repr):
    """Representation class for the library."""

    T = typing.TypeVar("T")

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        increase_factor = int(1e6)  # very long in order to generate valid Python from large objects
        self.maxlevel = 10  # except for recursion level
        self.maxtuple *= increase_factor
        self.maxlist *= increase_factor
        self.maxarray *= increase_factor
        self.maxdict *= increase_factor
        self.maxset *= increase_factor
        self.maxfrozenset *= increase_factor
        self.maxdeque *= increase_factor
        self.maxstring *= increase_factor
        self.maxlong *= increase_factor
        self.maxother *= increase_factor

        from .exporter import FllExporter

        self.fll: Final = FllExporter()

    def __repr__(self) -> str:
        """Return code to construct the representation in Python.

        Returns:
            code to construct the representation in Python
        """
        return representation.as_constructor(self)

    def package_of(self, x: Any, /) -> str:
        """Return the qualified class name of the object.

        Args:
            x: object to get package of

        Returns:
             qualified class name of the object.
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
        """Return the library's import statement based on the alias in the settings.

        info: related
            - [fuzzylite.library.Settings.alias][]

        Returns:
            library's import statement based on the alias in the settings.
        """
        if not settings.alias:
            return "import fuzzylite"
        elif settings.alias == "*":
            return "from fuzzylite import *"
        else:
            return f"import fuzzylite as {settings.alias}"

    def as_constructor(  # noqa: D417 # Missing argument description in the docstring: `self`
        self,
        x: T,
        /,
        fields: dict[str, Any] | None = None,
        *,
        positional: bool = False,
        cast_as: type[T] | None = None,
    ) -> str:
        """Return the Python code to use the constructor of the object.

        Args:
            x: object to construct
            fields: override the parameters and arguments to use in the constructor
            positional: use positional parameters if `true`, and keyword parameters otherwise
            cast_as: type to upcast the object (useful in inheritance approaches)

        Returns:
            Python code to use the constructor of the object.
        """
        arguments = self.construction_arguments(
            x, fields=fields, positional=positional, cast_as=cast_as
        )
        return f"{self.package_of(cast_as or x)}{(cast_as or x.__class__).__name__}({', '.join(arguments)})"

    def construction_arguments(  # noqa: D417 # Missing argument description in the docstring: `self`
        self,
        x: T,
        /,
        fields: dict[str, Any] | None = None,
        *,
        positional: bool = False,
        cast_as: type[T] | None = None,
    ) -> list[str]:
        """Return the list of parameters and arguments for the constructor of the object.

        Args:
            x: object to construct
            fields: override the parameters and arguments to use in the constructor
            positional: use positional parameters if `true`, and keyword parameters otherwise
            cast_as: type to upcast the object (useful in inheritance approaches)

        Returns:
             list of parameters and arguments for the constructor of the object.
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

    def repr_float(self, x: float | np.floating[Any], level: int) -> str:
        """Return the string representation of the floating-point value in Python.

        Args:
            x: float to represent
            level: irrelevant

        Returns:
            string representation of the floating-point value in Python.
        """
        from .operation import Op

        if Op.isinf(x):
            infinity = f"{self.package_of(settings)}{np.abs(x)!r}"
            return infinity if x > 0 else f"-{infinity}"
        if Op.isnan(x):
            return f"{self.package_of(settings)}{x!r}"
        # TODO: should it be `return Op.str(x)`?
        return builtins.repr(x)

    repr_float16 = repr_float
    repr_float32 = repr_float
    repr_float64 = repr_float
    repr_float128 = repr_float

    def repr_ndarray(self, x: Array[Any], level: int) -> str:
        """Return the string representation of the numpy array in Python.

        Args:
            x: numpy array to represent
            level: level for recursion control

        Returns:
            string representation of the numpy array in Python
        """
        if x.ndim == 0:
            return self.repr1(x.item(), level)
        elements = ", ".join(self.repr1(y, level) for y in x)
        return f"{self.package_of(settings)}{array.__name__}([{elements}])"


representation: Final = Representation()
repr: Final = representation.repr
