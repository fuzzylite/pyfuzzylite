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
    "settings",
    "scalar",
    "to_float",
    "array",
    "isinf",
    "isnan",
    "inf",
    "nan",
]

import logging
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy import inf, isinf, isnan, nan

if typing.TYPE_CHECKING:
    from typing import Any, Final

    from .factory import FactoryManager
    from .types import Array, Scalar, ScalarArray

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


def scalar(x: Sequence[Any] | Array[Any] | Any, /) -> ScalarArray | Scalar:
    """Convert the values into a floating point value  defined by the library
    @param x is the value to convert.
    """
    return np.asarray(x, dtype=settings.float_type)


def array(x: Any, /, **kwargs: Any) -> Array[Any]:
    """Convert the value into a floating point defined by the library
    @param x is the value to convert.
    """
    return np.asarray(x, **kwargs)


@dataclass
class Settings:
    """Settings for the library."""

    float_type: Any = np.float64
    decimals: int = 3
    atol: float = 1e-03
    rtol: float = 0.0
    logger: logging.Logger = logging.getLogger("fuzzylite")
    _factory_manager: FactoryManager | None = None

    @property
    def factory_manager(self) -> FactoryManager:
        """The factory manager for the library."""
        if self._factory_manager is None:
            from .factory import FactoryManager

            self._factory_manager = FactoryManager()
        return self._factory_manager

    @factory_manager.setter
    def factory_manager(self, value: FactoryManager) -> None:
        """Set the factory manager for the library."""
        self._factory_manager = value

    @property
    def debugging(self) -> bool:
        """Whether the library is in debug mode."""
        return self.logger.level == logging.DEBUG

    @debugging.setter
    def debugging(self, value: bool | int) -> None:
        """Set the library debugging mode."""
        if isinstance(value, bool):
            self.logger.setLevel(logging.DEBUG if value else logging.ERROR)
        elif isinstance(value, int):
            self.logger.setLevel(value)
        else:
            raise TypeError(f"Expected bool or int, got {type(value)}")


settings: Final[Settings] = Settings()


@dataclass(frozen=True)
class Information:
    """Information about the library."""

    name: str = "pyfuzzylite"
    description: str = "a fuzzy logic control library in Python"
    license: str = "FuzzyLite License"
    author: str = "Juan Rada-Vilela, Ph.D."
    author_email: str = "jcrada@fuzzylite.com"
    company: str = "FuzzyLite Limited"
    website: str = "https://fuzzylite.com/"

    @property
    def version(self) -> str:
        """The version of the library."""
        __version__ = "8.0.0"
        return __version__


information: Final[Information] = Information()
