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

import logging
from typing import Optional, SupportsFloat, Type, Union

from .factory import FactoryManager

__all__ = ["Library"]


class Library:
    """The Library class contains global settings and information about the library.
    @author Juan Rada-Vilela, Ph.D.
    @since 4.0.
    """

    def __init__(
        self,
        decimals: int,
        abs_tolerance: float,
        floating_point_type: Type[float] = float,
        factory_manager: Optional["FactoryManager"] = None,
    ) -> None:
        """Creates an instance of the library.
        @param decimals is the number of decimals utilized when formatting scalar values
        @param abs_tolerance is the minimum difference at which two scalar values are considered equivalent
        @param floating_point_type is the type of floating point (default is float, but numpy.float_ can also be used)
        @param factory_manager is the central manager of fuzzylite object factories
        @param logger is the logger of fuzzylite.
        """
        self.decimals = decimals
        self.abs_tolerance: float = abs_tolerance
        self.floating_point_type = floating_point_type
        self.factory_manager = factory_manager if factory_manager else FactoryManager()
        self.logger = logging.getLogger("fuzzylite")

    def floating_point(self, value: Union[SupportsFloat, str, bytes]) -> float:
        """Convert the value into a floating point defined by the library
        @param value is the value to convert.
        """
        return self.floating_point_type(value)

    def configure_logging(self, level: int, reset: bool = True) -> None:
        """Configure the logging service.
        @param level is the level of logging (see levels in Logging module)
        @param reset is whether to remove all previous logger handlers before configuring.
        """
        if reset:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
                handler.close()
        logging.basicConfig(
            level=level,
            datefmt="%Y-%m-%d %H:%M:%S",
            format="%(asctime)s %(levelname)s "
            "%(module)s::%(funcName)s()[%(lineno)d]"
            "\n%(message)s",
        )

    @property
    def debugging(self) -> bool:
        """Return whether the logger level is debugging."""
        return self.logger.level == logging.DEBUG

    @property
    def name(self) -> str:
        """Return the name of the `fuzzylite` library."""
        return "pyfuzzylite"

    @property
    def version(self) -> str:
        """Return the version of the `fuzzylite` library."""
        __version__ = "7.1.0"
        return __version__

    @property
    def license(self) -> str:
        """Return the license of the `fuzzylite` library."""
        return "GNU Affero General Public License v3"

    @property
    def description(self) -> str:
        """Return the description of the `fuzzylite` library."""
        return "a fuzzy logic control library in Python"

    @property
    def author(self) -> str:
        """Return the author of the `fuzzylite` library."""
        return "Juan Rada-Vilela, Ph.D."

    @property
    def author_email(self) -> str:
        """Return the email of the author of the `fuzzylite` library."""
        return "jcrada@fuzzylite.com"

    @property
    def company(self) -> str:
        """Return the name of the company that owns the `fuzzylite` library."""
        return "FuzzyLite Limited"

    @property
    def website(self) -> str:
        """Return the website of the `fuzzylite` library."""
        return "https://fuzzylite.com/"

    @property
    def summary(self) -> str:
        """Return a Markdown summary of the `fuzzylite` library."""
        result = """\
# pyfuzzylite: A Fuzzy Logic Control Library in Python

##  Introduction

**`fuzzylite`** is a free and open-source fuzzy logic control library
programmed in C++ for multiple platforms (e.g., Windows, Linux, Mac, iOS).
**`jfuzzylite`** is the equivalent `fuzzylite` library for Java and Android
platforms. **`pyfuzzylite`** is the equivalent `fuzzylite` library for Python.
**`QtFuzzyLite 6`** is (very likely) the best application available to easily
design and directly operate fuzzy logic controllers in real time.

If you are going to cite us in your article, please do so as:

```
Juan Rada-Vilela. The FuzzyLite Libraries for Fuzzy Logic Control, 2018. URL https://fuzzylite.com/.
```

```bibtex
 @misc{fl::fuzzylite,
 author={Juan Rada-Vilela},
 title={The FuzzyLite Libraries for Fuzzy Logic Control},
 url={https://fuzzylite.com/},
 year={2018}}
```

##  License of the FuzzyLite Libraries

The FuzzyLite Libraries, namely **`fuzzylite 6.0`** and **`jfuzzylite 6.0`**,
are licensed under the [**GNU General Public License (GPL)
3.0**](https://www.gnu.org/licenses/gpl.html), and **`pyfuzzylite 7.0`** is released under the [
**GNU Affero General Public License v3**](https://www.gnu.org/licenses/agpl.html). The FuzzyLite
Libraries are also offered under a **paid license for commercial purposes**. If you are using
them under a free license, please consider purchasing a license of **QtFuzzyLite** to support
the development of the libraries. If you want a commercial license of `fuzzylite`, `jfuzzylite`,
or `pyfuzzylite`, please contact [sales@fuzzylite.com](mailto:sales@fuzzylite.com).

## Features

The FuzzyLite Libraries have the following features:

**(6) Controllers**: Mamdani, Takagi-Sugeno, Larsen, Tsukamoto, Inverse
Tsukamoto, Hybrids

**(21) Linguistic terms**: (4) _Basic_: triangle, trapezoid, rectangle,
discrete. (9) _Extended_: bell, cosine, gaussian, gaussian product, pi-shape,
sigmoid difference, sigmoid product, spike. (5) _Edges_: binary, concave, ramp,
sigmoid, s-shape, z-shape. (3) _Functions_: constant, linear, function.

**(7) Activation methods**: general, proportional, threshold, first, last,
lowest, highest.

**(8) Conjunction and Implication (T-Norms)**: minimum, algebraic product,
bounded difference, drastic product, einstein product, hamacher product,
nilpotent minimum, function.

**(10) Disjunction and Aggregation (S-Norms)**: maximum, algebraic sum, bounded
sum, drastic sum, einstein sum, hamacher sum, nilpotent maximum, normalized
sum, unbounded sum, function.

**(7) Defuzzifiers**: (5) _Integral_: centroid, bisector, smallest of maximum,
largest of maximum, mean of maximum. (2) _Weighted_: weighted average, weighted
sum.

**(7) Hedges**: any, not, extremely, seldom, somewhat, very, function.

**(3) Importers**: FuzzyLite Language `fll`, Fuzzy Inference System `fis`,
Fuzzy Control Language `fcl`.

**(7) Exporters**: `C++`, `Java`, FuzzyLite Language `fll`, FuzzyLite Dataset
`fld`, `R` script, Fuzzy Inference System `fis`, Fuzzy Control Language `fcl`.

**(30+) Examples** of Mamdani, Takagi-Sugeno, Tsukamoto, and Hybrid controllers
from `fuzzylite`, Octave, and Matlab, each included in the following formats:
`C++`, `Java`, `fll`, `fld`, `R`, `fis`, and `fcl`.

In addition, you can easily:

* Create your own classes inheriting from `fuzzylite`, register them in the
  factories, and incorporate them to operate in `fuzzylite`.

* Utilize multiple rule blocks within a single engine, each containing any
  number of (possibly weighted) rule, and different conjunction, disjunction
  and activation operators.

* Write inference rules just naturally, e.g., `"if obstacle is left then steer
  is right"`.

* Return a default output value, lock the output values to be within specific
  ranges, lock the previous valid output value when no rules are activated.

* Explore the function space of your controller.

* Utilize the entire library across multiple threads as it is thread-safe.

* Download the sources, documentation, and binaries for the major platforms in
  the [**Downloads**](www.fuzzylite.com/downloads) tab.\
"""
        return result
