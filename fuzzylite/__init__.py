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


from math import inf, isinf, isnan, nan  # noqa


from .activation import *
from .defuzzifier import *
from .engine import *
from .exporter import *
from .factory import *
from .hedge import *
from .importer import *
from .library import *
from .norm import *
from .operation import *
from .rule import *
from .term import *
from .variable import *

lib: Library = Library(
    decimals=3,
    abs_tolerance=1e-5,
)

# Import examples here to avoid circular imports with fl.lib
from . import examples  # noqa

__name__ = lib.name
__version__ = lib.version
__doc__ = lib.summary
