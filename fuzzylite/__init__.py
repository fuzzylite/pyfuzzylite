"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, PhD <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited.

fuzzylite is a registered trademark of FuzzyLite Limited.
"""
from . import examples  # noqa
from .activation import *
from .benchmark import *
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
from .types import *
from .variable import *

__name__ = information.name
__doc__ = information.description
__version__ = information.version
