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

# flake8: noqa

import typing

from fuzzylite.activation import *
from fuzzylite.defuzzifier import *
from fuzzylite.engine import *
from fuzzylite.exporter import *
from fuzzylite.factory import *
from fuzzylite.hedge import *
from fuzzylite.importer import *
from fuzzylite.library_ import *
from fuzzylite.norm import *
from fuzzylite.operation import *
from fuzzylite.rule import *
from fuzzylite.term import *
from fuzzylite.variable import *

if typing.TYPE_CHECKING:
    from fuzzylite.library_ import Library

library: Library = Library()
__name__ = library.name
__version__ = library.version
__doc__ = library.summary
