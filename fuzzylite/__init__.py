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

# TODO: Find out better practices for these global variables...
DECIMALS = 3
MACHEPS = 1e-6

from .activation import *
from .defuzzifier import *
from .engine import *
from .exporter import *
from .factory import *
from .fuzzylite import FuzzyLite
from .hedge import *
from .importer import *
from .norm import *
from .operation import *
from .rule import *
from .term import *
from .variable import *
