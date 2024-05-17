"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from . import examples  # noqa: F401
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
__author__ = information.author
__copyright__ = information.copyright
__license__ = information.license
