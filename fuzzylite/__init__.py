import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)-20s - %(levelname)s - %(message)s')

#TODO: Find out better practices for these global variables...
DECIMALS = 3
MACHEPS = 1e-6

from .engine import *
from .exporter import *
from .hedge import *
from .importer import *
from .norm import *
from .operation import *
from .rule import *
from .term import *
from .variable import *

