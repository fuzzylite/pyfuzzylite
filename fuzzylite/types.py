from __future__ import annotations

__all__ = ["Array", "Object", "Scalar", "ScalarArray"]

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray as Array

from .activation import Activation
from .defuzzifier import Defuzzifier
from .engine import Engine
from .exporter import Exporter
from .factory import CloningFactory, ConstructionFactory
from .hedge import Hedge
from .importer import Importer
from .library import Information, Settings
from .norm import Norm
from .rule import Antecedent, Consequent, Rule, RuleBlock
from .term import Term
from .variable import Variable

Scalar = Union[float, np.floating[Any], Array[np.floating[Any]]]
ScalarArray = Array[np.floating[Any]]

Object = Union[
    Activation,
    Antecedent,
    Consequent,
    CloningFactory,
    ConstructionFactory,
    Defuzzifier,
    Engine,
    Exporter,
    Hedge,
    Importer,
    Information,
    Norm,
    Rule,
    RuleBlock,
    Settings,
    Term,
    Variable,
]
