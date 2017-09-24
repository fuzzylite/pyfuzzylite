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

decimals = 3


def valid_name(name: str) -> str:
    result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
    return result if result else "unnamed"


def str_(x: float) -> str:
    global decimals
    return ("{:.%sf}" % decimals).format(x)


def scale(x: float, from_minimum: float, from_maximum: float, to_minimum: float, to_maximum: float) -> float:
    return (to_maximum - to_minimum) / (from_maximum - from_minimum) * (x - from_minimum) + to_minimum


def bound(x: float, minimum: float, maximum: float):
    if x > maximum: return maximum
    if x < minimum: return minimum
    return x
