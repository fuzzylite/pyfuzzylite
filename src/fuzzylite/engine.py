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


class Engine(object):
    __slots__ = "input", "output", "rule"

    def __init__(self, inputs=None, outputs=None, rules=None):
        self.input = {iv.name: iv for iv in inputs} if inputs else {}
        self.output = {ov.name: ov for ov in outputs} if outputs else {}
        self.rule = []
