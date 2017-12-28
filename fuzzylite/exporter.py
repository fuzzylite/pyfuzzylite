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

from .operation import Operation as Op


class Exporter(object):
    pass


class FllExporter(Exporter):
    __slots__ = "indent", "separator"

    def __init__(self, indent="  ", separator="\n"):
        self.indent = indent
        self.separator = separator

    def input_variable(self, iv: 'InputVariable'):
        result = [f"InputVariable: {iv.name}",
                  f"{self.indent}description: {iv.description}",
                  f"{self.indent}enabled: {str(iv.enabled).lower()}",
                  f"{self.indent}range: {' '.join([Op.str(iv.minimum), Op.str(iv.maximum)])}",
                  f"{self.indent}lock-range: {str(iv.enabled).lower()}",
                  *[f"{self.indent}{str(term)}" for term in iv.terms]
                  ]
        return self.separator.join(result)

    def term(self, term: 'Term') -> str:
        result = ["term:", Op.valid_name(term.name), term.__class__.__name__, term.parameters()]
        return " ".join(result)

    def norm(self, norm: 'Norm') -> str:
        return type(norm).__name__ if norm else "none"

    def rule(self, rule: 'Rule') -> str:
        return "rule: %s" % rule.text
