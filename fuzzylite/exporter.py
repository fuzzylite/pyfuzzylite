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

    def engine(self, engine: 'Engine') -> str:
        pass

    def variable(self, v: 'Variable') -> str:
        result = [f"Variable: {v.name}",
                  f"{self.indent}description: {v.description}",
                  f"{self.indent}enabled: {str(v.enabled).lower()}",
                  f"{self.indent}range: {' '.join([Op.str(v.minimum), Op.str(v.maximum)])}",
                  f"{self.indent}lock-range: {str(v.enabled).lower()}",
                  *[f"{self.indent}{self.term(term)}" for term in v.terms]
                  ]
        return self.separator.join(result)

    def input_variable(self, iv: 'InputVariable') -> str:
        result = [f"InputVariable: {iv.name}",
                  f"{self.indent}description: {iv.description}",
                  f"{self.indent}enabled: {str(iv.enabled).lower()}",
                  f"{self.indent}range: {' '.join([Op.str(iv.minimum), Op.str(iv.maximum)])}",
                  f"{self.indent}lock-range: {str(iv.enabled).lower()}",
                  *[f"{self.indent}{self.term(term)}" for term in iv.terms]
                  ]
        return self.separator.join(result)

    def output_variable(self, ov: 'OutputVariable') -> str:
        result = [f"OutputVariable: {ov.name}",
                  f"{self.indent}description: {ov.description}",
                  f"{self.indent}enabled: {str(ov.enabled).lower()}",
                  f"{self.indent}range: {' '.join([Op.str(ov.minimum), Op.str(ov.maximum)])}",
                  f"{self.indent}lock-range: {str(ov.enabled).lower()}",
                  f"{self.indent}aggregation: {self.norm(ov.aggregation)}",
                  f"{self.indent}defuzzifier: {self.defuzzifier(ov.defuzzifier)}",
                  f"{self.indent}default: {Op.str(ov.default_value)}",
                  f"{self.indent}lock-previous: {str(ov.lock_previous_value).lower()}",
                  *[f"{self.indent}{self.term(term)}" for term in ov.terms]
                  ]
        return self.separator.join(result)

    def term(self, term: 'Term') -> str:
        result = ["term:", Op.valid_name(term.name), term.__class__.__name__, term.parameters()]
        return " ".join(result)

    def norm(self, norm: 'Norm') -> str:
        return type(norm).__name__ if norm else "none"

    def defuzzifier(self, defuzzifier: 'Defuzzifier') -> str:
        if not defuzzifier: return "none"
        from .defuzzifier import IntegralDefuzzifier, WeightedDefuzzifier
        result = [defuzzifier.__class__.__name__]
        if isinstance(defuzzifier, IntegralDefuzzifier):
            result.append(defuzzifier.resolution)
        elif isinstance(defuzzifier, WeightedDefuzzifier):
            result.append(defuzzifier.type)
        return " ".join(result)

    def rule(self, rule: 'Rule') -> str:
        return "rule: %s" % rule.text
