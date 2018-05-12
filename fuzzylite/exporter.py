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
    __slots__ = ["indent", "separator"]

    def __init__(self, indent="  ", separator="\n"):
        self.indent = indent
        self.separator = separator

    def to_string(self, instance: object):
        from .engine import Engine
        if isinstance(instance, Engine):
            return self.engine(instance)

        from .variable import InputVariable, OutputVariable, Variable
        if isinstance(instance, InputVariable):
            return self.input_variable(instance)
        if isinstance(instance, OutputVariable):
            return self.output_variable(instance)
        if isinstance(instance, Variable):
            return self.variable(instance)

        from .term import Term
        if isinstance(instance, Term):
            return self.term(instance)

        from .defuzzifier import Defuzzifier
        if isinstance(instance, Defuzzifier):
            return self.defuzzifier(instance)

        from .rule import RuleBlock, Rule
        if isinstance(instance, RuleBlock):
            return self.rule_block(instance)
        if isinstance(instance, Rule):
            return self.rule(instance)

        from .norm import Norm
        if isinstance(instance, Norm):
            return self.norm(instance)

        from .activation import Activation
        if isinstance(instance, Activation):
            return self.activation(instance)

        raise ValueError(f"expected a fuzzylite object, but found '{type(instance).__name__}'")

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

    def rule_block(self, rb: 'RuleBlock') -> str:
        result = [f"RuleBlock: {rb.name}",
                  f"{self.indent}description: {rb.description}",
                  f"{self.indent}enabled: {str(rb.enabled).lower()}",
                  f"{self.indent}conjunction: {self.norm(rb.conjunction)}",
                  f"{self.indent}disjunction: {self.norm(rb.disjunction)}",
                  f"{self.indent}implication: {self.norm(rb.implication)}",
                  f"{self.indent}activation: {self.activation(rb.activation)}",
                  *[f"{self.indent}{self.rule(rule)}" for rule in rb.rules]
                  ]
        return self.separator.join(result)

    def term(self, term: 'Term') -> str:
        result = ["term:", Op.valid_name(term.name), term.__class__.__name__]
        parameters = term.parameters()
        if parameters:
            result.append(parameters)
        return " ".join(result)

    def norm(self, norm: 'Norm') -> str:
        return type(norm).__name__ if norm else "none"

    def activation(self, activation: 'Activation') -> str:
        return type(activation).__name__ if activation else "none"

    def defuzzifier(self, defuzzifier: 'Defuzzifier') -> str:
        if not defuzzifier: return "none"
        from .defuzzifier import IntegralDefuzzifier, WeightedDefuzzifier
        result = [defuzzifier.__class__.__name__]
        if isinstance(defuzzifier, IntegralDefuzzifier):
            result.append(str(defuzzifier.resolution))
        elif isinstance(defuzzifier, WeightedDefuzzifier):
            result.append(defuzzifier.type.name)
        return " ".join(result)

    def rule(self, rule: 'Rule') -> str:
        return "rule: %s" % rule.text
