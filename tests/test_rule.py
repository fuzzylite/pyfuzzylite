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

import unittest

from fuzzylite import *
from tests.assert_component import ComponentAssert


class RuleBlockAssert(ComponentAssert):
    pass


class TestRuleBlock(unittest.TestCase):
    def test_constructor(self):
        RuleBlockAssert(self, RuleBlock()) \
            .exports_fll("\n".join(
                ["RuleBlock: ",
                 "  description: ",
                 "  enabled: true",
                 "  conjunction: none",
                 "  disjunction: none",
                 "  implication: none",
                 "  activation: none",
                 ]))

        RuleBlockAssert(self, RuleBlock("rb", "a ruleblock", enabled=False,
                                        conjunction=TNorm(), disjunction=SNorm(),
                                        implication=TNorm(), activation=Activation(),
                                        rules=[Rule.parse("if a then z"),
                                               Rule.parse("if b then y")])) \
            .exports_fll("\n".join(
                ["RuleBlock: rb",
                 "  description: a ruleblock",
                 "  enabled: false",
                 "  conjunction: TNorm",
                 "  disjunction: SNorm",
                 "  implication: TNorm",
                 "  activation: Activation",
                 "  rule: if a then z",
                 "  rule: if b then y",
                 ]))


class TestExpression(unittest.TestCase):
    def test_proposition(self):
        self.assertEqual(str(Proposition()), "? is ?")

        proposition = Proposition()
        proposition.variable = Variable("variable")
        proposition.hedges = [Very()]
        proposition.term = Term("term")

        self.assertEqual(str(proposition), "variable is very term")

    def test_operator(self):
        self.assertEqual(Operator().name, "")

        operator = Operator()
        operator.name = "operator"
        operator.left = "left"
        operator.right = "right"
        self.assertEqual(str(operator), "operator")


class RuleAssert(ComponentAssert):
    def parser_fails(self, text, exception=SyntaxError, regex=""):
        with self.test.assertRaisesRegex(exception, regex):
            self.actual.text = text
        return self

    def has_text(self, text: str):
        self.test.assertEqual(self.actual.text, text)
        return self


class TestRule(unittest.TestCase):
    def test_rule_parser_exceptions(self):
        RuleAssert(self, Rule()) \
            .parser_fails("", SyntaxError, "expected an if-then rule") \
            .parser_fails("then", SyntaxError, "expected keyword 'if'") \
            .parser_fails("if", SyntaxError, "expected keyword 'then'") \
            .parser_fails("if then", SyntaxError, "expected an antecedent in rule") \
            .parser_fails("if antecedent then", SyntaxError, "expected a consequent in rule") \
            .parser_fails("if antecedent then consequent with", SyntaxError, "expected the rule weight") \
            .parser_fails("if antecedent then consequent with 1.0 extra", SyntaxError, "unexpected token 'extra'")

    def test_rule_parser_exceptions(self):
        RuleAssert(self, Rule()) \



if __name__ == '__main__':
    unittest.main()
