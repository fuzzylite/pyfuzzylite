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
from typing import Type

import fuzzylite as fl
from tests.assert_component import BaseAssert


class RuleBlockAssert(BaseAssert[fl.RuleBlock]):
    pass


class TestRuleBlock(unittest.TestCase):
    def test_constructor(self) -> None:
        RuleBlockAssert(self, fl.RuleBlock()) \
            .exports_fll(
            "\n".join(
                [
                    "RuleBlock: ",
                    "  enabled: true",
                    "  conjunction: none",
                    "  disjunction: none",
                    "  implication: none",
                    "  activation: none",
                ]))

        RuleBlockAssert(self, fl.RuleBlock("rb", "a ruleblock",
                                           rules=[fl.Rule.parse("if a then z"),
                                                  fl.Rule.parse("if b then y")],
                                           conjunction=fl.TNorm(), disjunction=fl.SNorm(),
                                           implication=fl.TNorm(), activation=fl.Activation())) \
            .exports_fll(
            "\n".join(
                [
                    "RuleBlock: rb",
                    "  description: a ruleblock",
                    "  enabled: true",
                    "  conjunction: TNorm",
                    "  disjunction: SNorm",
                    "  implication: TNorm",
                    "  activation: Activation",
                    "  rule: if a then z",
                    "  rule: if b then y",
                ]))


class TestExpression(unittest.TestCase):
    def test_proposition(self) -> None:
        self.assertEqual(str(fl.Proposition()), "? is ?")

        proposition = fl.Proposition()
        proposition.variable = fl.Variable("variable")
        proposition.hedges = [fl.Very()]
        proposition.term = fl.Term("term")

        self.assertEqual("variable is very term", str(proposition))

    def test_operator(self) -> None:
        self.assertEqual("", fl.Operator().name)

        operator = fl.Operator()
        operator.name = "AND"
        operator.left = fl.Proposition()
        operator.right = fl.Proposition()
        self.assertEqual("AND", str(operator))


class RuleAssert(BaseAssert[fl.Rule]):

    def parser_fails(self, text: str, exception: Type[Exception] = SyntaxError,
                     regex: str = "") -> 'RuleAssert':
        with self.test.assertRaisesRegex(exception, regex):
            self.actual.text = text
        return self

    def has_text(self, text: str) -> 'RuleAssert':
        self.test.assertEqual(self.actual.text, text)
        return self


class TestRule(unittest.TestCase):

    def test_rule_parser(self) -> None:
        pass

    def test_rule_parser_exceptions(self) -> None:
        RuleAssert(self, fl.Rule()) \
            .parser_fails("", SyntaxError, "expected an if-then rule") \
            .parser_fails("then", SyntaxError, "expected keyword 'if'") \
            .parser_fails("if", SyntaxError, "expected keyword 'then'") \
            .parser_fails("if then", SyntaxError, "expected an antecedent in rule") \
            .parser_fails("if antecedent then", SyntaxError, "expected a consequent in rule") \
            .parser_fails("if antecedent then consequent with", SyntaxError,
                          "expected the rule weight") \
            .parser_fails("if antecedent then consequent with 1.0 extra", SyntaxError,
                          "unexpected token 'extra'")


if __name__ == '__main__':
    unittest.main()
