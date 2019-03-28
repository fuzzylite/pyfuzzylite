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
from typing import Dict, List, Optional, Type, Union
from unittest.mock import MagicMock

import fuzzylite as fl
from tests.assert_component import BaseAssert

SimpleDimmer = """
Engine: SimpleDimmer
InputVariable: Ambient
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: DARK Triangle 0.000 0.250 0.500
  term: MEDIUM Triangle 0.250 0.500 0.750
  term: BRIGHT Triangle 0.500 0.750 1.000
OutputVariable: Power
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 200
  default: nan
  lock-previous: false
  term: LOW Triangle 0.000 0.250 0.500
  term: MEDIUM Triangle 0.250 0.500 0.750
  term: HIGH Triangle 0.500 0.750 1.000
#RuleBlock:
  # rule: if Ambient is DARK then Power is HIGH
  # rule: if Ambient is MEDIUM then Power is MEDIUM
  # rule: if Ambient is BRIGHT then Power is LOW
  """


class TestExpression(unittest.TestCase):

    def test_proposition(self) -> None:
        self.assertEqual("", str(fl.Proposition()))

        proposition = fl.Proposition()
        proposition.variable = fl.Variable("variable")
        proposition.hedges = [fl.Very()]
        proposition.term = fl.Term("term")

        self.assertEqual("variable is very term", str(proposition))

        proposition = fl.Proposition(fl.Variable("variable"), [fl.Very()], fl.Term("term"))
        self.assertEqual("variable is very term", str(proposition))

    def test_operator(self) -> None:
        self.assertEqual("", fl.Operator().name)

        operator = fl.Operator()
        operator.name = "AND"
        self.assertEqual("AND", str(operator))

        operator = fl.Operator()
        operator.name = "OR"
        operator.left = fl.Proposition(fl.Variable("variable_a"), [fl.Very()], fl.Term("term_a"))
        operator.right = fl.Proposition(fl.Variable("variable_b"), [fl.Very()], fl.Term("term_b"))
        self.assertEqual("OR", str(operator))


class AssertAntecedent:

    def __init__(self, test: unittest.TestCase, engine: fl.Engine) -> None:
        self.test = test
        self.engine = engine

    def can_load_antecedent(self,
                            text: str,
                            postfix: Optional[str] = None,
                            prefix: Optional[str] = None,
                            infix: Optional[str] = None) -> 'AssertAntecedent':
        antecedent = fl.Antecedent(text)
        antecedent.load(self.engine)
        self.test.assertTrue(antecedent.is_loaded())
        self.test.assertTrue(postfix or prefix or infix, "expected one of {postfix, prefix, infix}")
        if postfix:
            self.test.assertEqual(postfix,
                                  antecedent.postfix(antecedent.expression))
        if prefix:
            self.test.assertEqual(prefix,
                                  antecedent.prefix(antecedent.expression))
        if infix:
            self.test.assertEqual(infix,
                                  antecedent.infix(antecedent.expression))
        return self

    def cannot_load_antecedent(self,
                               text: str,
                               exception: Type[Exception],
                               regex: str) -> 'AssertAntecedent':
        antecedent = fl.Antecedent(text)
        with self.test.assertRaisesRegex(exception, regex):
            antecedent.load(self.engine)
        return self

    def has_activation_degrees(self,
                               inputs: Union[
                                   Dict[fl.InputVariable, List[float]],
                                   Dict[fl.OutputVariable, List[List[fl.Activated]]]],
                               rules: Dict[str, List[float]],
                               conjunction: Optional[fl.TNorm] = None,
                               disjunction: Optional[fl.SNorm] = None,
                               decimal_places: int = 3
                               ) -> 'AssertAntecedent':
        self.test.assertTrue(inputs, msg="inputs is empty")
        self.test.assertTrue(rules, msg="rules is empty")

        index = 0
        more_values = True
        while more_values:
            for variable, values in inputs.items():
                more_values &= index + 1 < len(values)
                if isinstance(variable, fl.InputVariable):
                    variable.value = values[index]  # type: ignore
                elif isinstance(variable, fl.OutputVariable):
                    variable.fuzzy.terms = values[index]  # type: ignore

            for text, values in rules.items():
                antecedent = fl.Antecedent(text)
                antecedent.load(self.engine)
                obtained = antecedent.activation_degree(
                    conjunction=conjunction, disjunction=disjunction)
                expected = values[index]
                self.test.assertAlmostEqual(expected, obtained, places=decimal_places,
                                            msg=f"at index {index}")
            index += 1

        return self


class TestAntecedent(unittest.TestCase):

    def test_loaded(self) -> None:
        antecedent = fl.Antecedent()
        self.assertFalse(antecedent.is_loaded())

        antecedent.expression = fl.Expression()
        self.assertTrue(antecedent.is_loaded())

        antecedent.unload()
        self.assertFalse(antecedent.is_loaded())

    def test_antecedent_load_input_variable(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is DARK", infix="Ambient is DARK")

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is very DARK", infix="Ambient is very DARK")

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is any", infix="Ambient is any")

    def test_antecedent_load_input_variables_connectors(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Ambient is DARK {fl.Rule.AND} Ambient is BRIGHT",
            postfix="Ambient is DARK Ambient is BRIGHT and")

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Ambient is very DARK {fl.Rule.OR} Ambient is very BRIGHT",
            postfix="Ambient is very DARK Ambient is very BRIGHT or")

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Ambient is any {fl.Rule.AND} Ambient is not any",
            postfix="Ambient is any Ambient is not any and")

    def test_antecedent_load_output_variables_connectors(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Power is HIGH {fl.Rule.AND} Power is LOW",
            postfix="Power is HIGH Power is LOW and")

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Power is very HIGH {fl.Rule.OR} Power is very LOW",
            postfix="Power is very HIGH Power is very LOW or")

        AssertAntecedent(self, engine).can_load_antecedent(
            f"Power is any {fl.Rule.AND} Power is not any",
            postfix="Power is any Power is not any and")

    def test_antecedent_load_fails(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).cannot_load_antecedent(
            "", SyntaxError,
            "expected the antecedent of a rule, but found none")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            f"Ambient is any {fl.Rule.AND}", SyntaxError,
            "operator 'and' expects 2 operands, but found 1")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            f"Ambient is any DARK", SyntaxError,
            "expected variable or logical operator, but found 'DARK'")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            "InvalidVariable is any", SyntaxError,
            "expected variable or logical operator, but found 'InvalidVariable'")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            "Ambient isn't", SyntaxError,
            "expected keyword 'is', but found 'isn't'")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            "Ambient is very invalid", SyntaxError,
            "expected hedge or term, but found 'invalid'")

        AssertAntecedent(self, engine).cannot_load_antecedent(
            "Ambient is invalid_term", SyntaxError,
            "expected hedge or term, but found 'invalid_term'")

    def test_antecedent_to_string(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is DARK and Ambient is BRIGHT",
            postfix="Ambient is DARK Ambient is BRIGHT and",
            prefix="and Ambient is DARK Ambient is BRIGHT",
            infix="Ambient is DARK and Ambient is BRIGHT",
        )

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is DARK and (Ambient is MEDIUM or Ambient is BRIGHT)",
            postfix="Ambient is DARK Ambient is MEDIUM Ambient is BRIGHT or and",
            prefix="and Ambient is DARK or Ambient is MEDIUM Ambient is BRIGHT",
            infix="Ambient is DARK and Ambient is MEDIUM or Ambient is BRIGHT",
        )

        AssertAntecedent(self, engine).can_load_antecedent(
            "Ambient is BRIGHT or Ambient is DARK and Ambient is MEDIUM",
            postfix="Ambient is BRIGHT Ambient is DARK Ambient is MEDIUM and or",
            prefix="or Ambient is BRIGHT and Ambient is DARK Ambient is MEDIUM",
            infix="Ambient is BRIGHT or Ambient is DARK and Ambient is MEDIUM",
        )

    def test_activation_degrees(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        # Test disabled variables
        engine.variable("Ambient").enabled = False
        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is DARK":
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Ambient is MEDIUM":
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Ambient is BRIGHT":
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        # Test enabled variables
        engine.variable("Ambient").enabled = True

        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is DARK":
                    [0.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0],
                "Ambient is MEDIUM":
                    [0.0, 0.0, 0.6, 1.0, 0.6, 0.0, 0.0],
                "Ambient is BRIGHT":
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.0],
            }
        )

        # Test hedges
        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is very DARK":
                    [0.0, 0.64, 0.160, 0.0, 0.000, 0.000, 0.0],
                "Ambient is somewhat MEDIUM":
                    [0.0, 0.00, 0.775, 1.0, 0.775, 0.000, 0.0],
                "Ambient is seldom BRIGHT":
                    [0.0, 0.00, 0.000, 0.0, 0.447, 0.684, 0.0],
            }
        )

        # Test multiple hedges
        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is very very DARK":
                    [0.0, 0.41, 0.026, 0.0, 0.000, 0.000, 0.0],
                "Ambient is somewhat very MEDIUM":
                    [0.0, 0.00, 0.600, 1.0, 0.600, 0.000, 0.0],
                "Ambient is seldom very BRIGHT":
                    [0.0, 0.00, 0.000, 0.0, 0.283, 0.576, 0.0],
            }
        )

        # Test special hedges
        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is any":
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "Ambient is not any":
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Ambient is not not any":
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_activation_degrees_with_norms(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is DARK and Ambient is MEDIUM":
                    [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                "Ambient is MEDIUM and Ambient is BRIGHT":
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                "Ambient is BRIGHT and Ambient is DARK":
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            conjunction=fl.Minimum()
        )

        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.input_variable("Ambient"):
                    [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }, {
                "Ambient is DARK or Ambient is MEDIUM":
                    [0.0, 0.8, 0.6, 1.0, 0.6, 0.0, 0.0],
                "Ambient is MEDIUM or Ambient is BRIGHT":
                    [0.0, 0.0, 0.6, 1.0, 0.6, 0.8, 0.0],
                "Ambient is BRIGHT or Ambient is DARK":
                    [0.0, 0.8, 0.4, 0.0, 0.4, 0.8, 0.0],
            },
            disjunction=fl.Maximum()
        )

    def test_activation_degrees_output(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        # Test enabled variables
        low = engine.output_variable("Power").term("LOW")
        medium = engine.output_variable("Power").term("MEDIUM")
        high = engine.output_variable("Power").term("HIGH")

        AssertAntecedent(self, engine).has_activation_degrees(
            {
                engine.output_variable("Power"): [
                    [fl.Activated(low, 0.0), fl.Activated(medium, 0.0), fl.Activated(high, 0.0)],
                    [fl.Activated(low, 0.0), fl.Activated(medium, 0.0), fl.Activated(high, 1.0)],
                    [fl.Activated(low, 0.0), fl.Activated(medium, 1.0), fl.Activated(high, 0.0)],
                    [fl.Activated(low, 0.0), fl.Activated(medium, 1.0), fl.Activated(high, 1.0)],
                    [fl.Activated(low, 1.0), fl.Activated(medium, 0.0), fl.Activated(high, 0.0)],
                    [fl.Activated(low, 1.0), fl.Activated(medium, 0.0), fl.Activated(high, 1.0)],
                    [fl.Activated(low, 1.0), fl.Activated(medium, 1.0), fl.Activated(high, 0.0)],
                    [fl.Activated(low, 1.0), fl.Activated(medium, 1.0), fl.Activated(high, 1.0)],
                ]
            }, {
                "Power is LOW":
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "Power is MEDIUM":
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "Power is HIGH":
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )


class AssertConsequent:

    def __init__(self, test: unittest.TestCase, engine: fl.Engine):
        self.test = test
        self.engine = engine

    def can_load_consequent(self, text: str) -> 'AssertConsequent':
        consequent = fl.Consequent(text)
        consequent.load(self.engine)
        self.test.assertTrue(consequent.is_loaded())
        self.test.assertTrue(consequent.conclusions)
        self.test.assertEqual(text, str(consequent))
        return self

    def cannot_load_consequent(self,
                               text: str,
                               exception: Type[Exception],
                               regex: str) -> 'AssertConsequent':
        consequent = fl.Consequent(text)
        with self.test.assertRaisesRegex(exception, regex):
            consequent.load(self.engine)
        return self

    def modify_consequent(self,
                          text: str,
                          activation_degree: float,
                          expected: Dict[fl.OutputVariable, List[fl.Activated]],
                          implication: Optional[fl.TNorm] = None,
                          decimal_places: int = 3
                          ) -> 'AssertConsequent':
        self.test.assertTrue(expected, "expected cannot be empty")

        consequent = fl.Consequent(text)
        consequent.load(self.engine)
        consequent.modify(activation_degree, implication)
        for variable in expected.keys():
            expected_terms = {t.term.name: t.degree
                              for t in expected[variable]}
            obtained_terms = {t.term.name: t.degree
                              for t in variable.fuzzy.terms}
            self.test.assertSetEqual(set(expected_terms.keys()), set(obtained_terms.keys()))

            for expected_term, expected_activation in expected_terms.items():
                obtained_activation = obtained_terms[expected_term]
                self.test.assertAlmostEqual(
                    expected_activation, obtained_activation,
                    places=decimal_places, msg=f"for activated term {expected_term}")

            for activated in variable.fuzzy.terms:
                self.test.assertEqual(implication, activated.implication,
                                      msg=f"in {str(activated)}")

            variable.fuzzy.clear()
        return self

    def cannot_modify_consequent(self,
                                 text: str,
                                 activation_degree: float,
                                 implication: Optional[fl.TNorm] = None
                                 ) -> 'AssertConsequent':
        pass


class TestConsequent(unittest.TestCase):

    def test_loaded(self) -> None:
        consequent = fl.Consequent()
        self.assertFalse(consequent.is_loaded())

        consequent.conclusions.append(fl.Proposition())
        self.assertTrue(consequent.is_loaded())

        consequent.unload()
        self.assertFalse(consequent.is_loaded())

    def test_consequent_load_output_variable(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertConsequent(self, engine).can_load_consequent("Power is HIGH")

        AssertConsequent(self, engine).can_load_consequent("Power is MEDIUM")

        AssertConsequent(self, engine).can_load_consequent("Power is LOW")

    def test_consequent_load_with_connectors(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertConsequent(self, engine).can_load_consequent(
            "Power is HIGH and Power is HIGH")

        AssertConsequent(self, engine).can_load_consequent(
            "Power is very HIGH and Power is very HIGH")

        AssertConsequent(self, engine).can_load_consequent(
            "Power is any LOW and Power is not any HIGH")

    def test_consequent_load_fails(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        AssertConsequent(self, engine).cannot_load_consequent(
            "", SyntaxError,
            "expected the consequent of a rule, but found none")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Ambient is DARK", SyntaxError,
            "consequent expected an output variable, but found 'Ambient'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power HIGH", SyntaxError,
            "consequent expected keyword 'is', but found 'HIGH'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is ALL", SyntaxError,
            "consequent expected a hedge or term, but found 'ALL'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is very ALL", SyntaxError,
            "consequent expected a hedge or term, but found 'ALL'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is very HIGH or Power is very HIGH", SyntaxError,
            "unexpected token 'or'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power", SyntaxError,
            "consequent expected keyword 'is' after 'Power'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is", SyntaxError,
            "consequent expected hedge or term after 'is'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is very", SyntaxError,
            "consequent expected hedge or term after 'very'")

        AssertConsequent(self, engine).cannot_load_consequent(
            "Power is very LOW and", SyntaxError,
            "consequent expected output variable after 'and'")

    def test_modify_consequent(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        power = engine.output_variable("Power")
        low = power.term("LOW")
        high = power.term("HIGH")

        AssertConsequent(self, engine).modify_consequent(
            "Power is LOW", 0.5, {power: [fl.Activated(low, 0.5)]})

        AssertConsequent(self, engine).modify_consequent(
            "Power is LOW", 0.5, {power: [fl.Activated(low, 0.5)]},
            implication=None)

        AssertConsequent(self, engine).modify_consequent(
            "Power is very LOW", 0.5, {power: [fl.Activated(low, 0.25)]},
            implication=fl.Minimum())

        AssertConsequent(self, engine).modify_consequent(
            "Power is LOW and Power is HIGH", 0.25,
            {power: [fl.Activated(low, 0.25), fl.Activated(high, 0.25)]},
            implication=fl.AlgebraicProduct())

        AssertConsequent(self, engine).modify_consequent(
            "Power is LOW and Power is very HIGH", 0.5,
            {power: [fl.Activated(low, 0.5), fl.Activated(high, 0.25)]})

        power.enabled = False
        AssertConsequent(self, engine).modify_consequent(
            "Power is LOW and Power is very HIGH", 0.5, {power: []})

    def test_cannot_modify_consequent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "consequent is not loaded"):
            fl.Consequent("").modify(fl.nan, None)


class RuleAssert:

    def __init__(self, test: unittest.TestCase):
        self.test = test

    def can_parse_rule(self, text: str, as_text: Optional[str] = None) -> 'RuleAssert':
        rule = fl.Rule()
        rule.parse(text)
        self.test.assertEqual(as_text if as_text else text, rule.text)
        return self

    def can_load_rule(self, text: str, engine: fl.Engine) -> 'RuleAssert':
        rule = fl.Rule()
        rule.parse(text)
        rule.load(engine)
        self.test.assertEqual(text, rule.text)
        self.test.assertEqual(rule.activation_degree, 0.0)
        self.test.assertEqual(rule.triggered, False)
        self.can_create_rule(text, engine)
        return self

    def can_create_rule(self, text: str, engine: fl.Engine) -> 'RuleAssert':
        rule = fl.Rule.create(text, engine)
        self.test.assertEqual(text, rule.text)
        return self

    def cannot_parse_rule(self,
                          text: str,
                          exception: Type[Exception] = SyntaxError,
                          regex: str = "") -> 'RuleAssert':
        with self.test.assertRaisesRegex(exception, regex):
            rule = fl.Rule()
            rule.parse(text)
        return self

    def cannot_create_rule(self,
                           text: str,
                           engine: fl.Engine,
                           exception: Type[Exception] = SyntaxError,
                           regex: str = "") -> 'RuleAssert':
        with self.test.assertRaisesRegex(exception, regex):
            fl.Rule.create(text, engine)
        return self


class TestRule(unittest.TestCase):

    def test_text_setter(self) -> None:
        rule = fl.Rule()
        rule.parse = MagicMock()  # type: ignore
        rule.text = "if a then b"
        rule.parse.assert_called_once_with("if a then b")  # type: ignore

    def test_can_parse_rule(self) -> None:
        RuleAssert(self).can_parse_rule("if a then b")
        RuleAssert(self).can_parse_rule("if a then b with 1.0", as_text="if a then b")
        RuleAssert(self).can_parse_rule(
            "if antecedent1 antecedent2 then consequent1 consequent2")

    def test_parser_exceptions(self) -> None:
        RuleAssert(self).cannot_parse_rule(
            "", SyntaxError, "expected an if-then rule")
        RuleAssert(self).cannot_parse_rule(
            "then", SyntaxError, "expected keyword 'if'")
        RuleAssert(self).cannot_parse_rule(
            "if", SyntaxError, "expected keyword 'then'")
        RuleAssert(self).cannot_parse_rule(
            "if then", SyntaxError, "expected an antecedent in rule")
        RuleAssert(self).cannot_parse_rule(
            "if antecedent then", SyntaxError, "expected a consequent in rule")
        RuleAssert(self).cannot_parse_rule(
            "if antecedent then consequent with", SyntaxError, "expected the rule weight")
        RuleAssert(self).cannot_parse_rule(
            "if antecedent then consequent with 1.0 extra", SyntaxError, "unexpected token 'extra'")

    def test_can_load_rule(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        RuleAssert(self).can_load_rule(
            "if Ambient is DARK then Power is HIGH", engine)
        RuleAssert(self).can_load_rule(
            "if Ambient is MEDIUM then Power is MEDIUM", engine)
        RuleAssert(self).can_load_rule(
            "if Ambient is BRIGHT then Power is LOW", engine)

    def test_deactivate(self) -> None:
        rule = fl.Rule()
        rule.activation_degree = fl.nan
        rule.triggered = True
        rule.deactivate()
        self.assertEqual(rule.activation_degree, 0.0)
        self.assertEqual(rule.triggered, False)

    def test_activate_with(self) -> None:
        rule = fl.Rule.create("if Ambient is DARK then Power is HIGH")

        self.assertFalse(rule.is_loaded())
        with self.assertRaisesRegex(RuntimeError, "rule is not loaded"):
            rule.activate_with(None, None)

        rule.load(fl.FllImporter().from_string(SimpleDimmer))
        self.assertTrue(rule.is_loaded())

        rule.antecedent.activation_degree = MagicMock(return_value=0.5)  # type: ignore
        weights = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, -fl.inf, fl.inf]
        expected = [0.0, 0.05, 0.125, 0.25, 0.375, 0.45, 0.5, -fl.inf, fl.inf]

        obtained = []
        for weight in weights:
            rule.weight = weight
            obtained.append(rule.activate_with(None, None))

        self.assertEqual(expected, obtained)

    def test_trigger(self) -> None:
        rule = fl.Rule.create("if Ambient is DARK then Power is HIGH")

        self.assertFalse(rule.is_loaded())
        with self.assertRaisesRegex(RuntimeError, "rule is not loaded"):
            rule.trigger(None)

        rule.load(fl.FllImporter().from_string(SimpleDimmer))
        self.assertTrue(rule.is_loaded())

        rule.consequent.modify = MagicMock()  # type: ignore
        # combinations: enabled and activation_degree
        rule.enabled = False
        rule.activation_degree = 0.0
        rule.trigger(None)
        expected = False
        self.assertEqual(expected, rule.triggered)

        rule.enabled = False
        rule.activation_degree = 1.0
        rule.trigger(None)
        expected = False
        self.assertEqual(expected, rule.triggered)

        rule.enabled = True
        rule.activation_degree = 0.0
        rule.trigger(None)
        expected = False
        self.assertEqual(expected, rule.triggered)

        rule.enabled = True
        rule.activation_degree = 1.0
        rule.trigger(None)
        expected = True
        self.assertEqual(expected, rule.triggered)

    def test_is_loaded(self) -> None:
        rule = fl.Rule()
        cases = {(True, True): True,
                 (True, False): False,
                 (False, True): False,
                 (False, False): False}
        for premise, expected in cases.items():
            rule.antecedent.is_loaded = MagicMock(return_value=premise[0])  # type: ignore
            rule.consequent.is_loaded = MagicMock(return_value=premise[1])  # type: ignore
            self.assertEqual(expected, rule.is_loaded())

    def test_unload(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)
        rule = fl.Rule.create("if Ambient is DARK then Power is HIGH", engine)
        self.assertTrue(rule.is_loaded())

        rule.unload()
        self.assertEqual(False, rule.triggered)
        self.assertEqual(0.0, rule.activation_degree)
        self.assertFalse(rule.antecedent.is_loaded())
        self.assertFalse(rule.consequent.is_loaded())

    def test_load(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)
        rule = fl.Rule.create("if Ambient is DARK then Power is HIGH")
        self.assertFalse(rule.is_loaded())

        rule.load(engine)
        self.assertEqual(False, rule.triggered)
        self.assertEqual(0.0, rule.activation_degree)
        self.assertTrue(rule.antecedent.is_loaded())
        self.assertTrue(rule.consequent.is_loaded())


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
                                           rules=[fl.Rule.create("if a then z"),
                                                  fl.Rule.create("if b then y")],
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

    def test_activate(self) -> None:
        activation = fl.General()
        activation.activate = MagicMock()  # type: ignore

        rb = fl.RuleBlock(activation=activation)

        rb.activate()
        activation.activate.assert_called_once_with(rb)  # type: ignore

        rb.activation = None
        with self.assertRaisesRegex(ValueError, "expected an activation method, but found none"):
            rb.activate()

    def test_unload_rules(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)
        rb = fl.RuleBlock(
            rules=[
                fl.Rule.create("if Ambient is DARK then Power is HIGH", engine),
                fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM", engine),
                fl.Rule.create("if Ambient is BRIGHT then Power is LOW", engine)
            ]
        )

        self.assertTrue(3, len(rb.rules))
        for rule in rb.rules:
            self.assertTrue(rule.is_loaded())

        rb.unload_rules()
        self.assertTrue(3, len(rb.rules))
        for rule in rb.rules:
            self.assertFalse(rule.is_loaded())

        rb.load_rules(engine)
        self.assertTrue(3, len(rb.rules))
        for rule in rb.rules:
            self.assertTrue(rule.is_loaded())

    def test_load_rules(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        rule1 = fl.Rule.create("if X then Y", engine=None)
        self.assertFalse(rule1.is_loaded())

        rule2 = fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM", engine=engine)
        self.assertTrue(rule2.is_loaded())

        rule3 = fl.Rule.create("if Ambient is BRIGHT then Power is Invalid", engine=None)
        self.assertFalse(rule3.is_loaded())

        rb = fl.RuleBlock(rules=[rule1, rule2, rule3])
        expected = ["failed to load the following rules:",
                    "['rule: if X then Y']: expected variable or logical operator, but found 'X'",
                    "['rule: if Ambient is BRIGHT then Power is Invalid']: consequent expected a "
                    "hedge or term, but found 'Invalid'"]
        try:
            rb.load_rules(engine)
            self.assertTrue(False)
        except RuntimeError as ex:
            self.assertEqual("\n".join(expected), str(ex))

    def test_reload_rules(self) -> None:
        engine = fl.FllImporter().from_string(SimpleDimmer)

        rule = fl.Rule.create("if Ambient is DARK then Power is HIGH", engine)
        rule.load = MagicMock()  # type: ignore
        rule.unload = MagicMock()  # type: ignore

        rb = fl.RuleBlock(rules=[rule])

        rb.reload_rules(engine)
        rule.unload.assert_called()  # type: ignore
        rule.load.assert_called_once_with(engine)  # type: ignore


if __name__ == '__main__':
    unittest.main()
