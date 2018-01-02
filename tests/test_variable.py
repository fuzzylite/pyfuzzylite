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


class VariableAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Variable):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def has_name(self, name: str):
        self.test.assertEqual(self.actual.name, name)
        return self

    def has_description(self, description: str):
        self.test.assertEqual(self.actual.description, description)
        return self

    def exports_fll(self, fll: str):
        self.test.assertEqual(FllExporter().variable(self.actual), fll)
        return self

    def fuzzy_values(self, fuzzification: Dict[float, str]):
        for x in fuzzification:
            self.test.assertEqual(self.actual.fuzzify(x), fuzzification[x], f"when x={x}")
        return self

    def highest_memberships(self, x_mf: Dict[float, Tuple[float, Term]]):
        for x in x_mf:
            self.test.assertEqual(self.actual.highest_membership(x), x_mf[x], f"when x={x}")
        return self


class TestVariable(unittest.TestCase):
    def test_constructor(self):
        VariableAssert(self, Variable("name", "description")) \
            .exports_fll("\n".join([
            "Variable: name",
            "  description: description",
            "  enabled: true",
            "  range: -inf inf",
            "  lock-range: true"
        ]))
        VariableAssert(self, Variable("name", "description", -1, 1,
                                      [Triangle('A', -1, 1), Triangle('B', -10, 10)])) \
            .exports_fll("\n".join([
            "Variable: name",
            "  description: description",
            "  enabled: true",
            "  range: -1.000 1.000",
            "  lock-range: true",
            "  term: A Triangle -1.000 0.000 1.000",
            "  term: B Triangle -10.000 0.000 10.000",
        ]))

    def test_lock_range(self):
        variable = Variable("name", "description")
        variable.range = (-1.0, 1.0)

        variable.lock_range = False
        variable.value = -10.0
        self.assertEqual(variable.value, -10.0)
        variable.value = 10.0
        self.assertEqual(variable.value, 10.0)

        minimum, maximum = variable.range
        self.assertEqual(minimum, -1.0)
        self.assertEqual(maximum, 1.0)

        variable.lock_range = True
        variable.value = -10.0
        self.assertEqual(variable.value, minimum)
        variable.value = 10.0
        self.assertEqual(variable.value, maximum)

    def test_fuzzify(self):
        VariableAssert(self, Variable("name", "description", -1.0, 1.0,
                                      [Triangle('Low', -1, -1, 0),
                                       Triangle('Medium', -0.5, 0, 0.5),
                                       Triangle('High', 0, 1, 1)])) \
            .fuzzy_values(
            {-1.00: "1.000/Low + 0.000/Medium + 0.000/High",
             -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
             -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
             0.00: "0.000/Low + 1.000/Medium + 0.000/High",
             0.25: "0.000/Low + 0.500/Medium + 0.250/High",
             0.50: "0.000/Low + 0.000/Medium + 0.500/High",
             0.75: "0.000/Low + 0.000/Medium + 0.750/High",
             1.00: "0.000/Low + 0.000/Medium + 1.000/High",
             nan: "nan/Low + nan/Medium + nan/High",
             inf: "0.000/Low + 0.000/Medium + 0.000/High",
             -inf: "0.000/Low + 0.000/Medium + 0.000/High",
             })

    def test_highest_membership(self):
        low, medium, high = (Triangle('Low', -1, -.5, 0),
                             Triangle('Medium', -0.5, 0, 0.5),
                             Triangle('High', 0, .5, 1))
        VariableAssert(self, Variable("name", "description", -1.0, 1.0, [low, medium, high])) \
            .highest_memberships(
            {-1.00: (0.0, None),
             -0.75: (0.5, low),
             -0.50: (1.0, low),
             -0.25: (0.5, low),
             0.00: (1.0, medium),
             0.25: (0.5, medium),
             0.50: (1.0, high),
             0.75: (0.5, high),
             1.00: (0.0, None),
             nan: (0.0, None),
             inf: (0.0, None),
             -inf: (0.0, None),
             })


class InputVariableAssert(VariableAssert):
    def __init__(self, test: unittest.TestCase, actual: InputVariable):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def exports_fll(self, fll: str):
        self.test.assertEqual(FllExporter().input_variable(self.actual), fll)
        return self

    def fuzzy_values(self, fuzzification: Dict[float, str]):
        for x in fuzzification:
            self.actual.value = x
            self.test.assertEqual(self.actual.fuzzy_value(), fuzzification[x], f"when x={x}")
        return self


class TestInputVariable(unittest.TestCase):
    def test_constructor(self):
        InputVariableAssert(self, InputVariable("name", "description")) \
            .exports_fll("\n".join([
            "InputVariable: name",
            "  description: description",
            "  enabled: true",
            "  range: -inf inf",
            "  lock-range: true"
        ]))
        InputVariableAssert(self, InputVariable("name", "description", -1, 1,
                                                [Triangle('A', -1, 1), Triangle('B', -10, 10)])) \
            .exports_fll("\n".join([
            "InputVariable: name",
            "  description: description",
            "  enabled: true",
            "  range: -1.000 1.000",
            "  lock-range: true",
            "  term: A Triangle -1.000 0.000 1.000",
            "  term: B Triangle -10.000 0.000 10.000",
        ]))

    def test_fuzzy_value(self):
        InputVariableAssert(self, InputVariable("name", "description", -1.0, 1.0,
                                                [Triangle('Low', -1, -1, 0),
                                                 Triangle('Medium', -0.5, 0, 0.5),
                                                 Triangle('High', 0, 1, 1)])) \
            .fuzzy_values(
            {-1.00: "1.000/Low + 0.000/Medium + 0.000/High",
             -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
             -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
             0.00: "0.000/Low + 1.000/Medium + 0.000/High",
             0.25: "0.000/Low + 0.500/Medium + 0.250/High",
             0.50: "0.000/Low + 0.000/Medium + 0.500/High",
             0.75: "0.000/Low + 0.000/Medium + 0.750/High",
             1.00: "0.000/Low + 0.000/Medium + 1.000/High",
             nan: "nan/Low + nan/Medium + nan/High",
             inf: "0.000/Low + 0.000/Medium + 0.000/High",
             -inf: "0.000/Low + 0.000/Medium + 0.000/High",
             })


class OutputVariableAssert(VariableAssert):
    def __init__(self, test: unittest.TestCase, actual: OutputVariable):
        self.test = test
        self.actual = actual
        self.test.maxDiff = None  # show all differences

    def exports_fll(self, fll: str):
        self.test.assertEqual(FllExporter().output_variable(self.actual), fll)
        return self

    def fuzzy_values(self, fuzzification: Dict[Tuple[Activated], str]):
        for x in fuzzification:
            self.actual.fuzzy.terms.clear()
            self.actual.fuzzy.terms.extend(x)
            self.test.assertEqual(self.actual.fuzzy_value(), fuzzification[x], f"when x={x}")
        return self


class TestOutputVariable(unittest.TestCase):
    def test_constructor(self):
        OutputVariableAssert(self, OutputVariable("name", "description")) \
            .exports_fll("\n".join([
            "OutputVariable: name",
            "  description: description",
            "  enabled: true",
            "  range: -inf inf",
            "  lock-range: true",
            "  aggregation: none",
            "  defuzzifier: none",
            "  default: nan",
            "  lock-previous: false"
        ]))
        OutputVariableAssert(self, OutputVariable("name", "description", -1, 1,
                                                  [Triangle('A', -1, 1), Triangle('B', -10, 10)])) \
            .exports_fll("\n".join([
            "OutputVariable: name",
            "  description: description",
            "  enabled: true",
            "  range: -1.000 1.000",
            "  lock-range: true",
            "  aggregation: none",
            "  defuzzifier: none",
            "  default: nan",
            "  lock-previous: false",
            "  term: A Triangle -1.000 0.000 1.000",
            "  term: B Triangle -10.000 0.000 10.000",
        ]))

    def test_fuzzy_value(self):
        low, medium, high = [Triangle('Low', -1, -1, 0),
                             Triangle('Medium', -0.5, 0, 0.5),
                             Triangle('High', 0, 1, 1)]
        OutputVariableAssert(self, OutputVariable("name", "description", -1.0, 1.0, [low, medium, high])) \
            .fuzzy_values({tuple(): "0.000/Low + 0.000/Medium + 0.000/High",
                           tuple([Activated(low, 0.5)]): "0.500/Low + 0.000/Medium + 0.000/High",
                           tuple([Activated(low, -1.0), Activated(medium, -0.5),
                                  Activated(high, -0.1)]): "-1.000/Low - 0.500/Medium - 0.100/High"})

    def test_clear(self):
        low, medium, high = [Triangle('Low', -1, -1, 0),
                             Triangle('Medium', -0.5, 0, 0.5),
                             Triangle('High', 0, 1, 1)]
        variable = OutputVariable("name", "description", -1.0, 1.0, [low, medium, high])
        variable.value = 0.0
        variable.previous_value = -1.0
        variable.fuzzy.terms.extend([Activated(term, 0.5) for term in variable.terms])
        OutputVariableAssert(self, variable).exports_fll("\n".join([
            "OutputVariable: name",
            "  description: description",
            "  enabled: true",
            "  range: -1.000 1.000",
            "  lock-range: true",
            "  aggregation: none",
            "  defuzzifier: none",
            "  default: nan",
            "  lock-previous: false",
            "  term: Low Triangle -1.000 -1.000 0.000",
            "  term: Medium Triangle -0.500 0.000 0.500",
            "  term: High Triangle 0.000 1.000 1.000",
        ]))

        self.assertEqual(variable.value, 0.0)
        self.assertEqual(variable.previous_value, -1.0)
        self.assertSequenceEqual([str(term) for term in variable.fuzzy.terms],
                                 ["(0.500*Low)", "(0.500*Medium)", "(0.500*High)"])
        variable.clear()
        self.assertEqual(isnan(variable.value), True)
        self.assertEqual(isnan(variable.previous_value), True)
        self.assertSequenceEqual(variable.fuzzy.terms, [])

    def test_defuzzify_invalid(self):
        low, medium, high = [Triangle('Low', -1, -1, 0),
                             Triangle('Medium', -0.5, 0, 0.5),
                             Triangle('High', 0, 1, 1)]
        variable = OutputVariable("name", "description", -1.0, 1.0, [low, medium, high])
        variable.default_value = 0.123
        variable.enabled = False
        variable.value = 0.0
        variable.previous_value = nan

        # test defuzzification on disabled variable changes nothing
        variable.defuzzify()
        self.assertEqual(variable.enabled, False)
        self.assertEqual(variable.value, 0.0)
        self.assertEqual(isnan(variable.previous_value), True)

        # tests default value is set for an invalid defuzzification
        variable.enabled = True
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.0)
        self.assertEqual(variable.value, 0.123)

        variable.default_value = 0.246
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.123)
        self.assertEqual(variable.value, 0.246)

        # tests locking of previous value
        variable.default_value = 0.123
        variable.lock_previous_value = True
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.246)
        self.assertEqual(variable.value, 0.246)

        variable.previous_value = 0.1
        variable.value = nan
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.1)
        self.assertEqual(variable.value, 0.1)

        #tests exception on defuzzification
        variable.fuzzy.terms.extend([Activated(term) for term in variable.terms])
        variable.lock_previous_value = False
        variable.value = 0.4
        variable.default_value = 0.5
        with self.assertRaisesRegex(ValueError, "expected a defuzzifier in output variable name, but found none"):
            variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.4)
        self.assertEqual(variable.value, 0.5)

        defuzzifier = Defuzzifier()
        from unittest.mock import MagicMock
        defuzzifier.defuzzify = MagicMock(side_effect=ValueError("mocking exception during defuzzification"))
        variable.defuzzifier = defuzzifier
        variable.default_value = 0.6
        with self.assertRaisesRegex(ValueError, "mocking exception during defuzzification"):
            variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.5)
        self.assertEqual(variable.value, 0.6)





if __name__ == '__main__':
    unittest.main()
