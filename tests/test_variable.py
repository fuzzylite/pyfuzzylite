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

import math
import unittest
from typing import Dict, Optional, Sequence, Tuple

import fuzzylite as fl
from tests.assert_component import BaseAssert


class VariableAssert(BaseAssert[fl.Variable]):

    def fuzzy_values(self, fuzzification: Dict[float, str]) -> 'VariableAssert':
        for x in fuzzification:
            self.test.assertEqual(self.actual.fuzzify(x), fuzzification[x], f"when x={x}")
        return self

    def highest_memberships(self,
                            x_mf: Dict[float, Tuple[float, Optional[fl.Term]]]) -> 'VariableAssert':
        for x in x_mf:
            self.test.assertEqual(self.actual.highest_membership(x), x_mf[x], f"when x={x}")
        return self


class TestVariable(unittest.TestCase):

    def test_constructor(self) -> None:
        VariableAssert(self, fl.Variable("name", "description")) \
            .exports_fll(
            "\n".join([
                "Variable: name",
                "  description: description",
                "  enabled: true",
                "  range: -inf inf",
                "  lock-range: false"
            ]))
        VariableAssert(self,
                       fl.Variable(name="name",
                                   description="description",
                                   minimum=-1.0,
                                   maximum=1.0,
                                   terms=[
                                       fl.Triangle('A', -1.0, 1.0),
                                       fl.Triangle('B', -10.0, 10.0)
                                   ])) \
            .exports_fll(
            "\n".join([
                "Variable: name",
                "  description: description",
                "  enabled: true",
                "  range: -1.000 1.000",
                "  lock-range: false",
                "  term: A Triangle -1.000 0.000 1.000",
                "  term: B Triangle -10.000 0.000 10.000",
            ]))

    def test_lock_range(self) -> None:
        variable = fl.Variable("name", "description")
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

    def test_fuzzify(self) -> None:
        VariableAssert(self, fl.Variable(name="name",
                                         description="description",
                                         minimum=-1.0,
                                         maximum=1.0,
                                         terms=[
                                             fl.Triangle('Low', -1.0, -1.0, 0.0),
                                             fl.Triangle('Medium', -0.5, 0.0, 0.5),
                                             fl.Triangle('High', 0.0, 1.0, 1.0)
                                         ])) \
            .fuzzy_values(
            {-1.00: "1.000/Low + 0.000/Medium + 0.000/High",
             -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
             -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
             0.00: "0.000/Low + 1.000/Medium + 0.000/High",
             0.25: "0.000/Low + 0.500/Medium + 0.250/High",
             0.50: "0.000/Low + 0.000/Medium + 0.500/High",
             0.75: "0.000/Low + 0.000/Medium + 0.750/High",
             1.00: "0.000/Low + 0.000/Medium + 1.000/High",
             math.nan: "nan/Low + nan/Medium + nan/High",
             math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
             -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
             })

    def test_highest_membership(self) -> None:
        low, medium, high = (fl.Triangle('Low', -1.0, -.5, 0.0),
                             fl.Triangle('Medium', -0.5, 0.0, 0.5),
                             fl.Triangle('High', 0.0, .5, 1.0))
        VariableAssert(self, fl.Variable(name="name",
                                         description="description",
                                         minimum=-1.0,
                                         maximum=1.0,
                                         terms=[low, medium, high])) \
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
             math.nan: (0.0, None),
             math.inf: (0.0, None),
             -math.inf: (0.0, None),
             })


class InputVariableAssert(BaseAssert[fl.InputVariable]):

    def exports_fll(self, fll: str) -> 'InputVariableAssert':
        self.test.assertEqual(fl.FllExporter().input_variable(self.actual), fll)
        return self

    def fuzzy_values(self, fuzzification: Dict[float, str]) -> 'InputVariableAssert':
        for x in fuzzification:
            self.actual.value = x
            self.test.assertEqual(self.actual.fuzzy_value(), fuzzification[x], f"when x={x}")
        return self


class TestInputVariable(unittest.TestCase):

    def test_constructor(self) -> None:
        InputVariableAssert(self, fl.InputVariable("name", "description")) \
            .exports_fll("\n".join(["InputVariable: name",
                                    "  description: description",
                                    "  enabled: true",
                                    "  range: -inf inf",
                                    "  lock-range: false"
                                    ]))
        InputVariableAssert(self, fl.InputVariable(name="name",
                                                   description="description",
                                                   minimum=-1.0,
                                                   maximum=1.0,
                                                   terms=[
                                                       fl.Triangle('A', -1.0, 1.0),
                                                       fl.Triangle('B', -10.0, 10.0)
                                                   ])) \
            .exports_fll("\n".join(["InputVariable: name",
                                    "  description: description",
                                    "  enabled: true",
                                    "  range: -1.000 1.000",
                                    "  lock-range: false",
                                    "  term: A Triangle -1.000 0.000 1.000",
                                    "  term: B Triangle -10.000 0.000 10.000",
                                    ]))

    def test_fuzzy_value(self) -> None:
        InputVariableAssert(self, fl.InputVariable(name="name",
                                                   description="description",
                                                   minimum=-1.0,
                                                   maximum=1.0,
                                                   terms=[
                                                       fl.Triangle('Low', -1.0, -1.0, 0.0),
                                                       fl.Triangle('Medium', -0.5, 0.0, 0.5),
                                                       fl.Triangle('High', 0.0, 1.0, 1.0)
                                                   ])) \
            .fuzzy_values({-1.00: "1.000/Low + 0.000/Medium + 0.000/High",
                           -0.50: "0.500/Low + 0.000/Medium + 0.000/High",
                           -0.25: "0.250/Low + 0.500/Medium + 0.000/High",
                           0.00: "0.000/Low + 1.000/Medium + 0.000/High",
                           0.25: "0.000/Low + 0.500/Medium + 0.250/High",
                           0.50: "0.000/Low + 0.000/Medium + 0.500/High",
                           0.75: "0.000/Low + 0.000/Medium + 0.750/High",
                           1.00: "0.000/Low + 0.000/Medium + 1.000/High",
                           math.nan: "nan/Low + nan/Medium + nan/High",
                           math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                           -math.inf: "0.000/Low + 0.000/Medium + 0.000/High",
                           })


class OutputVariableAssert(BaseAssert[fl.OutputVariable]):

    def exports_fll(self, fll: str) -> 'OutputVariableAssert':
        self.test.assertEqual(fl.FllExporter().output_variable(self.actual), fll)
        return self

    def activated_values(self,
                         fuzzification: Dict[
                             Sequence[fl.Activated], str]) -> 'OutputVariableAssert':
        for x in fuzzification:
            self.actual.fuzzy.terms.clear()
            self.actual.fuzzy.terms.extend(x)
            self.test.assertEqual(self.actual.fuzzy_value(), fuzzification[x], f"when x={x}")
        return self


class TestOutputVariable(unittest.TestCase):

    def test_constructor(self) -> None:
        OutputVariableAssert(self, fl.OutputVariable("name", "description")) \
            .exports_fll("\n".join(["OutputVariable: name",
                                    "  description: description",
                                    "  enabled: true",
                                    "  range: -inf inf",
                                    "  lock-range: false",
                                    "  aggregation: none",
                                    "  defuzzifier: none",
                                    "  default: nan",
                                    "  lock-previous: false"
                                    ]))
        OutputVariableAssert(self, fl.OutputVariable(name="name",
                                                     description="description",
                                                     minimum=-1.0,
                                                     maximum=1.0,
                                                     terms=[
                                                         fl.Triangle('A', -1.0, 1.0),
                                                         fl.Triangle('B', -10.0, 10.0)
                                                     ])) \
            .exports_fll("\n".join(["OutputVariable: name",
                                    "  description: description",
                                    "  enabled: true",
                                    "  range: -1.000 1.000",
                                    "  lock-range: false",
                                    "  aggregation: none",
                                    "  defuzzifier: none",
                                    "  default: nan",
                                    "  lock-previous: false",
                                    "  term: A Triangle -1.000 0.000 1.000",
                                    "  term: B Triangle -10.000 0.000 10.000",
                                    ]))

    def test_fuzzy_value(self) -> None:
        low, medium, high = [fl.Triangle('Low', -1.0, -1.0, 0.0),
                             fl.Triangle('Medium', -0.5, 0.0, 0.5),
                             fl.Triangle('High', 0.0, 1.0, 1.0)]
        OutputVariableAssert(self,
                             fl.OutputVariable(name="name",
                                               description="description",
                                               minimum=-1.0,
                                               maximum=1.0,
                                               terms=[low, medium, high])) \
            .activated_values({tuple(): "0.000/Low + 0.000/Medium + 0.000/High",
                               tuple([fl.Activated(low, 0.5)]):
                                   "0.500/Low + 0.000/Medium + 0.000/High",
                               tuple([fl.Activated(low, -1.0),
                                      fl.Activated(medium, -0.5),
                                      fl.Activated(high, -0.1)]):
                                   "-1.000/Low - 0.500/Medium - 0.100/High"})

    def test_clear(self) -> None:
        low, medium, high = [fl.Triangle('Low', -1.0, -1.0, 0.0),
                             fl.Triangle('Medium', -0.5, 0.0, 0.5),
                             fl.Triangle('High', 0.0, 1.0, 1.0)]
        variable = fl.OutputVariable(name="name",
                                     description="description",
                                     minimum=-1.0,
                                     maximum=1.0,
                                     terms=[low, medium, high])
        variable.value = 0.0
        variable.previous_value = -1.0
        variable.fuzzy.terms.extend([fl.Activated(term, 0.5) for term in variable.terms])
        OutputVariableAssert(self, variable) \
            .exports_fll("\n".join(["OutputVariable: name",
                                    "  description: description",
                                    "  enabled: true",
                                    "  range: -1.000 1.000",
                                    "  lock-range: false",
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
        self.assertSequenceEqual([term.parameters() for term in variable.fuzzy.terms],
                                 ["(0.500*Low)", "(0.500*Medium)", "(0.500*High)"])
        variable.clear()
        self.assertEqual(math.isnan(variable.value), True)
        self.assertEqual(math.isnan(variable.previous_value), True)
        self.assertSequenceEqual(variable.fuzzy.terms, [])

    def test_defuzzify_invalid(self) -> None:
        low, medium, high = [fl.Triangle('Low', -1.0, -1.0, 0.0),
                             fl.Triangle('Medium', -0.5, 0.0, 0.5),
                             fl.Triangle('High', 0.0, 1.0, 1.0)]
        variable = fl.OutputVariable(name="name",
                                     description="description",
                                     minimum=-1.0,
                                     maximum=1.0,
                                     terms=[low, medium, high])
        variable.default_value = 0.123
        variable.enabled = False
        variable.value = 0.0
        variable.previous_value = math.nan

        # test defuzzification on disabled variable changes nothing
        variable.defuzzify()
        self.assertEqual(variable.enabled, False)
        self.assertEqual(variable.value, 0.0)
        self.assertEqual(math.isnan(variable.previous_value), True)

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
        variable.lock_previous = True
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.246)
        self.assertEqual(variable.value, 0.246)

        variable.previous_value = 0.1
        variable.value = math.nan
        variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.1)
        self.assertEqual(variable.value, 0.1)

        # tests exception on defuzzification
        variable.fuzzy.terms.extend([fl.Activated(term) for term in variable.terms])
        variable.lock_previous = False
        variable.value = 0.4
        variable.default_value = 0.5
        with self.assertRaisesRegex(ValueError, "expected a defuzzifier in output variable name, "
                                                "but found none"):
            variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.4)
        self.assertEqual(variable.value, 0.5)

        defuzzifier = fl.Defuzzifier()
        from unittest.mock import MagicMock
        setattr(defuzzifier, 'defuzzify', MagicMock(
            side_effect=ValueError("mocking exception during defuzzification")))
        # defuzzifier.defuzzify = MagicMock(
        #     side_effect=ValueError("mocking exception during defuzzification"))
        variable.defuzzifier = defuzzifier
        variable.default_value = 0.6
        with self.assertRaisesRegex(ValueError, "mocking exception during defuzzification"):
            variable.defuzzify()
        self.assertEqual(variable.previous_value, 0.5)
        self.assertEqual(variable.value, 0.6)


if __name__ == '__main__':
    unittest.main()
