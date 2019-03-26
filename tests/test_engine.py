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
from typing import List, Optional

import fuzzylite as fl
from tests.assert_component import BaseAssert


class EngineAssert(BaseAssert[fl.Engine]):

    def has_type(self, expected: fl.Engine.Type) -> 'EngineAssert':
        type = self.actual.infer_type()
        self.test.assertEqual(type, expected,
                              f"expected engine of type {expected}, but found {type}")
        return self

    def is_ready(self, expected: bool, status: str = "") -> 'EngineAssert':
        ready, message = self.actual.is_ready()
        self.test.assertEqual(ready, expected,
                              (f"expected engine {'*not*' if not expected else ''} to be ready,"
                               f"but was {'*not*' if not ready else ''} ready"))
        self.test.assertEqual(message, status)
        return self

    def has_n_inputs(self, n: int) -> 'EngineAssert':
        n_inputs = len(self.actual.input_variables)
        self.test.assertEqual(n_inputs, n, f"expected {n} input variables, but found {n_inputs}")
        return self

    def has_inputs(self, names: List[str]) -> 'EngineAssert':
        self.test.assertSequenceEqual([iv.name for iv in self.actual.input_variables], names)
        return self

    def has_n_outputs(self, n: int) -> 'EngineAssert':
        n_outputs = len(self.actual.output_variables)
        self.test.assertEqual(n_outputs, n, f"expected {n} output variables, but found {n_outputs}")
        return self

    def has_outputs(self, names: List[str]) -> 'EngineAssert':
        self.test.assertSequenceEqual([ov.name for ov in self.actual.output_variables], names)
        return self

    def has_n_blocks(self, n: int) -> 'EngineAssert':
        n_blocks = len(self.actual.rule_blocks)
        self.test.assertEqual(n_blocks, n, f"expected {n} rule blocks, but found {n_blocks}")
        return self

    def has_blocks(self, names: List[str]) -> 'EngineAssert':
        self.test.assertSequenceEqual([rb.name for rb in self.actual.rule_blocks], names)
        return self

    def evaluate_fld(self, fld: str, decimals: Optional[int] = None) -> 'EngineAssert':
        if decimals is None:
            decimals = fl.lib.decimals
        for line, evaluation in enumerate(fld.split("\n")):
            comment_index = evaluation.find("#")
            if comment_index != -1:
                evaluation = evaluation[:comment_index]
            if not evaluation:
                continue

            expected = evaluation.split()
            if len(expected) != len(self.actual.variables):
                raise ValueError(f"expected {len(self.actual.variables)} values, "
                                 f"but got {len(expected)}: [line: {line}] {evaluation}")

            obtained: List[str] = []
            for i, input_variable in enumerate(self.actual.input_variables):
                input_variable.value = fl.scalar(expected[i])
                obtained.append(expected[i])

            self.actual.process()

            obtained.extend(fl.Op.str(ov.value, decimals) for ov in self.actual.output_variables)

            self.test.assertListEqual(expected, obtained, msg=f"in evaluation line {line}")
        return self


class TestEngine(unittest.TestCase):

    def test_empty_engine(self) -> None:
        flc = fl.Engine("name", "description")
        EngineAssert(self, flc) \
            .has_name("name").has_description("description") \
            .has_n_inputs(0).has_inputs([]) \
            .has_n_outputs(0).has_outputs([]) \
            .has_n_blocks(0).has_blocks([])

    def test_engine(self) -> None:
        engine = fl.Engine()
        engine.name = "tipper"
        engine.description = "(service and food) -> (tip)"

        service = fl.InputVariable()
        service.name = "service"
        service.description = "quality of service"
        service.enabled = True
        service.range = (0.000, 10.000)
        service.lock_range = True
        service.terms.append(fl.Trapezoid("poor", 0.000, 0.000, 2.500, 5.000))
        service.terms.append(fl.Triangle("good", 2.500, 5.000, 7.500))
        service.terms.append(fl.Trapezoid("excellent", 5.000, 7.500, 10.000, 10.000))
        engine.input_variables.append(service)

        food = fl.InputVariable()
        food.name = "food"
        food.description = "quality of food"
        food.enabled = True
        food.range = (0.000, 10.000)
        food.lock_range = True
        food.terms.append(fl.Trapezoid("rancid", 0.000, 0.000, 2.500, 7.500))
        food.terms.append(fl.Trapezoid("delicious", 2.500, 7.500, 10.000, 10.000))
        engine.input_variables.append(food)

        mTip = fl.OutputVariable()  # noqa N806 should be lowercase
        mTip.name = "mTip"
        mTip.description = "tip based on Mamdani inference"
        mTip.enabled = True
        mTip.range = (0.000, 30.000)
        mTip.lock_range = False
        mTip.aggregation = fl.Maximum()
        mTip.defuzzifier = fl.Centroid(100)
        mTip.default_value = fl.nan
        mTip.lock_previous = False
        mTip.terms.append(fl.Triangle("cheap", 0.000, 5.000, 10.000))
        mTip.terms.append(fl.Triangle("average", 10.000, 15.000, 20.000))
        mTip.terms.append(fl.Triangle("generous", 20.000, 25.000, 30.000))
        engine.output_variables.append(mTip)

        tsTip = fl.OutputVariable()  # noqa N806 should be lowercase
        tsTip.name = "tsTip"
        tsTip.description = "tip based on Takagi-Sugeno inference"
        tsTip.enabled = True
        tsTip.range = (0.000, 30.000)
        tsTip.lock_range = False
        tsTip.aggregation = None
        tsTip.defuzzifier = fl.WeightedAverage("TakagiSugeno")
        tsTip.default_value = fl.nan
        tsTip.lock_previous = False
        tsTip.terms.append(fl.Constant("cheap", 5.000))
        tsTip.terms.append(fl.Constant("average", 15.000))
        tsTip.terms.append(fl.Constant("generous", 25.000))
        engine.output_variables.append(tsTip)

        mamdani = fl.RuleBlock()
        mamdani.name = "mamdani"
        mamdani.description = "Mamdani inference"
        mamdani.enabled = True
        mamdani.conjunction = fl.AlgebraicProduct()
        mamdani.disjunction = fl.AlgebraicSum()
        mamdani.implication = fl.Minimum()
        mamdani.activation = fl.General()
        mamdani.rules.append(
            fl.Rule.create("if service is poor or food is rancid then mTip is cheap", engine))
        mamdani.rules.append(fl.Rule.create("if service is good then mTip is average", engine))
        mamdani.rules.append(fl.Rule.create(
            "if service is excellent or food is delicious then mTip is generous with 0.5", engine))
        mamdani.rules.append(fl.Rule.create(
            "if service is excellent and food is delicious then mTip is generous with 1.0", engine))
        engine.rule_blocks.append(mamdani)

        takagiSugeno = fl.RuleBlock()  # noqa N806 should be lowercase
        takagiSugeno.name = "takagiSugeno"
        takagiSugeno.description = "Takagi-Sugeno inference"
        takagiSugeno.enabled = True
        takagiSugeno.conjunction = fl.AlgebraicProduct()
        takagiSugeno.disjunction = fl.AlgebraicSum()
        takagiSugeno.implication = None
        takagiSugeno.activation = fl.General()
        takagiSugeno.rules.append(fl.Rule.create(
            "if service is poor or food is rancid then tsTip is cheap", engine))
        takagiSugeno.rules.append(fl.Rule.create(
            "if service is good then tsTip is average", engine))
        takagiSugeno.rules.append(fl.Rule.create(
            "if service is excellent or food is delicious then tsTip is generous with 0.5", engine))
        takagiSugeno.rules.append(fl.Rule.create(
            "if service is excellent and food is delicious then tsTip is generous with 1.0",
            engine))
        engine.rule_blocks.append(takagiSugeno)

        EngineAssert(self, engine) \
            .has_name("tipper") \
            .has_description("(service and food) -> (tip)") \
            .has_n_inputs(2).has_inputs(["service", "food"]) \
            .has_n_outputs(2).has_outputs(["mTip", "tsTip"]) \
            .has_n_blocks(2).has_blocks(["mamdani", "takagiSugeno"]) \
            .evaluate_fld(
            """\
#service food mTip tsTip
0.0000000000000000 0.0000000000000000 4.9989502099580099 5.0000000000000000
0.0000000000000000 3.3333333333333335 7.7561896551724301 6.5384615384615392
0.0000000000000000 6.6666666666666670 12.9489036144578247 10.8823529411764728
0.0000000000000000 10.0000000000000000 13.5707062050051448 11.6666666666666661
3.3333333333333335 0.0000000000000000 8.5688247396168276 7.5000000000000000
3.3333333333333335 3.3333333333333335 10.1101355034654858 8.6734693877551035
3.3333333333333335 6.6666666666666670 13.7695060342408198 12.9245283018867916
3.3333333333333335 10.0000000000000000 14.3676481312670976 13.8888888888888911
6.6666666666666670 0.0000000000000000 12.8954528230390419 11.0000000000000000
6.6666666666666670 3.3333333333333335 13.2040624705105234 12.7966101694915260
6.6666666666666670 6.6666666666666670 17.9862390284958273 20.6363636363636367
6.6666666666666670 10.0000000000000000 21.1557340720221632 22.7777777777777821
10.0000000000000000 0.0000000000000000 13.5707062050051448 11.6666666666666661
10.0000000000000000 3.3333333333333335 13.7092196934510024 13.8888888888888875
10.0000000000000000 6.6666666666666670 20.2157800031293959 22.7777777777777821
10.0000000000000000 10.0000000000000000 25.0010497900419928 25.0000000000000000
""", decimals=16)

    def test_engine_from_fll(self) -> None:
        pass

    def test_inputs(self) -> None:
        flc = fl.Engine("name", "description",
                        [fl.InputVariable("A"), fl.InputVariable("B")])
        EngineAssert(self, flc) \
            .has_name("name").has_description("description") \
            .has_n_inputs(2).has_inputs(["A", "B"])

        flc.input_variables = []
        EngineAssert(self, flc).has_n_inputs(0).has_inputs([])

        flc.input_variables = [fl.InputVariable("X"), fl.InputVariable("Y"), fl.InputVariable("Z")]
        EngineAssert(self, flc).has_n_inputs(3).has_inputs(["X", "Y", "Z"])

        names = ["X", "Y", "Z"]
        for i, iv in enumerate(flc.input_variables):
            self.assertEqual(iv.name, names[i])

    def test_outputs(self) -> None:
        flc = fl.Engine("name", "description", [], [fl.OutputVariable("A"), fl.OutputVariable("B")])
        EngineAssert(self, flc) \
            .has_name("name").has_description("description") \
            .has_n_outputs(2).has_outputs(["A", "B"])

        flc.output_variables = []
        EngineAssert(self, flc).has_n_outputs(0).has_outputs([])

        flc.output_variables = [fl.OutputVariable("X"),
                                fl.OutputVariable("Y"),
                                fl.OutputVariable("Z")]
        EngineAssert(self, flc).has_n_outputs(3).has_outputs(["X", "Y", "Z"])

        names = ["X", "Y", "Z"]
        for i, iv in enumerate(flc.output_variables):
            self.assertEqual(iv.name, names[i])


if __name__ == '__main__':
    unittest.main()
