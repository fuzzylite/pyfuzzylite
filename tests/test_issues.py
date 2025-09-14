import unittest

import numpy as np

import fuzzylite as fl


class TestIssue91(unittest.TestCase):
    """Test case for https://github.com/fuzzylite/pyfuzzylite/issues/91"""

    def engine(self) -> fl.Engine:
        fll = """
Engine: tipper
description: (service and food) -> (tip)
InputVariable: service
  description: quality of service
  enabled: true
  range: 0.000 10.000
  lock-range: true
  term: poor Discrete 0.000 1.000 2.500 1.000 5.000 0.000 10.000 0.000
  term: good Discrete 0.000 0.000 2.500 0.000 5.000 1.000 7.500 0.000 10.000 0.000
  term: excellent Discrete 0.000 0.000 5.000 0.000 7.500 1.000 10.000 1.000
InputVariable: food
  description: quality of food
  enabled: true
  range: 0.000 10.000
  lock-range: true
  term: rancid Discrete 0.000 1.000 2.500 1.000 7.500 0.000 10.000 0.000
  term: delicious Discrete 0.000 0.000 2.500 0.000 7.500 1.000 10.000 1.000
OutputVariable: mTip
  description: tip based on Mamdani inference
  enabled: true
  range: 0.000 30.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 100
  default: 0.0
  lock-previous: false
  term: cheap Discrete 0.000 0.000 5.000 1.000 10.000 0.000 30.000 0.000
  term: average Discrete 0.000 0.000 10.000 0.000 15.000 1.000 20.000 0.000 30.000 0.000
  term: generous Discrete 0.000 0.000 20.000 0.000 25.000 1.000 30.000 0.000
OutputVariable: tsTip
  description: tip based on Takagi-Sugeno inference
  enabled: true
  range: 0.000 30.000
  lock-range: false
  aggregation: none
  defuzzifier: WeightedAverage TakagiSugeno
  default: 0.0
  lock-previous: false
  term: cheap Constant 5.000
  term: average Constant 15.000
  term: generous Constant 25.000
RuleBlock: mamdani
  description: Mamdani inference
  enabled: true
  conjunction: AlgebraicProduct
  disjunction: AlgebraicSum
  implication: Minimum
  activation: General
  rule: if service is poor or food is rancid then mTip is cheap
  rule: if service is good then mTip is average
  rule: if service is excellent or food is delicious then mTip is generous with 0.5
  rule: if service is excellent and food is delicious then mTip is generous with 1.0
RuleBlock: takagiSugeno
  description: Takagi-Sugeno inference
  enabled: true
  conjunction: AlgebraicProduct
  disjunction: AlgebraicSum
  implication: none
  activation: General
  rule: if service is poor or food is rancid then tsTip is cheap
  rule: if service is good then tsTip is average
  rule: if service is excellent or food is delicious then tsTip is generous with 0.5
  rule: if service is excellent and food is delicious then tsTip is generous with 1.0
        """
        return fl.FllImporter().from_string(fll)

    def test_issue_91(self) -> None:
        # "'numpy.float64' object does not support item assignment"
        engine = self.engine()
        engine.variable("food").value = 10
        engine.variable("service").value = 10
        engine.process()
        np.testing.assert_almost_equal(engine.variable("mTip").value, 25.00104979)
        np.testing.assert_almost_equal(engine.variable("tsTip").value, 25)

    def test_issue_92(self) -> None:
        class Approximation:
            def __init__(self) -> None:
                self.engine = fl.Engine(
                    name="approximation",
                    input_variables=[
                        fl.InputVariable(
                            name="inputX",
                            minimum=0.0,
                            maximum=10.0,
                            lock_range=False,
                            terms=[
                                fl.Triangle("NEAR_1", 0.0, 1.0, 2.0),
                                fl.Triangle("NEAR_2", 1.0, 2.0, 3.0),
                                fl.Triangle("NEAR_3", 2.0, 3.0, 4.0),
                                fl.Triangle("NEAR_4", 3.0, 4.0, 5.0),
                                fl.Triangle("NEAR_5", 4.0, 5.0, 6.0),
                                fl.Triangle("NEAR_6", 5.0, 6.0, 7.0),
                                fl.Triangle("NEAR_7", 6.0, 7.0, 8.0),
                                fl.Triangle("NEAR_8", 7.0, 8.0, 9.0),
                                fl.Triangle("NEAR_9", 8.0, 9.0, 10.0),
                            ],
                        )
                    ],
                    output_variables=[
                        fl.OutputVariable(
                            name="outputFx",
                            minimum=-1.0,
                            maximum=1.0,
                            lock_range=False,
                            lock_previous=True,
                            default_value=fl.nan,
                            aggregation=None,
                            defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                            terms=[
                                fl.Constant("f1", 0.84),
                                fl.Constant("f2", 0.45),
                                fl.Constant("f3", 0.04),
                                fl.Constant("f4", -0.18),
                                fl.Constant("f5", -0.19),
                                fl.Constant("f6", -0.04),
                                fl.Constant("f7", 0.09),
                                fl.Constant("f8", 0.12),
                                fl.Constant("f9", 0.04),
                            ],
                        ),
                        fl.OutputVariable(
                            name="trueFx",
                            minimum=-1.0,
                            maximum=1.0,
                            lock_range=False,
                            lock_previous=True,
                            default_value=fl.nan,
                            aggregation=None,
                            defuzzifier=fl.WeightedAverage(),
                            terms=[fl.Function("fx", "sin(inputX)/inputX")],
                        ),
                        fl.OutputVariable(
                            name="diffFx",
                            minimum=-1.0,
                            maximum=1.0,
                            lock_range=False,
                            lock_previous=False,
                            default_value=fl.nan,
                            aggregation=None,
                            defuzzifier=fl.WeightedAverage(),
                            terms=[fl.Function("diff", "fabs(outputFx-trueFx)")],
                        ),
                    ],
                    rule_blocks=[
                        fl.RuleBlock(
                            name="",
                            conjunction=None,
                            disjunction=None,
                            implication=None,
                            activation=fl.General(),
                            rules=[
                                fl.Rule.create("if inputX is NEAR_1 then outputFx is f1"),
                                fl.Rule.create("if inputX is NEAR_2 then outputFx is f2"),
                                fl.Rule.create("if inputX is NEAR_3 then outputFx is f3"),
                                fl.Rule.create("if inputX is NEAR_4 then outputFx is f4"),
                                fl.Rule.create("if inputX is NEAR_5 then outputFx is f5"),
                                fl.Rule.create("if inputX is NEAR_6 then outputFx is f6"),
                                fl.Rule.create("if inputX is NEAR_7 then outputFx is f7"),
                                fl.Rule.create("if inputX is NEAR_8 then outputFx is f8"),
                                fl.Rule.create("if inputX is NEAR_9 then outputFx is f9"),
                                fl.Rule.create(
                                    "if inputX is any then trueFx is fx and diffFx is diff"
                                ),
                            ],
                        )
                    ],
                )

        test = Approximation()
        test.engine.input_variable("inputX").value = 5.0
        test.engine.process()
