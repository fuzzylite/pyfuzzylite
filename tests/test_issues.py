import fuzzylite as fl
import unittest
import numpy as np


class TestIssue91(unittest.TestCase):
    """
    Test case for https://github.com/fuzzylite/pyfuzzylite/issues/91
    """

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
