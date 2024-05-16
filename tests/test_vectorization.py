"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

import fuzzylite as fl
from fuzzylite.examples import mamdani, takagi_sugeno, tsukamoto


class AssertIntegration:
    """Asserts integration of a Mamdani or Takagi-Sugeno system with a defuzzifier."""

    def __init__(self, engine: fl.Engine, vectorize: bool = True) -> None:
        """Constructor.

        @param engine is the engine to test on.
        @param vectorize is whether to test vectorization.
        """
        self.engine = copy.deepcopy(engine)
        self.vectorize = vectorize

    def assert_that(self, defuzzifier: fl.Defuzzifier, input_expected: dict[float, float]) -> None:
        """Asserts integration of a Mamdani or Takagi-Sugeno system with a defuzzifier."""
        for output in self.engine.output_variables:
            output.defuzzifier = defuzzifier

        self.engine.restart()
        for input, expected_output in input_expected.items():
            self.engine.input_variables[0].value = input
            self.engine.process()
            obtained = self.engine.output_variables[0].value
            np.testing.assert_allclose(
                obtained,
                expected_output,
                err_msg=f"{fl.Op.class_name(defuzzifier)}({input}) = {obtained}, but expected {expected_output}",
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
            )
        if self.vectorize:
            self.engine.restart()
            inputs = fl.array([x for x in input_expected])
            expected_outputs = fl.array([x for x in input_expected.values()])
            self.engine.input_variables[0].value = inputs
            self.engine.process()
            obtained = self.engine.output_variables[0].value
            np.testing.assert_allclose(
                obtained,
                expected_outputs,
                err_msg=f"{fl.Op.class_name(defuzzifier)}([{inputs}]) = {obtained}, but expected {expected_outputs}",
                atol=fl.settings.atol,
                rtol=fl.settings.rtol,
            )


class TestMamdani(unittest.TestCase):
    """Tests for Integral Defuzzifiers."""

    def test_simple_mamdani_bisector_integration(self) -> None:
        """Test a simple integration with Bisector."""
        AssertIntegration(mamdani.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.Bisector(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.625,
                0.5: 0.5,
                0.625: 0.375,
                0.675: 0.304,
                0.75: 0.25,
            },
        )

    def test_simple_mamdani_centroid_integration(self) -> None:
        """Test a simple integration with Centroid."""
        AssertIntegration(mamdani.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.Centroid(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.625,
                0.5: 0.5,
                0.625: 0.375,
                0.675: 0.334,
                0.75: 0.25,
            },
        )

    def test_simple_mamdani_lom_integration(self) -> None:
        """Test a simple integration with LargestOfMaximum."""
        AssertIntegration(mamdani.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.LargestOfMaximum(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.875,
                0.5: 0.5,
                0.625: 0.625,
                0.675: 0.325,
                0.75: 0.25,
            },
        )

    def test_simple_mamdani_mom_integration(self) -> None:
        """Test a simple integration with MeanOfMaximum."""
        AssertIntegration(mamdani.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.MeanOfMaximum(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.625,
                0.5: 0.5,
                0.625: 0.375,
                0.675: 0.25,
                0.75: 0.25,
            },
        )

    def test_simple_mamdani_som_integration(self) -> None:
        """Test a simple integration without vectorization."""
        AssertIntegration(mamdani.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.SmallestOfMaximum(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.375,
                0.5: 0.5,
                0.625: 0.125,
                0.675: 0.175,
                0.75: 0.25,
            },
        )


class TestWeightedDefuzzifier(unittest.TestCase):
    """Tests for Weighted defuzzifiers."""

    def test_simple_takagisugeno_avg_integration(self) -> None:
        """Test a simple integration with WeightedAverage."""
        AssertIntegration(takagi_sugeno.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.WeightedAverage(),
            {
                0.0: fl.nan,
                1.0: fl.nan,
                0.25: 0.75,
                0.375: 0.625,
                0.5: 0.5,
                0.625: 0.375,
                0.675: 0.325,
                0.75: 0.25,
            },
        )

    def test_simple_takagisugeno_sum_integration(self) -> None:
        """Test a simple integration with WeightedSum."""
        AssertIntegration(takagi_sugeno.simple_dimmer.SimpleDimmer().engine).assert_that(
            fl.WeightedSum(),
            {
                0.0: np.nan,
                1.0: np.nan,
                0.25: 0.75,
                0.375: 0.625,
                0.5: 0.5,
                0.625: 0.375,
                0.675: 0.325,
                0.75: 0.25,
            },
        )

    def test_simple_tsukamoto_avg_integration(self) -> None:
        """Test a simple integration with WeightedAverage."""
        AssertIntegration(tsukamoto.tsukamoto.Tsukamoto().engine).assert_that(
            fl.WeightedAverage(),
            {
                -20: 0.014,
                -10.0: 0.255,
                -7.5: 0.272,
                -5: 0.313,
                -2.5: 0.375,
                -1: 0.392,
                0: 0.399,
                1: 0.405,
                2.5: 0.426,
                5: 0.674,
                7.5: 0.964,
                10: 0.994,
                20: 0.702,
                -np.inf: np.nan,
                np.inf: np.nan,
                np.nan: np.nan,
            },
        )

    def test_simple_tsukamoto_sum_integration(self) -> None:
        """Test a simple integration with WeightedSum."""
        AssertIntegration(tsukamoto.tsukamoto.Tsukamoto().engine).assert_that(
            fl.WeightedSum(),
            {
                -20: 0.0,
                -10.0: 0.259,
                -7.5: 0.290,
                -5: 0.313,
                -2.5: 0.401,
                -1: 0.406,
                0: 0.411,
                1: 0.420,
                2.5: 0.455,
                5: 0.675,
                7.5: 1.027,
                10: 1.009,
                20: 0.011,
                -np.inf: np.nan,
                np.inf: np.nan,
                np.nan: np.nan,
            },
        )


if __name__ == "__main__":
    unittest.main()
