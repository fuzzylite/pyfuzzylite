import unittest

import numpy as np

import fuzzylite as fl


class TestVectorization(unittest.TestCase):
    """Tests for vectorization."""

    def test_simple_mamdani_integration(self) -> None:
        """Test a simple integration without vectorization."""
        from fuzzylite.examples.mamdani.SimpleDimmer import engine

        expected = {
            0.0: fl.nan,
            1.0: fl.nan,
            0.25: 0.75,
            0.5: 0.5,
            0.75: 0.25,
        }
        for input, output in expected.items():
            engine.input_variables[0].value = input
            engine.process()
            np.testing.assert_almost_equal(engine.output_variables[0].value, output)

    def test_simple_mamdani_vectorization(self) -> None:
        """Test a simple integration with vectorization."""
        from fuzzylite.examples.mamdani.SimpleDimmer import engine

        expected = {
            0.0: fl.nan,
            1.0: fl.nan,
            0.25: 0.75,
            0.5: 0.5,
            0.75: 0.25,
        }
        engine.input_variables[0].value = np.array([x for x in expected])
        engine.process()
        output = engine.output_variables[0].value
        np.testing.assert_almost_equal(np.array([x for x in expected.values()]), output)

    def test_simple_takagisugeno_integration(self) -> None:
        """Test a simple integration without vectorization."""
        from fuzzylite.examples.takagi_sugeno.SimpleDimmer import engine

        expected = {
            0.0: fl.nan,
            1.0: fl.nan,
            0.25: 0.75,
            0.5: 0.5,
            0.75: 0.25,
        }
        for input, output in expected.items():
            engine.input_variables[0].value = input
            engine.process()
            if np.isnan(output):
                self.assertTrue(np.isnan(engine.output_variables[0].value))
            else:
                self.assertAlmostEqual(engine.output_variables[0].value, output)

    def test_simple_takagisugeno_vectorization(self) -> None:
        """Test a simple integration with vectorization."""
        from fuzzylite.examples.takagi_sugeno.SimpleDimmer import engine

        expected = {
            0.0: fl.nan,
            1.0: fl.nan,
            0.25: 0.75,
            0.5: 0.5,
            0.75: 0.25,
        }
        engine.input_variables[0].value = np.array([x for x in expected])
        engine.process()
        output = engine.output_variables[0].value
        np.testing.assert_almost_equal(np.array([x for x in expected.values()]), output)
