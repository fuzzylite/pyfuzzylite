"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np

import fuzzylite as fl
from fuzzylite import inf, nan
from tests.assert_component import BaseAssert


class HedgeAssert(BaseAssert[fl.Hedge]):
    """Assert hedge."""

    def evaluates(self, az: dict[float, float]) -> HedgeAssert:
        """Assert the hedge produces the expected values from the keys."""
        for a, z in az.items():
            np.testing.assert_equal(self.actual.hedge(a), z, err_msg=f"when x={a}")
        vector_a = fl.array(list(az.keys()))
        vector_z = fl.array(list(az.values()))
        np.testing.assert_equal(vector_z, self.actual.hedge(vector_a))
        return self


class TestHedge(unittest.TestCase):
    """Test hedges."""

    def test_any(self) -> None:
        """Test the Any hedge."""
        HedgeAssert(self, fl.Any()).has_name("any").evaluates(
            {
                -1.0: 1.0,
                -0.5: 1.0,
                0.00: 1.0,
                0.25: 1.0,
                0.50: 1.0,
                0.75: 1.0,
                1.00: 1.0,
                inf: 1.0,
                -inf: 1.0,
                nan: 1.0,
            }
        )

    def test_extremely(self) -> None:
        """Test the Extremely hedge."""
        HedgeAssert(self, fl.Extremely()).has_name("extremely").evaluates(
            {
                -1.0: 2.0,
                -0.5: 0.5,
                0.00: 0.0,
                0.25: 0.125,
                0.50: 0.5,
                0.75: 0.875,
                1.00: 1.0,
                inf: -inf,
                -inf: inf,
                nan: nan,
            }
        )

    def test_not(self) -> None:
        """Test the Not hedge."""
        HedgeAssert(self, fl.Not()).has_name("not").evaluates(
            {
                -1.0: 2.0,
                -0.5: 1.5,
                0.00: 1.0,
                0.25: 0.75,
                0.50: 0.50,
                0.75: 0.25,
                1.00: 0.00,
                inf: -inf,
                -inf: inf,
                nan: nan,
            }
        )

    def test_seldom(self) -> None:
        """Test the Seldom hedge."""
        HedgeAssert(self, fl.Seldom()).has_name("seldom").evaluates(
            {
                -1.0: nan,
                -0.5: nan,
                0.00: 0.0,
                0.25: 0.3535533905932738,
                0.50: 0.5,
                0.75: 0.6464466094067263,
                1.00: 1.0,
                inf: nan,
                -inf: nan,
                nan: nan,
            }
        )

    def test_somewhat(self) -> None:
        """Test the Somewhat hedge."""
        HedgeAssert(self, fl.Somewhat()).has_name("somewhat").evaluates(
            {
                -1.0: nan,
                -0.5: nan,
                0.00: 0.0,
                0.25: 0.5,
                0.50: 0.7071067811865476,
                0.75: 0.8660254037844386,
                1.00: 1.0,
                inf: inf,
                -inf: nan,
                nan: nan,
            }
        )

    def test_very(self) -> None:
        """Test the Very hedge."""
        HedgeAssert(self, fl.Very()).has_name("very").evaluates(
            {
                -1.0: 1.0,
                -0.5: 0.25,
                0.00: 0.0,
                0.25: 0.0625,
                0.50: 0.25,
                0.75: 0.5625,
                1.00: 1.0,
                inf: inf,
                -inf: inf,
                nan: nan,
            }
        )

    def test_function(self) -> None:
        """Test the Function hedge."""
        HedgeAssert(self, fl.HedgeFunction(fl.Function.create("my_hedge", "x**2"))).has_name(
            "my_hedge"
        ).evaluates(
            {
                -1.0: 1.0,
                -0.5: 0.25,
                0.00: 0.0,
                0.25: 0.0625,
                0.50: 0.25,
                0.75: 0.5625,
                1.00: 1.0,
                inf: inf,
                -inf: inf,
                nan: nan,
            }
        )

    def test_lambda(self) -> None:
        """Test the Lambda hedge."""
        HedgeAssert(
            self,
            fl.HedgeLambda(
                "my_hedge",
                lambda x: (np.where(x <= 0.5, 2 * x**2, 1 - 2 * (1 - x) ** 2)),
            ),
        ).has_name("my_hedge").evaluates(
            {
                -1.0: 2.0,
                -0.5: 0.5,
                0.00: 0.0,
                0.25: 0.125,
                0.50: 0.5,
                0.75: 0.875,
                1.00: 1.0,
                inf: -inf,
                -inf: inf,
                nan: nan,
            }
        )


if __name__ == "__main__":
    unittest.main()
