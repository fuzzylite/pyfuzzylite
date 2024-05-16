"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import math
import unittest
from types import ModuleType
from typing import Callable

import numpy as np
import numpy.testing as npt

import fuzzylite as fl


class AssertOperator:
    """Assert operator."""

    def __init__(self, commutative: bool = True, scalable: bool = True) -> None:
        """Construct assert operator."""
        self.commutative = commutative
        self.scalable = scalable

    def assert_operator(
        self,
        operator: Callable[[fl.Scalar, fl.Scalar], fl.Scalar],
        a: fl.Scalar,
        b: fl.Scalar,
        expected: fl.Scalar,
    ) -> AssertOperator:
        """Assert operator."""
        obtained = operator(a, b)
        npt.assert_equal(obtained, expected)
        return self

    def assert_that(
        self,
        operator: Callable[[fl.Scalar, fl.Scalar], fl.Scalar],
        operands_result: dict[tuple[fl.Scalar, fl.Scalar], fl.Scalar],
    ) -> AssertOperator:
        """Assert that."""
        for (a, b), expected in operands_result.items():
            obtained = operator(a, b)
            npt.assert_equal(
                obtained,
                expected,
                err_msg=f"{str(operator.__qualname__)}(a={a}, b={b}) = {obtained}, but expected {expected}",
            )
            if self.commutative:
                obtained = operator(b, a)
                npt.assert_equal(
                    obtained,
                    expected,
                    err_msg=f"{str(operator.__qualname__)}(b={b}, a={a}) = {obtained}, but expected {expected}",
                )

        if self.scalable:
            values = operands_result.keys()
            a = fl.array([a for a, _ in values])
            b = fl.array([b for _, b in values])
            expected = fl.array([z for z in operands_result.values()])
            obtained = operator(a, b)
            npt.assert_equal(
                obtained,
                expected,
                err_msg=f"{str(operator.__qualname__)}(a={a}, b={b}) = {obtained}, but expected {expected}",
            )
            if self.commutative:
                obtained = operator(b, a)
                npt.assert_equal(
                    obtained,
                    expected,
                    err_msg=f"{str(operator.__qualname__)}(b={b}, a={a}) = {obtained}, but expected {expected}",
                )
        return self


class TestOperation(unittest.TestCase):
    """Test operation."""

    def test_operator_eq(self) -> None:
        """Test operator eq."""
        AssertOperator(commutative=True, scalable=True).assert_that(
            fl.Op.eq,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (fl.inf, fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 0.0,
                (-0.0, -0.0): 1.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_operator_neq(self) -> None:
        """Test operator neq."""
        AssertOperator(commutative=True, scalable=True).assert_that(
            fl.Op.neq,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 1.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 1.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_gt(self) -> None:
        """Test operator gt."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.gt,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 1.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_lt(self) -> None:
        """Test operator lt."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.lt,
            {
                (fl.nan, fl.nan): 0.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (fl.inf, fl.inf): 0.0,
                (1.0, 1.0): 0.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 0.0,
                (-1.0, -1.0): 0.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 0.0,
                (-0.0, -0.0): 0.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 0.0,
            },
        )

    def test_operator_ge(self) -> None:
        """Test operator ge."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.ge,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 1.0,
                (fl.inf, fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 1.0,
                (0.0, 1.0): 0.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 0.0,
                (-0.0, -1.0): 1.0,
                (-0.0, 0.0): 1.0,
                (1.0 + 1e-15, 1.0): 1.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_operator_le(self) -> None:
        """Test operator le."""
        AssertOperator(commutative=False, scalable=True).assert_that(
            fl.Op.le,
            {
                (fl.nan, fl.nan): 1.0,
                (fl.nan, 0.0): 0.0,
                (fl.inf, -fl.inf): 0.0,
                (-fl.inf, fl.inf): 1.0,
                (-fl.inf, -fl.inf): 1.0,
                (1.0, 1.0): 1.0,
                (1.0, 0.0): 0.0,
                (0.0, 1.0): 1.0,
                (0.0, 0.0): 1.0,
                (-1.0, -1.0): 1.0,
                (-1.0, -0.0): 1.0,
                (-0.0, -1.0): 0.0,
                (-0.0, 0.0): 1.0,
                (1.0 + 1e-15, 1.0): 0.0,
                (1.0 + 1e-16, 1.0): 1.0,
            },
        )

    def test_valid_identifier(self) -> None:
        """Test what a valid identifier is."""
        self.assertEqual(fl.Op.as_identifier("  xx  "), "xx")  # trims
        self.assertEqual(fl.Op.as_identifier("   ~!@#$%^&*()+{}[]:;\"'<>?/,   "), "_")
        self.assertEqual(fl.Op.as_identifier("abc123_.ABC"), "abc123_ABC")
        self.assertEqual(fl.Op.as_identifier("      "), "_")
        self.assertEqual(fl.Op.as_identifier("9abc_ABC"), "_9abc_ABC")

    def test_snake_case(self) -> None:
        """Test conversion to snake_case."""
        self.assertEqual("", fl.Op.snake_case(""))
        self.assertEqual("hello_world", fl.Op.snake_case("helloWorld"))
        self.assertEqual("hello_world", fl.Op.snake_case("HelloWorld"))
        self.assertEqual("hello_world", fl.Op.snake_case("Hello World!"))
        self.assertEqual("hello_world", fl.Op.snake_case(" ¡Hello World! "))
        self.assertEqual("hello_world_123", fl.Op.snake_case(" ¡Hello World! @\t123"))
        self.assertEqual("123_456_789_00_x", fl.Op.snake_case(" 123@@456..789--00_x "))

    def test_pascal_case(self) -> None:
        """Test conversion of strings to CamelCase."""
        self.assertEqual("", fl.Op.pascal_case(""))
        self.assertEqual("HelloWorld", fl.Op.pascal_case("HelloWorld"))
        self.assertEqual("HelloWorld", fl.Op.pascal_case("Hello World!"))
        self.assertEqual("HelloWorld", fl.Op.pascal_case(" ¡Hello World! "))
        self.assertEqual("HelloWorld123", fl.Op.pascal_case(" ¡Hello World! @\t123"))

        self.assertEqual("HelloWorld", fl.Op.pascal_case("hello_world"))
        self.assertEqual("HelloWorld", fl.Op.pascal_case("__hello_world__!"))
        self.assertEqual("HelloWorld", fl.Op.pascal_case(" ¡Hello World! "))
        self.assertEqual("HelloWorld123", fl.Op.pascal_case(" ¡Hello World! @\t123"))

    def test_str(self) -> None:
        """Test string operation uses global decimals."""
        with fl.library.settings.context(decimals=3):
            self.assertEqual(fl.Op.str(0.3), "0.300")
            self.assertEqual(fl.Op.str(-0.3), "-0.300")
            self.assertEqual(fl.Op.str(3), "3")
            self.assertEqual(fl.Op.str(3.0001), "3.000")

        self.assertEqual(fl.Op.str(math.inf), "inf")
        self.assertEqual(fl.Op.str(-math.inf), "-inf")
        self.assertEqual(fl.Op.str(math.nan), "nan")

        with fl.library.settings.context(decimals=5):
            self.assertEqual(fl.Op.str(0.3), "0.30000")

        with fl.library.settings.context(decimals=0):
            self.assertEqual(fl.Op.str(0.3), "0")

        self.assertEqual("0.000 1.000 2.000 3.000", fl.Op.str([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual("0.000 1.000 2.000 3.000", fl.Op.str(fl.scalar([0.0, 1.0, 2.0, 3.0])))
        self.assertEqual(
            "0.000 0.000\n1.000 1.000\n2.000 2.000",
            fl.Op.str(fl.scalar([[0, 0], [1, 1], [2, 2]])),
        )
        self.assertEqual(
            "[[[0.000 0.000]\n  [1.000 1.000]\n  [2.000 2.000]\n  [3.000 3.000]]]",
            fl.Op.str(fl.scalar([[[0, 0], [1, 1], [2, 2], [3, 3]]])),
        )

    def test_scale(self) -> None:
        """Test linear interpolation."""
        self.assertEqual(fl.Op.scale(0, 0, 1, -10, 10), -10.0)
        self.assertEqual(fl.Op.scale(0.5, 0, 1, -10, 10), 0.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, -10, 10), 10)

        self.assertEqual(fl.Op.scale(0, 0, 1, 0, 10), 0.0)
        self.assertEqual(fl.Op.scale(0.5, 0, 1, 0, 10), 5.0)
        self.assertEqual(fl.Op.scale(1, 0, 1, 0, 10), 10)

        self.assertEqual(fl.Op.scale(-1, 0, 1, 0, 10), -10.0)
        self.assertEqual(fl.Op.scale(2, 0, 1, 0, 10), 20)

        self.assertEqual(math.isnan(fl.Op.scale(math.nan, 0, 1, 0, 10)), True)
        self.assertEqual(fl.Op.scale(math.inf, 0, 1, 0, 10), math.inf)
        self.assertEqual(fl.Op.scale(-math.inf, 0, 1, 0, 10), -math.inf)

    def test_decimals(self) -> None:
        """Test decimals."""
        x = fl.Op.str(1.0)
        self.assertEqual("1.000", x)
        with fl.library.settings.context(decimals=6):
            x = fl.Op.str(1.0)
        self.assertEqual("1.000000", x)

    def test_glob_examples(self) -> None:
        """Test globbing examples."""
        # Modules
        modules = list(fl.Op.glob_examples("module"))
        self.assertEqual(61, len(modules))
        self.assertSetEqual({ModuleType}, {m.__class__ for m in modules})

        modules = list(fl.Op.glob_examples("module", fl.examples, recursive=False))
        self.assertEqual(0, len(modules))

        modules = list(fl.Op.glob_examples("module", fl.examples.hybrid, recursive=False))
        self.assertEqual(
            [
                "fuzzylite.examples.hybrid.obstacle_avoidance",
                "fuzzylite.examples.hybrid.tipper",
            ],
            [m.__name__ for m in modules],
        )

        # Engines
        engines = list(fl.Op.glob_examples("engine", module=fl.examples.mamdani, recursive=False))
        self.assertEqual(
            [
                "AllTerms",
                "Laundry",
                "ObstacleAvoidance",
                "SimpleDimmer",
                "SimpleDimmerChained",
                "SimpleDimmerInverse",
            ],
            [e.name for e in engines],
        )

        engines = list(fl.Op.glob_examples("engine"))
        self.assertEqual(61, len(engines))
        self.assertSetEqual({fl.Engine}, {e.__class__ for e in engines})

        engines = list(fl.Op.glob_examples("engine", module=fl.examples.mamdani, recursive=False))
        self.assertEqual(
            [
                "AllTerms",
                "Laundry",
                "ObstacleAvoidance",
                "SimpleDimmer",
                "SimpleDimmerChained",
                "SimpleDimmerInverse",
            ],
            [e.name for e in engines],
        )

        # Datasets
        datasets = fl.array([d for d in fl.Op.glob_examples("dataset", fl.examples.tsukamoto)])
        np.testing.assert_allclose(
            datasets,
            fl.array([d for d in fl.Op.glob_examples("fld", fl.examples.tsukamoto)]),
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )
        self.assertEqual(1, len(datasets))
        tsukamoto_fld = fl.array(
            [
                [-10.000000000, 0.255363311, fl.inf, 0.255012132, 0.250961670],
                [-9.980449658, 0.255423538, 1.229664676, 0.255066538, 0.250971506],
                [-9.960899316, 0.255484536, 1.092913074, 0.255121547, 0.250981462],
            ]
        )
        np.testing.assert_allclose(
            tsukamoto_fld,
            datasets[0][: len(tsukamoto_fld), :],
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

        # Language
        flls = list(fl.Op.glob_examples("fll", fl.examples.terms))
        self.assertEqual(flls, list(fl.Op.glob_examples("language", fl.examples.terms)))
        self.assertEqual(22, len(flls))
        self.assertEqual(
            flls[0],
            expected_arc_fll := (
                "Engine: Arc\n"
                "  description: obstacle avoidance for self-driving cars\n"
                "InputVariable: obstacle\n"
                "  description: location of obstacle relative to vehicle\n"
                "  enabled: true\n"
                "  range: 0.000 1.000\n"
                "  lock-range: false\n"
                "  term: left Triangle 0.000 0.333 0.666\n"
                "  term: right Triangle 0.333 0.666 1.000\n"
                "OutputVariable: steer\n"
                "  description: direction to steer the vehicle to\n"
                "  enabled: true\n"
                "  range: 0.000 1.000\n"
                "  lock-range: false\n"
                "  aggregation: Maximum\n"
                "  defuzzifier: Centroid\n"
                "  default: nan\n"
                "  lock-previous: false\n"
                "  term: left Arc 0.666 0.000\n"
                "  term: right Arc 0.333 1.000\n"
                "RuleBlock: steer_away\n"
                "  description: steer away from obstacles\n"
                "  enabled: true\n"
                "  conjunction: none\n"
                "  disjunction: none\n"
                "  implication: Minimum\n"
                "  activation: General\n"
                "  rule: if obstacle is left then steer is right\n"
                "  rule: if obstacle is right then steer is left\n"
            ),
        )

        # Files
        files = list(fl.Op.glob_examples("files", fl.examples.takagi_sugeno.octave))
        expected_files = []
        for example in [
            "cubic_approximator",
            "heart_disease_risk",
            "linear_tip_calculator",
            "sugeno_tip_calculator",
        ]:
            expected_files.extend([f"{example}.fld", f"{example}.fll", f"{example}.py"])
        self.assertEqual([f.name for f in files], expected_files)

        # Modules, not packages:
        # engine
        engines = list(fl.Op.glob_examples("engine", fl.examples.terms.arc))
        self.assertEqual(1, len(engines))
        self.assertEqual("Arc", engines[0].name)

        # module
        modules = list(fl.Op.glob_examples("module", fl.examples.mamdani.simple_dimmer))
        self.assertEqual(1, len(modules))
        self.assertEqual(modules[0], fl.examples.mamdani.simple_dimmer)

        # fll
        flls = list(fl.Op.glob_examples("fll", fl.examples.terms.arc))
        self.assertEqual(1, len(flls))
        self.assertEqual(flls[0], expected_arc_fll)

        # fld
        flds = list(fl.Op.glob_examples("fld", fl.examples.tsukamoto.tsukamoto))
        self.assertEqual(1, len(flds))
        np.testing.assert_allclose(tsukamoto_fld, flds[0][: len(tsukamoto_fld), :])

        # files
        files = list(fl.Op.glob_examples("files", fl.examples.tsukamoto.tsukamoto))
        self.assertSetEqual(
            {"tsukamoto.py", "tsukamoto.fld", "tsukamoto.fll"},
            {file.name for file in files},
        )

    @unittest.skip("Revisit describe() method")
    def test_describe(self) -> None:
        """Test describe."""
        # TODO: Revisit describe method
        self.assertEqual(
            "OutputVariable[{"
            "'default_value': 'nan', 'defuzzifier': 'None', "
            "'fuzzy': 'term: x Aggregated []', "
            "'lock_previous': 'False', 'previous_value': 'nan'"
            "}]",
            fl.Op.describe(fl.OutputVariable("x", "an x", terms=[fl.Triangle("t")])),
        )
        self.assertEqual(
            "InputVariable[{}]",
            fl.Op.describe(fl.InputVariable("x", "an x", terms=[fl.Triangle("t")])),
        )


if __name__ == "__main__":
    unittest.main()
