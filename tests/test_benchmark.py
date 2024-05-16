"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

import time
import unittest

import numpy as np

import fuzzylite as fl
from fuzzylite.examples.mamdani import simple_dimmer


class TestBenchmark(unittest.TestCase):
    """Test the benchmark class."""

    def test_repr(self) -> None:
        """Test the representation as Python code."""
        engine = fl.Engine(
            "test",
            input_variables=[fl.InputVariable("A")],
            output_variables=[fl.OutputVariable("Z")],
        )
        data = fl.array(
            [
                [0.0, fl.nan],
                [0.5, fl.nan],
                [1.0, fl.nan],
            ]
        )
        self.assertEqual(
            (
                "fl.Benchmark(name='test', engine=fl.Engine(name='test', "
                "input_variables=[fl.InputVariable(name='A', minimum=-fl.inf, maximum=fl.inf, "
                "lock_range=False, terms=[])], output_variables=[fl.OutputVariable(name='Z', "
                "minimum=-fl.inf, maximum=fl.inf, lock_range=False, lock_previous=False, "
                "default_value=fl.nan, aggregation=None, defuzzifier=None, terms=[])], "
                "rule_blocks=[]), data=fl.array([fl.array([0.0, fl.nan]), fl.array([0.5, "
                "fl.nan]), fl.array([1.0, fl.nan])]), rows=1.0, shuffle=True, seed=None)"
            ),
            repr(fl.Benchmark("test", engine, data)),
        )

    def test_for_example(self) -> None:
        """Test the creation of benchmarks from examples."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, 0.5, False)

        self.assertEqual(repr(benchmark.engine), repr(simple_dimmer.SimpleDimmer().engine))
        expected_data = fl.array(
            [
                [0.000000000, fl.nan],
                [0.000977517, 0.750000000],
                [0.001955034, 0.750000000],
            ]
        )
        np.testing.assert_allclose(
            expected_data,
            benchmark.data[0 : len(expected_data), :],
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )
        self.assertEqual(benchmark.rows, 0.5)
        self.assertEqual(benchmark.shuffle, False)

    def test_engine_and_data(self) -> None:
        """Test the engine and dataset can be created from an example."""
        obtained_engine, obtained_data = fl.Benchmark.engine_and_data(simple_dimmer)

        self.assertEqual(repr(obtained_engine), repr(simple_dimmer.SimpleDimmer().engine))

        expected_data = fl.array(
            [
                [0.000000000, fl.nan],
                [0.000977517, 0.750000000],
                [0.001955034, 0.750000000],
            ]
        )
        np.testing.assert_allclose(
            expected_data,
            obtained_data[0 : len(expected_data), :],
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

    def test_prepare_engine(self) -> None:
        """Test the preparation of the engine."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, 0.5, False)
        benchmark.engine.input_variable(0).value = 123
        benchmark.engine.rule_block(0).rules[0].activation_degree = 0.123987
        benchmark.engine.output_variable(0).value = 987

        benchmark.prepare_engine()

        np.testing.assert_allclose(benchmark.engine.input_variable(0).value, fl.nan)
        np.testing.assert_allclose(benchmark.engine.output_variable(0).value, fl.nan)
        np.testing.assert_allclose(benchmark.engine.rule_block(0).rules[0].activation_degree, 0.0)

    def test_prepare_data(self) -> None:
        """Test the preparation of the dataset."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, 3, False)
        benchmark.prepare_data()

        expected_data = fl.array(
            [
                [0.000000000, fl.nan],
                [0.000977517, 0.750000000],
                [0.001955034, 0.750000000],
            ]
        )
        np.testing.assert_allclose(
            expected_data,
            benchmark.test_data,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

        benchmark.rows = 0.25
        benchmark.prepare_data()
        self.assertEqual(len(benchmark.test_data), 256)

    def test_prepare(self) -> None:
        """Test prepare gets the engine and dataset ready."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, 3, False)
        benchmark.engine.input_variable(0).value = 123.456

        benchmark.prepare()

        np.testing.assert_allclose(benchmark.engine.input_values, fl.array([[fl.nan]]))

        expected_data = fl.array(
            [
                [0.000000000, fl.nan],
                [0.000977517, 0.750000000],
                [0.001955034, 0.750000000],
            ]
        )
        np.testing.assert_allclose(
            expected_data,
            benchmark.test_data,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

    def test_run(self) -> None:
        """Test the run."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, rows=2, shuffle=False)
        np.testing.assert_allclose(fl.array([[fl.nan, fl.nan]]), benchmark.engine.values)

        # without preparing, the whole dataset is used
        self.assertEqual(len(benchmark.test_data), 1024)

        benchmark.run()

        np.testing.assert_allclose(
            benchmark.engine.values,
            benchmark.test_data,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

    def test_measure(self) -> None:
        """Test the benchmark can be measured over a number of runs."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, rows=1, shuffle=False)
        benchmark.run = lambda: time.sleep(1e-3)  # type: ignore

        benchmark.measure(runs=10)

        np.testing.assert_allclose(
            [1e-3] * 10, benchmark.time, atol=fl.settings.atol, rtol=fl.settings.rtol
        )
        np.testing.assert_allclose(
            [fl.nan] * 10, benchmark.error, atol=fl.settings.atol, rtol=fl.settings.rtol
        )

    def test_reset(self) -> None:
        """Test the benchmark can be reset to an initial state."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, rows=1, shuffle=True, seed=0)
        benchmark.prepare()
        # benchmark.measure(runs=10)
        benchmark.time = list(float(x) for x in range(10 + 1))
        benchmark.error = list(2.0 * e for e in range(10 + 1))
        self.assertEqual(1, len(benchmark.test_data))
        first_dataset = benchmark.test_data.copy()

        benchmark.reset()
        self.assertEqual([], benchmark.time)
        self.assertEqual([], benchmark.error)
        self.assertEqual(1024, len(benchmark.test_data))

        benchmark.prepare()
        np.testing.assert_allclose(first_dataset, benchmark.test_data)

    def test_summary(self) -> None:
        """Test the summary of the benchmark is a correct dictionary."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, rows=0, shuffle=False)
        benchmark.time = list(float(x) for x in range(10 + 1))
        benchmark.error = list(2.0 * e for e in range(10 + 1))
        self.assertDictEqual(
            benchmark.summary(),
            dict(
                name="fuzzylite.examples.mamdani.simple_dimmer",
                runs=11,
                time=dict(  # pyright: ignore
                    sum=55,
                    mean=5.0,
                    std=3.1622776601683795,
                    min=0,
                    q1=2.5,
                    median=5.0,
                    q3=7.5,
                    max=10,
                ),
                error=dict(  # pyright: ignore
                    sum=110,
                    mean=10.0,
                    std=6.324555320336759,
                    min=0,
                    q1=5.0,
                    median=10.0,
                    q3=15.0,
                    max=20.0,
                ),
            ),
        )

    def test_summary_markdown(self) -> None:
        """Test the summary of the benchmark can be exported as Markdown."""
        benchmark = fl.Benchmark.for_example(simple_dimmer, rows=0, shuffle=False)
        benchmark.time = list(float(x) for x in range(10 + 1))
        benchmark.error = list(2.0 * e for e in range(10 + 1))

        self.assertEqual(
            (
                "| name | runs | time_sum | time_mean | time_std | time_min | time_q1 | "
                "time_median | time_q3 | time_max | error_sum | error_mean | error_std | "
                "error_min | error_q1 | error_median | error_q3 | error_max |\n"
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
                "| fuzzylite.examples.mamdani.simple_dimmer | 11 | 55.000 | 5.000 | 3.162 | "
                "0.000 | 2.500 | 5.000 | 7.500 | 10.000 | 110.000 | 10.000 | 6.325 | 0.000 | "
                "5.000 | 10.000 | 15.000 | 20.000 |"
            ),
            benchmark.summary_markdown(header=True),
        )

        self.assertEqual(
            (
                "| fuzzylite.examples.mamdani.simple_dimmer | 11 | 55.000 | 5.000 | 3.162 | "
                "0.000 | 2.500 | 5.000 | 7.500 | 10.000 | 110.000 | 10.000 | 6.325 | 0.000 | "
                "5.000 | 10.000 | 15.000 | 20.000 |"
            ),
            benchmark.summary_markdown(),
        )

        benchmark.reset()
        self.assertEqual(
            (
                "| fuzzylite.examples.mamdani.simple_dimmer | 0 | nan | nan | nan | nan | nan "
                "| nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |"
            ),
            benchmark.summary_markdown(),
        )

    def test_multiple_benchmarks(self) -> None:
        """Test how multiple benchmarks can be created."""
        examples = [fl.examples.terms.arc, fl.examples.terms.zs_shape]
        benchmarks = []
        for index, example in enumerate(examples):
            benchmarks.append(fl.Benchmark.for_example(example, rows=3, shuffle=False))
        for benchmark in benchmarks:
            # benchmark.measure(runs=10)
            benchmark.time = [float(x) for x in range(10)]
            benchmark.error = [2 * float(x) for x in range(10)]
        summary = []
        for index, benchmark in enumerate(benchmarks):
            summary.append(benchmark.summary_markdown(header=index == 0))

        self.assertEqual(
            (
                "| name | runs | time_sum | time_mean | time_std | time_min | time_q1 | "
                "time_median | time_q3 | time_max | error_sum | error_mean | error_std | "
                "error_min | error_q1 | error_median | error_q3 | error_max |\n"
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
                "| fuzzylite.examples.terms.arc | 10 | 45.000 | 4.500 | 2.872 | 0.000 | 2.250 "
                "| 4.500 | 6.750 | 9.000 | 90.000 | 9.000 | 5.745 | 0.000 | 4.500 | 9.000 | "
                "13.500 | 18.000 |\n"
                "| fuzzylite.examples.terms.zs_shape | 10 | 45.000 | 4.500 | 2.872 | 0.000 | "
                "2.250 | 4.500 | 6.750 | 9.000 | 90.000 | 9.000 | 5.745 | 0.000 | 4.500 | "
                "9.000 | 13.500 | 18.000 |"
            ),
            "\n".join(summary),
        )


if __name__ == "__main__":
    unittest.main()
