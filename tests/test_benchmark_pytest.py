"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any, Callable

from pytest_benchmark.fixture import BenchmarkFixture

import fuzzylite as fl


def generate_tests(package: ModuleType) -> str:
    """Automatically generate benchmark tests for every example in the package.

    @param package is the package containing the examples (eg, fuzzylite.examples.terms)
    @return Python code containing the tests
    """
    tests = [
        f"""\
class Test{fl.Op.pascal_case(package.__name__)}:
    \"\"\"Benchmark suite for {package.__name__}\"\"\"

    def setup_method(self, method: Callable[[BenchmarkFixture], None]) -> None:
        \"\"\"Configures the benchmark before execution\"\"\"
        self.parameters: dict[str, Any] = dict(
            rows=1e-3, shuffle=True, rounds=30, iterations=30, warmup_rounds=0
        )
"""
    ]
    for name, module in inspect.getmembers(package, predicate=inspect.ismodule):
        tests.append(
            f"""\
    def test_{name}(self, benchmark: BenchmarkFixture) -> None:
        \"\"\"Benchmark the {module.__name__}\"\"\"
        PytestBenchmark({str(module.__name__).replace("fuzzylite", "fl")}).start(benchmark, **self.parameters)\n
"""
        )
    return "\n".join(tests)


class PytestBenchmark(fl.Benchmark):
    """Class for pytest benchmark session."""

    def __init__(self, example: ModuleType) -> None:
        """@param example is the module containing the example."""
        super().__init__(example.__name__, *fl.Benchmark.engine_and_data(example))

    def start(
        self,
        benchmark: BenchmarkFixture,
        rows: int | float,
        shuffle: bool,
        rounds: int,
        iterations: int,
        warmup_rounds: int = 0,
    ) -> None:
        """Start the benchmark.

        @param benchmark is the fixture from the pytest-benchmark library
        @param rows is the number (int) or ratio (float) of rows to use from the data
        @param shuffle whether to shuffle the data
        @param rounds is the number of rounds to perform the evaluations
        @param iterations is the number of iterations to perform in a single round
        @param warmup_rounds is the number of rounds to warm up.
        """
        self.rows = rows
        self.shuffle = shuffle
        self.prepare()
        benchmark.pedantic(
            self.run,
            rounds=rounds,
            iterations=iterations,
            warmup_rounds=warmup_rounds,
        )


class TestFuzzyliteExamplesTerms:
    """Benchmark suite for fuzzylite.examples.terms."""

    def setup_method(self, method: Callable[[BenchmarkFixture], None]) -> None:
        """Configures the benchmark before execution."""
        self.parameters: dict[str, Any] = dict(
            rows=0.1, shuffle=True, rounds=30, iterations=30, warmup_rounds=0
        )

    def test_arc(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.arc."""
        PytestBenchmark(fl.examples.terms.arc).start(benchmark, **self.parameters)

    def test_bell(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.bell."""
        PytestBenchmark(fl.examples.terms.bell).start(benchmark, **self.parameters)

    def test_binary(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.binary."""
        PytestBenchmark(fl.examples.terms.binary).start(benchmark, **self.parameters)

    def test_concave(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.concave."""
        PytestBenchmark(fl.examples.terms.concave).start(benchmark, **self.parameters)

    def test_constant(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.constant."""
        PytestBenchmark(fl.examples.terms.constant).start(benchmark, **self.parameters)

    def test_cosine(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.cosine."""
        PytestBenchmark(fl.examples.terms.cosine).start(benchmark, **self.parameters)

    def test_discrete(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.discrete."""
        PytestBenchmark(fl.examples.terms.discrete).start(benchmark, **self.parameters)

    def test_function(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.function."""
        PytestBenchmark(fl.examples.terms.function).start(benchmark, **self.parameters)

    def test_gaussian(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.gaussian."""
        PytestBenchmark(fl.examples.terms.gaussian).start(benchmark, **self.parameters)

    def test_gaussian_product(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.gaussian_product."""
        PytestBenchmark(fl.examples.terms.gaussian_product).start(benchmark, **self.parameters)

    def test_linear(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.linear."""
        PytestBenchmark(fl.examples.terms.linear).start(benchmark, **self.parameters)

    def test_pi_shape(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.pi_shape."""
        PytestBenchmark(fl.examples.terms.pi_shape).start(benchmark, **self.parameters)

    def test_ramp(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.ramp."""
        PytestBenchmark(fl.examples.terms.ramp).start(benchmark, **self.parameters)

    def test_rectangle(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.rectangle."""
        PytestBenchmark(fl.examples.terms.rectangle).start(benchmark, **self.parameters)

    def test_semi_ellipse(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.semi_ellipse."""
        PytestBenchmark(fl.examples.terms.semi_ellipse).start(benchmark, **self.parameters)

    def test_sigmoid(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid."""
        PytestBenchmark(fl.examples.terms.sigmoid).start(benchmark, **self.parameters)

    def test_sigmoid_difference(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid_difference."""
        PytestBenchmark(fl.examples.terms.sigmoid_difference).start(benchmark, **self.parameters)

    def test_sigmoid_product(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid_product."""
        PytestBenchmark(fl.examples.terms.sigmoid_product).start(benchmark, **self.parameters)

    def test_spike(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.spike."""
        PytestBenchmark(fl.examples.terms.spike).start(benchmark, **self.parameters)

    def test_trapezoid(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.trapezoid."""
        PytestBenchmark(fl.examples.terms.trapezoid).start(benchmark, **self.parameters)

    def test_triangle(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.triangle."""
        PytestBenchmark(fl.examples.terms.triangle).start(benchmark, **self.parameters)

    def test_zs_shape(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.zs_shape."""
        PytestBenchmark(fl.examples.terms.zs_shape).start(benchmark, **self.parameters)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
