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

from pytest_codspeed.plugin import BenchmarkFixture

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
        self.parameters: dict[str, Any] = dict(rows=1e-3, shuffle=True)
"""
    ]
    for name, module in inspect.getmembers(package, predicate=inspect.ismodule):
        tests.append(
            f"""\
    def test_{name}(self, benchmark: BenchmarkFixture) -> None:
        \"\"\"Benchmark the {module.__name__}\"\"\"
        CodspeedBenchmark({str(module.__name__).replace("fuzzylite", "fl")}).start(benchmark, **self.parameters)\n
"""
        )
    return "\n".join(tests)


class CodspeedBenchmark(fl.Benchmark):
    """Class for Codspeed benchmark session."""

    def __init__(self, example: ModuleType) -> None:
        """@param example is the module containing the example."""
        super().__init__(example.__name__, *fl.Benchmark.engine_and_data(example))

    def start(self, benchmark: BenchmarkFixture, rows: int | float, shuffle: bool) -> None:
        """Start the benchmark.

        @param benchmark is the fixture from the pytest-codspeed library
        @param rows is the number (int) or ratio (float) of rows to use from the data
        @param shuffle whether to shuffle the data.
        """
        self.rows = rows
        self.shuffle = shuffle
        self.prepare()
        benchmark(lambda: self.run())


class TestFuzzyliteExamplesTerms:
    """Benchmark suite for fuzzylite.examples.terms."""

    def setup_method(self, method: Callable[[BenchmarkFixture], None]) -> None:
        """Configures the benchmark before execution."""
        self.parameters: dict[str, Any] = dict(rows=0.1, shuffle=True)

    def test_arc(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.arc."""
        CodspeedBenchmark(fl.examples.terms.arc).start(benchmark, **self.parameters)

    def test_bell(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.bell."""
        CodspeedBenchmark(fl.examples.terms.bell).start(benchmark, **self.parameters)

    def test_binary(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.binary."""
        CodspeedBenchmark(fl.examples.terms.binary).start(benchmark, **self.parameters)

    def test_concave(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.concave."""
        CodspeedBenchmark(fl.examples.terms.concave).start(benchmark, **self.parameters)

    def test_constant(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.constant."""
        CodspeedBenchmark(fl.examples.terms.constant).start(benchmark, **self.parameters)

    def test_cosine(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.cosine."""
        CodspeedBenchmark(fl.examples.terms.cosine).start(benchmark, **self.parameters)

    def test_discrete(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.discrete."""
        CodspeedBenchmark(fl.examples.terms.discrete).start(benchmark, **self.parameters)

    def test_function(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.function."""
        CodspeedBenchmark(fl.examples.terms.function).start(benchmark, **self.parameters)

    def test_gaussian(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.gaussian."""
        CodspeedBenchmark(fl.examples.terms.gaussian).start(benchmark, **self.parameters)

    def test_gaussian_product(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.gaussian_product."""
        CodspeedBenchmark(fl.examples.terms.gaussian_product).start(benchmark, **self.parameters)

    def test_linear(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.linear."""
        CodspeedBenchmark(fl.examples.terms.linear).start(benchmark, **self.parameters)

    def test_pi_shape(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.pi_shape."""
        CodspeedBenchmark(fl.examples.terms.pi_shape).start(benchmark, **self.parameters)

    def test_ramp(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.ramp."""
        CodspeedBenchmark(fl.examples.terms.ramp).start(benchmark, **self.parameters)

    def test_rectangle(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.rectangle."""
        CodspeedBenchmark(fl.examples.terms.rectangle).start(benchmark, **self.parameters)

    def test_semi_ellipse(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.semi_ellipse."""
        CodspeedBenchmark(fl.examples.terms.semi_ellipse).start(benchmark, **self.parameters)

    def test_sigmoid(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid."""
        CodspeedBenchmark(fl.examples.terms.sigmoid).start(benchmark, **self.parameters)

    def test_sigmoid_difference(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid_difference."""
        CodspeedBenchmark(fl.examples.terms.sigmoid_difference).start(benchmark, **self.parameters)

    def test_sigmoid_product(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.sigmoid_product."""
        CodspeedBenchmark(fl.examples.terms.sigmoid_product).start(benchmark, **self.parameters)

    def test_spike(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.spike."""
        CodspeedBenchmark(fl.examples.terms.spike).start(benchmark, **self.parameters)

    def test_trapezoid(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.trapezoid."""
        CodspeedBenchmark(fl.examples.terms.trapezoid).start(benchmark, **self.parameters)

    def test_triangle(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.triangle."""
        CodspeedBenchmark(fl.examples.terms.triangle).start(benchmark, **self.parameters)

    def test_zs_shape(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark the fuzzylite.examples.terms.zs_shape."""
        CodspeedBenchmark(fl.examples.terms.zs_shape).start(benchmark, **self.parameters)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
