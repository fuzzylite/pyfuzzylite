"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited
fuzzylite is a registered trademark of FuzzyLite Limited.
"""

from __future__ import annotations

__all__ = ["Benchmark"]

import inspect
from pathlib import Path
from types import ModuleType
from typing import Any
from .operation import Op
import numpy as np
from typing_extensions import Self

from .engine import Engine
from .library import representation, settings
from .types import ScalarArray


class Benchmark:
    """The Benchmark class evaluates the performance of an Engine on a given dataset.

    @author Juan Rada-Vilela, Ph.D.
    @since 8.0
    """

    def __init__(
        self,
        name: str,
        engine: Engine,
        data: ScalarArray,
        *,
        rows: int | float = 1.0,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """@param name is the name of the benchmark
        @param engine is the engine to benchmark
        @param data is the data to benchmark the engine on
        @param rows is the number (int) or ratio (float) of rows to use from the data
        @param shuffle whether to shuffle the data.
        @param seed is the seed to shuffle the data.
        """
        self.name = name
        self.engine = engine
        self.data = data
        self.test_data = data.view()
        self.rows = rows
        self.shuffle = shuffle
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.results: list[float] = []

    def __repr__(self) -> str:
        """@return Python code to construct the benchmark."""
        fields = vars(self).copy()
        for field in "test_data random results".split():
            fields.pop(field)
        return representation.as_constructor(self)

    @classmethod
    def from_example(
        cls, example: ModuleType, rows: int | float = 1.0, shuffle: bool = True
    ) -> Self:
        """Create benchmark from the example
        @param example is the example to benchmark (eg, fuzzylite.examples.terms.arc)
        @param rows is the number (int) or ratio (float) of rows to use from the data
        @param shuffle whether to shuffle the data
        @return a benchmark ready for the example.
        """
        engine, data = cls.engine_and_data(example)
        return cls(example.__name__, engine, data, rows=rows, shuffle=shuffle)

    @classmethod
    def engine_and_data(cls, example: ModuleType) -> tuple[Engine, ScalarArray]:
        """Create the engine and load data from the example
        @param example is the module to benchmark (eg, fuzzylite.examples.terms.arc)
        @return tuple of engine and dataset.
        """
        name_class, *_ = inspect.getmembers(example, predicate=inspect.isclass)
        name, engine_class = name_class
        engine = engine_class().engine

        if not example.__file__:
            raise ValueError(
                f"expected valid '__file__' in the example, but none was found: {example}"
            )
        fld_file = Path(example.__file__).with_suffix(".fld")
        data = np.loadtxt(fld_file, skiprows=1)

        return engine, data

    def prepare(self) -> None:
        """Prepare the engine and dataset to benchmark."""
        self.prepare_engine()
        self.prepare_data()

    def prepare_engine(self) -> None:
        """Prepare the engine to benchmark."""
        self.engine.restart()

    def prepare_data(self) -> None:
        """Prepare the dataset to benchmark on."""
        data = self.data.view()
        if self.shuffle:
            rows = data.shape[0]
            data = data[self.random.choice(rows, size=rows, replace=False), :]

        if isinstance(self.rows, float):
            rows = int(self.rows * data.shape[0])
            data = data[0:rows, :]
        else:
            data = data[0 : self.rows, :]
        self.test_data = data

    def measure(self, runs: int = 1) -> None:
        """Measure the performance of the engine on the dataset for a number of runs
        @param runs is the number of runs to evaluate the engine on the test data
        .
        """
        from timeit import default_timer as timer

        self.prepare()
        self.results = []
        for run in range(runs):
            start = timer()
            self.run()
            end = timer()
            self.results.append(end - start)

    def summary(self) -> dict[str, Any]:
        """Summarize the benchmark results."""
        results = np.asarray(self.results)
        result: dict[str, Any] = dict(
            name=self.name,
            runs=len(self.results),
            mean=np.mean(results),
            std=np.std(results),
            min=np.min(results),
            q1=np.percentile(results, [25]),
            median=np.median(results),
            q3=np.percentile(results, [75]),
            max=np.max(results),
        )
        return result

    def summary_markdown(self, header: bool = False) -> str:
        """Summarize the benchmark results and format them using markdown.
        @param header whether to include table header in summary"""
        markdown = []
        summary = self.summary()
        if header:
            markdown.append(f"|{'|'.join(summary.keys())}|")
            markdown.append(f"|{'---'.join(['|'] * len(summary))}|")
        stats = [Op.str(x) for x in summary]
        markdown.append(f"|{' | '.join(stats)}|")
        result = "\n".join(markdown)
        return result

    def run(self) -> None:
        """Run the benchmark once."""
        self.engine.input_values = self.test_data[:, : len(self.engine.input_variables)]
        self.engine.process()
        np.testing.assert_allclose(
            self.test_data, self.engine.values, atol=settings.atol, rtol=settings.rtol
        )