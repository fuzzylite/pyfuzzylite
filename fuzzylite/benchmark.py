"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = ["Benchmark"]

import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .engine import Engine
from .library import nan, representation, settings
from .operation import Op
from .types import Scalar, ScalarArray


class Benchmark:
    """Evaluate the performance of an engine on a dataset.

    The performance time is measured in seconds and the error is measured as the mean squared error
    over the differences between the expected dataset output values and the obtained output values.

    info: related
        - [fuzzylite.engine.Engine][]
        - [fuzzylite.library.Settings][]
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
        """Constructor.

        Args:
            name: name of the benchmark
            engine: engine to benchmark
            data: data to benchmark the engine on
            rows: number (int) or ratio (float) of rows to use from the data
            shuffle: shuffles the data
            seed: seed to shuffle the data.
        """
        self.name = name
        self.engine = engine
        self.data = data
        self.test_data = data.view()
        self.rows = rows
        self.shuffle = shuffle
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)
        self.time: list[float] = []
        self.error: list[float] = []

    def __repr__(self) -> str:
        """Return the code to construct the benchmark in Python.

        Returns:
            code to construct the benchmark in Python.
        """
        fields = vars(self).copy()
        for field in "test_data random time error".split():
            fields.pop(field)
        return representation.as_constructor(self)

    @classmethod
    def for_example(
        cls,
        example: ModuleType,
        rows: int | float = 1.0,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Benchmark:
        """Create benchmark for the example.

        Args:
            example: example to benchmark (eg, `fuzzylite.examples.terms.arc`)
            rows: number (int) or ratio (float) of rows to use from the data
            shuffle: whether to shuffle the data
            seed: seed to shuffle the data

        Returns:
             a benchmark ready for the example
        """
        engine, data = cls.engine_and_data(example)
        return cls(example.__name__, engine, data, rows=rows, shuffle=shuffle, seed=seed)

    @classmethod
    def engine_and_data(cls, example: ModuleType) -> tuple[Engine, ScalarArray]:
        """Create the engine and load the dataset for the example.

        Args:
            example: is the module to benchmark (eg, fuzzylite.examples.terms.arc)

        Returns:
             tuple of engine and dataset
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
            rows = len(data)
            data = data[self.random.choice(rows, size=rows, replace=False), :]

        if isinstance(self.rows, float):
            rows = int(self.rows * len(data))
            data = data[0:rows, :]
        else:
            data = data[0 : self.rows, :]
        self.test_data = data

    def measure(self, *, runs: int = 1) -> None:
        """Measure the performance of the engine on the dataset for a number of runs.

        Args:
             runs: number of runs to evaluate the engine on the test data
        """
        from timeit import default_timer as timer

        self.time = []
        self.error = []
        for run in range(runs):
            self.prepare()
            start = timer()
            self.run()
            end = timer()
            self.time.append(end - start)
            self.error.append(
                np.mean(
                    np.square(
                        self.engine.output_values
                        - self.test_data[:, len(self.engine.input_variables) :]
                    )
                ).astype(float)
            )

    def run(self) -> None:
        """Run the benchmark once (without computing statistics)."""
        self.engine.input_values = self.test_data[:, : len(self.engine.input_variables)]
        self.engine.process()
        np.testing.assert_allclose(
            self.test_data, self.engine.values, atol=settings.atol, rtol=settings.rtol
        )

    def reset(self) -> None:
        """Reset the benchmark."""
        self.time = []
        self.error = []
        self.random = np.random.RandomState(seed=self.seed)
        self.test_data = self.data.view()

    def summary(self) -> dict[str, Any]:
        """Summarize the benchmark results.

        Returns:
             dictionary of statistics containing the performance time in seconds and the mean squared error
        """
        time = np.asarray(self.time or [nan])
        error = np.asarray(self.error or [nan])
        result: dict[str, Any] = dict(name=self.name, runs=len(self.time))
        for metric, values in {"time": time, "error": error}.items():
            result[metric] = dict(
                sum=np.sum(values),
                mean=np.mean(values),
                std=np.std(values),
                min=np.min(values),
                q1=np.percentile(values, q=[25.0]).item(),
                median=np.median(values),
                q3=np.percentile(values, q=[75.0]).item(),
                max=np.max(values),
            )
        return result

    def summary_markdown(self, *, header: bool = False) -> str:
        """Summarize the benchmark results and format them using markdown.

        Args:
             header: whether to include table header in summary
        """
        markdown = []
        summary = self.summary()
        # flattening the summary
        for metric in ["time", "error"]:
            metrics: dict[str, Scalar] = summary.pop(metric)
            for name, value in metrics.items():
                summary[f"{metric}_{name}"] = value

        if header:
            markdown.append(f"| {' | '.join(summary.keys())} |")
            markdown.append(f"| {' | '.join(['---:'] * len(summary))} |")
        stats = [Op.str(x) for x in summary.values()]
        markdown.append(f"| {' | '.join(stats)} |")
        result = "\n".join(markdown)
        return result
