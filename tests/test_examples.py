import importlib
import logging
import pathlib
import unittest

logger = logging.getLogger()


class TestTerms(unittest.TestCase):

    def test_examples(self) -> None:
        import fuzzylite.examples.terms
        examples = pathlib.Path(next(iter(fuzzylite.examples.terms.__path__)))  # type: ignore
        self.assertTrue(examples.exists() and examples.is_dir())

        expected = {"__init__", "Bell", "Binary", "Concave", "Constant", "Cosine", "Discrete",
                    "Function", "Gaussian", "GaussianProduct", "Linear", "PiShape", "Ramp",
                    "Rectangle", "Sigmoid", "SigmoidDifference", "SigmoidProduct", "Spike",
                    "Trapezoid", "Triangle", "ZSShape"}
        self.assertSetEqual(expected, {file.stem for file in examples.glob("*.py")})

        for file_py in examples.iterdir():
            if file_py.suffix == ".py" and file_py.name != '__init__.py':
                module = f"{fuzzylite.examples.terms.__name__}.{file_py.stem}"
                # if an example is incorrect, an exception will be thrown below
                engine = importlib.import_module(module).engine  # type: ignore
                logger.info(str(engine))
