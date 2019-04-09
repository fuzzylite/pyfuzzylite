import importlib
import logging
import pathlib
import unittest

logger = logging.getLogger()


class TestTerms(unittest.TestCase):

    def test_examples(self) -> None:
        import fuzzylite.examples
        examples = pathlib.Path(next(iter(fuzzylite.examples.__path__)))  # type: ignore
        self.assertTrue(examples.exists() and examples.is_dir())

        expected = set("""\
fuzzylite.examples.hybrid.ObstacleAvoidance
fuzzylite.examples.hybrid.tipper
fuzzylite.examples.mamdani.AllTerms
fuzzylite.examples.mamdani.Laundry
fuzzylite.examples.mamdani.ObstacleAvoidance
fuzzylite.examples.mamdani.SimpleDimmer
fuzzylite.examples.mamdani.SimpleDimmerChained
fuzzylite.examples.mamdani.SimpleDimmerInverse
fuzzylite.examples.mamdani.matlab.mam21
fuzzylite.examples.mamdani.matlab.mam22
fuzzylite.examples.mamdani.matlab.shower
fuzzylite.examples.mamdani.matlab.tank
fuzzylite.examples.mamdani.matlab.tank2
fuzzylite.examples.mamdani.matlab.tipper
fuzzylite.examples.mamdani.matlab.tipper1
fuzzylite.examples.mamdani.octave.investment_portfolio
fuzzylite.examples.mamdani.octave.mamdani_tip_calculator
fuzzylite.examples.takagi_sugeno.ObstacleAvoidance
fuzzylite.examples.takagi_sugeno.SimpleDimmer
fuzzylite.examples.takagi_sugeno.approximation
fuzzylite.examples.takagi_sugeno.matlab.fpeaks
fuzzylite.examples.takagi_sugeno.matlab.invkine1
fuzzylite.examples.takagi_sugeno.matlab.invkine2
fuzzylite.examples.takagi_sugeno.matlab.juggler
fuzzylite.examples.takagi_sugeno.matlab.membrn1
fuzzylite.examples.takagi_sugeno.matlab.membrn2
fuzzylite.examples.takagi_sugeno.matlab.slbb
fuzzylite.examples.takagi_sugeno.matlab.slcp
fuzzylite.examples.takagi_sugeno.matlab.slcp1
fuzzylite.examples.takagi_sugeno.matlab.slcpp1
fuzzylite.examples.takagi_sugeno.matlab.sltbu_fl
fuzzylite.examples.takagi_sugeno.matlab.sugeno1
fuzzylite.examples.takagi_sugeno.matlab.tanksg
fuzzylite.examples.takagi_sugeno.matlab.tippersg
fuzzylite.examples.takagi_sugeno.octave.cubic_approximator
fuzzylite.examples.takagi_sugeno.octave.heart_disease_risk
fuzzylite.examples.takagi_sugeno.octave.linear_tip_calculator
fuzzylite.examples.takagi_sugeno.octave.sugeno_tip_calculator
fuzzylite.examples.terms.Bell
fuzzylite.examples.terms.Binary
fuzzylite.examples.terms.Concave
fuzzylite.examples.terms.Constant
fuzzylite.examples.terms.Cosine
fuzzylite.examples.terms.Discrete
fuzzylite.examples.terms.Function
fuzzylite.examples.terms.Gaussian
fuzzylite.examples.terms.GaussianProduct
fuzzylite.examples.terms.Linear
fuzzylite.examples.terms.PiShape
fuzzylite.examples.terms.Ramp
fuzzylite.examples.terms.Rectangle
fuzzylite.examples.terms.Sigmoid
fuzzylite.examples.terms.SigmoidDifference
fuzzylite.examples.terms.SigmoidProduct
fuzzylite.examples.terms.Spike
fuzzylite.examples.terms.Trapezoid
fuzzylite.examples.terms.Triangle
fuzzylite.examples.terms.ZSShape
fuzzylite.examples.tsukamoto.tsukamoto
""".split())

        obtained = set()
        for file_py in examples.rglob("*.py"):
            if file_py.suffix == ".py" and file_py.name != '__init__.py':
                package = []
                for parent in file_py.parents:
                    package.append(parent.stem)
                    if parent.stem == "fuzzylite":
                        break
                module = ".".join(reversed(package)) + f".{file_py.stem}"
                obtained.add(module)
        self.assertSetEqual(expected, obtained)

        for module in obtained:
            logger.info(f"Importing: {module}")
            # if an example is incorrect, an exception will be thrown below
            engine = importlib.import_module(module).engine  # type: ignore
            logger.info(str(engine))
