"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import unittest
from typing import cast

import black
import numpy as np

import fuzzylite as fl
from fuzzylite.examples.mamdani.simple_dimmer import SimpleDimmer
from fuzzylite.types import Self
from tests.assert_component import BaseAssert


class EngineAssert(BaseAssert[fl.Engine]):
    """Engine assert."""

    def has_type(
        self,
        expected: fl.Engine.Type | set[fl.Engine.Type],
        /,
        reasons: list[str] | None = None,
    ) -> Self:
        """Asserts the engine has the expected type."""
        obtained_reasons: list[str] = []
        inferred_type = self.actual.infer_type(obtained_reasons)

        if isinstance(expected, fl.Engine.Type):
            expected = {expected}
        self.test.assertIn(
            inferred_type,
            expected,
            f"expected engine type in {expected}, but found {type}",
        )
        if reasons is not None:
            self.test.assertEqual(obtained_reasons, reasons)
        return self

    def is_ready(self, expected: bool = True, /, reasons: list[str] | None = None) -> Self:
        """Test engine is ready."""
        obtained_reasons: list[str] = []
        obtained = self.actual.is_ready(obtained_reasons)
        self.test.assertEqual(expected, obtained)
        if reasons is not None:
            self.test.assertEqual(reasons, obtained_reasons)
        return self

    def has_n_inputs(self, n: int) -> Self:
        """Asserts the engine has the expected number of input variables."""
        n_inputs = len(self.actual.input_variables)
        self.test.assertEqual(n_inputs, n, f"expected {n} input variables, but found {n_inputs}")
        return self

    def has_inputs(self, names: list[str]) -> Self:
        """Asserts the engine has the expected input variables by name."""
        self.test.assertSequenceEqual([iv.name for iv in self.actual.input_variables], names)
        return self

    def has_n_outputs(self, n: int) -> Self:
        """Asserts the engine has the expected number of output variables."""
        n_outputs = len(self.actual.output_variables)
        self.test.assertEqual(n_outputs, n, f"expected {n} output variables, but found {n_outputs}")
        return self

    def has_outputs(self, names: list[str]) -> Self:
        """Asserts the engine has the expected output variables by name."""
        self.test.assertSequenceEqual([ov.name for ov in self.actual.output_variables], names)
        return self

    def has_n_blocks(self, n: int) -> Self:
        """Asserts the engine has the expected number of rule blocks."""
        n_blocks = len(self.actual.rule_blocks)
        self.test.assertEqual(n_blocks, n, f"expected {n} rule blocks, but found {n_blocks}")
        return self

    def has_blocks(self, names: list[str]) -> Self:
        """Asserts the engine has the expected number of rule blocks by name."""
        self.test.assertSequenceEqual([rb.name for rb in self.actual.rule_blocks], names)
        return self

    def when_input_values(self, x: fl.ScalarArray, /, raises: Exception | None = None) -> Self:
        """Sets the input values of the engine."""
        if raises:
            with self.test.assertRaises(type(raises)) as error:
                self.actual.input_values = x
            self.test.assertEqual(str(error.exception), str(raises))
        else:
            self.actual.input_values = x
        return self

    def then_input_variables(self, x: dict[int | str, fl.Scalar]) -> Self:
        """Sets the input variable of the engine."""
        for key, value in x.items():
            np.testing.assert_allclose(value, self.actual.input_variable(key).value)

        expected = np.atleast_2d([np.atleast_1d(value) for value in x.values()]).T
        obtained = self.actual.input_values
        np.testing.assert_allclose(expected, obtained)
        return self

    def evaluate_fld(self, fld: str, decimals: int) -> Self:
        """Asserts the engine produces the expected fld."""
        for line, evaluation in enumerate(fld.split("\n")):
            comment_index = evaluation.find("#")
            if comment_index != -1:
                evaluation = evaluation[:comment_index]
            if not evaluation:
                continue

            expected = fl.array([x for x in evaluation.split()], dtype=np.float64)
            if len(expected) != len(self.actual.variables):
                raise ValueError(
                    f"expected {len(self.actual.variables)} values, "
                    f"but got {len(expected)}: [line: {line}] {evaluation}"
                )

            for i, input_variable in enumerate(self.actual.input_variables):
                input_variable.value = expected[i]

            self.actual.process()

            obtained = np.hstack(
                (
                    self.actual.input_values.flatten(),
                    self.actual.output_values.flatten(),
                )
            )

            np.testing.assert_allclose(
                obtained, expected, rtol=fl.settings.rtol, atol=10 ** (-decimals)
            )
        return self


class TestEngine(unittest.TestCase):
    """Tests the engine."""

    def test_empty_engine(self) -> None:
        """Tests the empty engine."""
        flc = fl.Engine("name", "description")
        EngineAssert(self, flc).has_name("name").has_description("description").has_n_inputs(
            0
        ).has_inputs([]).has_n_outputs(0).has_outputs([]).has_n_blocks(0).has_blocks([])

    def test_engine(self) -> None:
        """Tests a basic engine."""
        engine = fl.Engine()
        engine.name = "tipper"
        engine.description = "(service and food) -> (tip)"

        service = fl.InputVariable()
        service.name = "service"
        service.description = "quality of service"
        service.enabled = True
        service.range = (0.000, 10.000)
        service.lock_range = True
        service.terms.append(fl.Trapezoid("poor", 0.000, 0.000, 2.500, 5.000))
        service.terms.append(fl.Triangle("good", 2.500, 5.000, 7.500))
        service.terms.append(fl.Trapezoid("excellent", 5.000, 7.500, 10.000, 10.000))
        engine.input_variables.append(service)

        food = fl.InputVariable()
        food.name = "food"
        food.description = "quality of food"
        food.enabled = True
        food.range = (0.000, 10.000)
        food.lock_range = True
        food.terms.append(fl.Trapezoid("rancid", 0.000, 0.000, 2.500, 7.500))
        food.terms.append(fl.Trapezoid("delicious", 2.500, 7.500, 10.000, 10.000))
        engine.input_variables.append(food)

        mTip = fl.OutputVariable()  # noqa N806 should be lowercase
        mTip.name = "mTip"
        mTip.description = "tip based on Mamdani inference"
        mTip.enabled = True
        mTip.range = (0.000, 30.000)
        mTip.lock_range = False
        mTip.aggregation = fl.Maximum()
        mTip.defuzzifier = fl.Centroid(100)
        mTip.default_value = fl.nan
        mTip.lock_previous = False
        mTip.terms.append(fl.Triangle("cheap", 0.000, 5.000, 10.000))
        mTip.terms.append(fl.Triangle("average", 10.000, 15.000, 20.000))
        mTip.terms.append(fl.Triangle("generous", 20.000, 25.000, 30.000))
        engine.output_variables.append(mTip)

        tsTip = fl.OutputVariable()  # noqa N806 should be lowercase
        tsTip.name = "tsTip"
        tsTip.description = "tip based on Takagi-Sugeno inference"
        tsTip.enabled = True
        tsTip.range = (0.000, 30.000)
        tsTip.lock_range = False
        tsTip.aggregation = None
        tsTip.defuzzifier = fl.WeightedAverage("TakagiSugeno")
        tsTip.default_value = fl.nan
        tsTip.lock_previous = False
        tsTip.terms.append(fl.Constant("cheap", 5.000))
        tsTip.terms.append(fl.Constant("average", 15.000))
        tsTip.terms.append(fl.Constant("generous", 25.000))
        engine.output_variables.append(tsTip)

        mamdani = fl.RuleBlock()
        mamdani.name = "mamdani"
        mamdani.description = "Mamdani inference"
        mamdani.enabled = True
        mamdani.conjunction = fl.AlgebraicProduct()
        mamdani.disjunction = fl.AlgebraicSum()
        mamdani.implication = fl.Minimum()
        mamdani.activation = fl.General()
        mamdani.rules.append(
            fl.Rule.create("if service is poor or food is rancid then mTip is cheap", engine)
        )
        mamdani.rules.append(fl.Rule.create("if service is good then mTip is average", engine))
        mamdani.rules.append(
            fl.Rule.create(
                "if service is excellent or food is delicious then mTip is generous with 0.5",
                engine,
            )
        )
        mamdani.rules.append(
            fl.Rule.create(
                "if service is excellent and food is delicious then mTip is generous with 1.0",
                engine,
            )
        )
        engine.rule_blocks.append(mamdani)

        takagiSugeno = fl.RuleBlock()  # noqa N806 should be lowercase
        takagiSugeno.name = "takagiSugeno"
        takagiSugeno.description = "Takagi-Sugeno inference"
        takagiSugeno.enabled = True
        takagiSugeno.conjunction = fl.AlgebraicProduct()
        takagiSugeno.disjunction = fl.AlgebraicSum()
        takagiSugeno.implication = None
        takagiSugeno.activation = fl.General()
        takagiSugeno.rules.append(
            fl.Rule.create("if service is poor or food is rancid then tsTip is cheap", engine)
        )
        takagiSugeno.rules.append(
            fl.Rule.create("if service is good then tsTip is average", engine)
        )
        takagiSugeno.rules.append(
            fl.Rule.create(
                "if service is excellent or food is delicious then tsTip is generous with 0.5",
                engine,
            )
        )
        takagiSugeno.rules.append(
            fl.Rule.create(
                "if service is excellent and food is delicious then tsTip is generous with 1.0",
                engine,
            )
        )
        engine.rule_blocks.append(takagiSugeno)

        EngineAssert(self, engine).has_name("tipper").has_description(
            "(service and food) -> (tip)"
        ).has_n_inputs(2).has_inputs(["service", "food"]).has_n_outputs(2).has_outputs(
            ["mTip", "tsTip"]
        ).has_n_blocks(
            2
        ).has_blocks(
            ["mamdani", "takagiSugeno"]
        ).evaluate_fld(
            """\
#service food mTip tsTip
0.0000000000000 0.0000000000000 4.9989502099580 5.0000000000000
0.0000000000000 3.3333333333333 7.7561896551724 6.5384615384615
0.0000000000000 6.6666666666666 12.9489036144578 10.8823529411764
0.0000000000000 10.0000000000000 13.5707062050051 11.6666666666666
3.3333333333333 0.0000000000000 8.5688247396168 7.5000000000000
3.3333333333333 3.3333333333333 10.1101355034654 8.6734693877551
3.3333333333333 6.6666666666666 13.7695060342408 12.9245283018867
3.3333333333333 10.0000000000000 14.3676481312670 13.8888888888888
6.6666666666666 0.0000000000000 12.8954528230390 11.0000000000000
6.6666666666666 3.3333333333333 13.2040624705105 12.7966101694915
6.6666666666666 6.6666666666666 17.9862390284958 20.6363636363636
6.6666666666666 10.0000000000000 21.1557340720221 22.7777777777777
10.0000000000000 0.0000000000000 13.5707062050051 11.6666666666666
10.0000000000000 3.3333333333333 13.7092196934510 13.8888888888888
10.0000000000000 6.6666666666666 20.2157800031293 22.7777777777777
10.0000000000000 10.0000000000000 25.0010497900419 25.0000000000000
""",
            decimals=12,
        )

    @unittest.skip("Not implemented yet")
    def test_engine_from_fll(self) -> None:
        """Not implemented yet."""
        pass

    def test_inputs(self) -> None:
        """Tests the input variables of an engine."""
        flc = fl.Engine("name", "description", [fl.InputVariable("A"), fl.InputVariable("B")])
        EngineAssert(self, flc).has_name("name").has_description("description").has_n_inputs(
            2
        ).has_inputs(["A", "B"])

        flc.input_variables = []
        EngineAssert(self, flc).has_n_inputs(0).has_inputs([])

        flc.input_variables = [
            fl.InputVariable("X"),
            fl.InputVariable("Y"),
            fl.InputVariable("Z"),
        ]
        EngineAssert(self, flc).has_n_inputs(3).has_inputs(["X", "Y", "Z"])

        names = ["X", "Y", "Z"]
        for i, iv in enumerate(flc.input_variables):
            self.assertEqual(iv.name, names[i])

    def test_outputs(self) -> None:
        """Tests the output variables of an engine."""
        flc = fl.Engine("name", "description", [], [fl.OutputVariable("A"), fl.OutputVariable("B")])
        EngineAssert(self, flc).has_name("name").has_description("description").has_n_outputs(
            2
        ).has_outputs(["A", "B"])

        flc.output_variables = []
        EngineAssert(self, flc).has_n_outputs(0).has_outputs([])

        flc.output_variables = [
            fl.OutputVariable("X"),
            fl.OutputVariable("Y"),
            fl.OutputVariable("Z"),
        ]
        EngineAssert(self, flc).has_n_outputs(3).has_outputs(["X", "Y", "Z"])

        names = ["X", "Y", "Z"]
        for i, iv in enumerate(flc.output_variables):
            self.assertEqual(iv.name, names[i])

    def test_values(self) -> None:
        """Test the engine exports the input and output values."""
        engine = fl.Engine(
            name="test",
            input_variables=[fl.InputVariable("A")],
            output_variables=[fl.OutputVariable("Z")],
        )
        engine.input_variable("A").value = fl.array([1, 2, 3])
        engine.output_variable("Z").value = fl.array([9, 8, 7])
        np.testing.assert_allclose(
            fl.scalar(
                [
                    [1, 9],
                    [2, 8],
                    [3, 7],
                ]
            ),
            engine.values,
        )
        engine.input_variable("A").value = fl.nan
        engine.output_variable("Z").value = fl.nan
        np.testing.assert_allclose(fl.array([[fl.nan, fl.nan]]), engine.values)

    def test_input_values_setter(self) -> None:
        """Tests the setter of input values through the engine."""
        EngineAssert(
            self, fl.Engine("1 input", input_variables=[fl.InputVariable("A")])
        ).when_input_values(fl.array(1.0)).then_input_variables({"A": 1.0}).when_input_values(
            fl.array([1, 2, 3, 4])
        ).then_input_variables(
            {"A": fl.array([1, 2, 3, 4])}
        )

        # Two inputs
        e2 = EngineAssert(
            self,
            fl.Engine(
                "2 inputs",
                input_variables=[fl.InputVariable("A"), fl.InputVariable("B")],
            ),
        )
        ## Single value
        e2.when_input_values(fl.array(1.0)).then_input_variables({"A": 1.0, "B": 1.0})
        ## 1D array
        e2.when_input_values(fl.array([1, -1])).then_input_variables({"A": 1.0, "B": -1.0})
        ## 2D array
        e2.when_input_values(
            fl.array(
                [
                    [1, -1],
                    [2, -2],
                    [3, -3],
                ]
            )
        ).then_input_variables({"A": fl.array([1, 2, 3]), "B": fl.array([-1, -2, -3])})

        ## Errors:
        EngineAssert(self, fl.Engine()).when_input_values(
            fl.array(1.0),
            raises=RuntimeError("can't set input values to an engine without input variables"),
        )
        e2.when_input_values(
            fl.array([[[1.0]]]),
            raises=ValueError(
                "expected a 0d-array (single value), 1d-array (vector), or 2d-array (matrix), "
                "but got a 3d-array: [[[1.]]]"
            ),
        )
        e2.when_input_values(
            fl.array([[1.0, 2.0, 3.0]]),
            raises=ValueError(
                "expected an array with 2 columns (one for each input variable), "
                "but got 3 columns: [[1. 2. 3.]]"
            ),
        )

    def test_input_values(self) -> None:
        """Test the engine.input_values."""
        engine = fl.Engine("Test")
        np.testing.assert_equal(fl.array([]), engine.input_values)

        engine = fl.Engine("Test", input_variables=[fl.InputVariable("A")])
        engine.input_variable(0).value = 0.6
        np.testing.assert_equal(fl.array([[0.6]]), engine.input_values)

        engine.input_variable(0).value = fl.array([0.6, 0.7])
        np.testing.assert_equal(fl.array([[0.6], [0.7]]), engine.input_values)

        engine = fl.Engine("Test", input_variables=[fl.InputVariable("A"), fl.InputVariable("B")])
        engine.input_variable(0).value = 0.6
        engine.input_variable(1).value = 0.2
        np.testing.assert_equal(fl.array([[0.6, 0.2]]), engine.input_values)

        engine.input_variable(0).value = fl.array([0.6, 0.7])
        engine.input_variable(1).value = fl.array([0.2, 0.3])
        np.testing.assert_equal(fl.array([[0.6, 0.2], [0.7, 0.3]]), engine.input_values)

    def test_output_values(self) -> None:
        """Test the engine.output_values."""
        engine = fl.Engine("Test")
        np.testing.assert_equal(fl.array([]), engine.output_values)

        engine = fl.Engine("Test", output_variables=[fl.OutputVariable("A")])
        engine.output_variable(0).value = 0.6
        np.testing.assert_equal(fl.array([[0.6]]), engine.output_values)

        engine.output_variable(0).value = fl.array([0.6, 0.7])
        np.testing.assert_equal(fl.array([[0.6], [0.7]]), engine.output_values)

        engine = fl.Engine(
            "Test", output_variables=[fl.OutputVariable("A"), fl.OutputVariable("B")]
        )
        engine.output_variable(0).value = 0.6
        engine.output_variable(1).value = 0.2
        np.testing.assert_equal(fl.array([[0.6, 0.2]]), engine.output_values)

        engine.output_variable(0).value = fl.array([0.6, 0.7])
        engine.output_variable(1).value = fl.array([0.2, 0.3])
        np.testing.assert_equal(fl.array([[0.6, 0.2], [0.7, 0.3]]), engine.output_values)

    def test_input_values_manual_bug(self) -> None:
        """Test the bug raised in https://github.com/fuzzylite/pyfuzzylite/issues/75."""
        fll = """\
Engine: ObstacleAvoidance
InputVariable: obstacle
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: left Ramp 1.000 0.000
  term: right Ramp 0.000 1.000
InputVariable: obstacle2
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: left Ramp 1.000 0.000
  term: right Ramp 0.000 1.000
OutputVariable: tsSteer
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: WeightedAverage
  default: nan
  lock-previous: false
  term: right Linear 1 1 0
  term: left Linear 0 1 1
RuleBlock: takagiSugeno
  enabled: true
  conjunction: Minimum
  disjunction: none
  implication: none
  activation: General
  rule: if obstacle is left and obstacle2 is left then tsSteer is right
  rule: if obstacle is left and obstacle2 is right then tsSteer is right"""
        engine = fl.FllImporter().from_string(fll)
        engine.input_variable(0).value = 0.2
        engine.input_variable(1).value = 0.6
        engine.process()
        np.testing.assert_allclose(0.8, engine.output_values)
        np.testing.assert_allclose(fl.array([[0.2, 0.6, 0.8]]), engine.values)

        engine.input_variable(0).value = fl.array([0.2, 0.2])
        engine.input_variable(1).value = fl.array([0.6, 0.6])
        engine.process()
        np.testing.assert_allclose(fl.array([[0.8], [0.8]]), engine.output_values)
        np.testing.assert_allclose(fl.array([[0.2, 0.6, 0.8], [0.2, 0.6, 0.8]]), engine.values)

    def test_repr(self) -> None:
        """Tests repr."""
        code = fl.repr(SimpleDimmer().engine)
        engine = eval(code)
        engine.input_variables[0].value = 1 / 3
        engine.process()
        np.testing.assert_allclose(
            engine.output_variables[0].value,
            0.659,
            atol=fl.settings.atol,
            rtol=fl.settings.rtol,
        )

    def test_takagi_sugeno_aggregation_operator(self) -> None:
        """Test to understand why we are aggregating same terms before defuzzifying."""
        fll = """\
Engine: problem1
InputVariable: in1
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: term Ramp 0.000 1.000
InputVariable: in2
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: term Ramp 0.000 1.000
InputVariable: in3
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: term Ramp 0.000 1.000
OutputVariable: output
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: BoundedSum
  defuzzifier: WeightedAverage Automatic
  default: nan
  lock-previous: false
  term: negative Constant -1.000
  term: positive Constant 1.000
RuleBlock:
  enabled: true
  conjunction: none
  disjunction: none
  implication: none
  activation: General
  rule: if in1 is term then output is positive
  rule: if in2 is term then output is negative
  rule: if in3 is term then output is positive
"""
        engine = fl.FllImporter().from_string(fll)
        engine.input_values = fl.array([0.75, 1.0, 0.75])
        engine.process()
        self.assertEqual(
            "1.000/negative + 1.000/positive", engine.output_variable(0).fuzzy_value().item()
        )
        # new behaviour
        np.testing.assert_allclose(
            0,
            engine.output_variable(0).value,
            rtol=fl.settings.rtol,
            atol=fl.settings.atol,
        )
        # old behaviour:
        engine.output_variable(0).aggregation = fl.UnboundedSum()
        engine.process()
        np.testing.assert_allclose(0.2, engine.output_variable(0).value)

    @unittest.skip(reason="__getattr__ has not been implemented properly yet.")
    def test_getattr(self) -> None:
        """Test components are accessible using `engine.component` style."""
        # Regular usage
        engine = SimpleDimmer().engine
        engine.rule_block(0).name = "Dimmer"
        engine.Ambient.value = fl.array([0.0, 0.25, 0.5, 0.75, 1.0])  # type: ignore
        engine.Dimmer.activate()  # type: ignore
        engine.Power.defuzzify()  # type: ignore
        np.testing.assert_allclose(
            [fl.nan, 0.75, 0.5, 0.25, fl.nan],
            engine.Power.value,  # type: ignore
        )

        # Non-existing component by name
        with self.assertRaises(AttributeError) as error:
            engine.Variable  # type: ignore # noqa: B018
        self.assertEqual("'Engine' object has no attribute 'Variable'", str(error.exception))

        # engine's members are retrieved first
        engine.name = "Test my name"
        engine.input_variables.append(fl.InputVariable("name"))
        self.assertEqual("Test my name", engine.name)

    def test_getitem(self) -> None:
        """Test components are accessible using `engine.component` style."""
        # Regular usage
        engine = SimpleDimmer().engine
        engine.rule_block(0).name = "Dimmer"
        engine["Ambient"].value = fl.array([0.0, 0.25, 0.5, 0.75, 1.0])  # type: ignore
        engine["Dimmer"].activate()  # type: ignore
        engine["Power"].defuzzify()  # type: ignore
        np.testing.assert_allclose(
            [fl.nan, 0.75, 0.5, 0.25, fl.nan],
            engine["Power"].value,  # type: ignore
        )

        # Non-existing component by name
        with self.assertRaises(ValueError) as error:
            engine["Variable"]
        self.assertEqual(
            str(error.exception),
            "engine 'SimpleDimmer' does not have a component named 'Variable'",
        )

        # engine's members are retrieved first, except in this case
        engine.name = "Test my name"
        expected = fl.InputVariable("name")
        engine.input_variables.append(expected)
        self.assertEqual(expected, engine["name"])
        self.assertNotEqual(engine.name, engine["name"])

        # side effect caught by type annotations
        self.assertEqual(engine.input_variable(0), engine[0])  # type:ignore

    def test_is_ready(self) -> None:
        """Test the engine is ready."""
        for engine in fl.Op.glob_examples("engine"):
            EngineAssert(self, engine).is_ready()

        EngineAssert(self, fl.Engine("test")).is_ready(
            False,
            [
                "Engine 'test' does not have any input variables",
                "Engine 'test' does not have any output variables",
                "Engine 'test' does not have any rule blocks",
            ],
        )

        EngineAssert(
            self,
            fl.Engine(
                "test",
                input_variables=[fl.InputVariable("A")],
                output_variables=[fl.OutputVariable("Z")],
                rule_blocks=[fl.RuleBlock("R")],
            ),
        ).is_ready(
            False,
            [
                "Output variable 'Z' does not have any terms",
                "Output variable 'Z' does not have any defuzzifier",
                "Rule block 'R' does not have any rules",
            ],
        )

        EngineAssert(
            self,
            fl.Engine(
                "test",
                input_variables=[fl.InputVariable("A", terms=[fl.Arc("a", 1, 0)])],
                output_variables=[
                    fl.OutputVariable("Z", terms=[fl.Arc("z", 0, 1)], defuzzifier=fl.Centroid()),
                ],
                rule_blocks=[
                    fl.RuleBlock(
                        "R",
                        rules=[
                            fl.Rule.create("if A is a then Z is z"),
                            fl.Rule.create("if A is a and Z is z then Z is z"),
                            fl.Rule.create("if A is a or Z is z then Z is z"),
                        ],
                    )
                ],
            ),
        ).is_ready(
            False,
            [
                "Output variable 'Z' does not have any aggregation operator",
                "Rule block 'R' does not have any conjunction operator and is needed by 1 rule",
                "Rule block 'R' does not have any disjunction operator and is needed by 1 rule",
                "Rule block 'R' does not have any implication operator and is needed by 3 rules",
            ],
        )

        EngineAssert(
            self,
            fl.Engine(
                "test",
                input_variables=[fl.InputVariable("A", terms=[fl.Arc("a", 1, 0)])],
                output_variables=[
                    fl.OutputVariable("Z", terms=[fl.Arc("z", 0, 1)], defuzzifier=fl.WeightedSum()),
                ],
                rule_blocks=[
                    fl.RuleBlock(
                        "R",
                        rules=[
                            fl.Rule.create("if A is a then Z is z"),
                            fl.Rule.create("if A is a and Z is z then Z is z"),
                            fl.Rule.create("if A is a or Z is z then Z is z"),
                        ],
                    )
                ],
            ),
        ).is_ready(
            False,
            [
                "Rule block 'R' does not have any conjunction operator and is needed by 1 rule",
                "Rule block 'R' does not have any disjunction operator and is needed by 1 rule",
            ],
        )

    def test_engine_type(self) -> None:
        """Test engine inferred types."""
        # Unknown
        EngineAssert(self, fl.Engine("test")).has_type(
            fl.Engine.Type.Unknown, ["Engine 'test' does not have any output variables"]
        )

        EngineAssert(self, fl.Engine("test", output_variables=[fl.OutputVariable("Z")])).has_type(
            fl.Engine.Type.Unknown,
            ["One or more output variables do not have a defuzzifier"],
        )

        # Mamdani
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[fl.OutputVariable("Z", defuzzifier=fl.Centroid())],
            ),
        ).has_type(fl.Engine.Type.Mamdani, ["Output variables have integral defuzzifiers"])

        # Larsen
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[fl.OutputVariable("Z", defuzzifier=fl.Centroid())],
                rule_blocks=[fl.RuleBlock("R", implication=fl.AlgebraicProduct())],
            ),
        ).has_type(
            fl.Engine.Type.Larsen,
            [
                "Output variables have integral defuzzifiers",
                "Implication in rule blocks is the AlgebraicProduct",
            ],
        )

        # Takagi-Sugeno
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[
                    fl.OutputVariable(
                        "Z",
                        terms=[fl.Constant("", 1.0)],
                        defuzzifier=fl.WeightedSum(),
                    )
                ],
            ),
        ).has_type(
            fl.Engine.Type.TakagiSugeno,
            [
                "Output variables have weighted defuzzifiers",
                "Output variables only have Constant, Linear, or Function terms",
            ],
        )

        # Tsukamoto
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[
                    fl.OutputVariable(
                        "Z",
                        terms=[fl.Arc("", 0, 1)],
                        defuzzifier=fl.WeightedAverage(),
                    )
                ],
            ),
        ).has_type(
            fl.Engine.Type.Tsukamoto,
            [
                "Output variables have weighted defuzzifiers",
                "Output variables only have monotonic terms",
            ],
        )

        # Inverse Tsukamoto
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[
                    fl.OutputVariable(
                        "Z",
                        terms=[fl.Triangle()],
                        defuzzifier=fl.WeightedSum(),
                    )
                ],
            ),
        ).has_type(
            fl.Engine.Type.InverseTsukamoto,
            [
                "Output variables have weighted defuzzifiers",
                "Output variables have non-monotonic terms",
                "Output variables have terms different from Constant, Linear, or Function terms",
            ],
        )

        # Hybrids
        EngineAssert(
            self,
            fl.Engine(
                "test",
                output_variables=[
                    fl.OutputVariable(
                        "Z1",
                        terms=[fl.Triangle()],
                        defuzzifier=fl.Centroid(),
                    ),
                    fl.OutputVariable("Z2", terms=[fl.Linear()], defuzzifier=fl.WeightedAverage()),
                ],
            ),
        ).has_type(
            fl.Engine.Type.Hybrid,
            ["Output variables have different types of defuzzifiers"],
        )

    def test_engine_type_from_examples(self) -> None:
        """Test types of engines from examples."""
        # Mamdani
        for engine in fl.Op.glob_examples("engine", fl.examples.mamdani):
            EngineAssert(self, engine).has_type({fl.Engine.Type.Mamdani, fl.Engine.Type.Larsen})
        # TakagiSugeno
        for engine in fl.Op.glob_examples("engine", fl.examples.takagi_sugeno):
            EngineAssert(self, engine).has_type(fl.Engine.Type.TakagiSugeno)

        # Tsukamoto
        for engine in fl.Op.glob_examples("engine", fl.examples.tsukamoto):
            EngineAssert(self, engine).has_type(fl.Engine.Type.Tsukamoto)

        # Hybrid
        for engine in fl.Op.glob_examples("engine", fl.examples.hybrid):
            EngineAssert(self, engine).has_type(fl.Engine.Type.Hybrid)

        # Mamdani or TakagiSugeno
        for engine in fl.Op.glob_examples("engine", fl.examples.terms):
            EngineAssert(self, engine).has_type(
                {fl.Engine.Type.Mamdani, fl.Engine.Type.TakagiSugeno}
            )

    @unittest.expectedFailure
    def test_copy_takagi_sugeno(self) -> None:
        """Test engine can do shallow copy of a Mamdani engine.

        Expected failure because a shallow copy of an engine will contain internal references to the original engine
        (eg, Linear terms referencing original engine).
        """
        import copy

        engine = fl.examples.terms.linear.Linear().engine
        expected = (
            "fl.Engine(\n"
            '    name="Linear",\n'
            '    description="obstacle avoidance for self-driving cars",\n'
            "    input_variables=[\n"
            "        fl.InputVariable(\n"
            '            name="obstacle",\n'
            '            description="location of obstacle relative to vehicle",\n'
            "            minimum=0.0,\n"
            "            maximum=1.0,\n"
            "            lock_range=False,\n"
            "            terms=[\n"
            '                fl.Triangle("left", 0.0, 0.333, 0.666),\n'
            '                fl.Triangle("right", 0.333, 0.666, 1.0),\n'
            "            ],\n"
            "        )\n"
            "    ],\n"
            "    output_variables=[\n"
            "        fl.OutputVariable(\n"
            '            name="steer",\n'
            '            description="direction to steer the vehicle to",\n'
            "            minimum=0.0,\n"
            "            maximum=1.0,\n"
            "            lock_range=False,\n"
            "            lock_previous=False,\n"
            "            default_value=fl.nan,\n"
            "            aggregation=None,\n"
            '            defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),\n'
            '            terms=[fl.Linear("left", [0.0, 0.333]), fl.Linear("right", [0.0, '
            "0.666])],\n"
            "        )\n"
            "    ],\n"
            "    rule_blocks=[\n"
            "        fl.RuleBlock(\n"
            '            name="steer_away",\n'
            '            description="steer away from obstacles",\n'
            "            conjunction=None,\n"
            "            disjunction=None,\n"
            "            implication=None,\n"
            "            activation=fl.General(),\n"
            "            rules=[\n"
            '                fl.Rule.create("if obstacle is left then steer is right"),\n'
            '                fl.Rule.create("if obstacle is right then steer is left"),\n'
            "            ],\n"
            "        )\n"
            "    ],\n"
            ")\n"
        )
        self.assertEqual(
            expected,
            black.format_str(
                repr(engine),
                mode=black.Mode(),
            ),
        )

        engine_copy = copy.copy(engine)

        self.assertEqual(
            expected,
            black.format_str(
                repr(engine_copy),
                mode=black.Mode(),
            ),
        )

        for variables in zip(engine.variables, engine_copy.variables):
            self.assertEqual(*variables)

        for rule_blocks in zip(engine.rule_blocks, engine_copy.rule_blocks):
            self.assertEqual(*rule_blocks)

        for output_variable in engine_copy.output_variables:
            for term in output_variable.terms:
                self.assertEqual(fl.Linear, term.__class__)
                self.assertEqual(id(engine_copy), id(cast(fl.Linear, term).engine))

    def test_deep_copy_takagi_sugeno(self) -> None:
        """Test engine can do shallow copy of a Mamdani engine."""
        engine = fl.examples.terms.linear.Linear().engine
        for output_variable in engine.output_variables:
            for term in output_variable.terms:
                output_variable.fuzzy.terms.append(fl.Activated(term, 1.0))

        expected = (
            "fl.Engine(\n"
            '    name="Linear",\n'
            '    description="obstacle avoidance for self-driving cars",\n'
            "    input_variables=[\n"
            "        fl.InputVariable(\n"
            '            name="obstacle",\n'
            '            description="location of obstacle relative to vehicle",\n'
            "            minimum=0.0,\n"
            "            maximum=1.0,\n"
            "            lock_range=False,\n"
            "            terms=[\n"
            '                fl.Triangle("left", 0.0, 0.333, 0.666),\n'
            '                fl.Triangle("right", 0.333, 0.666, 1.0),\n'
            "            ],\n"
            "        )\n"
            "    ],\n"
            "    output_variables=[\n"
            "        fl.OutputVariable(\n"
            '            name="steer",\n'
            '            description="direction to steer the vehicle to",\n'
            "            minimum=0.0,\n"
            "            maximum=1.0,\n"
            "            lock_range=False,\n"
            "            lock_previous=False,\n"
            "            default_value=fl.nan,\n"
            "            aggregation=None,\n"
            '            defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),\n'
            '            terms=[fl.Linear("left", [0.0, 0.333]), fl.Linear("right", [0.0, '
            "0.666])],\n"
            "        )\n"
            "    ],\n"
            "    rule_blocks=[\n"
            "        fl.RuleBlock(\n"
            '            name="steer_away",\n'
            '            description="steer away from obstacles",\n'
            "            conjunction=None,\n"
            "            disjunction=None,\n"
            "            implication=None,\n"
            "            activation=fl.General(),\n"
            "            rules=[\n"
            '                fl.Rule.create("if obstacle is left then steer is right"),\n'
            '                fl.Rule.create("if obstacle is right then steer is left"),\n'
            "            ],\n"
            "        )\n"
            "    ],\n"
            ")\n"
        )
        self.assertEqual(
            expected,
            black.format_str(
                repr(engine),
                mode=black.Mode(),
            ),
        )

        engine_copy = engine.copy()

        self.assertEqual(
            expected,
            black.format_str(
                repr(engine_copy),
                mode=black.Mode(),
            ),
        )

        for variables in zip(engine.variables, engine_copy.variables):
            self.assertEqual(repr(variables[0]), repr(variables[1]))
            self.assertTrue(id(variables[0]) != id(variables[1]))

        for rule_blocks in zip(engine.rule_blocks, engine_copy.rule_blocks):
            self.assertEqual(repr(rule_blocks[0]), repr(rule_blocks[1]))
            self.assertTrue(id(rule_blocks[0]) != id(rule_blocks[1]))

        self.assertTrue(engine_copy.output_variables)
        for output_variable in engine_copy.output_variables:
            self.assertTrue(output_variable.terms)
            for term in output_variable.terms:
                self.assertEqual(fl.Linear, term.__class__)
                self.assertEqual(id(engine_copy), id(cast(fl.Linear, term).engine))
            self.assertTrue(output_variable.fuzzy.terms)
            for index, activated in enumerate(output_variable.fuzzy.terms):
                self.assertEqual(id(activated.term), id(output_variable.term(index)))
                self.assertEqual(id(engine_copy), id(cast(fl.Linear, activated.term).engine))


if __name__ == "__main__":
    unittest.main()
