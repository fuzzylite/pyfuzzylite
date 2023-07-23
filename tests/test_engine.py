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

import unittest

import numpy as np
from typing_extensions import Self

import fuzzylite as fl
from fuzzylite.examples.mamdani.simple_dimmer import SimpleDimmer
from tests.assert_component import BaseAssert


class EngineAssert(BaseAssert[fl.Engine]):
    """Engine assert."""

    def has_type(self, expected: fl.Engine.Type) -> EngineAssert:
        """Asserts the engine has the expectd type."""
        type = self.actual.infer_type()
        self.test.assertEqual(
            type, expected, f"expected engine of type {expected}, but found {type}"
        )
        return self

    def is_ready(self, expected: bool, status: str = "") -> EngineAssert:
        """Asserts whether the engine is ready and its status."""
        ready, message = self.actual.is_ready()
        self.test.assertEqual(
            ready,
            expected,
            (
                f"expected engine {'*not*' if not expected else ''} to be ready,"
                f"but was {'*not*' if not ready else ''} ready"
            ),
        )
        self.test.assertEqual(message, status)
        return self

    def has_n_inputs(self, n: int) -> EngineAssert:
        """Asserts the engine has the expected number of input variables."""
        n_inputs = len(self.actual.input_variables)
        self.test.assertEqual(
            n_inputs, n, f"expected {n} input variables, but found {n_inputs}"
        )
        return self

    def has_inputs(self, names: list[str]) -> EngineAssert:
        """Asserts the engine has the expected input variables by name."""
        self.test.assertSequenceEqual(
            [iv.name for iv in self.actual.input_variables], names
        )
        return self

    def has_n_outputs(self, n: int) -> EngineAssert:
        """Asserts the engine has the expected number of output variables."""
        n_outputs = len(self.actual.output_variables)
        self.test.assertEqual(
            n_outputs, n, f"expected {n} output variables, but found {n_outputs}"
        )
        return self

    def has_outputs(self, names: list[str]) -> EngineAssert:
        """Asserts the engine has the expected output variables by name."""
        self.test.assertSequenceEqual(
            [ov.name for ov in self.actual.output_variables], names
        )
        return self

    def has_n_blocks(self, n: int) -> EngineAssert:
        """Asserts the engine has the expected number of rule blocks."""
        n_blocks = len(self.actual.rule_blocks)
        self.test.assertEqual(
            n_blocks, n, f"expected {n} rule blocks, but found {n_blocks}"
        )
        return self

    def has_blocks(self, names: list[str]) -> EngineAssert:
        """Asserts the engine has the expected number of rule blocks by name."""
        self.test.assertSequenceEqual(
            [rb.name for rb in self.actual.rule_blocks], names
        )
        return self

    def when_input_values(
        self, x: fl.ScalarArray, /, raises: Exception | None = None
    ) -> Self:
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

    def evaluate_fld(self, fld: str, decimals: int) -> EngineAssert:
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
                obtained, expected, rtol=fl.settings.rtol, atol=fl.settings.atol
            )
        return self


class TestEngine(unittest.TestCase):
    """Tests the engine."""

    def test_empty_engine(self) -> None:
        """Tests the empty engine."""
        flc = fl.Engine("name", "description")
        EngineAssert(self, flc).has_name("name").has_description(
            "description"
        ).has_n_inputs(0).has_inputs([]).has_n_outputs(0).has_outputs([]).has_n_blocks(
            0
        ).has_blocks(
            []
        )

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
            fl.Rule.create(
                "if service is poor or food is rancid then mTip is cheap", engine
            )
        )
        mamdani.rules.append(
            fl.Rule.create("if service is good then mTip is average", engine)
        )
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
            fl.Rule.create(
                "if service is poor or food is rancid then tsTip is cheap", engine
            )
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
            decimals=13,
        )

    @unittest.skip("Not implemented yet")
    def test_engine_from_fll(self) -> None:
        """Not implemented yet."""
        pass

    def test_inputs(self) -> None:
        """Tests the input variables of an engine."""
        flc = fl.Engine(
            "name", "description", [fl.InputVariable("A"), fl.InputVariable("B")]
        )
        EngineAssert(self, flc).has_name("name").has_description(
            "description"
        ).has_n_inputs(2).has_inputs(["A", "B"])

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
        flc = fl.Engine(
            "name", "description", [], [fl.OutputVariable("A"), fl.OutputVariable("B")]
        )
        EngineAssert(self, flc).has_name("name").has_description(
            "description"
        ).has_n_outputs(2).has_outputs(["A", "B"])

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

    def test_input_values_setter(self) -> None:
        """Tests the setter of input values through the engine."""
        EngineAssert(
            self, fl.Engine("1 input", input_variables=[fl.InputVariable("A")])
        ).when_input_values(fl.array(1.0)).then_input_variables(
            {"A": 1.0}
        ).when_input_values(
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
        e2.when_input_values(fl.array([1, -1])).then_input_variables(
            {"A": 1.0, "B": -1.0}
        )
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
            raises=RuntimeError(
                "can't set input values to an engine without input variables"
            ),
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
                "expected a value with 2 columns (one for each input variable), "
                "but got 3 columns: [[1. 2. 3.]]"
            ),
        )

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
            "1.000/negative + 1.000/positive", engine.output_variable(0).fuzzy_value()
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
            engine.Variable  # type: ignore
        self.assertEqual(
            "'Engine' object has no attribute 'Variable'", str(error.exception)
        )

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


if __name__ == "__main__":
    unittest.main()
