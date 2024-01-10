import fuzzylite as fl


class Approximation:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="approximation",
            input_variables=[
                fl.InputVariable(
                    name="inputX",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Triangle("NEAR_1", 0.0, 1.0, 2.0),
                        fl.Triangle("NEAR_2", 1.0, 2.0, 3.0),
                        fl.Triangle("NEAR_3", 2.0, 3.0, 4.0),
                        fl.Triangle("NEAR_4", 3.0, 4.0, 5.0),
                        fl.Triangle("NEAR_5", 4.0, 5.0, 6.0),
                        fl.Triangle("NEAR_6", 5.0, 6.0, 7.0),
                        fl.Triangle("NEAR_7", 6.0, 7.0, 8.0),
                        fl.Triangle("NEAR_8", 7.0, 8.0, 9.0),
                        fl.Triangle("NEAR_9", 8.0, 9.0, 10.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="outputFx",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=True,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("f1", 0.84),
                        fl.Constant("f2", 0.45),
                        fl.Constant("f3", 0.04),
                        fl.Constant("f4", -0.18),
                        fl.Constant("f5", -0.19),
                        fl.Constant("f6", -0.04),
                        fl.Constant("f7", 0.09),
                        fl.Constant("f8", 0.12),
                        fl.Constant("f9", 0.04),
                    ],
                ),
                fl.OutputVariable(
                    name="trueFx",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=True,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[fl.Function("fx", "sin(inputX)/inputX")],
                ),
                fl.OutputVariable(
                    name="diffFx",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[fl.Function("diff", "fabs(outputFx-trueFx)")],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=None,
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if inputX is NEAR_1 then outputFx is f1"),
                        fl.Rule.create("if inputX is NEAR_2 then outputFx is f2"),
                        fl.Rule.create("if inputX is NEAR_3 then outputFx is f3"),
                        fl.Rule.create("if inputX is NEAR_4 then outputFx is f4"),
                        fl.Rule.create("if inputX is NEAR_5 then outputFx is f5"),
                        fl.Rule.create("if inputX is NEAR_6 then outputFx is f6"),
                        fl.Rule.create("if inputX is NEAR_7 then outputFx is f7"),
                        fl.Rule.create("if inputX is NEAR_8 then outputFx is f8"),
                        fl.Rule.create("if inputX is NEAR_9 then outputFx is f9"),
                        fl.Rule.create("if inputX is any then trueFx is fx and diffFx is diff"),
                    ],
                )
            ],
        )
