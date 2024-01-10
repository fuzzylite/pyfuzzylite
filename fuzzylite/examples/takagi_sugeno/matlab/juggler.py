import fuzzylite as fl


class Juggler:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="juggler",
            input_variables=[
                fl.InputVariable(
                    name="xHit",
                    minimum=-4.0,
                    maximum=4.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", -4.0, 2.0, 4.0),
                        fl.Bell("in1mf2", 0.0, 2.0, 4.0),
                        fl.Bell("in1mf3", 4.0, 2.0, 4.0),
                    ],
                ),
                fl.InputVariable(
                    name="projectAngle",
                    minimum=0.0,
                    maximum=3.142,
                    lock_range=False,
                    terms=[
                        fl.Bell("in2mf1", 0.0, 0.785, 4.0),
                        fl.Bell("in2mf2", 1.571, 0.785, 4.0),
                        fl.Bell("in2mf3", 3.142, 0.785, 4.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="theta",
                    minimum=0.0,
                    maximum=0.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("out1mf", [-0.022, -0.5, 0.315]),
                        fl.Linear("out1mf", [-0.022, -0.5, 0.315]),
                        fl.Linear("out1mf", [-0.022, -0.5, 0.315]),
                        fl.Linear("out1mf", [0.114, -0.5, 0.785]),
                        fl.Linear("out1mf", [0.114, -0.5, 0.785]),
                        fl.Linear("out1mf", [0.114, -0.5, 0.785]),
                        fl.Linear("out1mf", [-0.022, -0.5, 1.256]),
                        fl.Linear("out1mf", [-0.022, -0.5, 1.256]),
                        fl.Linear("out1mf", [-0.022, -0.5, 1.256]),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if xHit is in1mf1 and projectAngle is in2mf1 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf1 and projectAngle is in2mf2 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf1 and projectAngle is in2mf3 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf2 and projectAngle is in2mf1 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf2 and projectAngle is in2mf2 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf2 and projectAngle is in2mf3 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf3 and projectAngle is in2mf1 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf3 and projectAngle is in2mf2 then theta is out1mf"
                        ),
                        fl.Rule.create(
                            "if xHit is in1mf3 and projectAngle is in2mf3 then theta is out1mf"
                        ),
                    ],
                )
            ],
        )
