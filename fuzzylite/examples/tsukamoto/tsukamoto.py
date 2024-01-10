import fuzzylite as fl


class Tsukamoto:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tsukamoto",
            input_variables=[
                fl.InputVariable(
                    name="X",
                    minimum=-10.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("small", -10.0, 5.0, 3.0),
                        fl.Bell("medium", 0.0, 5.0, 3.0),
                        fl.Bell("large", 10.0, 5.0, 3.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="Ramps",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[fl.Ramp("b", 0.6, 0.4), fl.Ramp("a", 0.0, 0.25), fl.Ramp("c", 0.7, 1.0)],
                ),
                fl.OutputVariable(
                    name="Sigmoids",
                    minimum=0.02,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[
                        fl.Sigmoid("b", 0.5, -30.0),
                        fl.Sigmoid("a", 0.13, 30.0),
                        fl.Sigmoid("c", 0.83, 30.0),
                    ],
                ),
                fl.OutputVariable(
                    name="ZSShapes",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[
                        fl.ZShape("b", 0.3, 0.6),
                        fl.SShape("a", 0.0, 0.25),
                        fl.SShape("c", 0.7, 1.0),
                    ],
                ),
                fl.OutputVariable(
                    name="Concaves",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(),
                    terms=[
                        fl.Concave("b", 0.5, 0.4),
                        fl.Concave("a", 0.24, 0.25),
                        fl.Concave("c", 0.9, 1.0),
                    ],
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
                        fl.Rule.create(
                            "if X is small then Ramps is a and Sigmoids is a and ZSShapes is a and Concaves is a"
                        ),
                        fl.Rule.create(
                            "if X is medium then Ramps is b and Sigmoids is b and ZSShapes is b and Concaves is b"
                        ),
                        fl.Rule.create(
                            "if X is large then Ramps is c and Sigmoids is c and ZSShapes is c and Concaves is c"
                        ),
                    ],
                )
            ],
        )
