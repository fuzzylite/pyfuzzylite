import fuzzylite as fl


class Membrn1:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="membrn1",
            input_variables=[
                fl.InputVariable(
                    name="in_n1",
                    minimum=1.0,
                    maximum=31.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", 2.253, 16.22, 5.05),
                        fl.Bell("in1mf2", 31.26, 15.021, 1.843),
                    ],
                ),
                fl.InputVariable(
                    name="in_n2",
                    minimum=1.0,
                    maximum=31.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in2mf1", 0.74, 15.021, 1.843),
                        fl.Bell("in2mf2", 29.747, 16.22, 5.05),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="out1",
                    minimum=-0.334,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("out1mf1", [0.026, 0.071, -0.615]),
                        fl.Linear("out1mf2", [-0.036, 0.036, -1.169]),
                        fl.Linear("out1mf3", [-0.094, 0.094, 2.231]),
                        fl.Linear("out1mf4", [-0.071, -0.026, 2.479]),
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
                            "if in_n1 is in1mf1 and in_n2 is in2mf1 then out1 is out1mf1"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf1 and in_n2 is in2mf2 then out1 is out1mf2"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf2 and in_n2 is in2mf1 then out1 is out1mf3"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf2 and in_n2 is in2mf2 then out1 is out1mf4"
                        ),
                    ],
                )
            ],
        )
