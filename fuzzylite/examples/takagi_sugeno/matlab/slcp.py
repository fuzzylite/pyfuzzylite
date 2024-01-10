import fuzzylite as fl


class Slcp:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="slcp",
            input_variables=[
                fl.InputVariable(
                    name="in1",
                    minimum=-0.3,
                    maximum=0.3,
                    lock_range=False,
                    terms=[fl.Bell("in1mf1", -0.3, 0.3, 2.0), fl.Bell("in1mf2", 0.3, 0.3, 2.0)],
                ),
                fl.InputVariable(
                    name="in2",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[fl.Bell("in2mf1", -1.0, 1.0, 2.0), fl.Bell("in2mf2", 1.0, 1.0, 2.0)],
                ),
                fl.InputVariable(
                    name="in3",
                    minimum=-3.0,
                    maximum=3.0,
                    lock_range=False,
                    terms=[fl.Bell("in3mf1", -3.0, 3.0, 2.0), fl.Bell("in3mf2", 3.0, 3.0, 2.0)],
                ),
                fl.InputVariable(
                    name="in4",
                    minimum=-3.0,
                    maximum=3.0,
                    lock_range=False,
                    terms=[fl.Bell("in4mf1", -3.0, 3.0, 2.0), fl.Bell("in4mf2", 3.0, 3.0, 2.0)],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="out",
                    minimum=-10.0,
                    maximum=10.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("outmf1", [41.373, 10.03, 3.162, 4.288, 0.339]),
                        fl.Linear("outmf2", [40.409, 10.053, 3.162, 4.288, 0.207]),
                        fl.Linear("outmf3", [41.373, 10.03, 3.162, 4.288, 0.339]),
                        fl.Linear("outmf4", [40.409, 10.053, 3.162, 4.288, 0.207]),
                        fl.Linear("outmf5", [38.561, 10.177, 3.162, 4.288, -0.049]),
                        fl.Linear("outmf6", [37.596, 10.154, 3.162, 4.288, -0.181]),
                        fl.Linear("outmf7", [38.561, 10.177, 3.162, 4.288, -0.049]),
                        fl.Linear("outmf8", [37.596, 10.154, 3.162, 4.288, -0.181]),
                        fl.Linear("outmf9", [37.596, 10.154, 3.162, 4.288, 0.181]),
                        fl.Linear("outmf10", [38.561, 10.177, 3.162, 4.288, 0.049]),
                        fl.Linear("outmf11", [37.596, 10.154, 3.162, 4.288, 0.181]),
                        fl.Linear("outmf12", [38.561, 10.177, 3.162, 4.288, 0.049]),
                        fl.Linear("outmf13", [40.408, 10.053, 3.162, 4.288, -0.207]),
                        fl.Linear("outmf14", [41.373, 10.03, 3.162, 4.288, -0.339]),
                        fl.Linear("outmf15", [40.408, 10.053, 3.162, 4.288, -0.207]),
                        fl.Linear("outmf16", [41.373, 10.03, 3.162, 4.288, -0.339]),
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
                            "if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf1 then out is outmf1"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf2 then out is outmf2"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf1 then out is outmf3"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf2 then out is outmf4"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf1 then out is outmf5"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf2 then out is outmf6"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf1 then out is outmf7"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf2 then out is outmf8"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf1 then out is outmf9"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf2 then out is outmf10"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf1 then out is outmf11"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf2 then out is outmf12"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf1 then out is outmf13"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf2 then out is outmf14"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf1 then out is outmf15"
                        ),
                        fl.Rule.create(
                            "if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf2 then out is outmf16"
                        ),
                    ],
                )
            ],
        )
