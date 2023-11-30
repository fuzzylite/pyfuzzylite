import fuzzylite as fl


class Slbb:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="slbb",
            input_variables=[
                fl.InputVariable(
                    name="in1",
                    minimum=-1.5,
                    maximum=1.5,
                    lock_range=False,
                    terms=[fl.Bell("in1mf1", -1.5, 1.5, 2.0), fl.Bell("in1mf2", 1.5, 1.5, 2.0)],
                ),
                fl.InputVariable(
                    name="in2",
                    minimum=-1.5,
                    maximum=1.5,
                    lock_range=False,
                    terms=[fl.Bell("in2mf1", -1.5, 1.5, 2.0), fl.Bell("in2mf2", 1.5, 1.5, 2.0)],
                ),
                fl.InputVariable(
                    name="in3",
                    minimum=-0.2,
                    maximum=0.2,
                    lock_range=False,
                    terms=[fl.Bell("in3mf1", -0.2, 0.2, 2.0), fl.Bell("in3mf2", 0.2, 0.2, 2.0)],
                ),
                fl.InputVariable(
                    name="in4",
                    minimum=-0.4,
                    maximum=0.4,
                    lock_range=False,
                    terms=[fl.Bell("in4mf1", -0.4, 0.4, 2.0), fl.Bell("in4mf2", 0.4, 0.4, 2.0)],
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
                        fl.Linear("outmf1", [1.015, 2.234, -12.665, -4.046, 0.026]),
                        fl.Linear("outmf2", [1.161, 1.969, -9.396, -6.165, 0.474]),
                        fl.Linear("outmf3", [1.506, 2.234, -12.99, -1.865, 1.426]),
                        fl.Linear("outmf4", [0.734, 1.969, -9.381, -4.688, -0.88]),
                        fl.Linear("outmf5", [0.734, 2.234, -12.853, -6.11, -1.034]),
                        fl.Linear("outmf6", [1.413, 1.969, -9.485, -6.592, 1.159]),
                        fl.Linear("outmf7", [1.225, 2.234, -12.801, -3.929, 0.366]),
                        fl.Linear("outmf8", [0.985, 1.969, -9.291, -5.115, -0.195]),
                        fl.Linear("outmf9", [0.985, 1.969, -9.292, -5.115, 0.195]),
                        fl.Linear("outmf10", [1.225, 2.234, -12.802, -3.929, -0.366]),
                        fl.Linear("outmf11", [1.413, 1.969, -9.485, -6.592, -1.159]),
                        fl.Linear("outmf12", [0.734, 2.234, -12.853, -6.11, 1.034]),
                        fl.Linear("outmf13", [0.734, 1.969, -9.381, -4.688, 0.88]),
                        fl.Linear("outmf14", [1.506, 2.234, -12.99, -1.865, -1.426]),
                        fl.Linear("outmf15", [1.161, 1.969, -9.396, -6.165, -0.474]),
                        fl.Linear("outmf16", [1.015, 2.234, -12.665, -4.046, -0.026]),
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
