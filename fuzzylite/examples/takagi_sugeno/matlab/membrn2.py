import fuzzylite as fl


class Membrn2:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="membrn2",
            input_variables=[
                fl.InputVariable(
                    name="in_n1",
                    minimum=1.0,
                    maximum=31.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", 1.152, 8.206, 0.874),
                        fl.Bell("in1mf2", 15.882, 7.078, 0.444),
                        fl.Bell("in1mf3", 30.575, 8.602, 0.818),
                    ],
                ),
                fl.InputVariable(
                    name="in_n2",
                    minimum=1.0,
                    maximum=31.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in2mf1", 1.426, 8.602, 0.818),
                        fl.Bell("in2mf2", 16.118, 7.078, 0.445),
                        fl.Bell("in2mf3", 30.847, 8.206, 0.875),
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
                        fl.Linear("out1mf1", [-0.035, 0.002, -0.352]),
                        fl.Linear("out1mf2", [0.044, 0.079, -0.028]),
                        fl.Linear("out1mf3", [-0.024, 0.024, -1.599]),
                        fl.Linear("out1mf4", [-0.067, 0.384, 0.007]),
                        fl.Linear("out1mf5", [0.351, -0.351, -3.663]),
                        fl.Linear("out1mf6", [-0.079, -0.044, 3.909]),
                        fl.Linear("out1mf7", [0.012, -0.012, -0.6]),
                        fl.Linear("out1mf8", [-0.384, 0.067, 10.158]),
                        fl.Linear("out1mf9", [-0.002, 0.035, -1.402]),
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
                            "if in_n1 is in1mf1 and in_n2 is in2mf3 then out1 is out1mf3"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf2 and in_n2 is in2mf1 then out1 is out1mf4"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf2 and in_n2 is in2mf2 then out1 is out1mf5"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf2 and in_n2 is in2mf3 then out1 is out1mf6"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf3 and in_n2 is in2mf1 then out1 is out1mf7"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf3 and in_n2 is in2mf2 then out1 is out1mf8"
                        ),
                        fl.Rule.create(
                            "if in_n1 is in1mf3 and in_n2 is in2mf3 then out1 is out1mf9"
                        ),
                    ],
                )
            ],
        )
