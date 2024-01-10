import fuzzylite as fl


class Slcp1:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="slcp1",
            input_variables=[
                fl.InputVariable(name="in1", minimum=-0.3, maximum=0.3, lock_range=False, terms=[]),
                fl.InputVariable(name="in2", minimum=-1.0, maximum=1.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in3", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in4", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(
                    name="in5",
                    minimum=0.5,
                    maximum=1.5,
                    lock_range=False,
                    terms=[
                        fl.Gaussian("small", 0.5, 0.2),
                        fl.Gaussian("medium", 1.0, 0.2),
                        fl.Gaussian("large", 1.5, 0.2),
                    ],
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
                        fl.Linear("outmf1", [32.166, 5.835, 3.162, 3.757, 0.0, 0.0]),
                        fl.Linear("outmf2", [39.012, 9.947, 3.162, 4.269, 0.0, 0.0]),
                        fl.Linear("outmf3", [45.009, 13.985, 3.162, 4.666, 0.0, 0.0]),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=None,
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if in5 is small then out is outmf1"),
                        fl.Rule.create("if in5 is medium then out is outmf2"),
                        fl.Rule.create("if in5 is large then out is outmf3"),
                    ],
                )
            ],
        )
