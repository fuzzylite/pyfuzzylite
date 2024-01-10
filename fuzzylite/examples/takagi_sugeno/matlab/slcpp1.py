import fuzzylite as fl


class Slcpp1:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="slcpp1",
            input_variables=[
                fl.InputVariable(name="in1", minimum=-0.3, maximum=0.3, lock_range=False, terms=[]),
                fl.InputVariable(name="in2", minimum=-1.0, maximum=1.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in3", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in4", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in5", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(name="in6", minimum=-3.0, maximum=3.0, lock_range=False, terms=[]),
                fl.InputVariable(
                    name="pole_length",
                    minimum=0.5,
                    maximum=1.5,
                    lock_range=False,
                    terms=[
                        fl.ZShape("mf1", 0.5, 0.6),
                        fl.PiShape("mf2", 0.5, 0.6, 0.6, 0.7),
                        fl.PiShape("mf3", 0.6, 0.7, 0.7, 0.8),
                        fl.PiShape("mf4", 0.7, 0.8, 0.8, 0.9),
                        fl.PiShape("mf5", 0.8, 0.9, 0.9, 1.0),
                        fl.PiShape("mf6", 0.9, 1.0, 1.0, 1.1),
                        fl.PiShape("mf7", 1.0, 1.1, 1.1, 1.2),
                        fl.PiShape("mf8", 1.1, 1.2, 1.2, 1.3),
                        fl.PiShape("mf9", 1.2, 1.3, 1.3, 1.4),
                        fl.PiShape("mf10", 1.3, 1.4, 1.4, 1.5),
                        fl.SShape("mf11", 1.4, 1.5),
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
                        fl.Linear("outmf1", [168.4, 31.0, -188.05, -49.25, -1.0, -2.7, 0.0, 0.0]),
                        fl.Linear(
                            "outmf2", [233.95, 47.19, -254.52, -66.58, -1.0, -2.74, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf3", [342.94, 74.73, -364.37, -95.23, -1.0, -2.78, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf4", [560.71, 130.67, -582.96, -152.24, -1.0, -2.81, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf5", [1213.88, 300.19, -1236.9, -322.8, -1.0, -2.84, 0.0, 0.0]
                        ),
                        fl.Linear("outmf6", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        fl.Linear(
                            "outmf7", [-1399.12, -382.95, 1374.63, 358.34, -1.0, -2.9, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf8", [-746.07, -213.42, 720.9, 187.84, -1.0, -2.93, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf9", [-528.52, -157.46, 502.68, 130.92, -1.0, -2.96, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf10", [-419.87, -129.89, 393.38, 102.41, -1.0, -2.98, 0.0, 0.0]
                        ),
                        fl.Linear(
                            "outmf11", [-354.77, -113.68, 327.65, 85.27, -1.0, -3.01, 0.0, 0.0]
                        ),
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
                        fl.Rule.create("if pole_length is mf1 then out is outmf1"),
                        fl.Rule.create("if pole_length is mf2 then out is outmf2"),
                        fl.Rule.create("if pole_length is mf3 then out is outmf3"),
                        fl.Rule.create("if pole_length is mf4 then out is outmf4"),
                        fl.Rule.create("if pole_length is mf5 then out is outmf5"),
                        fl.Rule.create("if pole_length is mf6 then out is outmf6"),
                        fl.Rule.create("if pole_length is mf7 then out is outmf7"),
                        fl.Rule.create("if pole_length is mf8 then out is outmf8"),
                        fl.Rule.create("if pole_length is mf9 then out is outmf9"),
                        fl.Rule.create("if pole_length is mf10 then out is outmf10"),
                        fl.Rule.create("if pole_length is mf11 then out is outmf11"),
                    ],
                )
            ],
        )
