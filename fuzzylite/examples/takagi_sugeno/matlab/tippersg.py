import fuzzylite as fl


class Tippersg:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tippersg",
            input_variables=[
                fl.InputVariable(
                    name="service",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Gaussian("poor", 0.0, 1.5),
                        fl.Gaussian("average", 5.0, 1.5),
                        fl.Gaussian("good", 10.0, 1.5),
                    ],
                ),
                fl.InputVariable(
                    name="food",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("rancid", -5.0, 0.0, 1.0, 3.0),
                        fl.Trapezoid("delicious", 7.0, 9.0, 10.0, 15.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="tip",
                    minimum=-30.0,
                    maximum=30.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("cheap", [0.0, 0.0, 5.0]),
                        fl.Linear("average", [0.0, 0.0, 15.0]),
                        fl.Linear("generous", [0.0, 0.0, 25.0]),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=None,
                    disjunction=fl.Maximum(),
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if service is poor or food is rancid then tip is cheap"),
                        fl.Rule.create("if service is average then tip is average"),
                        fl.Rule.create(
                            "if service is good or food is delicious then tip is generous"
                        ),
                    ],
                )
            ],
        )
