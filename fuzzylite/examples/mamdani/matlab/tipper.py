import fuzzylite as fl


class Tipper:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tipper",
            input_variables=[
                fl.InputVariable(
                    name="service",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Gaussian("poor", 0.0, 1.5),
                        fl.Gaussian("good", 5.0, 1.5),
                        fl.Gaussian("excellent", 10.0, 1.5),
                    ],
                ),
                fl.InputVariable(
                    name="food",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("rancid", 0.0, 0.0, 1.0, 3.0),
                        fl.Trapezoid("delicious", 7.0, 9.0, 10.0, 10.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="tip",
                    minimum=0.0,
                    maximum=30.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("cheap", 0.0, 5.0, 10.0),
                        fl.Triangle("average", 10.0, 15.0, 20.0),
                        fl.Triangle("generous", 20.0, 25.0, 30.0),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.Minimum(),
                    disjunction=fl.Maximum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if service is poor or food is rancid then tip is cheap"),
                        fl.Rule.create("if service is good then tip is average"),
                        fl.Rule.create(
                            "if service is excellent or food is delicious then tip is generous"
                        ),
                    ],
                )
            ],
        )
