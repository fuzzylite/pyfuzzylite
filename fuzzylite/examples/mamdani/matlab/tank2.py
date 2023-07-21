import fuzzylite as fl


class Tank2:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tank2",
            input_variables=[
                fl.InputVariable(
                    name="level",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("high", -2.0, -1.0, -0.8, -0.001),
                        fl.Triangle("good", -0.15, 0.0, 0.5),
                        fl.Trapezoid("low", 0.001, 0.8, 1.0, 1.5),
                    ],
                ),
                fl.InputVariable(
                    name="change",
                    minimum=-0.1,
                    maximum=0.1,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("falling", -0.14, -0.1, -0.06, 0.0),
                        fl.Trapezoid("rising", -0.001, 0.06, 0.1, 0.14),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="valve",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("close_fast", -1.0, -0.9, -0.8),
                        fl.Triangle("close_slow", -0.6, -0.5, -0.4),
                        fl.Triangle("no_change", -0.1, 0.0, 0.1),
                        fl.Triangle("open_slow", 0.4, 0.5, 0.6),
                        fl.Triangle("open_fast", 0.8, 0.9, 1.0),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=fl.AlgebraicSum(),
                    implication=fl.AlgebraicProduct(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if level is low then valve is open_fast"),
                        fl.Rule.create("if level is high then valve is close_fast"),
                        fl.Rule.create(
                            "if level is good and change is rising then valve is close_slow"
                        ),
                        fl.Rule.create(
                            "if level is good and change is falling then valve is open_slow"
                        ),
                    ],
                )
            ],
        )
