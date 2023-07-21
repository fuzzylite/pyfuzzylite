import fuzzylite as fl


class Tank:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tank",
            input_variables=[
                fl.InputVariable(
                    name="level",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[
                        fl.Gaussian("high", -1.0, 0.3),
                        fl.Gaussian("okay", 0.0, 0.3),
                        fl.Gaussian("low", 1.0, 0.3),
                    ],
                ),
                fl.InputVariable(
                    name="rate",
                    minimum=-0.1,
                    maximum=0.1,
                    lock_range=False,
                    terms=[
                        fl.Gaussian("negative", -0.1, 0.03),
                        fl.Gaussian("none", 0.0, 0.03),
                        fl.Gaussian("positive", 0.1, 0.03),
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
                        fl.Triangle("open_slow", 0.2, 0.3, 0.4),
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
                        fl.Rule.create("if level is okay then valve is no_change"),
                        fl.Rule.create("if level is low then valve is open_fast"),
                        fl.Rule.create("if level is high then valve is close_fast"),
                        fl.Rule.create(
                            "if level is okay and rate is positive then valve is close_slow"
                        ),
                        fl.Rule.create(
                            "if level is okay and rate is negative then valve is open_slow"
                        ),
                    ],
                )
            ],
        )
