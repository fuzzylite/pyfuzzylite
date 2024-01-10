import fuzzylite as fl


class MamdaniTipCalculator:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="mamdani_tip_calculator",
            input_variables=[
                fl.InputVariable(
                    name="FoodQuality",
                    minimum=1.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("Bad", 0.0, 1.0, 3.0, 7.0),
                        fl.Trapezoid("Good", 3.0, 7.0, 10.0, 11.0),
                    ],
                ),
                fl.InputVariable(
                    name="Service",
                    minimum=1.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("Bad", 0.0, 1.0, 3.0, 7.0),
                        fl.Trapezoid("Good", 3.0, 7.0, 10.0, 11.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="Tip",
                    minimum=0.0,
                    maximum=30.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.AlgebraicSum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Gaussian("AboutTenPercent", 10.0, 2.0),
                        fl.Gaussian("AboutFifteenPercent", 15.0, 2.0),
                        fl.Gaussian("AboutTwentyPercent", 20.0, 2.0),
                    ],
                ),
                fl.OutputVariable(
                    name="CheckPlusTip",
                    minimum=1.0,
                    maximum=1.3,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.AlgebraicSum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Gaussian("PlusAboutTenPercent", 1.1, 0.02),
                        fl.Gaussian("PlusAboutFifteenPercent", 1.15, 0.02),
                        fl.Gaussian("PlusAboutTwentyPercent", 1.2, 0.02),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=fl.Maximum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Bad then Tip is AboutTenPercent and CheckPlusTip is PlusAboutTenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Good then Tip is AboutFifteenPercent and CheckPlusTip is PlusAboutFifteenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Bad then Tip is AboutFifteenPercent and CheckPlusTip is PlusAboutFifteenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Good then Tip is AboutTwentyPercent and CheckPlusTip is PlusAboutTwentyPercent"
                        ),
                    ],
                )
            ],
        )
