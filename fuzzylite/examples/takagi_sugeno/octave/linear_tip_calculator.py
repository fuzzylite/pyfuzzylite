import fuzzylite as fl


class LinearTipCalculator:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="linear_tip_calculator",
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
                    minimum=10.0,
                    maximum=20.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("TenPercent", [0.0, 0.0, 10.0]),
                        fl.Linear("FifteenPercent", [0.0, 0.0, 15.0]),
                        fl.Linear("TwentyPercent", [0.0, 0.0, 20.0]),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.Minimum(),
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Bad then Tip is TenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Good then Tip is FifteenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Bad then Tip is FifteenPercent"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Good then Tip is TwentyPercent"
                        ),
                    ],
                )
            ],
        )
