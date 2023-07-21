import fuzzylite as fl


class SugenoTipCalculator:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="sugeno_tip_calculator",
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
                    name="CheapTip",
                    minimum=5.0,
                    maximum=25.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("Low", 10.0),
                        fl.Constant("Medium", 15.0),
                        fl.Constant("High", 20.0),
                    ],
                ),
                fl.OutputVariable(
                    name="AverageTip",
                    minimum=5.0,
                    maximum=25.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("Low", 10.0),
                        fl.Constant("Medium", 15.0),
                        fl.Constant("High", 20.0),
                    ],
                ),
                fl.OutputVariable(
                    name="GenerousTip",
                    minimum=5.0,
                    maximum=25.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("Low", 10.0),
                        fl.Constant("Medium", 15.0),
                        fl.Constant("High", 20.0),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.EinsteinProduct(),
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if FoodQuality is extremely Bad and Service is extremely Bad then CheapTip is extremely Low and AverageTip is very Low and GenerousTip is Low"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is extremely Bad then CheapTip is Low and AverageTip is Low and GenerousTip is Medium"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is very Good and Service is very Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Bad then CheapTip is Low and AverageTip is Low and GenerousTip is Medium"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is extremely Good and Service is Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is very High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Bad and Service is Good then CheapTip is Low and AverageTip is Medium and GenerousTip is High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is Good and Service is Good then CheapTip is Medium and AverageTip is Medium and GenerousTip is very High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is very Bad and Service is very Good then CheapTip is Low and AverageTip is Medium and GenerousTip is High"
                        ),
                        fl.Rule.create(
                            "if FoodQuality is very very Good and Service is very very Good then CheapTip is High and AverageTip is very High and GenerousTip is extremely High"
                        ),
                    ],
                )
            ],
        )
