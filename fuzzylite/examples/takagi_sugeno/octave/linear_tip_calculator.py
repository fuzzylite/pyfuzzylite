import fuzzylite as fl

engine = fl.Engine(
    name="linear_tip_calculator",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="FoodQuality",
        description="",
        enabled=True,
        minimum=1.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Trapezoid("Bad", 0.000, 1.000, 3.000, 7.000),
            fl.Trapezoid("Good", 3.000, 7.000, 10.000, 11.000)
        ]
    ),
    fl.InputVariable(
        name="Service",
        description="",
        enabled=True,
        minimum=1.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Trapezoid("Bad", 0.000, 1.000, 3.000, 7.000),
            fl.Trapezoid("Good", 3.000, 7.000, 10.000, 11.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="Tip",
        description="",
        enabled=True,
        minimum=10.000,
        maximum=20.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("TenPercent", [0.000, 0.000, 10.000], engine),
            fl.Linear("FifteenPercent", [0.000, 0.000, 15.000], engine),
            fl.Linear("TwentyPercent", [0.000, 0.000, 20.000], engine)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if FoodQuality is Bad and Service is Bad then Tip is TenPercent", engine),
            fl.Rule.create("if FoodQuality is Bad and Service is Good then Tip is FifteenPercent", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Bad then Tip is FifteenPercent", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Good then Tip is TwentyPercent", engine)
        ]
    )
]
