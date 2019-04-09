import fuzzylite as fl

engine = fl.Engine(
    name="mamdani_tip_calculator",
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
        minimum=0.000,
        maximum=30.000,
        lock_range=False,
        aggregation=fl.AlgebraicSum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Gaussian("AboutTenPercent", 10.000, 2.000),
            fl.Gaussian("AboutFifteenPercent", 15.000, 2.000),
            fl.Gaussian("AboutTwentyPercent", 20.000, 2.000)
        ]
    ),
    fl.OutputVariable(
        name="CheckPlusTip",
        description="",
        enabled=True,
        minimum=1.000,
        maximum=1.300,
        lock_range=False,
        aggregation=fl.AlgebraicSum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Gaussian("PlusAboutTenPercent", 1.100, 0.020),
            fl.Gaussian("PlusAboutFifteenPercent", 1.150, 0.020),
            fl.Gaussian("PlusAboutTwentyPercent", 1.200, 0.020)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=fl.Maximum(),
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if FoodQuality is Bad and Service is Bad then Tip is AboutTenPercent and CheckPlusTip is PlusAboutTenPercent", engine),
            fl.Rule.create("if FoodQuality is Bad and Service is Good then Tip is AboutFifteenPercent and CheckPlusTip is PlusAboutFifteenPercent", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Bad then Tip is AboutFifteenPercent and CheckPlusTip is PlusAboutFifteenPercent", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Good then Tip is AboutTwentyPercent and CheckPlusTip is PlusAboutTwentyPercent", engine)
        ]
    )
]
