import fuzzylite as fl

engine = fl.Engine(
    name="investment_portfolio",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="Age",
        description="",
        enabled=True,
        minimum=20.000,
        maximum=100.000,
        lock_range=False,
        terms=[
            fl.ZShape("Young", 30.000, 90.000),
            fl.SShape("Old", 30.000, 90.000)
        ]
    ),
    fl.InputVariable(
        name="RiskTolerance",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.ZShape("Low", 2.000, 8.000),
            fl.SShape("High", 2.000, 8.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="PercentageInStocks",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=100.000,
        lock_range=False,
        aggregation=fl.EinsteinSum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Gaussian("AboutFifteen", 15.000, 10.000),
            fl.Gaussian("AboutFifty", 50.000, 10.000),
            fl.Gaussian("AboutEightyFive", 85.000, 10.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.EinsteinProduct(),
        disjunction=fl.EinsteinSum(),
        implication=fl.EinsteinProduct(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if Age is Young or RiskTolerance is High then PercentageInStocks is AboutEightyFive", engine),
            fl.Rule.create("if Age is Old or RiskTolerance is Low then PercentageInStocks is AboutFifteen", engine),
            fl.Rule.create("if Age is not extremely Old and RiskTolerance is not extremely Low then PercentageInStocks is AboutFifty with 0.500", engine),
            fl.Rule.create("if Age is not extremely Young and RiskTolerance is not extremely High then PercentageInStocks is AboutFifty with 0.500", engine)
        ]
    )
]
