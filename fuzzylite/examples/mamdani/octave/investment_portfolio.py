import fuzzylite as fl


class InvestmentPortfolio:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="investment_portfolio",
            input_variables=[
                fl.InputVariable(
                    name="Age",
                    minimum=20.0,
                    maximum=100.0,
                    lock_range=False,
                    terms=[fl.ZShape("Young", 30.0, 90.0), fl.SShape("Old", 30.0, 90.0)],
                ),
                fl.InputVariable(
                    name="RiskTolerance",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    terms=[fl.ZShape("Low", 2.0, 8.0), fl.SShape("High", 2.0, 8.0)],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="PercentageInStocks",
                    minimum=0.0,
                    maximum=100.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.EinsteinSum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Gaussian("AboutFifteen", 15.0, 10.0),
                        fl.Gaussian("AboutFifty", 50.0, 10.0),
                        fl.Gaussian("AboutEightyFive", 85.0, 10.0),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.EinsteinProduct(),
                    disjunction=fl.EinsteinSum(),
                    implication=fl.EinsteinProduct(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if Age is Young or RiskTolerance is High then PercentageInStocks is AboutEightyFive"
                        ),
                        fl.Rule.create(
                            "if Age is Old or RiskTolerance is Low then PercentageInStocks is AboutFifteen"
                        ),
                        fl.Rule.create(
                            "if Age is not extremely Old and RiskTolerance is not extremely Low then PercentageInStocks is AboutFifty with 0.500"
                        ),
                        fl.Rule.create(
                            "if Age is not extremely Young and RiskTolerance is not extremely High then PercentageInStocks is AboutFifty with 0.500"
                        ),
                    ],
                )
            ],
        )
