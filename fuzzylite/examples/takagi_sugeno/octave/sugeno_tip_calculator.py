import fuzzylite as fl

engine = fl.Engine(
    name="sugeno_tip_calculator",
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
        name="CheapTip",
        description="",
        enabled=True,
        minimum=5.000,
        maximum=25.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("Low", 10.000),
            fl.Constant("Medium", 15.000),
            fl.Constant("High", 20.000)
        ]
    ),
    fl.OutputVariable(
        name="AverageTip",
        description="",
        enabled=True,
        minimum=5.000,
        maximum=25.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("Low", 10.000),
            fl.Constant("Medium", 15.000),
            fl.Constant("High", 20.000)
        ]
    ),
    fl.OutputVariable(
        name="GenerousTip",
        description="",
        enabled=True,
        minimum=5.000,
        maximum=25.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("Low", 10.000),
            fl.Constant("Medium", 15.000),
            fl.Constant("High", 20.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.EinsteinProduct(),
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if FoodQuality is extremely Bad and Service is extremely Bad then CheapTip is extremely Low and AverageTip is very Low and GenerousTip is Low", engine),
            fl.Rule.create("if FoodQuality is Good and Service is extremely Bad then CheapTip is Low and AverageTip is Low and GenerousTip is Medium", engine),
            fl.Rule.create("if FoodQuality is very Good and Service is very Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is High", engine),
            fl.Rule.create("if FoodQuality is Bad and Service is Bad then CheapTip is Low and AverageTip is Low and GenerousTip is Medium", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is High", engine),
            fl.Rule.create("if FoodQuality is extremely Good and Service is Bad then CheapTip is Low and AverageTip is Medium and GenerousTip is very High", engine),
            fl.Rule.create("if FoodQuality is Bad and Service is Good then CheapTip is Low and AverageTip is Medium and GenerousTip is High", engine),
            fl.Rule.create("if FoodQuality is Good and Service is Good then CheapTip is Medium and AverageTip is Medium and GenerousTip is very High", engine),
            fl.Rule.create("if FoodQuality is very Bad and Service is very Good then CheapTip is Low and AverageTip is Medium and GenerousTip is High", engine),
            fl.Rule.create("if FoodQuality is very very Good and Service is very very Good then CheapTip is High and AverageTip is very High and GenerousTip is extremely High", engine)
        ]
    )
]
