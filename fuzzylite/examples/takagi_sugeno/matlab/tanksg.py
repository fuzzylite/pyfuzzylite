import fuzzylite as fl

engine = fl.Engine(
    name="tanksg",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="level",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        terms=[
            fl.Gaussian("high", -1.000, 0.300),
            fl.Gaussian("okay", 0.004, 0.300),
            fl.Gaussian("low", 1.000, 0.300)
        ]
    ),
    fl.InputVariable(
        name="rate",
        description="",
        enabled=True,
        minimum=-0.100,
        maximum=0.100,
        lock_range=False,
        terms=[
            fl.Gaussian("negative", -0.100, 0.030),
            fl.Gaussian("none", 0.000, 0.030),
            fl.Gaussian("positive", 0.100, 0.030)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="valve",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("close_fast", -0.900),
            fl.Constant("close_slow", -0.500),
            fl.Constant("no_change", 0.000),
            fl.Constant("open_slow", 0.300),
            fl.Constant("open_fast", 0.900)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if level is okay then valve is no_change", engine),
            fl.Rule.create("if level is low then valve is open_fast", engine),
            fl.Rule.create("if level is high then valve is close_fast", engine),
            fl.Rule.create("if level is okay and rate is positive then valve is close_slow", engine),
            fl.Rule.create("if level is okay and rate is negative then valve is open_slow", engine)
        ]
    )
]
