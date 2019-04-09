import fuzzylite as fl

engine = fl.Engine(
    name="tank",
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
            fl.Gaussian("okay", 0.000, 0.300),
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
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Triangle("close_fast", -1.000, -0.900, -0.800),
            fl.Triangle("close_slow", -0.600, -0.500, -0.400),
            fl.Triangle("no_change", -0.100, 0.000, 0.100),
            fl.Triangle("open_slow", 0.200, 0.300, 0.400),
            fl.Triangle("open_fast", 0.800, 0.900, 1.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=fl.AlgebraicSum(),
        implication=fl.AlgebraicProduct(),
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
