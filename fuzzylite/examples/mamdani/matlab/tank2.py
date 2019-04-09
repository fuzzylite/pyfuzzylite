import fuzzylite as fl

engine = fl.Engine(
    name="tank2",
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
            fl.Trapezoid("high", -2.000, -1.000, -0.800, -0.001),
            fl.Triangle("good", -0.150, 0.000, 0.500),
            fl.Trapezoid("low", 0.001, 0.800, 1.000, 1.500)
        ]
    ),
    fl.InputVariable(
        name="change",
        description="",
        enabled=True,
        minimum=-0.100,
        maximum=0.100,
        lock_range=False,
        terms=[
            fl.Trapezoid("falling", -0.140, -0.100, -0.060, 0.000),
            fl.Trapezoid("rising", -0.001, 0.060, 0.100, 0.140)
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
            fl.Triangle("open_slow", 0.400, 0.500, 0.600),
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
            fl.Rule.create("if level is low then valve is open_fast", engine),
            fl.Rule.create("if level is high then valve is close_fast", engine),
            fl.Rule.create("if level is good and change is rising then valve is close_slow", engine),
            fl.Rule.create("if level is good and change is falling then valve is open_slow", engine)
        ]
    )
]
