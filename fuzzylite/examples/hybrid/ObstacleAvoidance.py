import fuzzylite as fl

engine = fl.Engine(
    name="ObstacleAvoidance",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="obstacle",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        terms=[
            fl.Ramp("left", 1.000, 0.000),
            fl.Ramp("right", 0.000, 1.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="mSteer",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(100),
        lock_previous=False,
        terms=[
            fl.Ramp("left", 1.000, 0.000),
            fl.Ramp("right", 0.000, 1.000)
        ]
    ),
    fl.OutputVariable(
        name="tsSteer",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[
            fl.Constant("left", 0.333),
            fl.Constant("right", 0.666)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="mamdani",
        description="",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=fl.AlgebraicProduct(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if obstacle is left then mSteer is right", engine),
            fl.Rule.create("if obstacle is right then mSteer is left", engine)
        ]
    ),
    fl.RuleBlock(
        name="takagiSugeno",
        description="",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if obstacle is left then tsSteer is right", engine),
            fl.Rule.create("if obstacle is right then tsSteer is left", engine)
        ]
    )
]
