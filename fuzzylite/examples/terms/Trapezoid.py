import fuzzylite as fl

engine = fl.Engine(
    name="Trapezoid",
    description="obstacle avoidance for self-driving cars"
)
engine.input_variables = [
    fl.InputVariable(
        name="obstacle",
        description="location of obstacle relative to vehicle",
        enabled=True,
        minimum=0.000000000,
        maximum=1.000000000,
        lock_range=False,
        terms=[
            fl.Triangle("left", 0.000000000, 0.333000000, 0.666000000),
            fl.Triangle("right", 0.333000000, 0.666000000, 1.000000000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="steer",
        description="direction to steer the vehicle to",
        enabled=True,
        minimum=0.000000000,
        maximum=1.000000000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(100),
        lock_previous=False,
        terms=[
            fl.Trapezoid("left", 0.000000000, 0.166500000, 0.499500000, 0.666000000),
            fl.Trapezoid("right", 0.333000000, 0.499750000, 0.833250000, 1.000000000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="steer_away",
        description="steer away from obstacles",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if obstacle is left then steer is right", engine),
            fl.Rule.create("if obstacle is right then steer is left", engine)
        ]
    )
]
