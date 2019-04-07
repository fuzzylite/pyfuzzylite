import fuzzylite as fl

engine = fl.Engine(
    name="Function",
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
            fl.Function.create("left",
                               "lt(x, 0.333) * (x - 0) / (0.333 - 0) + gt(x, 0.333) * (0.666 - x) "
                               "/ (0.666 - 0.333) + eq(x, 0.333)",
                               engine),
            fl.Function.create("right",
                               "lt(x, 0.666) * (x - 0.333) / (0.666 - 0.333) + gt(x, 0.666) * (1 "
                               "- x) / (1 - 0.666) + eq(x, 0.666)",
                               engine)
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
