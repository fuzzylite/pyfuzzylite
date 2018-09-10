import fuzzylite as fl

engine = fl.Engine(
    name="Linear",
    description="obstacle avoidance for self-driving cars"
)

obstacle = fl.InputVariable(
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
engine.input_variables = [obstacle]

steer = fl.OutputVariable(
    name="steer",
    description="direction to steer the vehicle to",
    enabled=True,
    minimum=0.000000000,
    maximum=1.000000000,
    lock_range=False,
    aggregation=None,
    defuzzifier=fl.WeightedAverage(TakagiSugeno),
    lock_previous=False,
    terms=[
        fl.Linear("left", 0.000000000, 0.333000000),
        fl.Linear("right", 0.000000000, 0.666500000)
    ]
)
engine.output_variables = [steer]

steer_away = fl.RuleBlock(
    name="steer_away",
    description="steer away from obstacles",
    enabled=True,
    conjunction=None,
    disjunction=None,
    implication=None,
    activation=fl.General(),
    rules=[
        fl.Rule.parse("if obstacle is left then steer is right"),
        fl.Rule.parse("if obstacle is right then steer is left")
    ]
)
engine.rule_blocks = [steer_away]