import fuzzylite as fl


class Linear:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="Linear",
            description="obstacle avoidance for self-driving cars",
            input_variables=[
                fl.InputVariable(
                    name="obstacle",
                    description="location of obstacle relative to vehicle",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[
                        fl.Triangle("left", 0.0, 0.333, 0.666),
                        fl.Triangle("right", 0.333, 0.666, 1.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="steer",
                    description="direction to steer the vehicle to",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[fl.Linear("left", [0.0, 0.333]), fl.Linear("right", [0.0, 0.666])],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="steer_away",
                    description="steer away from obstacles",
                    conjunction=None,
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if obstacle is left then steer is right"),
                        fl.Rule.create("if obstacle is right then steer is left"),
                    ],
                )
            ],
        )
