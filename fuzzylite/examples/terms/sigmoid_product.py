import fuzzylite as fl


class SigmoidProduct:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="SigmoidProduct",
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
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.SigmoidProduct("left", 0.167, 30.0, -30.0, 0.5),
                        fl.SigmoidProduct("right", 0.5, 30.0, -30.0, 0.833),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="steer_away",
                    description="steer away from obstacles",
                    conjunction=None,
                    disjunction=None,
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if obstacle is left then steer is right"),
                        fl.Rule.create("if obstacle is right then steer is left"),
                    ],
                )
            ],
        )
