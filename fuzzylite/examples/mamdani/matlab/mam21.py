import fuzzylite as fl


class Mam21:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="mam21",
            input_variables=[
                fl.InputVariable(
                    name="angle",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    terms=[fl.Bell("small", -5.0, 5.0, 8.0), fl.Bell("big", 5.0, 5.0, 8.0)],
                ),
                fl.InputVariable(
                    name="velocity",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    terms=[fl.Bell("small", -5.0, 5.0, 2.0), fl.Bell("big", 5.0, 5.0, 2.0)],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="force",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Bell("negBig", -5.0, 1.67, 8.0),
                        fl.Bell("negSmall", -1.67, 1.67, 8.0),
                        fl.Bell("posSmall", 1.67, 1.67, 8.0),
                        fl.Bell("posBig", 5.0, 1.67, 8.0),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.Minimum(),
                    disjunction=fl.Maximum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if angle is small and velocity is small then force is negBig"
                        ),
                        fl.Rule.create(
                            "if angle is small and velocity is big then force is negSmall"
                        ),
                        fl.Rule.create(
                            "if angle is big and velocity is small then force is posSmall"
                        ),
                        fl.Rule.create("if angle is big and velocity is big then force is posBig"),
                    ],
                )
            ],
        )
