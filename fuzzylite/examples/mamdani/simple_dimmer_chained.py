import fuzzylite as fl


class SimpleDimmerChained:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="SimpleDimmerChained",
            input_variables=[
                fl.InputVariable(
                    name="Ambient",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[
                        fl.Triangle("DARK", 0.0, 0.25, 0.5),
                        fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
                        fl.Triangle("BRIGHT", 0.5, 0.75, 1.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="Power",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("LOW", 0.0, 0.25, 0.5),
                        fl.Triangle("MEDIUM", 0.25, 0.5, 0.75),
                        fl.Triangle("HIGH", 0.5, 0.75, 1.0),
                    ],
                ),
                fl.OutputVariable(
                    name="InversePower",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Cosine("LOW", 0.2, 0.5),
                        fl.Cosine("MEDIUM", 0.5, 0.5),
                        fl.Cosine("HIGH", 0.8, 0.5),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=None,
                    disjunction=None,
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if Ambient is DARK then Power is HIGH"),
                        fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM"),
                        fl.Rule.create("if Ambient is BRIGHT then Power is LOW"),
                        fl.Rule.create("if Power is LOW then InversePower is HIGH"),
                        fl.Rule.create("if Power is MEDIUM then InversePower is MEDIUM"),
                        fl.Rule.create("if Power is HIGH then InversePower is LOW"),
                    ],
                )
            ],
        )
