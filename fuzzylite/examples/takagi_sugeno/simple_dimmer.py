import fuzzylite as fl


class SimpleDimmer:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="SimpleDimmer",
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
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("LOW", 0.25),
                        fl.Constant("MEDIUM", 0.5),
                        fl.Constant("HIGH", 0.75),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=None,
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if Ambient is DARK then Power is HIGH"),
                        fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM"),
                        fl.Rule.create("if Ambient is BRIGHT then Power is LOW"),
                    ],
                )
            ],
        )
