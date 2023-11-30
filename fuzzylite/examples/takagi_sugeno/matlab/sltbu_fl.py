import fuzzylite as fl


class SltbuFl:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="sltbu_fl",
            input_variables=[
                fl.InputVariable(
                    name="distance",
                    minimum=0.0,
                    maximum=25.0,
                    lock_range=False,
                    terms=[fl.ZShape("near", 1.0, 2.0), fl.SShape("far", 1.0, 2.0)],
                ),
                fl.InputVariable(
                    name="control1", minimum=-0.785, maximum=0.785, lock_range=False, terms=[]
                ),
                fl.InputVariable(
                    name="control2", minimum=-0.785, maximum=0.785, lock_range=False, terms=[]
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="control",
                    minimum=-0.785,
                    maximum=0.785,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("out1mf1", [0.0, 0.0, 1.0, 0.0]),
                        fl.Linear("out1mf2", [0.0, 1.0, 0.0, 0.0]),
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
                        fl.Rule.create("if distance is near then control is out1mf1"),
                        fl.Rule.create("if distance is far then control is out1mf2"),
                    ],
                )
            ],
        )
