import fuzzylite as fl


class Sugeno1:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="sugeno1",
            input_variables=[
                fl.InputVariable(
                    name="input",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    terms=[fl.Gaussian("low", -5.0, 4.0), fl.Gaussian("high", 5.0, 4.0)],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="output",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[fl.Linear("line1", [-1.0, -1.0]), fl.Linear("line2", [1.0, -1.0])],
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
                        fl.Rule.create("if input is low then output is line1"),
                        fl.Rule.create("if input is high then output is line2"),
                    ],
                )
            ],
        )
