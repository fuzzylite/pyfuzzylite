import fuzzylite as fl


class CubicApproximator:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="cubic_approximator",
            input_variables=[
                fl.InputVariable(
                    name="X",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    terms=[
                        fl.Triangle("AboutNegFive", -6.0, -5.0, -4.0),
                        fl.Triangle("AboutNegFour", -5.0, -4.0, -3.0),
                        fl.Triangle("AboutNegThree", -4.0, -3.0, -2.0),
                        fl.Triangle("AboutNegTwo", -3.0, -2.0, -1.0),
                        fl.Triangle("AboutNegOne", -2.0, -1.0, 0.0),
                        fl.Triangle("AboutZero", -1.0, 0.0, 1.0),
                        fl.Triangle("AboutOne", 0.0, 1.0, 2.0),
                        fl.Triangle("AboutTwo", 1.0, 2.0, 3.0),
                        fl.Triangle("AboutThree", 2.0, 3.0, 4.0),
                        fl.Triangle("AboutFour", 3.0, 4.0, 5.0),
                        fl.Triangle("AboutFive", 4.0, 5.0, 6.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="ApproxXCubed",
                    minimum=-5.0,
                    maximum=5.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("TangentatNegFive", [75.0, 250.0]),
                        fl.Linear("TangentatNegFour", [48.0, 128.0]),
                        fl.Linear("TangentatNegThree", [27.0, 54.0]),
                        fl.Linear("TangentatNegTwo", [12.0, 16.0]),
                        fl.Linear("TangentatNegOne", [3.0, 2.0]),
                        fl.Linear("TangentatZero", [0.0, 0.0]),
                        fl.Linear("TangentatOne", [3.0, -2.0]),
                        fl.Linear("TangentatTwo", [12.0, -16.0]),
                        fl.Linear("TangentatThree", [27.0, -54.0]),
                        fl.Linear("TangentatFour", [48.0, -128.0]),
                        fl.Linear("TangentatFive", [75.0, -250.0]),
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
                        fl.Rule.create(
                            "if X is AboutNegFive then ApproxXCubed is TangentatNegFive"
                        ),
                        fl.Rule.create(
                            "if X is AboutNegFour then ApproxXCubed is TangentatNegFour"
                        ),
                        fl.Rule.create(
                            "if X is AboutNegThree then ApproxXCubed is TangentatNegThree"
                        ),
                        fl.Rule.create("if X is AboutNegTwo then ApproxXCubed is TangentatNegTwo"),
                        fl.Rule.create("if X is AboutNegOne then ApproxXCubed is TangentatNegOne"),
                        fl.Rule.create("if X is AboutZero then ApproxXCubed is TangentatZero"),
                        fl.Rule.create("if X is AboutOne then ApproxXCubed is TangentatOne"),
                        fl.Rule.create("if X is AboutTwo then ApproxXCubed is TangentatTwo"),
                        fl.Rule.create("if X is AboutThree then ApproxXCubed is TangentatThree"),
                        fl.Rule.create("if X is AboutFour then ApproxXCubed is TangentatFour"),
                        fl.Rule.create("if X is AboutFive then ApproxXCubed is TangentatFive"),
                    ],
                )
            ],
        )
