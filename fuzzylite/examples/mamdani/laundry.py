import fuzzylite as fl


class Laundry:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="Laundry",
            input_variables=[
                fl.InputVariable(
                    name="Load",
                    minimum=0.0,
                    maximum=6.0,
                    lock_range=False,
                    terms=[
                        fl.Discrete(
                            "small",
                            fl.array(
                                [
                                    fl.array([0.0, 1.0]),
                                    fl.array([1.0, 1.0]),
                                    fl.array([2.0, 0.8]),
                                    fl.array([5.0, 0.0]),
                                ]
                            ),
                        ),
                        fl.Discrete(
                            "normal",
                            fl.array(
                                [fl.array([3.0, 0.0]), fl.array([4.0, 1.0]), fl.array([6.0, 0.0])]
                            ),
                        ),
                    ],
                ),
                fl.InputVariable(
                    name="Dirt",
                    minimum=0.0,
                    maximum=6.0,
                    lock_range=False,
                    terms=[
                        fl.Discrete(
                            "low",
                            fl.array(
                                [fl.array([0.0, 1.0]), fl.array([2.0, 0.8]), fl.array([5.0, 0.0])]
                            ),
                        ),
                        fl.Discrete(
                            "high",
                            fl.array(
                                [
                                    fl.array([1.0, 0.0]),
                                    fl.array([2.0, 0.2]),
                                    fl.array([4.0, 0.8]),
                                    fl.array([6.0, 1.0]),
                                ]
                            ),
                        ),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="Detergent",
                    minimum=0.0,
                    maximum=80.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.MeanOfMaximum(),
                    terms=[
                        fl.Discrete(
                            "less_than_usual",
                            fl.array(
                                [
                                    fl.array([10.0, 0.0]),
                                    fl.array([40.0, 1.0]),
                                    fl.array([50.0, 0.0]),
                                ]
                            ),
                        ),
                        fl.Discrete(
                            "usual",
                            fl.array(
                                [
                                    fl.array([40.0, 0.0]),
                                    fl.array([50.0, 1.0]),
                                    fl.array([60.0, 1.0]),
                                    fl.array([80.0, 0.0]),
                                ]
                            ),
                        ),
                        fl.Discrete(
                            "more_than_usual",
                            fl.array([fl.array([50.0, 0.0]), fl.array([80.0, 1.0])]),
                        ),
                    ],
                ),
                fl.OutputVariable(
                    name="Cycle",
                    minimum=0.0,
                    maximum=20.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.MeanOfMaximum(),
                    terms=[
                        fl.Discrete(
                            "short",
                            fl.array(
                                [fl.array([0.0, 1.0]), fl.array([10.0, 1.0]), fl.array([20.0, 0.0])]
                            ),
                        ),
                        fl.Discrete(
                            "long", fl.array([fl.array([10.0, 0.0]), fl.array([20.0, 1.0])])
                        ),
                    ],
                ),
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
                            "if Load is small and Dirt is not high then Detergent is less_than_usual"
                        ),
                        fl.Rule.create("if Load is small and Dirt is high then Detergent is usual"),
                        fl.Rule.create(
                            "if Load is normal and Dirt is low then Detergent is less_than_usual"
                        ),
                        fl.Rule.create(
                            "if Load is normal and Dirt is high then Detergent is more_than_usual"
                        ),
                        fl.Rule.create(
                            "if Detergent is usual or Detergent is less_than_usual then Cycle is short"
                        ),
                        fl.Rule.create("if Detergent is more_than_usual then Cycle is long"),
                    ],
                )
            ],
        )
