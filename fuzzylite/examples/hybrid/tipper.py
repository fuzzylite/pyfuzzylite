import fuzzylite as fl


class Tipper:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="tipper",
            description="(service and food) -> (tip)",
            input_variables=[
                fl.InputVariable(
                    name="service",
                    description="quality of service",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=True,
                    terms=[
                        fl.Trapezoid("poor", 0.0, 0.0, 2.5, 5.0),
                        fl.Triangle("good", 2.5, 5.0, 7.5),
                        fl.Trapezoid("excellent", 5.0, 7.5, 10.0, 10.0),
                    ],
                ),
                fl.InputVariable(
                    name="food",
                    description="quality of food",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=True,
                    terms=[
                        fl.Trapezoid("rancid", 0.0, 0.0, 2.5, 7.5),
                        fl.Trapezoid("delicious", 2.5, 7.5, 10.0, 10.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="mTip",
                    description="tip based on Mamdani inference",
                    minimum=0.0,
                    maximum=30.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("cheap", 0.0, 5.0, 10.0),
                        fl.Triangle("average", 10.0, 15.0, 20.0),
                        fl.Triangle("generous", 20.0, 25.0, 30.0),
                    ],
                ),
                fl.OutputVariable(
                    name="tsTip",
                    description="tip based on Takagi-Sugeno inference",
                    minimum=0.0,
                    maximum=30.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("cheap", 5.0),
                        fl.Constant("average", 15.0),
                        fl.Constant("generous", 25.0),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="mamdani",
                    description="Mamdani inference",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=fl.AlgebraicSum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if service is poor or food is rancid then mTip is cheap"),
                        fl.Rule.create("if service is good then mTip is average"),
                        fl.Rule.create(
                            "if service is excellent or food is delicious then mTip is generous with 0.500"
                        ),
                        fl.Rule.create(
                            "if service is excellent and food is delicious then mTip is generous"
                        ),
                    ],
                ),
                fl.RuleBlock(
                    name="takagiSugeno",
                    description="Takagi-Sugeno inference",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=fl.AlgebraicSum(),
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if service is poor or food is rancid then tsTip is cheap"),
                        fl.Rule.create("if service is good then tsTip is average"),
                        fl.Rule.create(
                            "if service is excellent or food is delicious then tsTip is generous with 0.500"
                        ),
                        fl.Rule.create(
                            "if service is excellent and food is delicious then tsTip is generous"
                        ),
                    ],
                ),
            ],
        )
