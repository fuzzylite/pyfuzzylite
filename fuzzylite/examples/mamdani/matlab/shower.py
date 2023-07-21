import fuzzylite as fl


class Shower:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="shower",
            input_variables=[
                fl.InputVariable(
                    name="temp",
                    minimum=-20.0,
                    maximum=20.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("cold", -30.0, -30.0, -15.0, 0.0),
                        fl.Triangle("good", -10.0, 0.0, 10.0),
                        fl.Trapezoid("hot", 0.0, 15.0, 30.0, 30.0),
                    ],
                ),
                fl.InputVariable(
                    name="flow",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("soft", -3.0, -3.0, -0.8, 0.0),
                        fl.Triangle("good", -0.4, 0.0, 0.4),
                        fl.Trapezoid("hard", 0.0, 0.8, 3.0, 3.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="cold",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("closeFast", -1.0, -0.6, -0.3),
                        fl.Triangle("closeSlow", -0.6, -0.3, 0.0),
                        fl.Triangle("steady", -0.3, 0.0, 0.3),
                        fl.Triangle("openSlow", 0.0, 0.3, 0.6),
                        fl.Triangle("openFast", 0.3, 0.6, 1.0),
                    ],
                ),
                fl.OutputVariable(
                    name="hot",
                    minimum=-1.0,
                    maximum=1.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Triangle("closeFast", -1.0, -0.6, -0.3),
                        fl.Triangle("closeSlow", -0.6, -0.3, 0.0),
                        fl.Triangle("steady", -0.3, 0.0, 0.3),
                        fl.Triangle("openSlow", 0.0, 0.3, 0.6),
                        fl.Triangle("openFast", 0.3, 0.6, 1.0),
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
                            "if temp is cold and flow is soft then cold is openSlow and hot is openFast"
                        ),
                        fl.Rule.create(
                            "if temp is cold and flow is good then cold is closeSlow and hot is openSlow"
                        ),
                        fl.Rule.create(
                            "if temp is cold and flow is hard then cold is closeFast and hot is closeSlow"
                        ),
                        fl.Rule.create(
                            "if temp is good and flow is soft then cold is openSlow and hot is openSlow"
                        ),
                        fl.Rule.create(
                            "if temp is good and flow is good then cold is steady and hot is steady"
                        ),
                        fl.Rule.create(
                            "if temp is good and flow is hard then cold is closeSlow and hot is closeSlow"
                        ),
                        fl.Rule.create(
                            "if temp is hot and flow is soft then cold is openFast and hot is openSlow"
                        ),
                        fl.Rule.create(
                            "if temp is hot and flow is good then cold is openSlow and hot is closeSlow"
                        ),
                        fl.Rule.create(
                            "if temp is hot and flow is hard then cold is closeSlow and hot is closeFast"
                        ),
                    ],
                )
            ],
        )
