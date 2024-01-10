import fuzzylite as fl


class AllTerms:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="AllTerms",
            input_variables=[
                fl.InputVariable(
                    name="AllInputTerms",
                    minimum=0.0,
                    maximum=6.5,
                    lock_range=False,
                    terms=[
                        fl.Sigmoid("A", 0.5, -20.0),
                        fl.ZShape("B", 0.0, 1.0),
                        fl.Ramp("C", 1.0, 0.0),
                        fl.Triangle("D", 0.5, 1.0, 1.5),
                        fl.Trapezoid("E", 1.0, 1.25, 1.75, 2.0),
                        fl.Concave("F", 0.85, 0.25),
                        fl.Rectangle("G", 1.75, 2.25),
                        fl.Discrete(
                            "H",
                            fl.array(
                                [
                                    fl.array([2.0, 0.0]),
                                    fl.array([2.25, 1.0]),
                                    fl.array([2.5, 0.5]),
                                    fl.array([2.75, 1.0]),
                                    fl.array([3.0, 0.0]),
                                ]
                            ),
                        ),
                        fl.Gaussian("I", 3.0, 0.2),
                        fl.Cosine("J", 3.25, 0.65),
                        fl.GaussianProduct("K", 3.5, 0.1, 3.3, 0.3),
                        fl.Spike("L", 3.64, 1.04),
                        fl.Bell("M", 4.0, 0.25, 3.0),
                        fl.PiShape("N", 4.0, 4.5, 4.5, 5.0),
                        fl.Concave("O", 5.65, 6.25),
                        fl.SigmoidDifference("P", 4.75, 10.0, 30.0, 5.25),
                        fl.SigmoidProduct("Q", 5.25, 20.0, -10.0, 5.75),
                        fl.Ramp("R", 5.5, 6.5),
                        fl.SShape("S", 5.5, 6.5),
                        fl.Sigmoid("T", 6.0, 20.0),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="AllOutputTerms",
                    minimum=0.0,
                    maximum=6.5,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(),
                    terms=[
                        fl.Sigmoid("A", 0.5, -20.0),
                        fl.ZShape("B", 0.0, 1.0),
                        fl.Ramp("C", 1.0, 0.0),
                        fl.Triangle("D", 0.5, 1.0, 1.5),
                        fl.Trapezoid("E", 1.0, 1.25, 1.75, 2.0),
                        fl.Concave("F", 0.85, 0.25),
                        fl.Rectangle("G", 1.75, 2.25),
                        fl.Discrete(
                            "H",
                            fl.array(
                                [
                                    fl.array([2.0, 0.0]),
                                    fl.array([2.25, 1.0]),
                                    fl.array([2.5, 0.5]),
                                    fl.array([2.75, 1.0]),
                                    fl.array([3.0, 0.0]),
                                ]
                            ),
                        ),
                        fl.Gaussian("I", 3.0, 0.2),
                        fl.Cosine("J", 3.25, 0.65),
                        fl.GaussianProduct("K", 3.5, 0.1, 3.3, 0.3),
                        fl.Spike("L", 3.64, 1.04),
                        fl.Bell("M", 4.0, 0.25, 3.0),
                        fl.PiShape("N", 4.0, 4.5, 4.5, 5.0),
                        fl.Concave("O", 5.65, 6.25),
                        fl.SigmoidDifference("P", 4.75, 10.0, 30.0, 5.25),
                        fl.SigmoidProduct("Q", 5.25, 20.0, -10.0, 5.75),
                        fl.Ramp("R", 5.5, 6.5),
                        fl.SShape("S", 5.5, 6.5),
                        fl.Sigmoid("T", 6.0, 20.0),
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
                        fl.Rule.create("if AllInputTerms is A then AllOutputTerms is T"),
                        fl.Rule.create("if AllInputTerms is B then AllOutputTerms is S"),
                        fl.Rule.create("if AllInputTerms is C then AllOutputTerms is R"),
                        fl.Rule.create("if AllInputTerms is D then AllOutputTerms is Q"),
                        fl.Rule.create("if AllInputTerms is E then AllOutputTerms is P"),
                        fl.Rule.create("if AllInputTerms is F then AllOutputTerms is O"),
                        fl.Rule.create("if AllInputTerms is G then AllOutputTerms is N"),
                        fl.Rule.create("if AllInputTerms is H then AllOutputTerms is M"),
                        fl.Rule.create("if AllInputTerms is I then AllOutputTerms is L"),
                        fl.Rule.create("if AllInputTerms is J then AllOutputTerms is K"),
                        fl.Rule.create("if AllInputTerms is K then AllOutputTerms is J"),
                        fl.Rule.create("if AllInputTerms is L then AllOutputTerms is I"),
                        fl.Rule.create("if AllInputTerms is M then AllOutputTerms is H"),
                        fl.Rule.create("if AllInputTerms is N then AllOutputTerms is G"),
                        fl.Rule.create("if AllInputTerms is O then AllOutputTerms is F"),
                        fl.Rule.create("if AllInputTerms is P then AllOutputTerms is E"),
                        fl.Rule.create("if AllInputTerms is Q then AllOutputTerms is D"),
                        fl.Rule.create("if AllInputTerms is R then AllOutputTerms is C"),
                        fl.Rule.create("if AllInputTerms is S then AllOutputTerms is B"),
                        fl.Rule.create("if AllInputTerms is T then AllOutputTerms is A"),
                    ],
                )
            ],
        )
