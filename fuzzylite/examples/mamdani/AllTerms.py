import fuzzylite as fl

engine = fl.Engine(
    name="AllTerms",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="AllInputTerms",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=6.500,
        lock_range=False,
        terms=[
            fl.Sigmoid("A", 0.500, -20.000),
            fl.ZShape("B", 0.000, 1.000),
            fl.Ramp("C", 1.000, 0.000),
            fl.Triangle("D", 0.500, 1.000, 1.500),
            fl.Trapezoid("E", 1.000, 1.250, 1.750, 2.000),
            fl.Concave("F", 0.850, 0.250),
            fl.Rectangle("G", 1.750, 2.250),
            fl.Discrete("H", [2.000, 0.000, 2.250, 1.000, 2.500, 0.500, 2.750, 1.000, 3.000, 0.000]),
            fl.Gaussian("I", 3.000, 0.200),
            fl.Cosine("J", 3.250, 0.650),
            fl.GaussianProduct("K", 3.500, 0.100, 3.300, 0.300),
            fl.Spike("L", 3.640, 1.040),
            fl.Bell("M", 4.000, 0.250, 3.000),
            fl.PiShape("N", 4.000, 4.500, 4.500, 5.000),
            fl.Concave("O", 5.650, 6.250),
            fl.SigmoidDifference("P", 4.750, 10.000, 30.000, 5.250),
            fl.SigmoidProduct("Q", 5.250, 20.000, -10.000, 5.750),
            fl.Ramp("R", 5.500, 6.500),
            fl.SShape("S", 5.500, 6.500),
            fl.Sigmoid("T", 6.000, 20.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="AllOutputTerms",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=6.500,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Sigmoid("A", 0.500, -20.000),
            fl.ZShape("B", 0.000, 1.000),
            fl.Ramp("C", 1.000, 0.000),
            fl.Triangle("D", 0.500, 1.000, 1.500),
            fl.Trapezoid("E", 1.000, 1.250, 1.750, 2.000),
            fl.Concave("F", 0.850, 0.250),
            fl.Rectangle("G", 1.750, 2.250),
            fl.Discrete("H", [2.000, 0.000, 2.250, 1.000, 2.500, 0.500, 2.750, 1.000, 3.000, 0.000]),
            fl.Gaussian("I", 3.000, 0.200),
            fl.Cosine("J", 3.250, 0.650),
            fl.GaussianProduct("K", 3.500, 0.100, 3.300, 0.300),
            fl.Spike("L", 3.640, 1.040),
            fl.Bell("M", 4.000, 0.250, 3.000),
            fl.PiShape("N", 4.000, 4.500, 4.500, 5.000),
            fl.Concave("O", 5.650, 6.250),
            fl.SigmoidDifference("P", 4.750, 10.000, 30.000, 5.250),
            fl.SigmoidProduct("Q", 5.250, 20.000, -10.000, 5.750),
            fl.Ramp("R", 5.500, 6.500),
            fl.SShape("S", 5.500, 6.500),
            fl.Sigmoid("T", 6.000, 20.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=fl.Maximum(),
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if AllInputTerms is A then AllOutputTerms is T", engine),
            fl.Rule.create("if AllInputTerms is B then AllOutputTerms is S", engine),
            fl.Rule.create("if AllInputTerms is C then AllOutputTerms is R", engine),
            fl.Rule.create("if AllInputTerms is D then AllOutputTerms is Q", engine),
            fl.Rule.create("if AllInputTerms is E then AllOutputTerms is P", engine),
            fl.Rule.create("if AllInputTerms is F then AllOutputTerms is O", engine),
            fl.Rule.create("if AllInputTerms is G then AllOutputTerms is N", engine),
            fl.Rule.create("if AllInputTerms is H then AllOutputTerms is M", engine),
            fl.Rule.create("if AllInputTerms is I then AllOutputTerms is L", engine),
            fl.Rule.create("if AllInputTerms is J then AllOutputTerms is K", engine),
            fl.Rule.create("if AllInputTerms is K then AllOutputTerms is J", engine),
            fl.Rule.create("if AllInputTerms is L then AllOutputTerms is I", engine),
            fl.Rule.create("if AllInputTerms is M then AllOutputTerms is H", engine),
            fl.Rule.create("if AllInputTerms is N then AllOutputTerms is G", engine),
            fl.Rule.create("if AllInputTerms is O then AllOutputTerms is F", engine),
            fl.Rule.create("if AllInputTerms is P then AllOutputTerms is E", engine),
            fl.Rule.create("if AllInputTerms is Q then AllOutputTerms is D", engine),
            fl.Rule.create("if AllInputTerms is R then AllOutputTerms is C", engine),
            fl.Rule.create("if AllInputTerms is S then AllOutputTerms is B", engine),
            fl.Rule.create("if AllInputTerms is T then AllOutputTerms is A", engine)
        ]
    )
]
