import fuzzylite as fl

engine = fl.Engine(
    name="cubic_approximator",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="X",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        terms=[
            fl.Triangle("AboutNegFive", -6.000, -5.000, -4.000),
            fl.Triangle("AboutNegFour", -5.000, -4.000, -3.000),
            fl.Triangle("AboutNegThree", -4.000, -3.000, -2.000),
            fl.Triangle("AboutNegTwo", -3.000, -2.000, -1.000),
            fl.Triangle("AboutNegOne", -2.000, -1.000, 0.000),
            fl.Triangle("AboutZero", -1.000, 0.000, 1.000),
            fl.Triangle("AboutOne", 0.000, 1.000, 2.000),
            fl.Triangle("AboutTwo", 1.000, 2.000, 3.000),
            fl.Triangle("AboutThree", 2.000, 3.000, 4.000),
            fl.Triangle("AboutFour", 3.000, 4.000, 5.000),
            fl.Triangle("AboutFive", 4.000, 5.000, 6.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="ApproxXCubed",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("TangentatNegFive", [75.000, 250.000], engine),
            fl.Linear("TangentatNegFour", [48.000, 128.000], engine),
            fl.Linear("TangentatNegThree", [27.000, 54.000], engine),
            fl.Linear("TangentatNegTwo", [12.000, 16.000], engine),
            fl.Linear("TangentatNegOne", [3.000, 2.000], engine),
            fl.Linear("TangentatZero", [0.000, 0.000], engine),
            fl.Linear("TangentatOne", [3.000, -2.000], engine),
            fl.Linear("TangentatTwo", [12.000, -16.000], engine),
            fl.Linear("TangentatThree", [27.000, -54.000], engine),
            fl.Linear("TangentatFour", [48.000, -128.000], engine),
            fl.Linear("TangentatFive", [75.000, -250.000], engine)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if X is AboutNegFive then ApproxXCubed is TangentatNegFive", engine),
            fl.Rule.create("if X is AboutNegFour then ApproxXCubed is TangentatNegFour", engine),
            fl.Rule.create("if X is AboutNegThree then ApproxXCubed is TangentatNegThree", engine),
            fl.Rule.create("if X is AboutNegTwo then ApproxXCubed is TangentatNegTwo", engine),
            fl.Rule.create("if X is AboutNegOne then ApproxXCubed is TangentatNegOne", engine),
            fl.Rule.create("if X is AboutZero then ApproxXCubed is TangentatZero", engine),
            fl.Rule.create("if X is AboutOne then ApproxXCubed is TangentatOne", engine),
            fl.Rule.create("if X is AboutTwo then ApproxXCubed is TangentatTwo", engine),
            fl.Rule.create("if X is AboutThree then ApproxXCubed is TangentatThree", engine),
            fl.Rule.create("if X is AboutFour then ApproxXCubed is TangentatFour", engine),
            fl.Rule.create("if X is AboutFive then ApproxXCubed is TangentatFive", engine)
        ]
    )
]
