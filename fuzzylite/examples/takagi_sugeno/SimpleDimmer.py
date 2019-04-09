import fuzzylite as fl

engine = fl.Engine(
    name="SimpleDimmer",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="Ambient",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        terms=[
            fl.Triangle("DARK", 0.000, 0.250, 0.500),
            fl.Triangle("MEDIUM", 0.250, 0.500, 0.750),
            fl.Triangle("BRIGHT", 0.500, 0.750, 1.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="Power",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("LOW", 0.250),
            fl.Constant("MEDIUM", 0.500),
            fl.Constant("HIGH", 0.750)
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
            fl.Rule.create("if Ambient is DARK then Power is HIGH", engine),
            fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM", engine),
            fl.Rule.create("if Ambient is BRIGHT then Power is LOW", engine)
        ]
    )
]
