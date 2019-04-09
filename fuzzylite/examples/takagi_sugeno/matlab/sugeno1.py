import fuzzylite as fl

engine = fl.Engine(
    name="sugeno1",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="input",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        terms=[
            fl.Gaussian("low", -5.000, 4.000),
            fl.Gaussian("high", 5.000, 4.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="output",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("line1", [-1.000, -1.000], engine),
            fl.Linear("line2", [1.000, -1.000], engine)
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
            fl.Rule.create("if input is low then output is line1", engine),
            fl.Rule.create("if input is high then output is line2", engine)
        ]
    )
]
