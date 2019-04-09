import fuzzylite as fl

engine = fl.Engine(
    name="sltbu_fl",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="distance",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=25.000,
        lock_range=False,
        terms=[
            fl.ZShape("near", 1.000, 2.000),
            fl.SShape("far", 1.000, 2.000)
        ]
    ),
    fl.InputVariable(
        name="control1",
        description="",
        enabled=True,
        minimum=-0.785,
        maximum=0.785,
        lock_range=False
    ),
    fl.InputVariable(
        name="control2",
        description="",
        enabled=True,
        minimum=-0.785,
        maximum=0.785,
        lock_range=False
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="control",
        description="",
        enabled=True,
        minimum=-0.785,
        maximum=0.785,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [0.000, 0.000, 1.000, 0.000], engine),
            fl.Linear("out1mf2", [0.000, 1.000, 0.000, 0.000], engine)
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
            fl.Rule.create("if distance is near then control is out1mf1", engine),
            fl.Rule.create("if distance is far then control is out1mf2", engine)
        ]
    )
]
