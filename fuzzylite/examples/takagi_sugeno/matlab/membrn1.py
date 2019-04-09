import fuzzylite as fl

engine = fl.Engine(
    name="membrn1",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in_n1",
        description="",
        enabled=True,
        minimum=1.000,
        maximum=31.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", 2.253, 16.220, 5.050),
            fl.Bell("in1mf2", 31.260, 15.021, 1.843)
        ]
    ),
    fl.InputVariable(
        name="in_n2",
        description="",
        enabled=True,
        minimum=1.000,
        maximum=31.000,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", 0.740, 15.021, 1.843),
            fl.Bell("in2mf2", 29.747, 16.220, 5.050)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="out1",
        description="",
        enabled=True,
        minimum=-0.334,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [0.026, 0.071, -0.615], engine),
            fl.Linear("out1mf2", [-0.036, 0.036, -1.169], engine),
            fl.Linear("out1mf3", [-0.094, 0.094, 2.231], engine),
            fl.Linear("out1mf4", [-0.071, -0.026, 2.479], engine)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if in_n1 is in1mf1 and in_n2 is in2mf1 then out1 is out1mf1", engine),
            fl.Rule.create("if in_n1 is in1mf1 and in_n2 is in2mf2 then out1 is out1mf2", engine),
            fl.Rule.create("if in_n1 is in1mf2 and in_n2 is in2mf1 then out1 is out1mf3", engine),
            fl.Rule.create("if in_n1 is in1mf2 and in_n2 is in2mf2 then out1 is out1mf4", engine)
        ]
    )
]
