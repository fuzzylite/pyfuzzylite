import fuzzylite as fl

engine = fl.Engine(
    name="slcp1",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in1",
        description="",
        enabled=True,
        minimum=-0.300,
        maximum=0.300,
        lock_range=False
    ),
    fl.InputVariable(
        name="in2",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False
    ),
    fl.InputVariable(
        name="in3",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False
    ),
    fl.InputVariable(
        name="in4",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False
    ),
    fl.InputVariable(
        name="in5",
        description="",
        enabled=True,
        minimum=0.500,
        maximum=1.500,
        lock_range=False,
        terms=[
            fl.Gaussian("small", 0.500, 0.200),
            fl.Gaussian("medium", 1.000, 0.200),
            fl.Gaussian("large", 1.500, 0.200)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="out",
        description="",
        enabled=True,
        minimum=-10.000,
        maximum=10.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("outmf1", [32.166, 5.835, 3.162, 3.757, 0.000, 0.000], engine),
            fl.Linear("outmf2", [39.012, 9.947, 3.162, 4.269, 0.000, 0.000], engine),
            fl.Linear("outmf3", [45.009, 13.985, 3.162, 4.666, 0.000, 0.000], engine)
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
            fl.Rule.create("if in5 is small then out is outmf1", engine),
            fl.Rule.create("if in5 is medium then out is outmf2", engine),
            fl.Rule.create("if in5 is large then out is outmf3", engine)
        ]
    )
]
