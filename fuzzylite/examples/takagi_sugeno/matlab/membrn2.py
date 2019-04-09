import fuzzylite as fl

engine = fl.Engine(
    name="membrn2",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in_n1",
        description="",
        enabled=True,
        minimum=1.000000000,
        maximum=31.000000000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", 1.152000000, 8.206000000, 0.874000000),
            fl.Bell("in1mf2", 15.882000000, 7.078000000, 0.444000000),
            fl.Bell("in1mf3", 30.575000000, 8.602000000, 0.818000000)
        ]
    ),
    fl.InputVariable(
        name="in_n2",
        description="",
        enabled=True,
        minimum=1.000000000,
        maximum=31.000000000,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", 1.426000000, 8.602000000, 0.818000000),
            fl.Bell("in2mf2", 16.118000000, 7.078000000, 0.445000000),
            fl.Bell("in2mf3", 30.847000000, 8.206000000, 0.875000000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="out1",
        description="",
        enabled=True,
        minimum=-0.334000000,
        maximum=1.000000000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [-0.035000000, 0.002000000, -0.352000000], engine),
            fl.Linear("out1mf2", [0.044000000, 0.079000000, -0.028000000], engine),
            fl.Linear("out1mf3", [-0.024000000, 0.024000000, -1.599000000], engine),
            fl.Linear("out1mf4", [-0.067000000, 0.384000000, 0.007000000], engine),
            fl.Linear("out1mf5", [0.351000000, -0.351000000, -3.663000000], engine),
            fl.Linear("out1mf6", [-0.079000000, -0.044000000, 3.909000000], engine),
            fl.Linear("out1mf7", [0.012000000, -0.012000000, -0.600000000], engine),
            fl.Linear("out1mf8", [-0.384000000, 0.067000000, 10.158000000], engine),
            fl.Linear("out1mf9", [-0.002000000, 0.035000000, -1.402000000], engine)
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
            fl.Rule.create("if in_n1 is in1mf1 and in_n2 is in2mf3 then out1 is out1mf3", engine),
            fl.Rule.create("if in_n1 is in1mf2 and in_n2 is in2mf1 then out1 is out1mf4", engine),
            fl.Rule.create("if in_n1 is in1mf2 and in_n2 is in2mf2 then out1 is out1mf5", engine),
            fl.Rule.create("if in_n1 is in1mf2 and in_n2 is in2mf3 then out1 is out1mf6", engine),
            fl.Rule.create("if in_n1 is in1mf3 and in_n2 is in2mf1 then out1 is out1mf7", engine),
            fl.Rule.create("if in_n1 is in1mf3 and in_n2 is in2mf2 then out1 is out1mf8", engine),
            fl.Rule.create("if in_n1 is in1mf3 and in_n2 is in2mf3 then out1 is out1mf9", engine)
        ]
    )
]
