import fuzzylite as fl

engine = fl.Engine(
    name="slbb",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in1",
        description="",
        enabled=True,
        minimum=-1.500,
        maximum=1.500,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -1.500, 1.500, 2.000),
            fl.Bell("in1mf2", 1.500, 1.500, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in2",
        description="",
        enabled=True,
        minimum=-1.500,
        maximum=1.500,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", -1.500, 1.500, 2.000),
            fl.Bell("in2mf2", 1.500, 1.500, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in3",
        description="",
        enabled=True,
        minimum=-0.200,
        maximum=0.200,
        lock_range=False,
        terms=[
            fl.Bell("in3mf1", -0.200, 0.200, 2.000),
            fl.Bell("in3mf2", 0.200, 0.200, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in4",
        description="",
        enabled=True,
        minimum=-0.400,
        maximum=0.400,
        lock_range=False,
        terms=[
            fl.Bell("in4mf1", -0.400, 0.400, 2.000),
            fl.Bell("in4mf2", 0.400, 0.400, 2.000)
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
            fl.Linear("outmf1", [1.015, 2.234, -12.665, -4.046, 0.026], engine),
            fl.Linear("outmf2", [1.161, 1.969, -9.396, -6.165, 0.474], engine),
            fl.Linear("outmf3", [1.506, 2.234, -12.990, -1.865, 1.426], engine),
            fl.Linear("outmf4", [0.734, 1.969, -9.381, -4.688, -0.880], engine),
            fl.Linear("outmf5", [0.734, 2.234, -12.853, -6.110, -1.034], engine),
            fl.Linear("outmf6", [1.413, 1.969, -9.485, -6.592, 1.159], engine),
            fl.Linear("outmf7", [1.225, 2.234, -12.801, -3.929, 0.366], engine),
            fl.Linear("outmf8", [0.985, 1.969, -9.291, -5.115, -0.195], engine),
            fl.Linear("outmf9", [0.985, 1.969, -9.292, -5.115, 0.195], engine),
            fl.Linear("outmf10", [1.225, 2.234, -12.802, -3.929, -0.366], engine),
            fl.Linear("outmf11", [1.413, 1.969, -9.485, -6.592, -1.159], engine),
            fl.Linear("outmf12", [0.734, 2.234, -12.853, -6.110, 1.034], engine),
            fl.Linear("outmf13", [0.734, 1.969, -9.381, -4.688, 0.880], engine),
            fl.Linear("outmf14", [1.506, 2.234, -12.990, -1.865, -1.426], engine),
            fl.Linear("outmf15", [1.161, 1.969, -9.396, -6.165, -0.474], engine),
            fl.Linear("outmf16", [1.015, 2.234, -12.665, -4.046, -0.026], engine)
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
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf1 then out is outmf1", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf2 then out is outmf2", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf1 then out is outmf3", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf2 then out is outmf4", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf1 then out is outmf5", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf2 then out is outmf6", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf1 then out is outmf7", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf2 then out is outmf8", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf1 then out is outmf9", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf1 and in4 is in4mf2 then out is outmf10", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf1 then out is outmf11", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf1 and in3 is in3mf2 and in4 is in4mf2 then out is outmf12", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf1 then out is outmf13", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf1 and in4 is in4mf2 then out is outmf14", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf1 then out is outmf15", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in2mf2 and in3 is in3mf2 and in4 is in4mf2 then out is outmf16", engine)
        ]
    )
]
