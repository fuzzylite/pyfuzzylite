import fuzzylite as fl

engine = fl.Engine(
    name="slcp",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in1",
        description="",
        enabled=True,
        minimum=-0.300,
        maximum=0.300,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -0.300, 0.300, 2.000),
            fl.Bell("in1mf2", 0.300, 0.300, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in2",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", -1.000, 1.000, 2.000),
            fl.Bell("in2mf2", 1.000, 1.000, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in3",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False,
        terms=[
            fl.Bell("in3mf1", -3.000, 3.000, 2.000),
            fl.Bell("in3mf2", 3.000, 3.000, 2.000)
        ]
    ),
    fl.InputVariable(
        name="in4",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False,
        terms=[
            fl.Bell("in4mf1", -3.000, 3.000, 2.000),
            fl.Bell("in4mf2", 3.000, 3.000, 2.000)
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
            fl.Linear("outmf1", [41.373, 10.030, 3.162, 4.288, 0.339], engine),
            fl.Linear("outmf2", [40.409, 10.053, 3.162, 4.288, 0.207], engine),
            fl.Linear("outmf3", [41.373, 10.030, 3.162, 4.288, 0.339], engine),
            fl.Linear("outmf4", [40.409, 10.053, 3.162, 4.288, 0.207], engine),
            fl.Linear("outmf5", [38.561, 10.177, 3.162, 4.288, -0.049], engine),
            fl.Linear("outmf6", [37.596, 10.154, 3.162, 4.288, -0.181], engine),
            fl.Linear("outmf7", [38.561, 10.177, 3.162, 4.288, -0.049], engine),
            fl.Linear("outmf8", [37.596, 10.154, 3.162, 4.288, -0.181], engine),
            fl.Linear("outmf9", [37.596, 10.154, 3.162, 4.288, 0.181], engine),
            fl.Linear("outmf10", [38.561, 10.177, 3.162, 4.288, 0.049], engine),
            fl.Linear("outmf11", [37.596, 10.154, 3.162, 4.288, 0.181], engine),
            fl.Linear("outmf12", [38.561, 10.177, 3.162, 4.288, 0.049], engine),
            fl.Linear("outmf13", [40.408, 10.053, 3.162, 4.288, -0.207], engine),
            fl.Linear("outmf14", [41.373, 10.030, 3.162, 4.288, -0.339], engine),
            fl.Linear("outmf15", [40.408, 10.053, 3.162, 4.288, -0.207], engine),
            fl.Linear("outmf16", [41.373, 10.030, 3.162, 4.288, -0.339], engine)
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
