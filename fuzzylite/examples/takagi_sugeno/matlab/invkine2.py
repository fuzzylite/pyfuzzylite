import fuzzylite as fl

engine = fl.Engine(
    name="invkine2",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="input1",
        description="",
        enabled=True,
        minimum=-6.287,
        maximum=17.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -5.763, 3.015, 1.851),
            fl.Bell("in1mf2", -1.624, 3.130, 2.111),
            fl.Bell("in1mf3", 3.552, 3.193, 2.104),
            fl.Bell("in1mf4", 8.273, 2.907, 1.985),
            fl.Bell("in1mf5", 13.232, 2.708, 2.056),
            fl.Bell("in1mf6", 17.783, 1.635, 1.897)
        ]
    ),
    fl.InputVariable(
        name="input2",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=16.972,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", 0.005, 1.877, 1.995),
            fl.Bell("in2mf2", 3.312, 2.017, 1.829),
            fl.Bell("in2mf3", 6.568, 2.261, 1.793),
            fl.Bell("in2mf4", 10.111, 2.741, 1.978),
            fl.Bell("in2mf5", 14.952, 2.045, 1.783),
            fl.Bell("in2mf6", 17.910, 0.824, 1.734)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="output",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=3.100,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [-0.048, 1.456, 2.222], engine),
            fl.Linear("out1mf2", [-0.218, -0.305, 2.042], engine),
            fl.Linear("out1mf3", [0.026, -0.141, 3.067], engine),
            fl.Linear("out1mf4", [0.052, -0.150, 3.419], engine),
            fl.Linear("out1mf5", [0.113, -0.189, 4.339], engine),
            fl.Linear("out1mf6", [2.543, 0.361, -2.738], engine),
            fl.Linear("out1mf7", [2.517, -6.809, 23.353], engine),
            fl.Linear("out1mf8", [-0.208, -0.394, 4.472], engine),
            fl.Linear("out1mf9", [-0.046, -0.300, 4.452], engine),
            fl.Linear("out1mf10", [-0.006, -0.217, 4.195], engine),
            fl.Linear("out1mf11", [0.089, -0.254, 4.992], engine),
            fl.Linear("out1mf12", [-0.033, 0.103, -2.012], engine),
            fl.Linear("out1mf13", [1.355, 1.228, -5.678], engine),
            fl.Linear("out1mf14", [-0.245, -0.124, 3.753], engine),
            fl.Linear("out1mf15", [-0.099, -0.111, 3.304], engine),
            fl.Linear("out1mf16", [-0.052, -0.163, 3.560], engine),
            fl.Linear("out1mf17", [0.099, -0.260, 4.662], engine),
            fl.Linear("out1mf18", [0.082, -1.849, 31.104], engine),
            fl.Linear("out1mf19", [2.180, -2.963, -0.061], engine),
            fl.Linear("out1mf20", [-0.982, 0.510, 5.657], engine),
            fl.Linear("out1mf21", [-0.087, -0.179, 3.744], engine),
            fl.Linear("out1mf22", [-0.124, -0.161, 4.094], engine),
            fl.Linear("out1mf23", [0.383, 0.007, -1.559], engine),
            fl.Linear("out1mf24", [-8.415, 2.083, 5.177], engine),
            fl.Linear("out1mf25", [1.721, -15.079, -0.687], engine),
            fl.Linear("out1mf26", [-1.043, -0.786, 20.510], engine),
            fl.Linear("out1mf27", [-0.249, -0.396, 6.995], engine),
            fl.Linear("out1mf28", [-0.076, -0.245, 4.416], engine),
            fl.Linear("out1mf29", [0.765, -1.488, 17.384], engine),
            fl.Linear("out1mf30", [-21.210, -43.022, -2.522], engine),
            fl.Linear("out1mf31", [-0.661, 3.523, 6.215], engine),
            fl.Linear("out1mf32", [-1.998, 1.582, 33.256], engine),
            fl.Linear("out1mf33", [-2.068, 5.673, 6.520], engine),
            fl.Linear("out1mf34", [-5.044, 7.093, 3.516], engine),
            fl.Linear("out1mf35", [-46.049, -35.021, -2.926], engine),
            fl.Linear("out1mf36", [-0.448, -0.770, -0.041], engine)
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
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf1 then output is out1mf1", engine),
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf2 then output is out1mf2", engine),
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf3 then output is out1mf3", engine),
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf4 then output is out1mf4", engine),
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf5 then output is out1mf5", engine),
            fl.Rule.create("if input1 is in1mf1 and input2 is in2mf6 then output is out1mf6", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf1 then output is out1mf7", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf2 then output is out1mf8", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf3 then output is out1mf9", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf4 then output is out1mf10", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf5 then output is out1mf11", engine),
            fl.Rule.create("if input1 is in1mf2 and input2 is in2mf6 then output is out1mf12", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf1 then output is out1mf13", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf2 then output is out1mf14", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf3 then output is out1mf15", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf4 then output is out1mf16", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf5 then output is out1mf17", engine),
            fl.Rule.create("if input1 is in1mf3 and input2 is in2mf6 then output is out1mf18", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf1 then output is out1mf19", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf2 then output is out1mf20", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf3 then output is out1mf21", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf4 then output is out1mf22", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf5 then output is out1mf23", engine),
            fl.Rule.create("if input1 is in1mf4 and input2 is in2mf6 then output is out1mf24", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf1 then output is out1mf25", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf2 then output is out1mf26", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf3 then output is out1mf27", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf4 then output is out1mf28", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf5 then output is out1mf29", engine),
            fl.Rule.create("if input1 is in1mf5 and input2 is in2mf6 then output is out1mf30", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf1 then output is out1mf31", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf2 then output is out1mf32", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf3 then output is out1mf33", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf4 then output is out1mf34", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf5 then output is out1mf35", engine),
            fl.Rule.create("if input1 is in1mf6 and input2 is in2mf6 then output is out1mf36", engine)
        ]
    )
]
