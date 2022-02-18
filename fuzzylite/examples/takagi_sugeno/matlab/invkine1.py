import fuzzylite as fl

engine = fl.Engine(name="invkine1", description="")
engine.input_variables = [
    fl.InputVariable(
        name="input1",
        description="",
        enabled=True,
        minimum=-6.287,
        maximum=17.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -6.122, 2.259, 1.761),
            fl.Bell("in1mf2", -2.181, 2.095, 2.232),
            fl.Bell("in1mf3", 2.080, 2.157, 1.314),
            fl.Bell("in1mf4", 4.962, 2.790, 2.508),
            fl.Bell("in1mf5", 9.338, 2.506, 1.812),
            fl.Bell("in1mf6", 13.150, 2.363, 2.267),
            fl.Bell("in1mf7", 17.789, 1.310, 1.756),
        ],
    ),
    fl.InputVariable(
        name="input2",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=16.972,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", 0.621, 1.741, 2.454),
            fl.Bell("in2mf2", 2.406, 0.866, 1.278),
            fl.Bell("in2mf3", 4.883, 1.814, 2.402),
            fl.Bell("in2mf4", 8.087, 1.941, 1.929),
            fl.Bell("in2mf5", 11.428, 2.333, 2.022),
            fl.Bell("in2mf6", 14.579, 2.221, 1.858),
            fl.Bell("in2mf7", 17.813, 0.820, 1.577),
        ],
    ),
]
engine.output_variables = [
    fl.OutputVariable(
        name="output",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.500,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [-0.912, 2.517, 0.061], engine),
            fl.Linear("out1mf2", [-2.153, -2.204, -4.037], engine),
            fl.Linear("out1mf3", [-0.107, -0.148, 1.920], engine),
            fl.Linear("out1mf4", [-0.088, -0.071, 1.593], engine),
            fl.Linear("out1mf5", [-0.098, -0.040, 1.361], engine),
            fl.Linear("out1mf6", [-0.068, -0.027, 1.617], engine),
            fl.Linear("out1mf7", [-1.901, -0.081, 0.185], engine),
            fl.Linear("out1mf8", [16.651, 11.713, 6.803], engine),
            fl.Linear("out1mf9", [-4.152, -1.033, -4.755], engine),
            fl.Linear("out1mf10", [-0.123, 0.004, 0.861], engine),
            fl.Linear("out1mf11", [-0.102, 0.006, 0.816], engine),
            fl.Linear("out1mf12", [-0.089, 0.038, 0.515], engine),
            fl.Linear("out1mf13", [-0.074, 0.082, -0.061], engine),
            fl.Linear("out1mf14", [-0.009, -0.173, 4.841], engine),
            fl.Linear("out1mf15", [-7.995, -2.818, 17.910], engine),
            fl.Linear("out1mf16", [0.674, 0.745, -2.167], engine),
            fl.Linear("out1mf17", [-0.130, -0.004, 0.869], engine),
            fl.Linear("out1mf18", [-0.094, 0.061, 0.366], engine),
            fl.Linear("out1mf19", [-0.087, 0.121, -0.395], engine),
            fl.Linear("out1mf20", [-0.061, 0.162, -1.312], engine),
            fl.Linear("out1mf21", [-0.163, 0.920, -13.253], engine),
            fl.Linear("out1mf22", [1.417, 3.072, 1.881], engine),
            fl.Linear("out1mf23", [-0.950, -0.732, 3.128], engine),
            fl.Linear("out1mf24", [-0.058, 0.162, -0.530], engine),
            fl.Linear("out1mf25", [-0.044, 0.077, -0.121], engine),
            fl.Linear("out1mf26", [-0.061, 0.054, 0.273], engine),
            fl.Linear("out1mf27", [-0.068, 0.099, -0.250], engine),
            fl.Linear("out1mf28", [0.429, 0.610, -9.118], engine),
            fl.Linear("out1mf29", [-6.661, -3.697, 0.015], engine),
            fl.Linear("out1mf30", [-3.544, 8.995, 0.396], engine),
            fl.Linear("out1mf31", [0.210, -0.137, -1.010], engine),
            fl.Linear("out1mf32", [-0.003, 0.137, -1.210], engine),
            fl.Linear("out1mf33", [-0.030, 0.018, 0.107], engine),
            fl.Linear("out1mf34", [-0.227, -0.306, 6.550], engine),
            fl.Linear("out1mf35", [11.005, -3.375, -1.135], engine),
            fl.Linear("out1mf36", [0.578, 0.033, -0.104], engine),
            fl.Linear("out1mf37", [0.895, -3.268, -0.992], engine),
            fl.Linear("out1mf38", [0.125, 0.006, -1.733], engine),
            fl.Linear("out1mf39", [0.044, 0.003, -0.303], engine),
            fl.Linear("out1mf40", [-0.179, -0.093, 3.458], engine),
            fl.Linear("out1mf41", [0.222, 0.597, -10.102], engine),
            fl.Linear("out1mf42", [9.320, 13.692, 0.858], engine),
            fl.Linear("out1mf43", [0.161, -0.117, -1.382], engine),
            fl.Linear("out1mf44", [0.495, -0.833, -6.564], engine),
            fl.Linear("out1mf45", [0.465, -0.787, -5.610], engine),
            fl.Linear("out1mf46", [1.334, -3.017, -2.871], engine),
            fl.Linear("out1mf47", [2.561, -0.864, -0.557], engine),
            fl.Linear("out1mf48", [17.123, 11.150, 1.006], engine),
            fl.Linear("out1mf49", [0.220, 0.154, 0.010], engine),
        ],
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
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf1 then output is out1mf1",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf2 then output is out1mf2",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf3 then output is out1mf3",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf4 then output is out1mf4",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf5 then output is out1mf5",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf6 then output is out1mf6",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf1 and input2 is in2mf7 then output is out1mf7",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf1 then output is out1mf8",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf2 then output is out1mf9",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf3 then output is out1mf10",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf4 then output is out1mf11",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf5 then output is out1mf12",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf6 then output is out1mf13",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf2 and input2 is in2mf7 then output is out1mf14",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf1 then output is out1mf15",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf2 then output is out1mf16",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf3 then output is out1mf17",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf4 then output is out1mf18",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf5 then output is out1mf19",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf6 then output is out1mf20",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf3 and input2 is in2mf7 then output is out1mf21",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf1 then output is out1mf22",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf2 then output is out1mf23",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf3 then output is out1mf24",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf4 then output is out1mf25",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf5 then output is out1mf26",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf6 then output is out1mf27",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf4 and input2 is in2mf7 then output is out1mf28",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf1 then output is out1mf29",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf2 then output is out1mf30",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf3 then output is out1mf31",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf4 then output is out1mf32",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf5 then output is out1mf33",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf6 then output is out1mf34",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf5 and input2 is in2mf7 then output is out1mf35",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf1 then output is out1mf36",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf2 then output is out1mf37",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf3 then output is out1mf38",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf4 then output is out1mf39",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf5 then output is out1mf40",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf6 then output is out1mf41",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf6 and input2 is in2mf7 then output is out1mf42",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf1 then output is out1mf43",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf2 then output is out1mf44",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf3 then output is out1mf45",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf4 then output is out1mf46",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf5 then output is out1mf47",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf6 then output is out1mf48",
                engine,
            ),
            fl.Rule.create(
                "if input1 is in1mf7 and input2 is in2mf7 then output is out1mf49",
                engine,
            ),
        ],
    )
]
