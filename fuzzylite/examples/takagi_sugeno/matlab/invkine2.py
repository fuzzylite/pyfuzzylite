import fuzzylite as fl


class Invkine2:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="invkine2",
            input_variables=[
                fl.InputVariable(
                    name="input1",
                    minimum=-6.287,
                    maximum=17.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", -5.763, 3.015, 1.851),
                        fl.Bell("in1mf2", -1.624, 3.13, 2.111),
                        fl.Bell("in1mf3", 3.552, 3.193, 2.104),
                        fl.Bell("in1mf4", 8.273, 2.907, 1.985),
                        fl.Bell("in1mf5", 13.232, 2.708, 2.056),
                        fl.Bell("in1mf6", 17.783, 1.635, 1.897),
                    ],
                ),
                fl.InputVariable(
                    name="input2",
                    minimum=0.0,
                    maximum=16.972,
                    lock_range=False,
                    terms=[
                        fl.Bell("in2mf1", 0.005, 1.877, 1.995),
                        fl.Bell("in2mf2", 3.312, 2.017, 1.829),
                        fl.Bell("in2mf3", 6.568, 2.261, 1.793),
                        fl.Bell("in2mf4", 10.111, 2.741, 1.978),
                        fl.Bell("in2mf5", 14.952, 2.045, 1.783),
                        fl.Bell("in2mf6", 17.91, 0.824, 1.734),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="output",
                    minimum=0.0,
                    maximum=3.1,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("out1mf1", [-0.048, 1.456, 2.222]),
                        fl.Linear("out1mf2", [-0.218, -0.305, 2.042]),
                        fl.Linear("out1mf3", [0.026, -0.141, 3.067]),
                        fl.Linear("out1mf4", [0.052, -0.15, 3.419]),
                        fl.Linear("out1mf5", [0.113, -0.189, 4.339]),
                        fl.Linear("out1mf6", [2.543, 0.361, -2.738]),
                        fl.Linear("out1mf7", [2.517, -6.809, 23.353]),
                        fl.Linear("out1mf8", [-0.208, -0.394, 4.472]),
                        fl.Linear("out1mf9", [-0.046, -0.3, 4.452]),
                        fl.Linear("out1mf10", [-0.006, -0.217, 4.195]),
                        fl.Linear("out1mf11", [0.089, -0.254, 4.992]),
                        fl.Linear("out1mf12", [-0.033, 0.103, -2.012]),
                        fl.Linear("out1mf13", [1.355, 1.228, -5.678]),
                        fl.Linear("out1mf14", [-0.245, -0.124, 3.753]),
                        fl.Linear("out1mf15", [-0.099, -0.111, 3.304]),
                        fl.Linear("out1mf16", [-0.052, -0.163, 3.56]),
                        fl.Linear("out1mf17", [0.099, -0.26, 4.662]),
                        fl.Linear("out1mf18", [0.082, -1.849, 31.104]),
                        fl.Linear("out1mf19", [2.18, -2.963, -0.061]),
                        fl.Linear("out1mf20", [-0.982, 0.51, 5.657]),
                        fl.Linear("out1mf21", [-0.087, -0.179, 3.744]),
                        fl.Linear("out1mf22", [-0.124, -0.161, 4.094]),
                        fl.Linear("out1mf23", [0.383, 0.007, -1.559]),
                        fl.Linear("out1mf24", [-8.415, 2.083, 5.177]),
                        fl.Linear("out1mf25", [1.721, -15.079, -0.687]),
                        fl.Linear("out1mf26", [-1.043, -0.786, 20.51]),
                        fl.Linear("out1mf27", [-0.249, -0.396, 6.995]),
                        fl.Linear("out1mf28", [-0.076, -0.245, 4.416]),
                        fl.Linear("out1mf29", [0.765, -1.488, 17.384]),
                        fl.Linear("out1mf30", [-21.21, -43.022, -2.522]),
                        fl.Linear("out1mf31", [-0.661, 3.523, 6.215]),
                        fl.Linear("out1mf32", [-1.998, 1.582, 33.256]),
                        fl.Linear("out1mf33", [-2.068, 5.673, 6.52]),
                        fl.Linear("out1mf34", [-5.044, 7.093, 3.516]),
                        fl.Linear("out1mf35", [-46.049, -35.021, -2.926]),
                        fl.Linear("out1mf36", [-0.448, -0.77, -0.041]),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf1 then output is out1mf1"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf2 then output is out1mf2"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf3 then output is out1mf3"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf4 then output is out1mf4"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf5 then output is out1mf5"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf1 and input2 is in2mf6 then output is out1mf6"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf1 then output is out1mf7"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf2 then output is out1mf8"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf3 then output is out1mf9"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf4 then output is out1mf10"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf5 then output is out1mf11"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf2 and input2 is in2mf6 then output is out1mf12"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf1 then output is out1mf13"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf2 then output is out1mf14"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf3 then output is out1mf15"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf4 then output is out1mf16"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf5 then output is out1mf17"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf3 and input2 is in2mf6 then output is out1mf18"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf1 then output is out1mf19"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf2 then output is out1mf20"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf3 then output is out1mf21"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf4 then output is out1mf22"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf5 then output is out1mf23"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf4 and input2 is in2mf6 then output is out1mf24"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf1 then output is out1mf25"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf2 then output is out1mf26"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf3 then output is out1mf27"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf4 then output is out1mf28"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf5 then output is out1mf29"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf5 and input2 is in2mf6 then output is out1mf30"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf1 then output is out1mf31"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf2 then output is out1mf32"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf3 then output is out1mf33"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf4 then output is out1mf34"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf5 then output is out1mf35"
                        ),
                        fl.Rule.create(
                            "if input1 is in1mf6 and input2 is in2mf6 then output is out1mf36"
                        ),
                    ],
                )
            ],
        )
