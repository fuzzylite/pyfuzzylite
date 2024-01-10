import fuzzylite as fl


class Fpeaks:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="fpeaks",
            input_variables=[
                fl.InputVariable(
                    name="in1",
                    minimum=-3.0,
                    maximum=3.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", -2.233, 1.578, 2.151),
                        fl.Bell("in1mf2", -0.394, 0.753, 1.838),
                        fl.Bell("in1mf3", 0.497, 0.689, 1.844),
                        fl.Bell("in1mf4", 2.27, 1.528, 2.156),
                    ],
                ),
                fl.InputVariable(
                    name="in2",
                    minimum=-3.0,
                    maximum=3.0,
                    lock_range=False,
                    terms=[
                        fl.Bell("in1mf1", -2.686, 1.267, 2.044),
                        fl.Bell("in1mf2", -0.836, 1.266, 1.796),
                        fl.Bell("in1mf3", 0.859, 1.314, 1.937),
                        fl.Bell("in1mf4", 2.727, 1.214, 2.047),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="out1",
                    minimum=-10.0,
                    maximum=10.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Linear("out1mf1", [0.155, -2.228, -8.974]),
                        fl.Linear("out1mf2", [-0.312, -7.705, -9.055]),
                        fl.Linear("out1mf3", [-0.454, -4.437, 6.93]),
                        fl.Linear("out1mf4", [0.248, -1.122, 5.081]),
                        fl.Linear("out1mf5", [-6.278, 25.211, 99.148]),
                        fl.Linear("out1mf6", [5.531, 105.916, 157.283]),
                        fl.Linear("out1mf7", [19.519, 112.333, -127.796]),
                        fl.Linear("out1mf8", [-5.079, 34.738, -143.414]),
                        fl.Linear("out1mf9", [-5.889, 27.311, 116.585]),
                        fl.Linear("out1mf10", [21.517, 97.266, 93.802]),
                        fl.Linear("out1mf11", [9.198, 79.853, -118.482]),
                        fl.Linear("out1mf12", [-6.571, 23.026, -87.747]),
                        fl.Linear("out1mf13", [0.092, -1.126, -4.527]),
                        fl.Linear("out1mf14", [-0.304, -4.434, -6.561]),
                        fl.Linear("out1mf15", [-0.166, -6.284, 7.307]),
                        fl.Linear("out1mf16", [0.107, -2.028, 8.159]),
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
                        fl.Rule.create("if in1 is in1mf1 and in2 is in1mf1 then out1 is out1mf1"),
                        fl.Rule.create("if in1 is in1mf1 and in2 is in1mf2 then out1 is out1mf2"),
                        fl.Rule.create("if in1 is in1mf1 and in2 is in1mf3 then out1 is out1mf3"),
                        fl.Rule.create("if in1 is in1mf1 and in2 is in1mf4 then out1 is out1mf4"),
                        fl.Rule.create("if in1 is in1mf2 and in2 is in1mf1 then out1 is out1mf5"),
                        fl.Rule.create("if in1 is in1mf2 and in2 is in1mf2 then out1 is out1mf6"),
                        fl.Rule.create("if in1 is in1mf2 and in2 is in1mf3 then out1 is out1mf7"),
                        fl.Rule.create("if in1 is in1mf2 and in2 is in1mf4 then out1 is out1mf8"),
                        fl.Rule.create("if in1 is in1mf3 and in2 is in1mf1 then out1 is out1mf9"),
                        fl.Rule.create("if in1 is in1mf3 and in2 is in1mf2 then out1 is out1mf10"),
                        fl.Rule.create("if in1 is in1mf3 and in2 is in1mf3 then out1 is out1mf11"),
                        fl.Rule.create("if in1 is in1mf3 and in2 is in1mf4 then out1 is out1mf12"),
                        fl.Rule.create("if in1 is in1mf4 and in2 is in1mf1 then out1 is out1mf13"),
                        fl.Rule.create("if in1 is in1mf4 and in2 is in1mf2 then out1 is out1mf14"),
                        fl.Rule.create("if in1 is in1mf4 and in2 is in1mf3 then out1 is out1mf15"),
                        fl.Rule.create("if in1 is in1mf4 and in2 is in1mf4 then out1 is out1mf16"),
                    ],
                )
            ],
        )
