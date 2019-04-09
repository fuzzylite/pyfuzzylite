import fuzzylite as fl

engine = fl.Engine(
    name="fpeaks",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="in1",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -2.233, 1.578, 2.151),
            fl.Bell("in1mf2", -0.394, 0.753, 1.838),
            fl.Bell("in1mf3", 0.497, 0.689, 1.844),
            fl.Bell("in1mf4", 2.270, 1.528, 2.156)
        ]
    ),
    fl.InputVariable(
        name="in2",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -2.686, 1.267, 2.044),
            fl.Bell("in1mf2", -0.836, 1.266, 1.796),
            fl.Bell("in1mf3", 0.859, 1.314, 1.937),
            fl.Bell("in1mf4", 2.727, 1.214, 2.047)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="out1",
        description="",
        enabled=True,
        minimum=-10.000,
        maximum=10.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf1", [0.155, -2.228, -8.974], engine),
            fl.Linear("out1mf2", [-0.312, -7.705, -9.055], engine),
            fl.Linear("out1mf3", [-0.454, -4.437, 6.930], engine),
            fl.Linear("out1mf4", [0.248, -1.122, 5.081], engine),
            fl.Linear("out1mf5", [-6.278, 25.211, 99.148], engine),
            fl.Linear("out1mf6", [5.531, 105.916, 157.283], engine),
            fl.Linear("out1mf7", [19.519, 112.333, -127.796], engine),
            fl.Linear("out1mf8", [-5.079, 34.738, -143.414], engine),
            fl.Linear("out1mf9", [-5.889, 27.311, 116.585], engine),
            fl.Linear("out1mf10", [21.517, 97.266, 93.802], engine),
            fl.Linear("out1mf11", [9.198, 79.853, -118.482], engine),
            fl.Linear("out1mf12", [-6.571, 23.026, -87.747], engine),
            fl.Linear("out1mf13", [0.092, -1.126, -4.527], engine),
            fl.Linear("out1mf14", [-0.304, -4.434, -6.561], engine),
            fl.Linear("out1mf15", [-0.166, -6.284, 7.307], engine),
            fl.Linear("out1mf16", [0.107, -2.028, 8.159], engine)
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
            fl.Rule.create("if in1 is in1mf1 and in2 is in1mf1 then out1 is out1mf1", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in1mf2 then out1 is out1mf2", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in1mf3 then out1 is out1mf3", engine),
            fl.Rule.create("if in1 is in1mf1 and in2 is in1mf4 then out1 is out1mf4", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in1mf1 then out1 is out1mf5", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in1mf2 then out1 is out1mf6", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in1mf3 then out1 is out1mf7", engine),
            fl.Rule.create("if in1 is in1mf2 and in2 is in1mf4 then out1 is out1mf8", engine),
            fl.Rule.create("if in1 is in1mf3 and in2 is in1mf1 then out1 is out1mf9", engine),
            fl.Rule.create("if in1 is in1mf3 and in2 is in1mf2 then out1 is out1mf10", engine),
            fl.Rule.create("if in1 is in1mf3 and in2 is in1mf3 then out1 is out1mf11", engine),
            fl.Rule.create("if in1 is in1mf3 and in2 is in1mf4 then out1 is out1mf12", engine),
            fl.Rule.create("if in1 is in1mf4 and in2 is in1mf1 then out1 is out1mf13", engine),
            fl.Rule.create("if in1 is in1mf4 and in2 is in1mf2 then out1 is out1mf14", engine),
            fl.Rule.create("if in1 is in1mf4 and in2 is in1mf3 then out1 is out1mf15", engine),
            fl.Rule.create("if in1 is in1mf4 and in2 is in1mf4 then out1 is out1mf16", engine)
        ]
    )
]
