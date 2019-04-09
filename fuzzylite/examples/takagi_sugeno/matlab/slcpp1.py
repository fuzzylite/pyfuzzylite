import fuzzylite as fl

engine = fl.Engine(
    name="slcpp1",
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
        minimum=-3.000,
        maximum=3.000,
        lock_range=False
    ),
    fl.InputVariable(
        name="in6",
        description="",
        enabled=True,
        minimum=-3.000,
        maximum=3.000,
        lock_range=False
    ),
    fl.InputVariable(
        name="pole_length",
        description="",
        enabled=True,
        minimum=0.500,
        maximum=1.500,
        lock_range=False,
        terms=[
            fl.ZShape("mf1", 0.500, 0.600),
            fl.PiShape("mf2", 0.500, 0.600, 0.600, 0.700),
            fl.PiShape("mf3", 0.600, 0.700, 0.700, 0.800),
            fl.PiShape("mf4", 0.700, 0.800, 0.800, 0.900),
            fl.PiShape("mf5", 0.800, 0.900, 0.900, 1.000),
            fl.PiShape("mf6", 0.900, 1.000, 1.000, 1.100),
            fl.PiShape("mf7", 1.000, 1.100, 1.100, 1.200),
            fl.PiShape("mf8", 1.100, 1.200, 1.200, 1.300),
            fl.PiShape("mf9", 1.200, 1.300, 1.300, 1.400),
            fl.PiShape("mf10", 1.300, 1.400, 1.400, 1.500),
            fl.SShape("mf11", 1.400, 1.500)
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
            fl.Linear("outmf1", [168.400, 31.000, -188.050, -49.250, -1.000, -2.700, 0.000, 0.000], engine),
            fl.Linear("outmf2", [233.950, 47.190, -254.520, -66.580, -1.000, -2.740, 0.000, 0.000], engine),
            fl.Linear("outmf3", [342.940, 74.730, -364.370, -95.230, -1.000, -2.780, 0.000, 0.000], engine),
            fl.Linear("outmf4", [560.710, 130.670, -582.960, -152.240, -1.000, -2.810, 0.000, 0.000], engine),
            fl.Linear("outmf5", [1213.880, 300.190, -1236.900, -322.800, -1.000, -2.840, 0.000, 0.000], engine),
            fl.Linear("outmf6", [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], engine),
            fl.Linear("outmf7", [-1399.120, -382.950, 1374.630, 358.340, -1.000, -2.900, 0.000, 0.000], engine),
            fl.Linear("outmf8", [-746.070, -213.420, 720.900, 187.840, -1.000, -2.930, 0.000, 0.000], engine),
            fl.Linear("outmf9", [-528.520, -157.460, 502.680, 130.920, -1.000, -2.960, 0.000, 0.000], engine),
            fl.Linear("outmf10", [-419.870, -129.890, 393.380, 102.410, -1.000, -2.980, 0.000, 0.000], engine),
            fl.Linear("outmf11", [-354.770, -113.680, 327.650, 85.270, -1.000, -3.010, 0.000, 0.000], engine)
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
            fl.Rule.create("if pole_length is mf1 then out is outmf1", engine),
            fl.Rule.create("if pole_length is mf2 then out is outmf2", engine),
            fl.Rule.create("if pole_length is mf3 then out is outmf3", engine),
            fl.Rule.create("if pole_length is mf4 then out is outmf4", engine),
            fl.Rule.create("if pole_length is mf5 then out is outmf5", engine),
            fl.Rule.create("if pole_length is mf6 then out is outmf6", engine),
            fl.Rule.create("if pole_length is mf7 then out is outmf7", engine),
            fl.Rule.create("if pole_length is mf8 then out is outmf8", engine),
            fl.Rule.create("if pole_length is mf9 then out is outmf9", engine),
            fl.Rule.create("if pole_length is mf10 then out is outmf10", engine),
            fl.Rule.create("if pole_length is mf11 then out is outmf11", engine)
        ]
    )
]
