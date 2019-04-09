import fuzzylite as fl

engine = fl.Engine(
    name="tsukamoto",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="X",
        description="",
        enabled=True,
        minimum=-10.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Bell("small", -10.000, 5.000, 3.000),
            fl.Bell("medium", 0.000, 5.000, 3.000),
            fl.Bell("large", 10.000, 5.000, 3.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="Ramps",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[
            fl.Ramp("b", 0.600, 0.400),
            fl.Ramp("a", 0.000, 0.250),
            fl.Ramp("c", 0.700, 1.000)
        ]
    ),
    fl.OutputVariable(
        name="Sigmoids",
        description="",
        enabled=True,
        minimum=0.020,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[
            fl.Sigmoid("b", 0.500, -30.000),
            fl.Sigmoid("a", 0.130, 30.000),
            fl.Sigmoid("c", 0.830, 30.000)
        ]
    ),
    fl.OutputVariable(
        name="ZSShapes",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[
            fl.ZShape("b", 0.300, 0.600),
            fl.SShape("a", 0.000, 0.250),
            fl.SShape("c", 0.700, 1.000)
        ]
    ),
    fl.OutputVariable(
        name="Concaves",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[
            fl.Concave("b", 0.500, 0.400),
            fl.Concave("a", 0.240, 0.250),
            fl.Concave("c", 0.900, 1.000)
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
            fl.Rule.create("if X is small then Ramps is a and Sigmoids is a and ZSShapes is a and Concaves is a", engine),
            fl.Rule.create("if X is medium then Ramps is b and Sigmoids is b and ZSShapes is b and Concaves is b", engine),
            fl.Rule.create("if X is large then Ramps is c and Sigmoids is c and ZSShapes is c and Concaves is c", engine)
        ]
    )
]
