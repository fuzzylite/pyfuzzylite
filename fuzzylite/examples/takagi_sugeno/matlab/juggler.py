import fuzzylite as fl

engine = fl.Engine(
    name="juggler",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="xHit",
        description="",
        enabled=True,
        minimum=-4.000,
        maximum=4.000,
        lock_range=False,
        terms=[
            fl.Bell("in1mf1", -4.000, 2.000, 4.000),
            fl.Bell("in1mf2", 0.000, 2.000, 4.000),
            fl.Bell("in1mf3", 4.000, 2.000, 4.000)
        ]
    ),
    fl.InputVariable(
        name="projectAngle",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=3.142,
        lock_range=False,
        terms=[
            fl.Bell("in2mf1", 0.000, 0.785, 4.000),
            fl.Bell("in2mf2", 1.571, 0.785, 4.000),
            fl.Bell("in2mf3", 3.142, 0.785, 4.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="theta",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=0.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("out1mf", [-0.022, -0.500, 0.315], engine),
            fl.Linear("out1mf", [-0.022, -0.500, 0.315], engine),
            fl.Linear("out1mf", [-0.022, -0.500, 0.315], engine),
            fl.Linear("out1mf", [0.114, -0.500, 0.785], engine),
            fl.Linear("out1mf", [0.114, -0.500, 0.785], engine),
            fl.Linear("out1mf", [0.114, -0.500, 0.785], engine),
            fl.Linear("out1mf", [-0.022, -0.500, 1.256], engine),
            fl.Linear("out1mf", [-0.022, -0.500, 1.256], engine),
            fl.Linear("out1mf", [-0.022, -0.500, 1.256], engine)
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
            fl.Rule.create("if xHit is in1mf1 and projectAngle is in2mf1 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf1 and projectAngle is in2mf2 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf1 and projectAngle is in2mf3 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf2 and projectAngle is in2mf1 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf2 and projectAngle is in2mf2 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf2 and projectAngle is in2mf3 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf3 and projectAngle is in2mf1 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf3 and projectAngle is in2mf2 then theta is out1mf", engine),
            fl.Rule.create("if xHit is in1mf3 and projectAngle is in2mf3 then theta is out1mf", engine)
        ]
    )
]
