import fuzzylite as fl

engine = fl.Engine(
    name="mam21",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="angle",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        terms=[
            fl.Bell("small", -5.000, 5.000, 8.000),
            fl.Bell("big", 5.000, 5.000, 8.000)
        ]
    ),
    fl.InputVariable(
        name="velocity",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        terms=[
            fl.Bell("small", -5.000, 5.000, 2.000),
            fl.Bell("big", 5.000, 5.000, 2.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="force",
        description="",
        enabled=True,
        minimum=-5.000,
        maximum=5.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Bell("negBig", -5.000, 1.670, 8.000),
            fl.Bell("negSmall", -1.670, 1.670, 8.000),
            fl.Bell("posSmall", 1.670, 1.670, 8.000),
            fl.Bell("posBig", 5.000, 1.670, 8.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=fl.Maximum(),
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if angle is small and velocity is small then force is negBig", engine),
            fl.Rule.create("if angle is small and velocity is big then force is negSmall", engine),
            fl.Rule.create("if angle is big and velocity is small then force is posSmall", engine),
            fl.Rule.create("if angle is big and velocity is big then force is posBig", engine)
        ]
    )
]
