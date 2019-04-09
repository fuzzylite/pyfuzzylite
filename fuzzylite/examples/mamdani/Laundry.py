import fuzzylite as fl

engine = fl.Engine(
    name="Laundry",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="Load",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=6.000,
        lock_range=False,
        terms=[
            fl.Discrete("small", [0.000, 1.000, 1.000, 1.000, 2.000, 0.800, 5.000, 0.000]),
            fl.Discrete("normal", [3.000, 0.000, 4.000, 1.000, 6.000, 0.000])
        ]
    ),
    fl.InputVariable(
        name="Dirt",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=6.000,
        lock_range=False,
        terms=[
            fl.Discrete("low", [0.000, 1.000, 2.000, 0.800, 5.000, 0.000]),
            fl.Discrete("high", [1.000, 0.000, 2.000, 0.200, 4.000, 0.800, 6.000, 1.000])
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="Detergent",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=80.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.MeanOfMaximum(500),
        lock_previous=False,
        terms=[
            fl.Discrete("less_than_usual", [10.000, 0.000, 40.000, 1.000, 50.000, 0.000]),
            fl.Discrete("usual", [40.000, 0.000, 50.000, 1.000, 60.000, 1.000, 80.000, 0.000]),
            fl.Discrete("more_than_usual", [50.000, 0.000, 80.000, 1.000])
        ]
    ),
    fl.OutputVariable(
        name="Cycle",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=20.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.MeanOfMaximum(500),
        lock_previous=False,
        terms=[
            fl.Discrete("short", [0.000, 1.000, 10.000, 1.000, 20.000, 0.000]),
            fl.Discrete("long", [10.000, 0.000, 20.000, 1.000])
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
            fl.Rule.create("if Load is small and Dirt is not high then Detergent is less_than_usual", engine),
            fl.Rule.create("if Load is small and Dirt is high then Detergent is usual", engine),
            fl.Rule.create("if Load is normal and Dirt is low then Detergent is less_than_usual", engine),
            fl.Rule.create("if Load is normal and Dirt is high then Detergent is more_than_usual", engine),
            fl.Rule.create("if Detergent is usual or Detergent is less_than_usual then Cycle is short", engine),
            fl.Rule.create("if Detergent is more_than_usual then Cycle is long", engine)
        ]
    )
]
