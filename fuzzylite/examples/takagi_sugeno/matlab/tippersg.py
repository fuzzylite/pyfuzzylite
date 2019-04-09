import fuzzylite as fl

engine = fl.Engine(
    name="tippersg",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="service",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Gaussian("poor", 0.000, 1.500),
            fl.Gaussian("average", 5.000, 1.500),
            fl.Gaussian("good", 10.000, 1.500)
        ]
    ),
    fl.InputVariable(
        name="food",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Trapezoid("rancid", -5.000, 0.000, 1.000, 3.000),
            fl.Trapezoid("delicious", 7.000, 9.000, 10.000, 15.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="tip",
        description="",
        enabled=True,
        minimum=-30.000,
        maximum=30.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Linear("cheap", [0.000, 0.000, 5.000], engine),
            fl.Linear("average", [0.000, 0.000, 15.000], engine),
            fl.Linear("generous", [0.000, 0.000, 25.000], engine)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=None,
        disjunction=fl.Maximum(),
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if service is poor or food is rancid then tip is cheap", engine),
            fl.Rule.create("if service is average then tip is average", engine),
            fl.Rule.create("if service is good or food is delicious then tip is generous", engine)
        ]
    )
]
