import fuzzylite as fl

engine = fl.Engine(
    name="tipper1",
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
            fl.Gaussian("good", 5.000, 1.500),
            fl.Gaussian("excellent", 10.000, 1.500)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="tip",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=30.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Triangle("cheap", 0.000, 5.000, 10.000),
            fl.Triangle("average", 10.000, 15.000, 20.000),
            fl.Triangle("generous", 20.000, 25.000, 30.000)
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
            fl.Rule.create("if service is poor then tip is cheap", engine),
            fl.Rule.create("if service is good then tip is average", engine),
            fl.Rule.create("if service is excellent then tip is generous", engine)
        ]
    )
]
