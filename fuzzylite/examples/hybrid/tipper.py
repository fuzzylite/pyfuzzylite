import fuzzylite as fl

engine = fl.Engine(
    name="tipper",
    description="(service and food) -> (tip)"
)
engine.input_variables = [
    fl.InputVariable(
        name="service",
        description="quality of service",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=True,
        terms=[
            fl.Trapezoid("poor", 0.000, 0.000, 2.500, 5.000),
            fl.Triangle("good", 2.500, 5.000, 7.500),
            fl.Trapezoid("excellent", 5.000, 7.500, 10.000, 10.000)
        ]
    ),
    fl.InputVariable(
        name="food",
        description="quality of food",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=True,
        terms=[
            fl.Trapezoid("rancid", 0.000, 0.000, 2.500, 7.500),
            fl.Trapezoid("delicious", 2.500, 7.500, 10.000, 10.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="mTip",
        description="tip based on Mamdani inference",
        enabled=True,
        minimum=0.000,
        maximum=30.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(100),
        lock_previous=False,
        terms=[
            fl.Triangle("cheap", 0.000, 5.000, 10.000),
            fl.Triangle("average", 10.000, 15.000, 20.000),
            fl.Triangle("generous", 20.000, 25.000, 30.000)
        ]
    ),
    fl.OutputVariable(
        name="tsTip",
        description="tip based on Takagi-Sugeno inference",
        enabled=True,
        minimum=0.000,
        maximum=30.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("cheap", 5.000),
            fl.Constant("average", 15.000),
            fl.Constant("generous", 25.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="mamdani",
        description="Mamdani inference",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=fl.AlgebraicSum(),
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if service is poor or food is rancid then mTip is cheap", engine),
            fl.Rule.create("if service is good then mTip is average", engine),
            fl.Rule.create("if service is excellent or food is delicious then mTip is generous with 0.500", engine),
            fl.Rule.create("if service is excellent and food is delicious then mTip is generous", engine)
        ]
    ),
    fl.RuleBlock(
        name="takagiSugeno",
        description="Takagi-Sugeno inference",
        enabled=True,
        conjunction=fl.AlgebraicProduct(),
        disjunction=fl.AlgebraicSum(),
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if service is poor or food is rancid then tsTip is cheap", engine),
            fl.Rule.create("if service is good then tsTip is average", engine),
            fl.Rule.create("if service is excellent or food is delicious then tsTip is generous with 0.500", engine),
            fl.Rule.create("if service is excellent and food is delicious then tsTip is generous", engine)
        ]
    )
]
