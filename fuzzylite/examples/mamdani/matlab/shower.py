import fuzzylite as fl

engine = fl.Engine(
    name="shower",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="temp",
        description="",
        enabled=True,
        minimum=-20.000000000,
        maximum=20.000000000,
        lock_range=False,
        terms=[
            fl.Trapezoid("cold", -30.000000000, -30.000000000, -15.000000000, 0.000000000),
            fl.Triangle("good", -10.000000000, 0.000000000, 10.000000000),
            fl.Trapezoid("hot", 0.000000000, 15.000000000, 30.000000000, 30.000000000)
        ]
    ),
    fl.InputVariable(
        name="flow",
        description="",
        enabled=True,
        minimum=-1.000000000,
        maximum=1.000000000,
        lock_range=False,
        terms=[
            fl.Trapezoid("soft", -3.000000000, -3.000000000, -0.800000000, 0.000000000),
            fl.Triangle("good", -0.400000000, 0.000000000, 0.400000000),
            fl.Trapezoid("hard", 0.000000000, 0.800000000, 3.000000000, 3.000000000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="cold",
        description="",
        enabled=True,
        minimum=-1.000000000,
        maximum=1.000000000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Triangle("closeFast", -1.000000000, -0.600000000, -0.300000000),
            fl.Triangle("closeSlow", -0.600000000, -0.300000000, 0.000000000),
            fl.Triangle("steady", -0.300000000, 0.000000000, 0.300000000),
            fl.Triangle("openSlow", 0.000000000, 0.300000000, 0.600000000),
            fl.Triangle("openFast", 0.300000000, 0.600000000, 1.000000000)
        ]
    ),
    fl.OutputVariable(
        name="hot",
        description="",
        enabled=True,
        minimum=-1.000000000,
        maximum=1.000000000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Triangle("closeFast", -1.000000000, -0.600000000, -0.300000000),
            fl.Triangle("closeSlow", -0.600000000, -0.300000000, 0.000000000),
            fl.Triangle("steady", -0.300000000, 0.000000000, 0.300000000),
            fl.Triangle("openSlow", 0.000000000, 0.300000000, 0.600000000),
            fl.Triangle("openFast", 0.300000000, 0.600000000, 1.000000000)
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
            fl.Rule.create("if temp is cold and flow is soft then cold is openSlow and hot is openFast", engine),
            fl.Rule.create("if temp is cold and flow is good then cold is closeSlow and hot is openSlow", engine),
            fl.Rule.create("if temp is cold and flow is hard then cold is closeFast and hot is closeSlow", engine),
            fl.Rule.create("if temp is good and flow is soft then cold is openSlow and hot is openSlow", engine),
            fl.Rule.create("if temp is good and flow is good then cold is steady and hot is steady", engine),
            fl.Rule.create("if temp is good and flow is hard then cold is closeSlow and hot is closeSlow", engine),
            fl.Rule.create("if temp is hot and flow is soft then cold is openFast and hot is openSlow", engine),
            fl.Rule.create("if temp is hot and flow is good then cold is openSlow and hot is closeSlow", engine),
            fl.Rule.create("if temp is hot and flow is hard then cold is closeSlow and hot is closeFast", engine)
        ]
    )
]
