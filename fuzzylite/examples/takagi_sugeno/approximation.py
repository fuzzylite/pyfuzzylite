import fuzzylite as fl

engine = fl.Engine(
    name="approximation",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="inputX",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        terms=[
            fl.Triangle("NEAR_1", 0.000, 1.000, 2.000),
            fl.Triangle("NEAR_2", 1.000, 2.000, 3.000),
            fl.Triangle("NEAR_3", 2.000, 3.000, 4.000),
            fl.Triangle("NEAR_4", 3.000, 4.000, 5.000),
            fl.Triangle("NEAR_5", 4.000, 5.000, 6.000),
            fl.Triangle("NEAR_6", 5.000, 6.000, 7.000),
            fl.Triangle("NEAR_7", 6.000, 7.000, 8.000),
            fl.Triangle("NEAR_8", 7.000, 8.000, 9.000),
            fl.Triangle("NEAR_9", 8.000, 9.000, 10.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="outputFx",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=True,
        terms=[
            fl.Constant("f1", 0.840),
            fl.Constant("f2", 0.450),
            fl.Constant("f3", 0.040),
            fl.Constant("f4", -0.180),
            fl.Constant("f5", -0.190),
            fl.Constant("f6", -0.040),
            fl.Constant("f7", 0.090),
            fl.Constant("f8", 0.120),
            fl.Constant("f9", 0.040)
        ]
    ),
    fl.OutputVariable(
        name="trueFx",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=True,
        terms=[fl.Function.create("fx", "sin(inputX)/inputX", engine)]
    ),
    fl.OutputVariable(
        name="diffFx",
        description="",
        enabled=True,
        minimum=-1.000,
        maximum=1.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("Automatic"),
        lock_previous=False,
        terms=[fl.Function.create("diff", "fabs(outputFx-trueFx)", engine)]
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
            fl.Rule.create("if inputX is NEAR_1 then outputFx is f1", engine),
            fl.Rule.create("if inputX is NEAR_2 then outputFx is f2", engine),
            fl.Rule.create("if inputX is NEAR_3 then outputFx is f3", engine),
            fl.Rule.create("if inputX is NEAR_4 then outputFx is f4", engine),
            fl.Rule.create("if inputX is NEAR_5 then outputFx is f5", engine),
            fl.Rule.create("if inputX is NEAR_6 then outputFx is f6", engine),
            fl.Rule.create("if inputX is NEAR_7 then outputFx is f7", engine),
            fl.Rule.create("if inputX is NEAR_8 then outputFx is f8", engine),
            fl.Rule.create("if inputX is NEAR_9 then outputFx is f9", engine),
            fl.Rule.create("if inputX is any then trueFx is fx and diffFx is diff", engine)
        ]
    )
]
