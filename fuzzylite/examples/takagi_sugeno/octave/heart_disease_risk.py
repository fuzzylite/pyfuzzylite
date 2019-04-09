import fuzzylite as fl

engine = fl.Engine(
    name="heart_disease_risk",
    description=""
)
engine.input_variables = [
    fl.InputVariable(
        name="LDLLevel",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=300.000,
        lock_range=False,
        terms=[
            fl.Trapezoid("Low", -1.000, 0.000, 90.000, 110.000),
            fl.Trapezoid("LowBorderline", 90.000, 110.000, 120.000, 140.000),
            fl.Trapezoid("Borderline", 120.000, 140.000, 150.000, 170.000),
            fl.Trapezoid("HighBorderline", 150.000, 170.000, 180.000, 200.000),
            fl.Trapezoid("High", 180.000, 200.000, 300.000, 301.000)
        ]
    ),
    fl.InputVariable(
        name="HDLLevel",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=100.000,
        lock_range=False,
        terms=[
            fl.Trapezoid("LowHDL", -1.000, 0.000, 35.000, 45.000),
            fl.Trapezoid("ModerateHDL", 35.000, 45.000, 55.000, 65.000),
            fl.Trapezoid("HighHDL", 55.000, 65.000, 100.000, 101.000)
        ]
    )
]
engine.output_variables = [
    fl.OutputVariable(
        name="HeartDiseaseRisk",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        aggregation=None,
        defuzzifier=fl.WeightedAverage("TakagiSugeno"),
        lock_previous=False,
        terms=[
            fl.Constant("NoRisk", 0.000),
            fl.Constant("LowRisk", 2.500),
            fl.Constant("MediumRisk", 5.000),
            fl.Constant("HighRisk", 7.500),
            fl.Constant("ExtremeRisk", 10.000)
        ]
    )
]
engine.rule_blocks = [
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=None,
        activation=fl.General(),
        rules=[
            fl.Rule.create("if LDLLevel is Low and HDLLevel is LowHDL then HeartDiseaseRisk is MediumRisk", engine),
            fl.Rule.create("if LDLLevel is Low and HDLLevel is ModerateHDL then HeartDiseaseRisk is LowRisk", engine),
            fl.Rule.create("if LDLLevel is Low and HDLLevel is HighHDL then HeartDiseaseRisk is NoRisk", engine),
            fl.Rule.create("if LDLLevel is LowBorderline and HDLLevel is LowHDL then HeartDiseaseRisk is MediumRisk", engine),
            fl.Rule.create("if LDLLevel is LowBorderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is LowRisk", engine),
            fl.Rule.create("if LDLLevel is LowBorderline and HDLLevel is HighHDL then HeartDiseaseRisk is LowRisk", engine),
            fl.Rule.create("if LDLLevel is Borderline and HDLLevel is LowHDL then HeartDiseaseRisk is HighRisk", engine),
            fl.Rule.create("if LDLLevel is Borderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is MediumRisk", engine),
            fl.Rule.create("if LDLLevel is Borderline and HDLLevel is HighHDL then HeartDiseaseRisk is LowRisk", engine),
            fl.Rule.create("if LDLLevel is HighBorderline and HDLLevel is LowHDL then HeartDiseaseRisk is HighRisk", engine),
            fl.Rule.create("if LDLLevel is HighBorderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is HighRisk", engine),
            fl.Rule.create("if LDLLevel is HighBorderline and HDLLevel is HighHDL then HeartDiseaseRisk is MediumRisk", engine),
            fl.Rule.create("if LDLLevel is High and HDLLevel is LowHDL then HeartDiseaseRisk is ExtremeRisk", engine),
            fl.Rule.create("if LDLLevel is High and HDLLevel is ModerateHDL then HeartDiseaseRisk is HighRisk", engine),
            fl.Rule.create("if LDLLevel is High and HDLLevel is HighHDL then HeartDiseaseRisk is MediumRisk", engine)
        ]
    )
]
