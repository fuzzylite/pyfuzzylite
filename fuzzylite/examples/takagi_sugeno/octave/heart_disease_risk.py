import fuzzylite as fl


class HeartDiseaseRisk:
    def __init__(self) -> None:
        self.engine = fl.Engine(
            name="heart_disease_risk",
            input_variables=[
                fl.InputVariable(
                    name="LDLLevel",
                    minimum=0.0,
                    maximum=300.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("Low", -1.0, 0.0, 90.0, 110.0),
                        fl.Trapezoid("LowBorderline", 90.0, 110.0, 120.0, 140.0),
                        fl.Trapezoid("Borderline", 120.0, 140.0, 150.0, 170.0),
                        fl.Trapezoid("HighBorderline", 150.0, 170.0, 180.0, 200.0),
                        fl.Trapezoid("High", 180.0, 200.0, 300.0, 301.0),
                    ],
                ),
                fl.InputVariable(
                    name="HDLLevel",
                    minimum=0.0,
                    maximum=100.0,
                    lock_range=False,
                    terms=[
                        fl.Trapezoid("LowHDL", -1.0, 0.0, 35.0, 45.0),
                        fl.Trapezoid("ModerateHDL", 35.0, 45.0, 55.0, 65.0),
                        fl.Trapezoid("HighHDL", 55.0, 65.0, 100.0, 101.0),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="HeartDiseaseRisk",
                    minimum=0.0,
                    maximum=10.0,
                    lock_range=False,
                    lock_previous=False,
                    default_value=fl.nan,
                    aggregation=None,
                    defuzzifier=fl.WeightedAverage(type="TakagiSugeno"),
                    terms=[
                        fl.Constant("NoRisk", 0.0),
                        fl.Constant("LowRisk", 2.5),
                        fl.Constant("MediumRisk", 5.0),
                        fl.Constant("HighRisk", 7.5),
                        fl.Constant("ExtremeRisk", 10.0),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="",
                    conjunction=fl.Minimum(),
                    disjunction=None,
                    implication=None,
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create(
                            "if LDLLevel is Low and HDLLevel is LowHDL then HeartDiseaseRisk is MediumRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is Low and HDLLevel is ModerateHDL then HeartDiseaseRisk is LowRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is Low and HDLLevel is HighHDL then HeartDiseaseRisk is NoRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is LowBorderline and HDLLevel is LowHDL then HeartDiseaseRisk is MediumRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is LowBorderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is LowRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is LowBorderline and HDLLevel is HighHDL then HeartDiseaseRisk is LowRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is Borderline and HDLLevel is LowHDL then HeartDiseaseRisk is HighRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is Borderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is MediumRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is Borderline and HDLLevel is HighHDL then HeartDiseaseRisk is LowRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is HighBorderline and HDLLevel is LowHDL then HeartDiseaseRisk is HighRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is HighBorderline and HDLLevel is ModerateHDL then HeartDiseaseRisk is HighRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is HighBorderline and HDLLevel is HighHDL then HeartDiseaseRisk is MediumRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is High and HDLLevel is LowHDL then HeartDiseaseRisk is ExtremeRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is High and HDLLevel is ModerateHDL then HeartDiseaseRisk is HighRisk"
                        ),
                        fl.Rule.create(
                            "if LDLLevel is High and HDLLevel is HighHDL then HeartDiseaseRisk is MediumRisk"
                        ),
                    ],
                )
            ],
        )
