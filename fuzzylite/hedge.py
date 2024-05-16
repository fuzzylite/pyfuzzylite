"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

__all__ = [
    "Hedge",
    "Any",
    "Extremely",
    "Not",
    "Seldom",
    "Somewhat",
    "Very",
    "HedgeLambda",
    "HedgeFunction",
]

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .library import representation, scalar
from .term import Function
from .types import Scalar


class Hedge(ABC):
    r"""Abstract class for hedges.

    Hedges are used in the antecedent and consequent of a rule to modify the membership function of the term it precedes.

    The hedges in the library can be ordered based on the difference between the
    membership function $\mu(x)$ and its hedge $h(\mu(x))$ as follows (from most similar to least):
    Seldom   $<$ Somewhat   $<$ Very  $<$ Extremely   $<$ Not   $<$ Any

    | `term`                                                	| Seldom                                                                                                     	| Somewhat                                                                                                        	| Very                                                                                                    	| Extremely                                                                                                         	| Not                                                                                                   	| Any                                                                                                   	|
    |-------------------------------------------------------	|------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------------------------------------------------------	|---------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------------------	|
    | [fuzzylite.term.Rectangle][]                          	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Rectangle][]                                                    	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Rectangle][]                                                       	| [fuzzylite.hedge.Very][] [fuzzylite.term.Rectangle][]                                                   	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Rectangle][]                                                        	| [fuzzylite.hedge.Not][] [fuzzylite.term.Rectangle][]                                                  	| [fuzzylite.hedge.Any][] [fuzzylite.term.Rectangle][]                                                  	|
    | ![](../../image/term/Rectangle.svg)                   	| ![](../../image/hedge/Seldom-Rectangle.svg)                                                                	| ![](../../image/hedge/Somewhat-Rectangle.svg)                                                                   	| ![](../../image/hedge/Very-Rectangle.svg)                                                               	| ![](../../image/hedge/Extremely-Rectangle.svg)                                                                    	| ![](../../image/hedge/Not-Rectangle.svg)                                                              	| ![](../../image/hedge/Any-Rectangle.svg)                                                              	|
    | [fuzzylite.term.SemiEllipse][]                        	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.SemiEllipse][]                                                  	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.SemiEllipse][]                                                     	| [fuzzylite.hedge.Very][] [fuzzylite.term.SemiEllipse][]                                                 	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.SemiEllipse][]                                                      	| [fuzzylite.hedge.Not][] [fuzzylite.term.SemiEllipse][]                                                	| [fuzzylite.hedge.Any][] [fuzzylite.term.SemiEllipse][]                                                	|
    | ![](../../image/term/SemiEllipse.svg )                	| ![](../../image/hedge/Seldom-SemiEllipse.svg )                                                             	| ![](../../image/hedge/Somewhat-SemiEllipse.svg )                                                                	| ![](../../image/hedge/Very-SemiEllipse.svg )                                                            	| ![](../../image/hedge/Extremely-SemiEllipse.svg )                                                                 	| ![](../../image/hedge/Not-SemiEllipse.svg )                                                           	| ![](../../image/hedge/Any-SemiEllipse.svg )                                                           	|
    | [fuzzylite.term.Triangle][]                           	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Triangle][]                                                     	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Triangle][]                                                        	| [fuzzylite.hedge.Very][] [fuzzylite.term.Triangle][]                                                    	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Triangle][]                                                         	| [fuzzylite.hedge.Not][] [fuzzylite.term.Triangle][]                                                   	| [fuzzylite.hedge.Any][] [fuzzylite.term.Triangle][]                                                   	|
    | ![](../../image/term/Triangle.svg )                   	| ![](../../image/hedge/Seldom-Triangle.svg )                                                                	| ![](../../image/hedge/Somewhat-Triangle.svg )                                                                   	| ![](../../image/hedge/Very-Triangle.svg )                                                               	| ![](../../image/hedge/Extremely-Triangle.svg )                                                                    	| ![](../../image/hedge/Not-Triangle.svg )                                                              	| ![](../../image/hedge/Any-Triangle.svg )                                                              	|
    | [fuzzylite.term.Trapezoid][]                          	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Trapezoid][]                                                    	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Trapezoid][]                                                       	| [fuzzylite.hedge.Very][] [fuzzylite.term.Trapezoid][]                                                   	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Trapezoid][]                                                        	| [fuzzylite.hedge.Not][] [fuzzylite.term.Trapezoid][]                                                  	| [fuzzylite.hedge.Any][] [fuzzylite.term.Trapezoid][]                                                  	|
    | ![](../../image/term/Trapezoid.svg)                   	| ![](../../image/hedge/Seldom-Trapezoid.svg)                                                                	| ![](../../image/hedge/Somewhat-Trapezoid.svg)                                                                   	| ![](../../image/hedge/Very-Trapezoid.svg)                                                               	| ![](../../image/hedge/Extremely-Trapezoid.svg)                                                                    	| ![](../../image/hedge/Not-Trapezoid.svg)                                                              	| ![](../../image/hedge/Any-Trapezoid.svg)                                                              	|
    | [fuzzylite.term.Discrete][]                           	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Discrete][]                                                     	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Discrete][]                                                        	| [fuzzylite.hedge.Very][] [fuzzylite.term.Discrete][]                                                    	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Discrete][]                                                         	| [fuzzylite.hedge.Not][] [fuzzylite.term.Discrete][]                                                   	| [fuzzylite.hedge.Any][] [fuzzylite.term.Discrete][]                                                   	|
    | ![](../../image/term/Discrete.svg )                   	| ![](../../image/hedge/Seldom-Discrete.svg )                                                                	| ![](../../image/hedge/Somewhat-Discrete.svg )                                                                   	| ![](../../image/hedge/Very-Discrete.svg )                                                               	| ![](../../image/hedge/Extremely-Discrete.svg )                                                                    	| ![](../../image/hedge/Not-Discrete.svg )                                                              	| ![](../../image/hedge/Any-Discrete.svg )                                                              	|
    | [fuzzylite.term.Bell][]                               	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Bell][]                                                         	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Bell][]                                                            	| [fuzzylite.hedge.Very][] [fuzzylite.term.Bell][]                                                        	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Bell][]                                                             	| [fuzzylite.hedge.Not][] [fuzzylite.term.Bell][]                                                       	| [fuzzylite.hedge.Any][] [fuzzylite.term.Bell][]                                                       	|
    | ![](../../image/term/Bell.svg)                        	| ![](../../image/hedge/Seldom-Bell.svg)                                                                     	| ![](../../image/hedge/Somewhat-Bell.svg)                                                                        	| ![](../../image/hedge/Very-Bell.svg)                                                                    	| ![](../../image/hedge/Extremely-Bell.svg)                                                                         	| ![](../../image/hedge/Not-Bell.svg)                                                                   	| ![](../../image/hedge/Any-Bell.svg)                                                                   	|
    | [fuzzylite.term.Cosine][]                             	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Cosine][]                                                       	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Cosine][]                                                          	| [fuzzylite.hedge.Very][] [fuzzylite.term.Cosine][]                                                      	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Cosine][]                                                           	| [fuzzylite.hedge.Not][] [fuzzylite.term.Cosine][]                                                     	| [fuzzylite.hedge.Any][] [fuzzylite.term.Cosine][]                                                     	|
    | ![](../../image/term/Cosine.svg)                      	| ![](../../image/hedge/Seldom-Cosine.svg)                                                                   	| ![](../../image/hedge/Somewhat-Cosine.svg)                                                                      	| ![](../../image/hedge/Very-Cosine.svg)                                                                  	| ![](../../image/hedge/Extremely-Cosine.svg)                                                                       	| ![](../../image/hedge/Not-Cosine.svg)                                                                 	| ![](../../image/hedge/Any-Cosine.svg)                                                                 	|
    | [fuzzylite.term.Gaussian][]                           	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Gaussian][]                                                     	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Gaussian][]                                                        	| [fuzzylite.hedge.Very][] [fuzzylite.term.Gaussian][]                                                    	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Gaussian][]                                                         	| [fuzzylite.hedge.Not][] [fuzzylite.term.Gaussian][]                                                   	| [fuzzylite.hedge.Any][] [fuzzylite.term.Gaussian][]                                                   	|
    | ![](../../image/term/Gaussian.svg)                    	| ![](../../image/hedge/Seldom-Gaussian.svg)                                                                 	| ![](../../image/hedge/Somewhat-Gaussian.svg)                                                                    	| ![](../../image/hedge/Very-Gaussian.svg)                                                                	| ![](../../image/hedge/Extremely-Gaussian.svg)                                                                     	| ![](../../image/hedge/Not-Gaussian.svg)                                                               	| ![](../../image/hedge/Any-Gaussian.svg)                                                               	|
    | [fuzzylite.term.GaussianProduct][]                    	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.GaussianProduct][]                                              	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.GaussianProduct][]                                                 	| [fuzzylite.hedge.Very][] [fuzzylite.term.GaussianProduct][]                                             	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.GaussianProduct][]                                                  	| [fuzzylite.hedge.Not][] [fuzzylite.term.GaussianProduct][]                                            	| [fuzzylite.hedge.Any][] [fuzzylite.term.GaussianProduct][]                                            	|
    | ![](../../image/term/GaussianProduct.svg)             	| ![](../../image/hedge/Seldom-GaussianProduct.svg)                                                          	| ![](../../image/hedge/Somewhat-GaussianProduct.svg)                                                             	| ![](../../image/hedge/Very-GaussianProduct.svg)                                                         	| ![](../../image/hedge/Extremely-GaussianProduct.svg)                                                              	| ![](../../image/hedge/Not-GaussianProduct.svg)                                                        	| ![](../../image/hedge/Any-GaussianProduct.svg)                                                        	|
    | [fuzzylite.term.PiShape][]                            	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.PiShape][]                                                      	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.PiShape][]                                                         	| [fuzzylite.hedge.Very][] [fuzzylite.term.PiShape][]                                                     	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.PiShape][]                                                          	| [fuzzylite.hedge.Not][] [fuzzylite.term.PiShape][]                                                    	| [fuzzylite.hedge.Any][] [fuzzylite.term.PiShape][]                                                    	|
    | ![](../../image/term/PiShape.svg)                     	| ![](../../image/hedge/Seldom-PiShape.svg)                                                                  	| ![](../../image/hedge/Somewhat-PiShape.svg)                                                                     	| ![](../../image/hedge/Very-PiShape.svg)                                                                 	| ![](../../image/hedge/Extremely-PiShape.svg)                                                                      	| ![](../../image/hedge/Not-PiShape.svg)                                                                	| ![](../../image/hedge/Any-PiShape.svg)                                                                	|
    | [fuzzylite.term.SigmoidDifference][]                  	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.SigmoidDifference][]                                            	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.SigmoidDifference][]                                               	| [fuzzylite.hedge.Very][] [fuzzylite.term.SigmoidDifference][]                                           	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.SigmoidDifference][]                                                	| [fuzzylite.hedge.Not][] [fuzzylite.term.SigmoidDifference][]                                          	| [fuzzylite.hedge.Any][] [fuzzylite.term.SigmoidDifference][]                                          	|
    | ![](../../image/term/SigmoidDifference.svg)           	| ![](../../image/hedge/Seldom-SigmoidDifference.svg)                                                        	| ![](../../image/hedge/Somewhat-SigmoidDifference.svg)                                                           	| ![](../../image/hedge/Very-SigmoidDifference.svg)                                                       	| ![](../../image/hedge/Extremely-SigmoidDifference.svg)                                                            	| ![](../../image/hedge/Not-SigmoidDifference.svg)                                                      	| ![](../../image/hedge/Any-SigmoidDifference.svg)                                                      	|
    | [fuzzylite.term.SigmoidProduct][]                     	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.SigmoidProduct][]                                               	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.SigmoidProduct][]                                                  	| [fuzzylite.hedge.Very][] [fuzzylite.term.SigmoidProduct][]                                              	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.SigmoidProduct][]                                                   	| [fuzzylite.hedge.Not][] [fuzzylite.term.SigmoidProduct][]                                             	| [fuzzylite.hedge.Any][] [fuzzylite.term.SigmoidProduct][]                                             	|
    | ![](../../image/term/SigmoidProduct.svg)              	| ![](../../image/hedge/Seldom-SigmoidProduct.svg)                                                           	| ![](../../image/hedge/Somewhat-SigmoidProduct.svg)                                                              	| ![](../../image/hedge/Very-SigmoidProduct.svg)                                                          	| ![](../../image/hedge/Extremely-SigmoidProduct.svg)                                                               	| ![](../../image/hedge/Not-SigmoidProduct.svg)                                                         	| ![](../../image/hedge/Any-SigmoidProduct.svg)                                                         	|
    | [fuzzylite.term.Spike][]                              	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Spike][]                                                        	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Spike][]                                                           	| [fuzzylite.hedge.Very][] [fuzzylite.term.Spike][]                                                       	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Spike][]                                                            	| [fuzzylite.hedge.Not][] [fuzzylite.term.Spike][]                                                      	| [fuzzylite.hedge.Any][] [fuzzylite.term.Spike][]                                                      	|
    | ![](../../image/term/Spike.svg)                       	| ![](../../image/hedge/Seldom-Spike.svg)                                                                    	| ![](../../image/hedge/Somewhat-Spike.svg)                                                                       	| ![](../../image/hedge/Very-Spike.svg)                                                                   	| ![](../../image/hedge/Extremely-Spike.svg)                                                                        	| ![](../../image/hedge/Not-Spike.svg)                                                                  	| ![](../../image/hedge/Any-Spike.svg)                                                                  	|
    | [fuzzylite.term.Arc][]                                	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Arc][]                                                          	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Arc][]                                                             	| [fuzzylite.hedge.Very][] [fuzzylite.term.Arc][]                                                         	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Arc][]                                                              	| [fuzzylite.hedge.Not][] [fuzzylite.term.Arc][]                                                        	| [fuzzylite.hedge.Any][] [fuzzylite.term.Arc][]                                                        	|
    | ![](../../image/term/Arc.svg)                         	| ![](../../image/hedge/Seldom-Arc.svg)                                                                      	| ![](../../image/hedge/Somewhat-Arc.svg)                                                                         	| ![](../../image/hedge/Very-Arc.svg)                                                                     	| ![](../../image/hedge/Extremely-Arc.svg)                                                                          	| ![](../../image/hedge/Not-Arc.svg)                                                                    	| ![](../../image/hedge/Any-Arc.svg)                                                                    	|
    | [fuzzylite.term.Binary][]                             	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Binary][]                                                       	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Binary][]                                                          	| [fuzzylite.hedge.Very][] [fuzzylite.term.Binary][]                                                      	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Binary][]                                                           	| [fuzzylite.hedge.Not][] [fuzzylite.term.Binary][]                                                     	| [fuzzylite.hedge.Any][] [fuzzylite.term.Binary][]                                                     	|
    | ![](../../image/term/Binary.svg)                      	| ![](../../image/hedge/Seldom-Binary.svg)                                                                   	| ![](../../image/hedge/Somewhat-Binary.svg)                                                                      	| ![](../../image/hedge/Very-Binary.svg)                                                                  	| ![](../../image/hedge/Extremely-Binary.svg)                                                                       	| ![](../../image/hedge/Not-Binary.svg)                                                                 	| ![](../../image/hedge/Any-Binary.svg)                                                                 	|
    | [fuzzylite.term.Concave][]                            	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Concave][]                                                      	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Concave][]                                                         	| [fuzzylite.hedge.Very][] [fuzzylite.term.Concave][]                                                     	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Concave][]                                                          	| [fuzzylite.hedge.Not][] [fuzzylite.term.Concave][]                                                    	| [fuzzylite.hedge.Any][] [fuzzylite.term.Concave][]                                                    	|
    | ![](../../image/term/Concave.svg)                     	| ![](../../image/hedge/Seldom-Concave.svg)                                                                  	| ![](../../image/hedge/Somewhat-Concave.svg)                                                                     	| ![](../../image/hedge/Very-Concave.svg)                                                                 	| ![](../../image/hedge/Extremely-Concave.svg)                                                                      	| ![](../../image/hedge/Not-Concave.svg)                                                                	| ![](../../image/hedge/Any-Concave.svg)                                                                	|
    | [fuzzylite.term.Ramp][]                               	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Ramp][]                                                         	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Ramp][]                                                            	| [fuzzylite.hedge.Very][] [fuzzylite.term.Ramp][]                                                        	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Ramp][]                                                             	| [fuzzylite.hedge.Not][] [fuzzylite.term.Ramp][]                                                       	| [fuzzylite.hedge.Any][] [fuzzylite.term.Ramp][]                                                       	|
    | ![](../../image/term/Ramp.svg)                        	| ![](../../image/hedge/Seldom-Ramp.svg)                                                                     	| ![](../../image/hedge/Somewhat-Ramp.svg)                                                                        	| ![](../../image/hedge/Very-Ramp.svg)                                                                    	| ![](../../image/hedge/Extremely-Ramp.svg)                                                                         	| ![](../../image/hedge/Not-Ramp.svg)                                                                   	| ![](../../image/hedge/Any-Ramp.svg)                                                                   	|
    | [fuzzylite.term.Sigmoid][]                            	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.Sigmoid][]                                                      	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.Sigmoid][]                                                         	| [fuzzylite.hedge.Very][] [fuzzylite.term.Sigmoid][]                                                     	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.Sigmoid][]                                                          	| [fuzzylite.hedge.Not][] [fuzzylite.term.Sigmoid][]                                                    	| [fuzzylite.hedge.Any][] [fuzzylite.term.Sigmoid][]                                                    	|
    | ![](../../image/term/Sigmoid.svg)                     	| ![](../../image/hedge/Seldom-Sigmoid.svg)                                                                  	| ![](../../image/hedge/Somewhat-Sigmoid.svg)                                                                     	| ![](../../image/hedge/Very-Sigmoid.svg)                                                                 	| ![](../../image/hedge/Extremely-Sigmoid.svg)                                                                      	| ![](../../image/hedge/Not-Sigmoid.svg)                                                                	| ![](../../image/hedge/Any-Sigmoid.svg)                                                                	|
    | [fuzzylite.term.SShape][] - [fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Seldom][] fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Somewhat][] [fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Very][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Very][] [fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Extremely][] [fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Not][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Not][] [fuzzylite.term.ZShape][] 	| [fuzzylite.hedge.Any][] [fuzzylite.term.SShape][] - [fuzzylite.hedge.Any][] [fuzzylite.term.ZShape][] 	|
    | ![](../../image/term/ZShape - SShape.svg)             	| ![](../../image/hedge/Seldom-ZShape - SShape.svg)                                                          	| ![](../../image/hedge/Somewhat-ZShape - SShape.svg)                                                             	| ![](../../image/hedge/Very-ZShape - SShape.svg)                                                         	| ![](../../image/hedge/Extremely-ZShape - SShape.svg)                                                              	| ![](../../image/hedge/Not-ZShape - SShape.svg)                                                        	| ![](../../image/hedge/Any-ZShape - SShape.svg)                                                        	|
    | [fuzzylite.term.SShape][]                             	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.SShape][]                                                       	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.SShape][]                                                          	| [fuzzylite.hedge.Very][] [fuzzylite.term.SShape][]                                                      	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.SShape][]                                                           	| [fuzzylite.hedge.Not][] [fuzzylite.term.SShape][]                                                     	| [fuzzylite.hedge.Any][] [fuzzylite.term.SShape][]                                                     	|
    | ![](../../image/term/SShape.svg)                      	| ![](../../image/hedge/Seldom-SShape.svg)                                                                   	| ![](../../image/hedge/Somewhat-SShape.svg)                                                                      	| ![](../../image/hedge/Very-SShape.svg)                                                                  	| ![](../../image/hedge/Extremely-SShape.svg)                                                                       	| ![](../../image/hedge/Not-SShape.svg)                                                                 	| ![](../../image/hedge/Any-SShape.svg)                                                                 	|
    | [fuzzylite.term.ZShape][]                             	| [fuzzylite.hedge.Seldom][] [fuzzylite.term.ZShape][]                                                       	| [fuzzylite.hedge.Somewhat][] [fuzzylite.term.ZShape][]                                                          	| [fuzzylite.hedge.Very][] [fuzzylite.term.ZShape][]                                                      	| [fuzzylite.hedge.Extremely][] [fuzzylite.term.ZShape][]                                                           	| [fuzzylite.hedge.Not][] [fuzzylite.term.ZShape][]                                                     	| [fuzzylite.hedge.Any][] [fuzzylite.term.ZShape][]                                                     	|
    | ![](../../image/term/ZShape.svg)                      	| ![](../../image/hedge/Seldom-ZShape.svg)                                                                   	| ![](../../image/hedge/Somewhat-ZShape.svg)                                                                      	| ![](../../image/hedge/Very-ZShape.svg)                                                                  	| ![](../../image/hedge/Extremely-ZShape.svg)                                                                       	| ![](../../image/hedge/Not-ZShape.svg)                                                                 	| ![](../../image/hedge/Any-ZShape.svg)                                                                 	|

    info: related
        - [fuzzylite.hedge.Not][]
        - [fuzzylite.hedge.Seldom][]
        - [fuzzylite.hedge.Somewhat][]
        - [fuzzylite.hedge.Very][]
        - [fuzzylite.hedge.Extremely][]
        - [fuzzylite.hedge.Any][]
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Consequent][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __str__(self) -> str:
        """Return the name of the hedge.

        Returns:
            name of the hedge.
        """
        return self.name

    def __repr__(self) -> str:
        """Return the Python code to construct the hedge.

        Returns:
            Python code to construct the hedge.
        """
        return representation.as_constructor(self)

    @property
    def name(self) -> str:
        """Return the name of the hedge.

        Returns:
            name of the hedge.
        """
        return self.__class__.__name__.lower()

    @abstractmethod
    def hedge(self, x: Scalar) -> Scalar:
        """Implement the hedge for the membership function value $x$.

        Args:
            x: membership function value

        Returns:
           hedge of $x$.
        """


class Any(Hedge):
    """Special hedge that always returns `1.0`.

    The antecedent of a rule considers `Any` to be a syntactically special hedge because it is not
    followed by a term (e.g., `if Variable is any then...` vs `if Variable is very term then...`)

    The hedge is useful for better documenting rules.

    info: related
        - [fuzzylite.rule.Antecedent][]
        - [fuzzylite.rule.Rule][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        """Return scalar of same shape of `x` filled with `1.0`.

        Args:
            x: irrelevant except for its shape

        Returns:
            $h(x)=1.0$
        """
        x = scalar(x)
        y = np.full_like(x, 1.0)
        return y


class Extremely(Hedge):
    r"""Hedge that modifies the membership function value of a term as follows.

    Note: Equation
        $h(x) = \begin{cases}
            2x^2 & \mbox{if } x \le 0.5 \cr
            1-2(1-x)^2 & \mbox{otherwise} \cr
            \end{cases}$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Extremely}(x)$.

        Args:
             x: membership function value

        Returns:
            $h(x) = \begin{cases} 2x^2 & \mbox{if } x \le 0.5 \cr 1-2(1-x)^2 & \mbox{otherwise} \cr \end{cases}$
        """
        x = scalar(x)
        y = np.where(x <= 0.5, 2 * x**2, 1 - 2 * (1 - x) ** 2)
        return y


class Not(Hedge):
    """Hedge that modifies the membership function value of a term by.

    Note: Equation
        $h(x) = 1-x$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Not}(x)$.

        Args:
            x: membership function value

        Returns:
             $h(x) = 1-x$
        """
        x = scalar(x)
        y = 1 - x
        return y


class Seldom(Hedge):
    r"""Hedge that modifies the membership function value of a term as follows.

    Note: Equation
        $h(x) = \begin{cases}
            \sqrt{\dfrac{x}{2}} & \mbox{if } x \le 0.5 \cr
            1-\sqrt{\dfrac{1-x}{2}} & \mbox{otherwise}
        \end{cases}$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Seldom(x)}$.

        Args:
            x: membership function value

        Returns:
            $h(x) = \begin{cases} \sqrt{\dfrac{x}{2}} & \mbox{if $x \le 0.5$} \cr 1-\sqrt{\dfrac{(1-x)}{2}} & \mbox{otherwise}\cr \end{cases}$
        """
        x = scalar(x)
        y = np.where(x <= 0.5, np.sqrt(0.5 * x), 1 - np.sqrt(0.5 * (1 - x)))
        return y


class Somewhat(Hedge):
    r"""Hedge that modifies the membership function value of a term by.

    Note: Equation
        $h(x) = \sqrt{x}$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Somewhat}(x)$.

        Args:
            x: membership function value

        Returns:
            $h(x) = \sqrt{x}$
        """
        x = scalar(x)
        y = np.sqrt(x)
        return y


class Very(Hedge):
    r"""Hedge that modifies the membership function value of a term by.

    Note: Equation
        $h(x) = x^2$

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\text{Very}(x)$.

        Args:
            x: membership function value

        Returns:
             $h(x) = x^2$
        """
        x = scalar(x)
        y = x**2
        return y


class HedgeLambda(Hedge):
    r"""Hedge that modifies the membership function value of a term according to a $\lambda$ function.

    This hedge is not registered with the HedgeFactory because the $\lambda$ function cannot be easily configured.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.hedge.HedgeFunction][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __init__(self, name: str, function: Callable[[Scalar], Scalar]) -> None:
        r"""Constructor.

        Args:
            name: name of the hedge
            function: $\lambda$ function.
        """
        self._name = name
        self.function = function

    @property
    def name(self) -> str:
        """Get the name of the hedge.

        Returns:
            name of the hedge
        """
        return self._name

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $\lambda(x)$.

        Args:
            x: membership function value

        Returns:
            $h(x) = \lambda(x)$
        """
        return self.function(x)


class HedgeFunction(Hedge):
    r"""Hedge that modifies the membership function value of a term according to the term Function.

    This hedge is not registered with the HedgeFactory because the Function cannot be easily configured.

    info: related
        - [fuzzylite.hedge.Hedge][]
        - [fuzzylite.hedge.HedgeLambda][]
        - [fuzzylite.factory.HedgeFactory][]
    """

    def __init__(self, function: Function) -> None:
        """Constructor.

        Args:
            function: function $f$.
        """
        self.function = function

    @property
    def name(self) -> str:
        """Get the name of the hedge.

        Returns:
            name of the hedge
        """
        return self.function.name

    def hedge(self, x: Scalar) -> Scalar:
        r"""Compute $f(x)$.

        Args:
            x: membership function value

        Returns:
            $h(x) = f(x)$
        """
        return self.function.membership(x)
