# pyfuzzylite&trade;

***

<img src="image/fuzzylite.svg" align="left" alt="fuzzylite">

## A Fuzzy Logic Control Library in Python

by [Juan Rada-Vilela](https://fuzzylite.com/jcrada), Ph.D.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://opensource.org/license/gpl-3-0/)
[![License: Paid](https://img.shields.io/badge/License-proprietary-blue)](mailto:sales@fuzzylite.com)

***

[main branch](https://github.com/fuzzylite/pyfuzzylite/tree/main)  
[![Python package](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml/badge.svg?branch=main)](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml)  
[![Coverage Status](
https://coveralls.io/repos/github/fuzzylite/pyfuzzylite/badge.svg?branch=main)](
https://coveralls.io/github/fuzzylite/pyfuzzylite?branch=main)

[development branch](https://github.com/fuzzylite/pyfuzzylite/tree/development)  
[![Python package](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml/badge.svg?branch=development)](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml)  
[![Coverage Status](
https://coveralls.io/repos/github/fuzzylite/pyfuzzylite/badge.svg?branch=development)](
https://coveralls.io/github/fuzzylite/pyfuzzylite?branch=development)

***

## <a name="license">License</a>

`pyfuzzylite` is dual-licensed under the [**GNU GPL 3.0**](https://opensource.org/license/gpl-3-0/) and under a
**proprietary license for commercial purposes**.

You are **strongly** encouraged to support the development of the FuzzyLite Libraries by purchasing a license
of [`QtFuzzyLite`](https://fuzzylite.com/downloads).

[`QtFuzzyLite`](https://fuzzylite.com/downloads/) is the best graphical user interface available to easily design and
directly operate fuzzy logic controllers in real time. Available for Windows, Mac, and Linux, its goal is to
significantly **speed up** the design of your fuzzy logic controllers, while providing a very **useful**, **functional**
and **beautiful** user interface.
Please, download it and check it out for free at [fuzzylite.com/downloads](https://fuzzylite.com/downloads).

## <a name="fuzzylite">FuzzyLite</a>

**The FuzzyLite Libraries for Fuzzy Logic Control** refer to [`fuzzylite`](https://github.com/fuzzylite/fuzzylite/)
(C++), [`pyfuzzylite`](https://github.com/fuzzylite/pyfuzzylite/) (Python),
and [`jfuzzylite`](https://github.com/fuzzylite/jfuzzylite/) (Java).

The **goal** of the FuzzyLite Libraries is to **easily** design and **efficiently** operate fuzzy logic controllers
following an **object-oriented** programming model with minimal dependency on external libraries.

## <a name="features">Features</a>

**(6) Controllers**: Mamdani, Takagi-Sugeno, Larsen, Tsukamoto, Inverse Tsukamoto, Hybrids

**(23) Linguistic terms**:  (5) *Basic*: triangle, trapezoid, rectangle, discrete, semi-ellipse.
(9) *Extended*: bell, cosine, gaussian, gaussian product, pi-shape, sigmoid difference, sigmoid product, spike.
(6) *Edges*: arc, binary, concave, ramp, sigmoid, s-shape, z-shape.
(3) *Functions*: constant, linear, function.

**(7) Activation methods**:  general, proportional, threshold, first, last, lowest, highest.

**(8) Conjunction and Implication (T-Norms)**: minimum, algebraic product, bounded difference, drastic product, einstein
product, hamacher product, nilpotent minimum, function.

**(10) Disjunction and Aggregation (S-Norms)**:  maximum, algebraic sum, bounded sum, drastic sum, einstein sum,
hamacher sum, nilpotent maximum, normalized sum, unbounded sum, function.

**(7) Defuzzifiers**:  (5) *Integral*: centroid, bisector, smallest of maximum, largest of maximum, mean of maximum.
(2) *Weighted*: weighted average, weighted sum.

**(7) Hedges**: any, not, extremely, seldom, somewhat, very, function.

**(3) Importers**: FuzzyLite Language `fll`, Fuzzy Inference System `fis`, Fuzzy Control Language `fcl`.

**(7) Exporters**: `C++`, `Java`, FuzzyLite Language `fll`, FuzzyLite Dataset `fld`, `R` script, Fuzzy Inference
System `fis`, Fuzzy Control Language `fcl`.

**(30+) Examples**  of Mamdani, Takagi-Sugeno, Tsukamoto, and Hybrid controllers from `fuzzylite`, Octave, and Matlab,
each included in the following formats: `C++`, `Java`, `fll`, `fld`, `R`, `fis`, and `fcl`.

## <a name="examples">Examples</a>

```python title="pyfuzzylite"
import fuzzylite as fl

engine = fl.Engine(
    name="ObstacleAvoidance",
    input_variables=[
        fl.InputVariable(
            name="obstacle",
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=[fl.Ramp("left", 1.0, 0.0), fl.Ramp("right", 0.0, 1.0)],
        )
    ],
    output_variables=[
        fl.OutputVariable(
            name="mSteer",
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            lock_previous=False,
            default_value=fl.nan,
            aggregation=fl.Maximum(),
            defuzzifier=fl.Centroid(resolution=100),
            terms=[fl.Ramp("left", 1.0, 0.0), fl.Ramp("right", 0.0, 1.0)],
        )
    ],
    rule_blocks=[
        fl.RuleBlock(
            name="mamdani",
            conjunction=None,
            disjunction=None,
            implication=fl.AlgebraicProduct(),
            activation=fl.General(),
            rules=[
                fl.Rule.create("if obstacle is left then mSteer is right"),
                fl.Rule.create("if obstacle is right then mSteer is left"),
            ],
        )
    ],
)
```

```yaml title="FuzzyLite Language"
#File: ObstacleAvoidance.fll
Engine: ObstacleAvoidance
InputVariable: obstacle
  enabled: true
  range: 0.000 1.000
  lock-range: false
  term: left Ramp 1.000 0.000
  term: right Ramp 0.000 1.000
OutputVariable: mSteer
  enabled: true
  range: 0.000 1.000
  lock-range: false
  aggregation: Maximum
  defuzzifier: Centroid 100
  default: nan
  lock-previous: false
  term: left Ramp 1.000 0.000
  term: right Ramp 0.000 1.000
RuleBlock: mamdani
  enabled: true
  conjunction: none
  disjunction: none
  implication: AlgebraicProduct
  activation: General
  rule: if obstacle is left then mSteer is right
  rule: if obstacle is right then mSteer is left
```

***

## <a name="reference">Reference</a>

If you are using the FuzzyLite Libraries, please cite the following reference in your article:

Juan Rada-Vilela. The FuzzyLite Libraries for Fuzzy Logic Control, 2018. URL https://fuzzylite.com.

```bibtex title="bibtex"
 @misc{fl::fuzzylite,
     author={Juan Rada-Vilela},
     title={The FuzzyLite Libraries for Fuzzy Logic Control},
     url={https://fuzzylite.com},
     year={2018}
 }
```

fuzzylite&reg; is a registered trademark of FuzzyLite Limited  
jfuzzylite&trade; is a trademark of FuzzyLite Limited  
pyfuzzylite&trade; is a trademark of FuzzyLite Limited  
QtFuzzyLite&trade; is a trademark of FuzzyLite Limited

Copyright &copy; 2010-2023 FuzzyLite Limited. All rights reserved.
