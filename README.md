<div align="center" markdown=1>
<img src="/fuzzylite.png" alt="fuzzylite" width="10%">
<h1>pyfuzzylite 8.0.4</h1>
<h2>A Fuzzy Logic Control Library in Python</h2>
<h3>by <a href="https://fuzzylite.com/about"><b>Juan Rada-Vilela, PhD</b></a></h3>

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://opensource.org/license/gpl-3-0/)
[![License: Paid](https://img.shields.io/badge/License-proprietary-blue)](mailto:sales@fuzzylite.com)
[![Coverage Status](
https://coveralls.io/repos/github/fuzzylite/pyfuzzylite/badge.svg?branch=main)](
https://coveralls.io/github/fuzzylite/pyfuzzylite?branch=main)  
[![Build](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/build.yml/badge.svg)](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/build.yml)
[![Test](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/test.yml/badge.svg)](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/test.yml)
[![Publish](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/publish.yml/badge.svg)](
https://github.com/fuzzylite/pyfuzzylite/actions/workflows/publish.yml)

</div>

## <a name="fuzzylite">FuzzyLite</a>

**The FuzzyLite Libraries for Fuzzy Logic Control** refer to [`fuzzylite`](https://github.com/fuzzylite/fuzzylite/)
(C++), [`pyfuzzylite`](https://github.com/fuzzylite/pyfuzzylite/) (Python),
and [`jfuzzylite`](https://github.com/fuzzylite/jfuzzylite/) (Java).

The **goal** of the FuzzyLite Libraries is to **easily** design and **efficiently** operate fuzzy logic controllers
following an **object-oriented** programming model with minimal dependency on external libraries.

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

## <a name="install">Install</a>

```commandline
pip install pyfuzzylite
```

## <a name="features">Features</a>

**Documentation**: [fuzzylite.github.io/pyfuzzylite/](https://fuzzylite.github.io/pyfuzzylite/)

**(6) Controllers**: Mamdani, Takagi-Sugeno, Larsen, Tsukamoto, Inverse Tsukamoto, Hybrid

**(25) Linguistic terms**:  (5) *Basic*: Triangle, Trapezoid, Rectangle, Discrete, SemiEllipse.
(8) *Extended*: Bell, Cosine, Gaussian, GaussianProduct, PiShape, SigmoidDifference, SigmoidProduct, Spike.
(7) *Edges*: Arc, Binary, Concave, Ramp, Sigmoid, SShape, ZShape.
(3) *Functions*: Constant, Linear, Function. (2) *Special*: Aggregated, Activated.

**(7) Activation methods**:  General, Proportional, Threshold, First, Last, Lowest, Highest.

**(9) Conjunction and Implication (T-Norms)**: Minimum, AlgebraicProduct, BoundedDifference, DrasticProduct,
EinsteinProduct, HamacherProduct, NilpotentMinimum, LambdaNorm, FunctionNorm.

**(11) Disjunction and Aggregation (S-Norms)**:  Maximum, AlgebraicSum, BoundedSum, DrasticSum, EinsteinSum,
HamacherSum, NilpotentMaximum, NormalizedSum, UnboundedSum, LambdaNorm, FunctionNorm.

**(7) Defuzzifiers**:  (5) *Integral*: Centroid, Bisector, SmallestOfMaximum, LargestOfMaximum, MeanOfMaximum.
(2) *Weighted*: WeightedAverage, WeightedSum.

**(7) Hedges**: Any, Not, Extremely, Seldom, Somewhat, Very, Function.

**(3) Importers**: FuzzyLite Language `fll`. With `fuzzylite`: Fuzzy Inference System `fis`, Fuzzy Control
Language `fcl`.

**(7) Exporters**: `Python`, FuzzyLite Language `fll`, FuzzyLite Dataset `fld`. With `fuzzylite`: `C++`, `Java`,
FuzzyLite Language `fll`, FuzzyLite Dataset `fld`, `R` script, Fuzzy Inference System `fis`, Fuzzy Control
Language `fcl`.

**(30+) Examples**  of Mamdani, Takagi-Sugeno, Tsukamoto, and Hybrid controllers from `fuzzylite`, Octave, and Matlab,
each included in the following formats: `py`, `fll`, `fld`. With `fuzzylite`: `C++`, `Java`, `R`, `fis`, and `fcl`.

## <a name="examples">Examples</a>

### FuzzyLite Language

```yaml
# File: examples/mamdani/ObstacleAvoidance.fll
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

```python
# Python
import fuzzylite as fl

engine = fl.FllImporter().from_file("examples/mamdani/ObstacleAvoidance.fll")
```

### Python

```python
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

### `float` and vectorization

```python
# single `float` operation
engine.input_variable("obstacle").value = 0.5
engine.process()
print("y =", engine.output_variable("mSteer").value)
# > y = 0.5
print("ỹ =", engine.output_variable("mSteer").fuzzy_value())
# > ỹ = 0.500/left + 0.500/right

# vectorization
engine.input_variable("obstacle").value = fl.array([0, 0.25, 0.5, 0.75, 1.0])
engine.process()
print("y =", repr(engine.output_variable("mSteer").value))
# > y = array([0.6666665 , 0.62179477, 0.5       , 0.37820523, 0.3333335 ])
print("ỹ =", repr(engine.output_variable("mSteer").fuzzy_value()))
# > ỹ = array(['0.000/left + 1.000/right',
#              '0.250/left + 0.750/right',
#              '0.500/left + 0.500/right',
#              '0.750/left + 0.250/right',
#              '1.000/left + 0.000/right'], dtype='<U26')
```

Please refer to the documentation for more
information: [**fuzzylite.github.io/pyfuzzylite/**](https://fuzzylite.github.io/pyfuzzylite/)

## <a name="contributing">Contributing</a>

All contributions are welcome, provided they follow the following guidelines:

- Source code is consistent with standards in the library
- Contribution is properly documented and tested, raising issues where appropriate
- Contribution is licensed under the FuzzyLite License

## <a name="reference">Reference</a>

If you are using the FuzzyLite Libraries, please cite the following reference in your article:

> Juan Rada-Vilela. The FuzzyLite Libraries for Fuzzy Logic Control, 2018. URL https://fuzzylite.com.

Or using `bibtex`:

```bibtex
@misc{fl::fuzzylite,
    author={Juan Rada-Vilela},
    title={The FuzzyLite Libraries for Fuzzy Logic Control},
    url={https://fuzzylite.com},
    year={2018}
}
```

***

fuzzylite&reg; is a registered trademark of FuzzyLite <br>
jfuzzylite&trade;, pyfuzzylite&trade; and QtFuzzyLite&trade; are trademarks of FuzzyLite
