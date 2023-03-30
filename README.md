
pyfuzzylite&trade; 
==================
<img src="https://raw.githubusercontent.com/fuzzylite/pyfuzzylite/master/fuzzylite.png" align="right" alt="fuzzylite">


A Fuzzy Logic Control Library in Python
---------------------------------------

By: [Juan Rada-Vilela](https://www.fuzzylite.com/jcrada), Ph.D.


[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://opensource.org/license/gpl-3-0/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit/)


[main branch](https://github.com/fuzzylite/pyfuzzylite/tree/main)  
[![](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml)   

[development branch](https://github.com/fuzzylite/pyfuzzylite/tree/development)  
[![](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml/badge.svg?branch=development)](https://github.com/fuzzylite/pyfuzzylite/actions/workflows/python-package.yml) 

***


### <a name="license">License</a>
`pyfuzzylite` is dual-licensed under the [**GNU GPL 3.0**](https://opensource.org/license/gpl-3-0/) and the [**MIT License**](https://opensource.org/license/mit/).

You are **strongly** encouraged to support the development of the FuzzyLite Libraries by purchasing a license of [`QtFuzzyLite 6`](https://www.fuzzylite.com/downloads).

[`QtFuzzyLite 6`](https://www.fuzzylite.com/downloads/) is the best graphical user interface available to  easily design and directly operate fuzzy logic controllers in real time. Available for Windows, Mac, and Linux, its goal is to significantly **speed up** the design of your fuzzy logic controllers, while providing a very **useful**, **functional** and **beautiful** user interface.
Please, download it and check it out for free at [www.fuzzylite.com/downloads/](https://www.fuzzylite.com/downloads/).




### <a name="introduction">Introduction</a>


`fuzzylite` is a free and open-source fuzzy logic control library programmed in C++ for multiple platforms (e.g., Windows, Linux, Mac, iOS). [`jfuzzylite`](https://github.com/fuzzylite/jfuzzylite/) is the equivalent library for Java and Android platforms. [`pyfuzzylite`](https://github.com/fuzzylite/pyfuzzylite/) is the equivalent library for Python. Together, they are **The FuzzyLite Libraries for Fuzzy Logic Control**.



 The **goal** of the FuzzyLite Libraries is to **easily** design and **efficiently** operate fuzzy logic controllers following an **object-oriented** programming model **without** relying on external libraries.


#### Reference
If you are using the FuzzyLite Libraries, please cite the following reference in your article:

Juan Rada-Vilela. The FuzzyLite Libraries for Fuzzy Logic Control, 2018. URL https://fuzzylite.com/.

```bibtex
 @misc{fl::fuzzylite,
 author={Juan Rada-Vilela},
 title={The FuzzyLite Libraries for Fuzzy Logic Control},
 url={https://fuzzylite.com/},
 year={2018}}
```

#### Documentation
The documentation for the `fuzzylite` library is available at: [fuzzylite.com/documentation/](https://fuzzylite.com/documentation/).

#### Contributing
All contributions are welcome, provided they follow the following guidelines:
 - Pull requests are made to the [development](https://github.com/fuzzylite/pyfuzzylite/tree/development) branch
 - Source code is consistent with standards in the library
 - Contribution is appropriately documented and tested, raising issues where appropriate
 - License of the contribution is waived to match the licenses of the FuzzyLite Libraries




### <a name="features">Features</a>

**(6) Controllers**: Mamdani, Takagi-Sugeno, Larsen, Tsukamoto, Inverse Tsukamoto, Hybrids

**(21) Linguistic terms**:  (4) *Basic*: triangle, trapezoid, rectangle, discrete.
(9) *Extended*: bell, cosine, gaussian, gaussian product, pi-shape, sigmoid difference, sigmoid product, spike.
(5) *Edges*: binary, concave, ramp, sigmoid, s-shape, z-shape.
(3) *Functions*: constant, linear, function.

**(7) Activation methods**:  general, proportional, threshold, first, last, lowest, highest.

**(8) Conjunction and Implication (T-Norms)**: minimum, algebraic product, bounded difference, drastic product, einstein product, hamacher product, nilpotent minimum, function.

**(10) Disjunction and Aggregation (S-Norms)**:  maximum, algebraic sum, bounded sum, drastic sum, einstein sum, hamacher sum, nilpotent maximum, normalized sum, unbounded sum,  function.

**(7) Defuzzifiers**:  (5) *Integral*: centroid, bisector, smallest of maximum, largest of maximum, mean of maximum.
(2) *Weighted*: weighted average, weighted sum.

**(7) Hedges**: any, not, extremely, seldom, somewhat, very, function.

**(3) Importers**: FuzzyLite Language `fll`, Fuzzy Inference System `fis`, Fuzzy Control Language `fcl`.

**(7) Exporters**: `C++`, `Java`, FuzzyLite Language `fll`, FuzzyLite Dataset `fld`, `R` script, Fuzzy Inference System `fis`, Fuzzy Control Language `fcl`.

**(30+) Examples**  of Mamdani, Takagi-Sugeno, Tsukamoto, and Hybrid controllers from `fuzzylite`, Octave, and Matlab, each included in the following formats: `C++`, `Java`, `fll`, `fld`, `R`, `fis`, and `fcl`.





### <a name="example">Example</a>
#### FuzzyLite Language
```yaml
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



fuzzylite&reg; is a registered trademark of FuzzyLite Limited.
jfuzzylite&trade; is a trademark of FuzzyLite Limited.
pyfuzzylite&trade; is a trademark of FuzzyLite Limited.
QtFuzzyLite&trade; is a trademark of FuzzyLite Limited.


Copyright &#xa9; 2010-2023 FuzzyLite Limited. All rights reserved.
