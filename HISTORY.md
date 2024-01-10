# Version 8.0.0

## Summary

### New

- integration with `numpy` means engines operate more efficiently thanks to vectorization, but the regular operation
  with Python `float` also works seamlessly
- `fuzzylite/types.py` to store library annotation types
- type `fl.Scalar = float | numpy.Array` replaces `float` everywhere vectorization can be used
  (see `fuzzylite/types.py`)
- type `fl.ScalarArray = numpy.Array[float]` used in places where an array of floats is needed (
  see `fuzzylite/types.py`)
- function `fl.scalar` to convert any argument to a `fl.Scalar`
- function `fl.array` maps to `np.array`
- function `fl.to_float` converts any argument to a `float`, which was the behaviour of `fl.Op.scalar`.
- linguistic terms `Arc` and `SemiEllipse`
- indexable components:
    - `Engine`: get variables or rule blocks by name using square brackets, eg, `engine["light"].value`
    - `Variable`: get terms by name using square brackets, eg, `variable["low"].membership(value)`
    - `Factory`: get constructors and objects by name using square brackets,
      eg, `factory["Triangle"]()`, `factory["sin"](3.14)`
- class `Benchmark` to benchmark engines
- `from __future__ import annotations` in every file to use better type annotations
- class `library.Settings` to configure general settings in singleton `library.settings`
- class `library.Representation` to easily convert objects to Python code
- documentation significantly improved and configured using `mkdocs-material`, available
  at: [fuzzylite.github.io/pyfuzzylite](https://fuzzylite.github.io/pyfuzzylite)
- function `__len__` in `Variable`, `RuleBlock` to get number of components with `len(x)`,
  but also means that implicit boolean statements like `if component` will evaluate to `False` when
  `component is None` or `len(component) == 0`
- tsukamoto functions `Sigmoid.tsukamoto`, `SShape.tsukamoto`, `ZShape.tsukamoto`
- tests, tests and more tests, reaching 95% of code coverage

### Changed

- Requires Python >= 3.9
- Dual license: GNU GPL and paid proprietary license for commercial purposes
- All enum values are automatically assigned with enum.auto(), instead of manual assignments
- Almost all classes redefine `__repr__` to return the object as a Python constructor using the
  new `fl.library.Representation` class
- Almost all classes redefine `__str__` to return the equivalent FuzzyLite Language (if possible)
- Many more tests and better structure for tests
- Aggregation operator is now used in TakagiSugeno/Tsukamoto engines on activations of the same term
  (see `tests/test_engine.py:test_takagi_sugeno_aggregation_operator`)
- Examples: all file names have been changed to `snake_case`, and engines wrapped in classes
- Examples: all examples can be easily imported,
  eg `from fuzzylite.examples.mamdani.obstacle_avoidance import ObstacleAvoidance`
- Documentation updated and format changed from `doxygen` to `google` style
- `library.settings` for default absolute tolerance changed from `1e-5` to `1e-3` to match the default 3 decimals
- Formatting: code `line-width=100` instead of `88`
- Many functions now map or use to `numpy` functions to support vectorization
- The following classes are now
  abstract: `Activation`, `Defuzzifier`, `Exporter`, `Hedge`, `Importer`, `Norm`, `SNorm`, `TNorm`, `Expression`
  and `Term`
- `IntegralDefuzzifier`s are simpler, support vectorization, use `numpy`, and the default resolution is 1000
- `WeightedDefuzzifier`s are simpler and use `Aggregated.grouped_terms` to iterate over the terms and aggregate them
  accordingly
- Get components by name or index with `Engine.input_variable()`, `Engine.output_variable()` or `Engine.rule_block()`
- `PythonExporter` exports code to Python using single-statement constructors
- `FldExporter` uses vectorization to export datasets, so for now only works with the `General` activation method
- Operators and functions in `FunctionFactory` map to their equivalent methods in `numpy`
- Vectorization support in class `activation.General`, and
  modules `defuzzifier`, `engine`, `hedge`, `norm`, `rule`, `term`, `variable`
- type `Rule.triggered` to an array of bools to support vectorization
- type `Rule.activation_degree` to `Scalar` to support vectorization
- renamed parameters of `Op.scale` from `from_[minimum|maximum]` to `x_[min|max]`, and `to_[minimum|maximum]`
  to `y_[min|max]`
- function `Term.discretize` uses `Op.midpoints` to discretize
- function `Term.discretize` uses `resolution = 10` instead of `100`
- function `Aggregated.activation_degree` to support vectorization, so now returns `fl.Scalar`
- function `fuzzy_value` returns `Array[np.str_]` instead of `str` to account for vectorization
- function `highest_membership` returns `Activated | None` instead of `tuple[float, Term | None]`

### Removed

- class `library.Library` removed and split into `library.Settings` and `library.Information`
- singleton `fl.lib`. See `library.py` for settings
- functions `fl.isnan` and `fl.isinf` moved to `Op` (use now: `fl.Op.isnan`, `fl.Op.isinf`)
- function `class_name`
  from `Activation`, `Defuzzifier`, `Exporter`, `Importer`, `ConstructionFactory`, `CloningFactory`, `Norm`, `Term`;
  use `Op.class_name` instead
- parameter `decimals` from function `Op.str()`
- class `Discrete.Pair`
- `pyhamcrest` dependency for tests

### Bug fixes

- Bug fix: `Rule.antecedent` and `Rule.consequent` instantiations in `Rule` constructor
- Bug fix: `Function.update_reference` loads function if not loaded
- Bug fix: function `Term.tsukamoto` uses the height in all monotonic terms
- Bug fix: function `Concave.tsukamoto` uses parameter `y` instead of incorrectly
  computing `membership(activation_degree)`
- Bug fix: function `Ramp.tsukamoto` uses height correctly

## Details

### __init__.py

- New: import `import benchmark`
- Changed: library `__name__ = "fuzzylite"`, instead of the incorrectly `pyfuzzylite`
- Changed: constants and functions `fl.inf`, `fl.nan`, `fl.isnan`, `fl.isinf` now map to their `numpy` equivalents
- Removed: functions `fl.isnan` and `fl.isinf` moved to `Op` (use now: `fl.Op.isnan`, `fl.Op.isinf`)
- Removed: properties `fl.inf` and `fl.nan` moved to `library.py`
- Removed: singleton `fl.lib`. See `library.py` for settings

### activation.py

- New: vectorization supported only in `General` activation method, other methods require float operations (unchanged)
- Changed: class `Activation` is abstract
- Removed: function `Activation.class_name`, use now `Op.class_name` when needed

### benchmark.py

- New: class `Benchmark` to measure performance time and errors of engines

### defuzzifier.py

- Bug fix: Weighted defuzzifiers infer the type correctly now (see WeightedDefuzzifier::infer_type)
- Changed: classes `Defuzzifier`, `IntegralDefuzzifier` and `WeightedDefuzzifier` are abstract
- Changed: all `IntegralDefuzzifier`s use vectors and numpy to compute defuzzifications
- Changed: default value `IntegralDefuzzifier.resolution = 1000` instead of `100`
- Changed: Simplified algorithms of all defuzzifiers
- Changed: values of `WeightedDefuzzifier.Type` are `enum.auto()` instead of manual assignments
- Changed: `WeightedDefuzzifier`s compute weighted sum and average on `Aggregated.grouped_terms()` instead of all terms
  (see `Aggregated` term and `tests/test_engine.py:test_takagi_sugeno_aggregation_operator`)
- Removed: function `Defuzzifier.class_name`, use now `Op.class_name` when needed

### engine.py

- New: properties `Engine.input_values` and `Engine.output_values` produce 2D matrices with their respective values
- New: property setter `Engine.input_values` provides shortcuts to setting input variable values
- New: constructor parameter `load:bool=True` to automatically update the reference of all terms in variables and load
  the rules
- New: function `Engine.__getitem__` to get variables or ruleblocks by name using square brackets,
  eg `engine["inputVariable1"].value`
- New: function `Engine.infer_type` to infer type of engine based on its configuration
- New: function `Engine.is_ready` to check whether engine is ready to operate
- New: function `Engine.copy` to deep copy an engine
- Changed: values of `Engine.Type` are `enum.auto()` instead of manual assignments
- Changed: function signatures for `Engine.[input_variable | output_variable | rule_block]` from `str` to `str|int`
  to also get the components by index
- Changed: function `Engine.restart` also reloads the rules of all rule blocks

### exporter.py

- New: setting `PythonExporter.formatted` formats the resulting code using `black` if it is available
- New: setting `PythonExporter.encapsulated` encapsulates code in a class if object is an engine, or in a method if
  object is anything else
- Changed: class `Exporter` is abstract
- Changed: class `PythonExporter` exports engine in a single-statement constructor
- Changed: constructor `PythonExporter` constructor signature changed from `indent: str`
  to `formatted: bool, encapsulated: bool`
- Changed: Examples in Python are wrapped in classes to avoid instantiation when importing the library
- Changed: Examples in Python are converted to single-statement constructor
- Changed: `PythonExporter` uses `library.Representation` to export objects to Python code
- Changed: `FldExporter` uses vectorization to export the dataset
- Removed: function `Exporter.class_name`, use now `Op.class_name` when needed

### factory.py

- New: components `Arc` and `SemiEllipse` added to `TermFactory`
- New: function `ConstructionFactory.import_from` automatically imports all the classes from the given `fuzzylite`
  module
- New: function `ConstructionFactory.construct` takes variadic parameters to configure the constructor
- New: function `[Construction|Cloning]Factory.__getitem__` to get constructors and objects using square brackets,
  eg `factory["Triangle"]`
- New: function `[Construction|Cloning]Factory.__len__` to get number of constructors and objects using `len(factory)`
- Changed: constructor `ConstructionFactory` takes an optional dictionary of constructors
- Changed: constructor `CloningFactory` takes an optional dictionary of objects
- Changed: operators and functions in `FunctionFactory` map to their equivalent methods in numpy, but preserving their
  names
- Removed: function `[Construction|Cloning]Factory.class_name`, use now `Op.class_name` when needed

### hedges.py

- Changed: class `Hedge` is abstract
- Changed: all hedges work with vectorization

### importer.py

- Changed: class `Importer` is abstract
- Removed: function `Importer.class_name`, use instead `Op.class_name`
- Removed: type `FllImporter.T`

### library.py

- New: functions `fl.scalar`, `fl.to_float`, and `fl.array`
- New: class `Settings` for the library configuration, and a `settings` singleton used across the library
- New: default `Settings.float_type` for the type of floats across the library is `np.float64`, instead of the previous
  Python
  `float`
- New: setting `Settings.debugging` is a property to get and set the debugging for logging purposes
- New: setting `Settings.rtol` for relative tolerance across the library is 0.0
- New: setting `Settings.atol` for absolute tolerance across the library is 1e-03, instead of the previous
  `abs_tolerance=1e-05`
- New: function `Settings.context` to change settings within a specific context (
  eg, `with settings.context(decimals=10): ...`)
- New: setting `Settings.alias` to change the alias of fuzzylite when exporting to Python (eg, `import fuzzylite as fl`)
- New: class `Information` that contains information about the library, and an `information` to access the information
- New: class `Representation` to represent objects as constructors in Python
- New: `fl.library.repr` is a shortcut to use the representation class
- New: class `Representation` contains a FuzzyLite Language Exporter
- New: `fl.inf` and `fl.nan` map to `numpy.inf` and `numpy.nan`, respectively
- Removed: class `Library`, use now `library.settings` and `library.information` instead

### norm.py

- Changed: class `Norm`, `SNorm`, and `TNorm` are abstract
- Changed: Vectorized all norms
- Removed: function `Norm.class_name`, use now `Op.class_name` when needed

### operation.py

- New: functions `Op.isinf` and `Op.isnan` map to numpy functions, respectively
- New: function `Op.midpoints` to discretize a range of values into a list of midpoints
- New: function `Op.is_close` to compare two values with absolute and relative tolerance given by the library settings
- New: function `Op.class_name` to get the class name of the given object
- New: function `Op.glob_examples` to glob over the `fuzzylite` examples
- New: function `Op.snake_case` to convert text to `snake_case`
- New: function `Op.pascal_case` to convert text to `PascalCase`
- New: function `Op.class_name` to get the class name of any `fuzzylite` object
- New: function `Op.to_fll` to convert any `fuzzylite` object to the FuzzyLite Language
- Changed: renamed parameters of `scale` from `from_[minimum|maximum]` to `x_[min|max]`, and `to_[minimum|maximum]`
  to `y_[min|max]`
- Changed: Vectorized all comparative operations (eq, neq, lt, le, gt, ge)
- Changed: function `Op.as_identifier` now returns `_` instead of `unnamed`
- Removed: parameter `decimals` from function `Op.str()`
- Removed: parameter tolerance from all comparative functions `eq`, `neq`, `lt`, `le`, `gt`, `ge`
- Removed: function `pi`
- Removed: parameter `slots` in function `describe`
- Removed: function `scalar`, moved to `library.py`

### rule.py

- New: vectorization in rules
- New: function `RuleBlock.__getitem__` to get rules by index (or slicing) using square brackets `rb[0]`
- New: function `RuleBlock.__len__` to get the number of rules using `len(rb)`
- New: function `RuleBlock.__iter__` to make the rule block iterable
- Changed: class `Expression` is abstract
- Changed: type `Rule.triggered` to an array of bools to support vectorization
- Changed: type `Rule.activation_degree` to `Scalar` to support vectorization
- Bug fix: `Rule.antecedent` and `Rule.consequent` instantiations in constructor

### term.py

#### General

- New: All terms are vectorized
- New: classes `Arc` and `SemiEllipse` for linguistic terms
- New: `Term._parse` method to easily parse configuration parameters
- New: function `Term._parameters` with variadic arguments to exclude `height` if `Op.is_close(height, 1.0)`
- New: function `Activated.fuzzy_value` to convert activation to an `Array[np.str_]`
- New: property `Activated.degree` is a property that automatically replaces `nan` and `inf` values
- New: `Aggregated.grouped_terms()` groups multiple activations of the same term and aggregates them using the
  aggregation operator (or `UnboundedSum` in its absence)
- New: functions `Discrete.to_list`, `Discrete.to_dict`, and `Discrete.to_xy`
- New: function `Discrete.create` to create a Discrete term from a list, dict or tuple of lists
- New: tsukamoto functions `Sigmoid.tsukamoto`, `SShape.tsukamoto`, `ZShape.tsukamoto`

- Changed: simplified all the terms' equations and they now use `numpy`
- Changed: class `Term` is abstract
- Changed: function `Term.membership` is abstract
- Changed: function signature `Term.tsukamoto` takes only `y: Scalar` as parameter
- Changed: function `Term.discretize` uses `Op.midpoints` to discretize
- Changed: function `Term.discretize` uses `resolution = 10` instead of `100`
- Changed: `Term.tsukamoto` raises exception if the term is not monotonic, before it returned the membership function
  value
- Changed: function `Aggregated.activation_degree` to support vectorization, so now returns `fl.Scalar`
- Changed: function `Aggregated.highest_activated_term` raises `ValueError` if using vectorization
- Changed: constructor `Discrete` to also take a `ScalarArray` to support vectorization
- Changed: return type of `Discrete.[x|y]()` to `ScalarArray` to support vectorization
- Changed: `Discrete.values` are now a `numpy` array instead of list of `Discrete.Pair`
- Changed: function `Discrete.membership` interpolates with numpy, which does not extrapolate and instead takes the
  first (or last) y-value for x-values out of range
- Changed: function `GaussianProduct.membership` to reuse `Gaussian` membership functions on computation
- Changed: constructor parameter type `coefficients` in `Linear` to take `Sequence` instead of `Iterable`
- Changed: function `PiShape.membership` to reuse `SShape` and `ZShape` membership functions on computation
- Changed: function `SigmoidDifference` to reuse `Sigmoid` membership function on computation
- Changed: function `SigmoidProduct` to reuse `Sigmoid` membership function on computation

- Removed: function `Term.class_name`, use `Op.class_name` when needed
- Removed: parameter `bounded_mf` from function `Term.discretize`
- Removed: class `Discrete.Pair`
- Removed: Discrete.pairs_from, Discrete.values_from, Discrete.dict_from, Discrete.list_from
- Removed: type `bytes` from `Discrete.Floatable`

- Bug fix: `Function.update_reference` loads function if not loaded
- Bug fix: function `Term.tsukamoto` uses the height in all monotonic terms
- Bug fix: function `Concave.tsukamoto` uses parameter `y` instead of incorrectly
  computing `membership(activation_degree)`
- Bug fix: function `Ramp.tsukamoto` uses height correctly

### types.py (new)

- New: type `fl.Scalar` for float and numpy arrays
- New: type `fl.ScalarArray` for numpy arrays of floats
- New: type `fl.Array` (mapping to `numpy.NDArray`) for numpy arrays of specific types

### variable.py

- New: function `__getitem__` to get terms by index or name using square brackets, eg, `variable[0]`
  or `variable["low"]`
- Changed: function `fuzzy_value` returns `Array[np.str_]` instead of `str` to account for vectorization
- Changed: function `highest_membership` returns `Activated | None` instead of `tuple[float, Term | None]`
- Changed: function `OutputVariable.defuzzify` is simpler and works with vectorized operations

### noxfile.py

- New: session `lint_pyright` to do static code analysis with `pyright`
- New: session `lint_markdown` to lint markdown files
- New: session `benchmark` to run benchmarks online (`codspeed`) or offline (`pytest`)
- New: session `docs` to build documentation and (optionally) publish it to GitHub Pages
- New: default sessions to run tests when using bare `nox`
- Changed: simplified command texts

### pyproject.toml

- New: dependency `numpy`
- New: configuration for `pyright`
- New: dependencies for development only
- Changed: configurations to support Python 3.9 instead of 3.7
- Removed: `pyhamcrest` dependency for tests

### documentation

- New: documentation built with `mkdocs-material` and hosted
  at  [fuzzylite.github.io/pyfuzzylite](https://fuzzylite.github.io/pyfuzzylite)
- Changed: documentation style is now `google` instead of `doxygen`
- Changed: documentation significantly improved and updated
