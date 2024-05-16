"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import fuzzylite as fl


def generate_documentation() -> str:
    """Generate markdown files in `/tmp/fl/docs` from the modules and exports of fuzzylite.

    @return mkdocs configuration linking to generated markdown files.
    """
    modules = [
        module
        for module in fl.Op.glob_examples("module", module=fl)
        if not module.__name__.startswith("fuzzylite.examples")
    ]

    target = Path("/tmp/fl/docs/")
    if target.exists():
        shutil.rmtree(target)

    mkdocs: dict[str, list[str]] = {}
    for module in modules:
        name = module.__name__.replace(".", "/")  # eg, fuzzylite/activation
        target_module = target / name
        target_module.mkdir(parents=True, exist_ok=True)
        if target_module.stem not in mkdocs:
            mkdocs[target_module.stem] = []
        for component in sorted(module.__all__):
            documentation = target_module.joinpath(component + ".md")
            if documentation.exists():
                # eg, library/information vs library/Information
                documentation = target_module.joinpath("_" + component + ".md")
            documentation.write_text(f"::: {module.__name__}.{component}")
            mkdocs[target_module.stem].append(component)

        all_module = target / "fuzzylite" / "__all__"
        all_module.mkdir(parents=True, exist_ok=True)
        all_module.joinpath(f"{target_module.stem}.md").write_text(
            "\n".join(f"::: {module.__name__}.{component}" for component in sorted(module.__all__))
        )

    result = []

    result.append("- __all__:")
    result.extend(f"  - {module}: fuzzylite/__all__/{module}.md" for module in mkdocs)

    for module_name, components in mkdocs.items():
        result.append(f"- {module_name}:")
        result.extend(
            f"  - {component}: fuzzylite/{module_name}/{component}.md" for component in components
        )
    mkdocs_yaml = "\n".join(result)
    return mkdocs_yaml


class GenerateDocumentation(unittest.TestCase):
    """Generate the documentation using `mkdocs` and `mkdocstrings`."""

    def test_generate_documentation(self) -> None:
        """Generate the documentation in /tmp/fl/docs and print the mkdocs index."""
        print(generate_documentation())


if __name__ == "__main__":
    unittest.main()
