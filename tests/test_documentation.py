import shutil
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
        target_module = Path(f"/tmp/fl/docs/{name}")
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
    result = []
    for module_name, components in mkdocs.items():
        result.append(f"- {module_name}:")
        result.extend(
            f"  - {component}: fuzzylite/{module_name}/{component}.md"
            for component in components
        )
    mkdocs_yaml = "\n".join(result)
    return mkdocs_yaml


if __name__ == "__main__":
    print(generate_documentation())
