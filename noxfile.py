"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited
fuzzylite is a registered trademark of FuzzyLite Limited.
"""

import os

import nox

nox.options.sessions = ["check", "freeze", "install", "lint", "test"]


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """Check the `pyproject.toml` is valid."""
    session.run(*"poetry check".split(), external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """Run code formatting."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]

    session.run(*"black --version".split(), external=True)
    session.run("black", *files, external=True)

    session.run(*"ruff --version".split(), external=True)
    session.run(*"ruff --fix".split(), *files, external=True)


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run static code analysis and checks format is correct."""
    linters = {
        "black": "lint_black",
        "ruff": "lint_ruff",
        "pyright": "lint_pyright",
        "mypy": "lint_mypy",
        "qodana": "lint_qodana",
    }
    # Case 1: posargs is empty, run all linters
    # Case 2: posargs only contains -qodana, run all linters except qodana (cloud linting)
    # Case 3: posargs is not empty, run only the specified linters
    posargs = list(session.posargs)
    if "-qodana" in posargs:
        posargs.remove("-qodana")
        linters.pop("qodana")
    if not posargs:
        posargs = list(linters.keys())
    for linter in posargs:
        session.notify(linters[linter])


@nox.session(python=False)
def lint_black(session: nox.Session) -> None:
    """Run black linter."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    session.run(*"black --version".split(), external=True)
    session.run(*"black --check".split(), *files, external=True)


@nox.session(python=False)
def lint_ruff(session: nox.Session) -> None:
    """Run ruff linter."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    session.run(*"ruff check".split(), *files, external=True)
    session.run(*"ruff --version".split(), external=True)


@nox.session(python=False)
def lint_pyright(session: nox.Session) -> None:
    """Run pyright linter."""
    session.run(*"pyright --version".split(), external=True)
    session.run("pyright", external=True)


@nox.session(python=False)
def lint_mypy(session: nox.Session) -> None:
    """Run mypy linter."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    session.run(*"mypy --version".split(), external=True)
    session.run("mypy", *files, external=True)


@nox.session(python=False)
def lint_qodana(session: nox.Session) -> None:
    """Run qodana linter."""
    if "QODANA_TOKEN" not in os.environ:
        session.warn(
            "Qodana linting failed to run because environment variable 'QODANA_TOKEN' is not present"
        )
    else:
        token = os.environ["QODANA_TOKEN"] or ""
        session.run(
            *" ".join(
                [
                    "qodana scan",
                    "--source-directory fuzzylite/",
                    "--results-dir .qodana/",
                    "--baseline .qodana/baseline/qodana.sarif.json",
                    "--clear-cache",
                ]
            ).split(),
            env={"QODANA_TOKEN": token},
            external=True,
        )


@nox.session(python=False)
def install(session: nox.Session) -> None:
    """Install the project using poetry."""
    session.run(*"poetry install -v --no-interaction".split(), external=True)


@nox.session(python=False)
def install_upgrade(session: nox.Session) -> None:
    """Install the project using poetry and upgraded dependencies."""
    session.run(*"poetry lock -v".split(), external=True)
    session.run(*"poetry install -v --no-interaction".split(), external=True)


@nox.session(python=False)
def freeze(session: nox.Session) -> None:
    """Print all the versions of dependencies."""
    session.run(*"poetry show".split(), external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """Run the tests in the project."""
    session.run(*"coverage run -m pytest".split(), external=True)
    session.run(*"coverage report".split(), external=True)


@nox.session(python=False)
def benchmark(session: nox.Session) -> None:
    """Run the benchmarks."""
    if "codspeed" in session.posargs:
        if "CODSPEED_TOKEN" not in os.environ:
            session.warn(
                "Codspeed benchmark failed to run because environment variable 'CODSPEED_TOKEN' is not present"
            )
        else:
            token = os.environ.get("CODSPEED_TOKEN") or ""
            session.run(
                *"pytest tests/test_benchmark_codspeed.py".split(),
                external=True,
                env={"CODSPEED_TOKEN": token},
            )
    else:
        session.run(*"pytest tests/test_benchmark_pytest.py".split(), external=True)


@nox.session
def test_publish(session: nox.Session) -> None:
    """Build the distributable and upload it to testpypi."""
    session.run(*"rm -rf dist/".split(), external=True)
    session.run(*"poetry build".split(), external=True)
    session.run(*"twine check --strict dist/*".split(), external=True)
    session.run(
        *"twine upload --repository testpypi dist/* --config-file .pypirc --verbose".split(),
        external=True,
    )


@nox.session
def publish(session: nox.Session) -> None:
    """Build the distributable and upload it to pypi."""
    session.run(*"rm -rf dist/".split(), external=True)
    session.run(*"poetry build".split(), external=True)
    session.run(*"twine check --strict dist/*".split(), external=True)
    session.run(
        *"twine upload --repository pypi dist/* --config-file .pypirc --verbose".split(),
        external=True,
    )
